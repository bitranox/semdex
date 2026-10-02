#!/usr/bin/env python3
"""Sweep each ANN store's own tuning knobs and report its recall-latency frontier.

Every approximate-store number published here was taken at whatever the driver happened to
default to. That makes the store comparison a comparison of DEFAULTS: a store that looks weak may
simply be untuned, and the speed it appears to win is bought at a recall cost that tuning could
partly buy back. One point is not a frontier.

Two kinds of knob, and the difference decides the cost of measuring them:

* **Query-time** (``nprobes``, ``refine_factor``, ``ef_search``) change per search, so the index
  is built ONCE and re-queried at every level. Cheap, and the frontier a deployment can move
  along without reindexing.
* **Build-time** (``m``, ``ef_construction``, ``num_partitions``) are baked into the index, so
  each level costs a full rebuild. Swept at fewer levels for that reason, and reported
  separately, because they are a different decision - made once, at index time.

Recall is measured against the exact top-k from the same tested kernel the rest of these
benchmarks use, over cached real embeddings. Synthetic vectors would understate both recall and
latency badly, because real embeddings cluster.

Usage::

    python scripts/score_ann_frontier.py                       # query-time sweep, lancedb
    python scripts/score_ann_frontier.py --stores lancedb,pgvector --phase both
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _provenance import source_git_sha
from _score_kernel import topk_stream
from preembed_vectors import cache_root, read_json
from score_chunk_sweep import dirsafe, metrics, query_vectors
from score_through_store import chunk_identity, fill_store, query_store, summarize

from semdex.composition import build_vector_store
from semdex.domain.ann_tuning import AnnParams
from semdex.domain.enums import StoreBackend
from semdex.domain.models import Collection

_COLLECTION = "annfront"
_OUT = Path(__file__).resolve().parents[1] / "tests" / "benchmarks" / "raw" / "ann-frontier.json"

# Query-time levels per store. `None` is the driver default and is measured explicitly, because
# it is the setting every published number so far was taken at and therefore the reference point
# the frontier has to be read against.
_QUERY_GRID: dict[StoreBackend, list[AnnParams]] = {
    StoreBackend.LANCEDB: [
        AnnParams(),
        AnnParams(nprobes=1),
        AnnParams(nprobes=5),
        AnnParams(nprobes=10),
        AnnParams(nprobes=20),
        AnnParams(nprobes=40),
        AnnParams(nprobes=80),
        AnnParams(nprobes=10, refine_factor=5),
        AnnParams(nprobes=20, refine_factor=5),
        AnnParams(nprobes=40, refine_factor=10),
    ],
    StoreBackend.PGVECTOR: [
        AnnParams(),
        AnnParams(ef_search=10),
        AnnParams(ef_search=20),
        AnnParams(ef_search=40),
        AnnParams(ef_search=100),
        AnnParams(ef_search=200),
        AnnParams(ef_search=400),
    ],
    StoreBackend.MARIADB: [
        AnnParams(),
        AnnParams(ef_search=10),
        AnnParams(ef_search=20),
        AnnParams(ef_search=40),
        AnnParams(ef_search=100),
        AnnParams(ef_search=200),
    ],
}

# Build-time levels. Each entry costs a full index rebuild, so the grids are deliberately short.
_BUILD_GRID: dict[StoreBackend, list[AnnParams]] = {
    StoreBackend.LANCEDB: [AnnParams(), AnnParams(num_partitions=64), AnnParams(num_partitions=512)],
    StoreBackend.PGVECTOR: [
        AnnParams(),
        AnnParams(m=8, ef_construction=32),
        AnnParams(m=32, ef_construction=200),
    ],
    StoreBackend.MARIADB: [AnnParams(), AnnParams(m=8), AnnParams(m=32)],
}

# The query-time level used while a build-time level is being judged: the driver default, so the
# two axes are never moved at once.
_BUILD_PROBE = AnnParams()


@dataclass(frozen=True, slots=True)
class Cell:
    """One cached (corpus, profile, embedding) cell, loaded once and reused by every store."""

    corpus: str
    profile: str
    embedding: str
    dim: int
    rows: int
    vectors: Path
    uris: list[str]
    ordinals: list[int]
    qids: list[str]
    queries: np.ndarray
    qrels: dict[str, dict[str, int]]


def load_cell(corpus: str, profile: str, embedding: str) -> Cell | None:
    """Load a cached cell, or None when it was never embedded."""
    cache = cache_root()
    vroot = cache / "vectors" / f"{corpus}__{profile}__{dirsafe(embedding)}"
    meta = read_json(vroot / "meta.json")
    if not meta or "count" not in meta:
        return None
    chunks_dir = cache / "chunks" / f"{corpus}__{profile}"
    queries_text = read_json(chunks_dir / "queries.json") or {}
    qrels = read_json(chunks_dir / "qrels.json") or {}
    uris, ordinals = chunk_identity(chunks_dir / "chunks.parquet")
    qids, matrix = query_vectors(corpus, embedding, meta["model_id"], None, queries_text)
    return Cell(
        corpus=corpus,
        profile=profile,
        embedding=embedding,
        dim=int(meta["dim"]),
        rows=int(meta["count"]),
        vectors=vroot / "vectors.npy",
        uris=uris,
        ordinals=ordinals,
        qids=qids,
        queries=matrix,
        qrels=qrels,
    )


def exact_documents(cell: Cell, k: int, fetch: int) -> dict[str, list[str]]:
    """The exact top-k documents per query, from the shared streamed kernel."""
    index, _score = topk_stream(cell.vectors, cell.queries, fetch=fetch)
    reference: dict[str, list[str]] = {}
    for row, qid in enumerate(cell.qids):
        seen: list[str] = []
        for position in index[row]:
            uri = cell.uris[int(position)]
            if uri not in seen:
                seen.append(uri)
            if len(seen) >= k:
                break
        reference[qid] = seen
    return reference


def recall_against(reference: dict[str, list[str]], candidate: dict[str, list[str]], k: int) -> float:
    """Share of the exact top-k documents the store actually returned, averaged over queries."""
    scores: list[float] = []
    for qid, want in reference.items():
        target = set(want[:k])
        if not target:
            continue
        got = set(candidate.get(qid, [])[:k])
        scores.append(len(target & got) / len(target))
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def _label(params: AnnParams) -> str:
    """A human-readable name for a knob setting; the empty setting is the driver default."""
    parts = [
        f"{name}={value}"
        for name, value in (
            ("nprobes", params.nprobes),
            ("refine", params.refine_factor),
            ("ef_search", params.ef_search),
            ("m", params.m),
            ("ef_construction", params.ef_construction),
            ("partitions", params.num_partitions),
        )
        if value is not None
    ]
    return " ".join(parts) if parts else "driver default"


def _merge(build: AnnParams, query: AnnParams) -> AnnParams:
    """Build-time and query-time knobs combined into the single object the factory takes."""
    return AnnParams(
        nprobes=query.nprobes,
        refine_factor=query.refine_factor,
        ef_search=query.ef_search,
        m=build.m,
        ef_construction=build.ef_construction,
        num_partitions=build.num_partitions,
    )


def _open_store(backend: StoreBackend, work: Path, dsn: str | None, params: AnnParams) -> Any:
    return build_vector_store(backend, work, dsn=dsn, ann_params=params)


def _store_bytes(work: Path) -> int:
    return sum(f.stat().st_size for f in work.rglob("*") if f.is_file())


def measure_point(
    store: Any,
    cell: Cell,
    reference: dict[str, list[str]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Query the store as configured and score recall, quality and latency from the same run."""
    rankings, latencies = query_store(store, cell.queries, cell.qids, config)
    per_query = {qid: metrics(rankings[qid], cell.qrels.get(qid, {}), config["k"]) for qid in cell.qids}
    ordered = sorted(latencies)
    return {
        "recall_vs_exact": recall_against(reference, rankings, config["k"]),
        "search_p50_ms": round(_percentile(ordered, 0.50), 2),
        "search_p95_ms": round(_percentile(ordered, 0.95), 2),
        **summarize(per_query),
    }


def _percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    position = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[position] * 1000


def sweep_backend(
    backend: StoreBackend,
    cell: Cell,
    reference: dict[str, list[str]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Every knob level for one store: build once per build level, re-query per query level."""
    rows: list[dict[str, Any]] = []
    dsn = config["dsn"].get(backend.value)
    work_root = Path(config["work"]) / backend.value
    build_levels = _BUILD_GRID[backend] if config["phase"] in ("build", "both") else [AnnParams()]
    for build in build_levels:
        query_levels = _QUERY_GRID[backend] if config["phase"] in ("query", "both") else [_BUILD_PROBE]
        # A build level is judged at the driver's own query default, so the two axes never move
        # together; only the default build level gets the full query sweep.
        if build != AnnParams() and config["phase"] == "both":
            query_levels = [_BUILD_PROBE]
        shutil.rmtree(work_root, ignore_errors=True)
        work_root.mkdir(parents=True, exist_ok=True)
        store = _open_store(backend, work_root, dsn, _merge(build, _BUILD_PROBE))
        store.ensure_collection(Collection(name=_COLLECTION, model_id=f"annf-{cell.dim}", dim=cell.dim))
        build_s = fill_store(store, cell.vectors, cell.uris, cell.ordinals, config["batch"], collection=_COLLECTION)
        size = _store_bytes(work_root)
        store.close()
        for query in query_levels:
            reopened = _open_store(backend, work_root, dsn, _merge(build, query))
            point = measure_point(reopened, cell, reference, config)
            reopened.close()
            rows.append(
                {
                    "corpus": cell.corpus,
                    "embedding": cell.embedding,
                    "dim": cell.dim,
                    "rows": cell.rows,
                    "store": backend.value,
                    "build_level": _label(build),
                    "query_level": _label(query),
                    "build_s": round(build_s, 1),
                    "store_mb": round(size / 1_000_000, 1),
                    **point,
                }
            )
            print(
                f"    {backend.value:9s} build[{_label(build):22s}] query[{_label(query):24s}] "
                f"recall={point['recall_vs_exact']:.3f} ndcg={point['ndcg@10']:.4f} "
                f"p50={point['search_p50_ms']}ms",
                flush=True,
            )
        shutil.rmtree(work_root, ignore_errors=True)
    return rows


def _environment() -> dict[str, Any]:
    return {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "semdex_git_sha": source_git_sha(),
        "host": platform.node(),
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stores", default="lancedb")
    parser.add_argument("--phase", choices=("query", "build", "both"), default="query")
    parser.add_argument("--corpus", default="mldr_en_8k_slice")
    parser.add_argument("--profile", default="recursive-t256-o0-gpt2")
    parser.add_argument("--embedding", default="fastembed:bge-base")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--fetch", type=int, default=200)
    parser.add_argument("--batch", type=int, default=5000)
    parser.add_argument("--pgvector-dsn", default=os.environ.get("SEMDEX_BENCH_PGVECTOR_DSN"))
    parser.add_argument("--mariadb-dsn", default=os.environ.get("SEMDEX_BENCH_MARIADB_DSN"))
    parser.add_argument("--work", type=Path, default=cache_root() / "tmp-annfront")
    parser.add_argument("--out", type=Path, default=_OUT)
    parser.add_argument("--merge", action="store_true", help="fold into an existing --out")
    return parser.parse_args(argv)


def _row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["corpus"], row["embedding"], row["store"], row["build_level"], row["query_level"])


def merge_rows(existing: list[dict[str, Any]], fresh: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold a run into an earlier one, newest measurement winning per knob setting."""
    by_key = {_row_key(row): row for row in existing}
    by_key.update({_row_key(row): row for row in fresh})
    return sorted(by_key.values(), key=lambda row: [str(v) for v in _row_key(row)])


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cell = load_cell(args.corpus, args.profile, args.embedding)
    if cell is None:
        print(f"[annf] not embedded: {args.corpus}__{args.profile}__{args.embedding}", file=sys.stderr)
        return 2
    print(f"[annf] {args.corpus} {cell.rows:,} rows dim {cell.dim}, {len(cell.qids)} queries", flush=True)

    dsn_by_store: dict[str, str | None] = {"pgvector": args.pgvector_dsn, "mariadb": args.mariadb_dsn}
    config: dict[str, Any] = {
        "k": args.k,
        "fetch": args.fetch,
        "batch": args.batch,
        "work": args.work,
        "phase": args.phase,
        "collection": _COLLECTION,
        "dsn": dsn_by_store,
    }
    reference = exact_documents(cell, args.k, args.fetch)
    print("[annf] exact reference built", flush=True)

    rows: list[dict[str, Any]] = []
    for name in [s.strip() for s in args.stores.split(",") if s.strip()]:
        backend = StoreBackend(name)
        if backend.value in dsn_by_store and not dsn_by_store[backend.value]:
            print(f"[annf] skip {backend.value}: no dsn given", file=sys.stderr)
            continue
        rows.extend(sweep_backend(backend, cell, reference, config))

    payload: dict[str, Any] = {**_environment(), "k": args.k, "points": rows}
    if args.merge and args.out.exists():
        previous = json.loads(args.out.read_text(encoding="utf-8"))
        payload["points"] = merge_rows(previous.get("points") or [], rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out} ({len(payload['points'])} points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
