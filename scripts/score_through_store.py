#!/usr/bin/env python
# pyright: basic
"""Score retrieval quality THROUGH a real vector store, not through an exact numpy scan.

Every quality number elsewhere in this benchmark set is produced by an exact brute-force cosine
scan, and every store number is latency, disk and recall-against-exact. The two halves are never
composed, so nobody has measured the nDCG a user actually gets out of an ANN store - and it cannot
be inferred by multiplying them, because the documents an approximate index misses are not a
random sample of the ranking. A recall of 0.8 does not imply 80 percent of the quality: an index
that drops the rank-1 document costs far more than one that drops rank 9.

So this runs the SAME eval queries through the SAME cached vectors, loaded into an actual store,
and reports quality and latency from one run, in one row.

The exact stores are the control. sqlite_vec scans exhaustively, so its nDCG must equal the
kernel's to within float noise; if it does not, the harness is wrong and no ANN number from the
same run can be trusted. That check runs every time rather than being assumed.

Env:
  CACHE_ROOT               cache dir (default /embeddings)
  SEMDEX_STOREQ_CORPORA    comma list (default the two MLDR slices)
  SEMDEX_STOREQ_PROFILES   comma list (default recursive-t256-o0-gpt2)
  SEMDEX_STOREQ_EMBEDDINGS comma list (default fastembed:bge-base)
  SEMDEX_STOREQ_STORES     comma list (default sqlite_vec,lancedb)
  SEMDEX_STOREQ_K          reported top-k (default 10)
  SEMDEX_STOREQ_FETCH      chunks fetched per query before dedup (default 200)
  SEMDEX_STOREQ_BATCH      upsert batch size (default 5000)
  SEMDEX_STOREQ_WORK       scratch dir for the stores (default <cache>/tmp-storeq)
  SEMDEX_STOREQ_OUT        results json (default <cache>/scores/store_quality.json)
  SEMDEX_STOREQ_FORCE      =1 to re-score cells already present
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _provenance import stamped
from _score_kernel import topk_stream
from _score_stats import bootstrap_ci
from preembed_vectors import _cache_root, _read_json
from score_chunk_sweep import _dirsafe, _metrics, _query_vectors

from semdex.adapters.config.vectorstore import VectorStoreConfig
from semdex.composition import build_vector_store
from semdex.domain.ann_tuning import AnnParams
from semdex.domain.enums import StoreBackend
from semdex.domain.models import Chunk, Collection, SourceRef

_METRICS = ("ndcg@10", "recall@10", "mrr", "p@1")
_COLLECTION = "storeq"
# sqlite_vec and json scan every row, so their ranking is exact by construction and must match the
# kernel. Anything else is approximate and is what this measurement exists to price.
_EXACT_BACKENDS = {StoreBackend.SQLITE_VEC, StoreBackend.JSON}
# How far an exhaustive store may differ from the kernel before the harness is suspect.
# "Exact" guarantees identical SCORES, not an identical order among equal ones, and duplicate
# chunk text is common: 73 of nfcorpus's 323 queries have an exact tie inside their top 20 and 2
# are tied across the top-10 boundary, which moves nDCG@10 by about 1e-5. A real wiring fault -
# the wrong vectors, the wrong queries, a broken uri mapping - is orders of magnitude larger, so
# this threshold still catches one while ignoring tie order.
_EXACT_TOLERANCE = 1e-3


def chunk_identity(parquet: Path) -> tuple[list[str], list[int]]:
    """Per chunk row, its document uri and ordinal - never the text, which is not needed here."""
    uris: list[str] = []
    ordinals: list[int] = []
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=8192, columns=["source_uri", "ordinal"]):
        data = batch.to_pydict()
        uris.extend(data["source_uri"])
        ordinals.extend(data["ordinal"])
    return uris, ordinals


def _chunks_for(uris: list[str], ordinals: list[int], start: int, stop: int) -> list[Chunk]:
    """Domain chunks carrying their REAL document uri, so a hit maps straight to a document."""
    return [
        Chunk(
            text="",
            source=SourceRef(uri=uris[i], label="", content_hash="", mtime=0.0),
            ordinal=ordinals[i],
            token_count=1,
        )
        for i in range(start, stop)
    ]


def shipped_ann_params(backend: StoreBackend) -> AnnParams:
    """The ANN knobs a stock semdex install actually uses for this backend.

    Taken from the config model's own defaults rather than named here, so the table cannot drift
    away from what ships. It used to build stores with no params at all, which measured each
    DRIVER's default instead - a different thing since lancedb's `balanced` preset stopped being
    the bare driver default, and the reason the published lancedb rows understated it.
    """
    return VectorStoreConfig().resolved_ann_params(backend)


def _describe(params: AnnParams) -> str:
    """The ANN setting a row was measured at, for the raw file and the published table."""
    parts = [
        f"{name}={value}"
        for name, value in (
            ("nprobes", params.nprobes),
            ("refine_factor", params.refine_factor),
            ("ef_search", params.ef_search),
        )
        if value is not None
    ]
    return " ".join(parts) if parts else "driver default"


def _load_store(backend: StoreBackend, work: Path, dim: int) -> Any:
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    store = build_vector_store(backend, work, ann_params=shipped_ann_params(backend))
    store.ensure_collection(Collection(name=_COLLECTION, model_id=f"storeq-{dim}", dim=dim))
    return store


def fill_store(
    store: Any,
    vectors_path: Path,
    uris: list[str],
    ordinals: list[int],
    batch: int,
    *,
    collection: str = _COLLECTION,
) -> float:
    """Stream the cached vectors into the store; never hold the whole cell in memory.

    ``collection`` is a parameter rather than this module's constant so a sibling harness can
    reuse this without silently having to name its collection "storeq" too.
    """
    mm = np.load(vectors_path, mmap_mode="r")
    started = time.perf_counter()
    for start in range(0, int(mm.shape[0]), batch):
        stop = min(start + batch, int(mm.shape[0]))
        block = np.asarray(mm[start:stop], dtype=np.float32)
        store.upsert(
            collection=collection,
            chunks=_chunks_for(uris, ordinals, start, stop),
            vectors=[row.tolist() for row in block],
        )
    store.compact(collection=collection)  # upsert only appends; compact builds the ANN index
    return time.perf_counter() - started


def _documents(hits: Any, k: int) -> list[str]:
    seen: list[str] = []
    for hit in hits:
        uri = str(getattr(hit, "uri", "") or "")
        if uri and uri not in seen:
            seen.append(uri)
            if len(seen) >= k:
                break
    return seen


def query_store(
    store: Any, queries: np.ndarray, qids: list[str], config: dict[str, Any]
) -> tuple[dict[str, list[str]], list[float]]:
    rankings: dict[str, list[str]] = {}
    latencies: list[float] = []
    for position, qid in enumerate(qids):
        vector = queries[position].tolist()
        started = time.perf_counter()
        hits = store.query(collection=config.get("collection", _COLLECTION), vector=vector, k=config["fetch"])
        latencies.append(time.perf_counter() - started)
        rankings[qid] = _documents(hits, config["k"])
    latencies.sort()
    return rankings, latencies


def _percentile(values: list[float], q: float) -> float:
    return values[min(len(values) - 1, int(q * len(values)))] * 1000 if values else 0.0


def _summarize(per_query: dict[str, dict[str, float]]) -> dict[str, Any]:
    row: dict[str, Any] = {"n_queries": len(per_query)}
    for metric in _METRICS:
        stats = bootstrap_ci([scores[metric] for scores in per_query.values()])
        row[metric] = round(stats["mean"], 6)
        row[f"{metric}_ci_lo"] = round(stats["ci_lo"], 6)
        row[f"{metric}_ci_hi"] = round(stats["ci_hi"], 6)
    return row


def _recall_against(reference: dict[str, list[str]], candidate: dict[str, list[str]], k: int) -> float:
    shared = [q for q in reference if q in candidate]
    if not shared:
        return 0.0
    hits = [len(set(candidate[q][:k]) & set(reference[q][:k])) / max(len(reference[q][:k]), 1) for q in shared]
    return round(float(np.mean(hits)), 4)


def exact_reference(
    vectors_path: Path, queries: np.ndarray, qids: list[str], uris: list[str], config: dict[str, Any]
) -> dict[str, list[str]]:
    """The exact ranking from the tested kernel: both the control and the recall reference."""
    index, _ = topk_stream(vectors_path, queries, fetch=config["fetch"])
    return {
        qid: _documents([_Hit(uris[int(row)]) for row in index[position]], config["k"])
        for position, qid in enumerate(qids)
    }


class _Hit:
    """Minimal stand-in so the exact path and the store path share one deduplication routine."""

    __slots__ = ("uri",)

    def __init__(self, uri: str) -> None:
        self.uri = uri


def score_cell(corpus: str, profile: str, label: str, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cache = _cache_root()
    vroot = cache / "vectors" / f"{corpus}__{profile}__{_dirsafe(label)}"
    meta = _read_json(vroot / "meta.json")
    if not meta or "count" not in meta:
        return {}
    chunks_dir = cache / "chunks" / f"{corpus}__{profile}"
    queries_text = _read_json(chunks_dir / "queries.json") or {}
    qrels = _read_json(chunks_dir / "qrels.json") or {}
    uris, ordinals = chunk_identity(chunks_dir / "chunks.parquet")
    qids, query_matrix = _query_vectors(corpus, label, meta["model_id"], None, queries_text)

    reference = exact_reference(vroot / "vectors.npy", query_matrix, qids, uris, config)
    exact_metrics = _summarize({q: _metrics(reference[q], qrels.get(q, {}), config["k"]) for q in qids})
    print(f"    exact kernel nDCG@10={exact_metrics['ndcg@10']:.4f}", flush=True)

    rows: dict[str, dict[str, Any]] = {}
    for backend in config["stores"]:
        store = _load_store(backend, config["work"] / backend.value, int(meta["dim"]))
        upsert_s = fill_store(store, vroot / "vectors.npy", uris, ordinals, config["batch"])
        rankings, latencies = query_store(store, query_matrix, qids, config)
        size = sum(f.stat().st_size for f in (config["work"] / backend.value).rglob("*") if f.is_file())
        store.close()
        per_query = {q: _metrics(rankings[q], qrels.get(q, {}), config["k"]) for q in qids}
        row = {
            "corpus": corpus,
            "profile": profile,
            "embedding": label,
            "dim": int(meta["dim"]),
            "store": backend.value,
            "rows": int(meta["count"]),
            "exact_scan": backend in _EXACT_BACKENDS,
            "ann_params": _describe(shipped_ann_params(backend)),
            "upsert_s": round(upsert_s, 1),
            "search_p50_ms": round(_percentile(latencies, 0.50), 2),
            "search_p95_ms": round(_percentile(latencies, 0.95), 2),
            "store_mb": round(size / 1_000_000, 1),
            "doc_recall_at_k": _recall_against(reference, rankings, config["k"]),
            "exact_ndcg@10": exact_metrics["ndcg@10"],
            **_summarize(per_query),
        }
        row["ndcg_delta_vs_exact"] = round(row["ndcg@10"] - exact_metrics["ndcg@10"], 6)
        rows[backend.value] = row
        _check_exact_control(row)
        print(
            f"    {backend.value:11s} nDCG@10={row['ndcg@10']:.4f} "
            f"(delta {row['ndcg_delta_vs_exact']:+.4f})  docrecall={row['doc_recall_at_k']:.3f}  "
            f"p50={row['search_p50_ms']}ms  {row['store_mb']}MB",
            flush=True,
        )
        shutil.rmtree(config["work"] / backend.value, ignore_errors=True)
    return rows


def _check_exact_control(row: dict[str, Any]) -> None:
    """An exhaustive store must reproduce the kernel. If it does not, nothing else here is valid."""
    if not row["exact_scan"]:
        return
    if abs(row["ndcg_delta_vs_exact"]) > _EXACT_TOLERANCE:
        row["control_failed"] = True
        print(
            f"    !! {row['store']} scans exhaustively yet differs from the exact kernel by "
            f"{row['ndcg_delta_vs_exact']:+.6f}, beyond what tie ordering explains. The harness is "
            f"wrong; ignore every store number in this run.",
            flush=True,
        )


def main() -> None:
    cache = _cache_root()
    config = {
        "k": int(os.environ.get("SEMDEX_STOREQ_K", "10")),
        "fetch": int(os.environ.get("SEMDEX_STOREQ_FETCH", "200")),
        "batch": int(os.environ.get("SEMDEX_STOREQ_BATCH", "5000")),
        "work": Path(os.environ.get("SEMDEX_STOREQ_WORK", str(cache / "tmp-storeq"))),
        "stores": [StoreBackend(s) for s in os.environ.get("SEMDEX_STOREQ_STORES", "sqlite_vec,lancedb").split(",")],
    }
    corpora = os.environ.get("SEMDEX_STOREQ_CORPORA", "mldr_en_8k_slice,mldr_de_3k_slice").split(",")
    profiles = os.environ.get("SEMDEX_STOREQ_PROFILES", "recursive-t256-o0-gpt2").split(",")
    embeddings = os.environ.get("SEMDEX_STOREQ_EMBEDDINGS", "fastembed:bge-base").split(",")
    force = os.environ.get("SEMDEX_STOREQ_FORCE") == "1"
    out = Path(os.environ.get("SEMDEX_STOREQ_OUT", str(cache / "scores" / "store_quality.json")))
    out.parent.mkdir(parents=True, exist_ok=True)
    results = _read_json(out) or {}

    for corpus in corpora:
        for profile in profiles:
            for label in embeddings:
                key = f"{corpus}__{profile}__{_dirsafe(label)}"
                if any(k.startswith(key) for k in results) and not force:
                    print(f"[storeq] skip (cached) {key}", flush=True)
                    continue
                print(f"[storeq] {key}", flush=True)
                rows = score_cell(corpus, profile, label, config)
                if not rows:
                    print(f"[storeq] not-embedded {key}", flush=True)
                    continue
                results.update({f"{key}__{store}": stamped(row) for store, row in rows.items()})
                out.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"\nwrote {out}", flush=True)


# --- shared harness API ---------------------------------------------------------------
# Mean of the per-query metric maps, shared with the ANN frontier sweep.
# Public aliases so a strictly-checked sibling can import them without reportPrivateUsage;
# the underscore names stay for the scripts that already import them.
summarize = _summarize
__all__ = ["summarize"]


if __name__ == "__main__":
    main()
