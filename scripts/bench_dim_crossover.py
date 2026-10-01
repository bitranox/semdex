#!/usr/bin/env python
"""How embedding DIMENSION moves the exact-vs-ANN store crossover, and what ANN recall costs.

For each dimension it sweeps the row count N and times sqlite_vec (exact O(N) scan) against
lancedb (IVF-PQ ANN), so you can read off the crossover N where ANN overtakes exact, and watch it
move earlier as D grows (exact scan cost is linear in D, ANN is nearly flat).

Two vector sources, and the difference between them is the whole point:

``SOURCE=synthetic`` generates random unit vectors. Latency depends only on (N, D), so randoms
reproduce the latency behaviour with no model, corpus or GPU. They do NOT reproduce RECALL:
random vectors are near-orthogonal and spread evenly over the sphere, which is the easiest
possible case for a partitioning index. Real embeddings cluster hard, so an ANN index built on
them puts many true neighbours outside the probed partitions. This mode is for the SHAPE of the
latency curve only, never for a recall number or an absolute millisecond figure.

``SOURCE=cache`` reads the pre-computed real embeddings. One corpus and chunk profile are held
fixed and only the embedding model changes, so the dimension ladder is a controlled comparison
over identical text: same documents, same chunk boundaries, same row count. Dimensions with no
native model are filled by truncating a longer cell and renormalising, recorded as
``truncated_from``.

Queries in cache mode are HELD-OUT PASSAGE vectors from the same corpus, not embedded search
strings. That is deliberate and it needs no GPU: recall@k here measures ANN FIDELITY against
exact search on a realistic query distribution, which is a property of the index and the data
geometry, not of any query text. It is not a retrieval-quality measurement, and this file never
reports it as one - retrieval quality lives in the chunk and embedding sweeps.

Recall is measured against exact top-k computed by ``_score_kernel.topk_stream`` over the same
row subset, so "exact" is this repo's own tested kernel rather than a second implementation.

lancedb's ANN index normally builds only above ``[vector_store].lance_index_threshold`` (default
100K); here it is lowered (LANCE_THRESHOLD, default 10000) so ANN is active at every measured
scale and the crossover reflects exact-vs-ANN, not the threshold gate.

Vectors are upserted in batches (never all N in RAM for the synthetic path); the cache path holds
one row subset, bounded by SCALES x D x 4 bytes.

Env: SOURCE (synthetic|cache), DIMS, SCALES, QUERIES (30 synthetic / 100 cache), K (10),
LANCE_THRESHOLD (10000), BATCH (5000), SEED (0), REPEATS (1), STORES, WORK (dir), OUT (json),
CACHE_ROOT, CACHE_CORPUS, CACHE_PROFILE.
"""

# pyright: basic
# Benchmark harness on numpy (no type stubs); strict mode would only add reportUnknown* noise.

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _score_kernel import normalize_rows, topk_stream

from semdex.composition import build_vector_store
from semdex.domain.enums import StoreBackend
from semdex.domain.models import Chunk, Collection, SourceRef

_COLLECTION = "dimx"


def _rng(seed: int):
    # A fixed-seed generator: the run is reproducible and every store at a scale sees identical
    # vectors, so the sqlite-vs-lance comparison is apples-to-apples.
    return np.random.default_rng(seed)


def _unit_batch(gen, n: int, dim: int) -> np.ndarray:
    v = gen.standard_normal((n, dim), dtype="float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v


def _chunks(start: int, n: int) -> list[Chunk]:
    # Unique source path per row so the store keys them distinctly AND so a returned hit maps back
    # to its row index, which is what makes the recall comparison possible.
    return [
        Chunk(
            text="x",
            source=SourceRef(uri=str(Path(f"/d{start + i}")), label="", content_hash="", mtime=0.0),
            ordinal=0,
            token_count=1,
        )
        for i in range(n)
    ]


def _pct(sorted_vals, q):
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))] if sorted_vals else 0.0


def _build(backend, store_dir: Path, threshold: int):
    if backend is StoreBackend.LANCEDB:
        return build_vector_store(backend, store_dir, lance_index_threshold=threshold)
    return build_vector_store(backend, store_dir)


def _cache_root() -> Path:
    return Path(os.environ.get("CACHE_ROOT", "/embeddings"))


def _cells_by_dim(corpus: str, profile: str) -> dict[int, Path]:
    """Every cached cell for one (corpus, profile), keyed by its embedding dimension."""
    found: dict[int, Path] = {}
    for cell in sorted((_cache_root() / "vectors").glob(f"{corpus}__{profile}__*")):
        meta_path = cell / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        found[int(meta["dim"])] = cell
    return found


def _source_for_dim(cells: dict[int, Path], dim: int) -> tuple[Path, int | None]:
    """The cell to read for a requested dimension, and the dim it was truncated from, if any."""
    if dim in cells:
        return cells[dim], None
    larger = sorted(d for d in cells if d > dim)
    if not larger:
        raise SystemExit(f"no cached cell at or above dim {dim}; have {sorted(cells)}")
    return cells[larger[0]], larger[0]


def load_cache_vectors(corpus: str, profile: str, dim: int, scale: int, seed: int) -> tuple[np.ndarray, int | None]:
    """A seeded row subset at the requested dimension, L2-normalised.

    Rows are sampled at RANDOM, never taken as the first N. The cells are written in document
    order, so the leading rows are one contiguous slice of the corpus and would understate how
    spread out a real index is.
    """
    cell, truncated_from = _source_for_dim(_cells_by_dim(corpus, profile), dim)
    mm = np.load(cell / "vectors.npy", mmap_mode="r")
    total = int(mm.shape[0])
    if scale > total:
        raise SystemExit(f"scale {scale} exceeds the {total} rows in {cell.name}")
    picks = np.sort(_rng(seed).choice(total, size=scale, replace=False))
    subset = np.asarray(mm[picks][:, :dim], dtype=np.float32)  # prefix truncation for the MRL fills
    return normalize_rows(subset), truncated_from


def _held_out_queries(vectors: np.ndarray, n_queries: int, seed: int) -> np.ndarray:
    """Query vectors drawn from the corpus itself, so the query distribution matches the data."""
    picks = _rng(seed + 999).choice(vectors.shape[0], size=min(n_queries, vectors.shape[0]), replace=False)
    return vectors[picks]


def _batches_from(vectors: np.ndarray, batch: int) -> Iterator[np.ndarray]:
    for start in range(0, vectors.shape[0], batch):
        yield vectors[start : start + batch]


def _synthetic_batches(gen, scale: int, dim: int, batch: int) -> Iterator[np.ndarray]:
    for start in range(0, scale, batch):
        yield _unit_batch(gen, min(batch, scale - start), dim)


def _upsert(store, batches: Iterator[np.ndarray]) -> float:
    start_time = time.perf_counter()
    offset = 0
    for block in batches:
        store.upsert(
            collection=_COLLECTION, chunks=_chunks(offset, block.shape[0]), vectors=[r.tolist() for r in block]
        )
        offset += block.shape[0]
    store.compact(collection=_COLLECTION)  # upsert only appends; compact builds/folds the ANN index
    return time.perf_counter() - start_time


def _row_index_of(hit: Any) -> int | None:
    """Map a hit back to the row it was upserted as (the uri is /d<row>).

    Hit carries ``uri`` directly. Reading it as ``hit.source.uri`` returns None for every hit and
    silently yields recall 0.0 for EVERY store, exact ones included - which is why the exact store
    is checked as a control below rather than trusted.
    """
    name = Path(str(getattr(hit, "uri", "") or "")).name
    return int(name[1:]) if name.startswith("d") and name[1:].isdigit() else None


def _query_once(store, vector: list[float], k: int) -> tuple[float, list[int]]:
    started = time.perf_counter()
    hits = store.query(collection=_COLLECTION, vector=vector, k=k)
    elapsed = time.perf_counter() - started
    rows = [r for r in (_row_index_of(h) for h in hits) if r is not None]
    return elapsed, rows


def _measure_latency(store, queries: np.ndarray, k: int, repeats: int) -> tuple[list[float], list[list[int]]]:
    latencies: list[float] = []
    returned: list[list[int]] = []
    for repeat in range(repeats):
        for i, query in enumerate(queries):
            elapsed, rows = _query_once(store, query.tolist(), k)
            latencies.append(elapsed)
            if repeat == 0:
                returned.append(rows)
            del i
    return latencies, returned


def _recall(returned: list[list[int]], truth: np.ndarray, k: int) -> float | None:
    """Mean fraction of the exact top-k that the store actually returned."""
    if not returned or truth.size == 0:
        return None
    scores = [
        len(set(rows[:k]) & set(truth[i][:k].tolist())) / k for i, rows in enumerate(returned) if i < truth.shape[0]
    ]
    return round(float(np.mean(scores)), 4) if scores else None


def measure(backend, store_dir: Path, *, dim: int, scale: int, plan: dict[str, Any]) -> dict[str, Any]:
    shutil.rmtree(store_dir, ignore_errors=True)
    store_dir.mkdir(parents=True, exist_ok=True)
    store = _build(backend, store_dir, plan["threshold"])
    store.ensure_collection(Collection(name=_COLLECTION, model_id=f"dimx-{dim}", dim=dim))

    upsert_s = _upsert(store, plan["batches"]())
    latencies, returned = _measure_latency(store, plan["queries"], plan["k"], plan["repeats"])
    latencies.sort()
    size = sum(f.stat().st_size for f in store_dir.rglob("*") if f.is_file())
    store.close()
    shutil.rmtree(store_dir, ignore_errors=True)
    result = {
        "store": backend.value,
        "search_p50_ms": round(_pct(latencies, 0.50) * 1000, 2),
        "search_p95_ms": round(_pct(latencies, 0.95) * 1000, 2),
        "search_min_ms": round(latencies[0] * 1000, 2) if latencies else 0.0,
        "search_max_ms": round(latencies[-1] * 1000, 2) if latencies else 0.0,
        "upsert_s": round(upsert_s, 1),
        "store_mb": round(size / 1_000_000, 1),
        "repeats": plan["repeats"],
    }
    if plan["truth"] is not None:
        result["recall_at_k"] = _recall(returned, plan["truth"], plan["k"])
    return result


def _plan_for(dim: int, scale: int, config: dict[str, Any]) -> dict[str, Any]:
    """Everything a store measurement needs, built once so both stores see identical input."""
    if config["source"] == "cache":
        vectors, truncated_from = load_cache_vectors(
            config["corpus"], config["profile"], dim, scale, config["seed"] + dim + scale
        )
        queries = _held_out_queries(vectors, config["n_queries"], config["seed"])
        truth, _ = topk_stream(vectors, queries, fetch=config["k"])
        return {
            "batches": lambda: _batches_from(vectors, config["batch"]),
            "queries": queries,
            "truth": truth,
            "truncated_from": truncated_from,
            **{key: config[key] for key in ("k", "threshold", "repeats")},
        }
    queries = _unit_batch(_rng(config["seed"] + 10_000), config["n_queries"], dim)
    return {
        # A fresh generator per store so both see the identical vectors at this (dim, scale).
        "batches": lambda: _synthetic_batches(_rng(config["seed"] + dim + scale), scale, dim, config["batch"]),
        "queries": queries,
        "truth": None,
        "truncated_from": None,
        **{key: config[key] for key in ("k", "threshold", "repeats")},
    }


def _config() -> dict[str, Any]:
    source = os.environ.get("SOURCE", "synthetic")
    return {
        "source": source,
        "dims": [int(x) for x in os.environ.get("DIMS", "384,768,1536,3072").split(",")],
        "scales": [int(x) for x in os.environ.get("SCALES", "50000,100000,250000,500000").split(",")],
        "n_queries": int(os.environ.get("QUERIES", "100" if source == "cache" else "30")),
        "k": int(os.environ.get("K", "10")),
        "threshold": int(os.environ.get("LANCE_THRESHOLD", "10000")),
        "batch": int(os.environ.get("BATCH", "5000")),
        "seed": int(os.environ.get("SEED", "0")),
        "repeats": int(os.environ.get("REPEATS", "1")),
        "corpus": os.environ.get("CACHE_CORPUS", "mldr_en_8k_slice"),
        "profile": os.environ.get("CACHE_PROFILE", "recursive-t256-o0-gpt2"),
        "stores": [StoreBackend(s) for s in os.environ.get("STORES", "sqlite_vec,lancedb").split(",")],
    }


def _measure_row(dim: int, scale: int, config: dict[str, Any], work: Path) -> dict[str, Any]:
    plan = _plan_for(dim, scale, config)
    row: dict[str, object] = {"dim": dim, "scale": scale, "source": config["source"]}
    if plan["truncated_from"]:
        row["truncated_from"] = plan["truncated_from"]
    for backend in config["stores"]:
        try:
            measured: dict[str, Any] = measure(backend, work / backend.value, dim=dim, scale=scale, plan=plan)
            row[backend.value] = measured
            recall = f" recall={measured['recall_at_k']}" if "recall_at_k" in measured else ""
            print(
                f"  dim={dim:>5} N={scale:>7} {backend.value:11} p50={measured['search_p50_ms']:>8}ms "
                f"p95={measured['search_p95_ms']:>8}ms upsert={measured['upsert_s']}s "
                f"{measured['store_mb']}MB{recall}",
                flush=True,
            )
        except Exception as exc:
            # A refusal is a RESULT, not a crash: pgvector's HNSW index caps at 2000 dimensions,
            # and "this store cannot index a 4096-dim model" belongs in the published table.
            row[backend.value] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  dim={dim} N={scale} {backend.value} FAILED: {type(exc).__name__}: {exc}", flush=True)
    _annotate_winner(row, config)
    return row


def _annotate_winner(row: dict[str, Any], config: dict[str, Any]) -> None:
    exact = row.get(StoreBackend.SQLITE_VEC.value)
    ann = row.get(StoreBackend.LANCEDB.value)
    if not (isinstance(exact, dict) and isinstance(ann, dict)):
        return
    # sqlite_vec is an EXACT store, so its recall against exact ground truth must be 1.0. It is
    # therefore a free positive control on the recall instrument itself: anything else means the
    # measurement is broken, not the store, and an ANN recall figure next to it would be fiction.
    exact_recall = exact.get("recall_at_k")
    if exact_recall is not None and exact_recall < 0.999:
        row["recall_instrument_broken"] = True
        print(
            f"  !! dim={row['dim']} N={row['scale']}: the EXACT store reports recall "
            f"{exact_recall}, which is impossible. The recall measurement is broken; ignore every "
            f"recall number in this run.",
            flush=True,
        )
    sq, ln = exact.get("search_p50_ms"), ann.get("search_p50_ms")
    if isinstance(sq, (int, float)) and isinstance(ln, (int, float)):
        row["ann_faster"] = ln < sq
        recall = ann.get("recall_at_k")
        note = f" at recall {recall}" if recall is not None else ""
        print(
            f"  -> dim={row['dim']} N={row['scale']}: {'ANN wins' if ln < sq else 'exact wins'} "
            f"(sqlite {sq}ms vs lance {ln}ms){note}",
            flush=True,
        )
    del config


def main():
    config = _config()
    work = Path(os.environ.get("WORK", "./dimx"))
    out = Path(os.environ.get("OUT", "dim-crossover.json"))

    results = []
    for dim in config["dims"]:
        for scale in config["scales"]:
            results.append(_measure_row(dim, scale, config, work))
            payload = {
                "lance_index_threshold": config["threshold"],
                "source": config["source"],
                "corpus": config["corpus"] if config["source"] == "cache" else None,
                "profile": config["profile"] if config["source"] == "cache" else None,
                "k": config["k"],
                "n_queries": config["n_queries"],
                "repeats": config["repeats"],
                "results": results,
            }
            out.write_text(json.dumps(payload, indent=2))

    print("\n=== crossover N (first scale where lancedb ANN beats sqlite_vec exact), per dim ===", flush=True)
    for dim in config["dims"]:
        rows = [r for r in results if r["dim"] == dim and "ann_faster" in r]
        cross = next((r["scale"] for r in rows if r["ann_faster"]), None)
        print(f"  dim={dim:>5}: crossover at N={cross if cross else '> ' + str(config['scales'][-1])}", flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
