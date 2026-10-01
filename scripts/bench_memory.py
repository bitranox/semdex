#!/usr/bin/env python
"""Resident memory of every vector store and every embedding provider.

Disk size is published for the stores and nothing here reports memory, so the question a reader
actually has - will this fit in an 8 GB container - has no answer. This measures it.

Two sweeps, one mechanism:

  stores      each backend built at N rows of dim D, then SERVED from a second process. Swept
              over N, because the embedded JSON store holds every vector in memory by design and
              the on-disk ones do not, and a single size cannot show which is which.
  embedders   each provider loaded and asked to embed a batch, so the model's footprint is
              separated from the interpreter's.

Two properties decide the process layout, and both were caught by measuring rather than by
reasoning:

* **Peak RSS is monotonic within a process**, so every cell needs its own interpreter or each
  inherits the high-water mark of everything before it (see scripts/_bench_memory.py).
* **Ingest and serving must also be separated.** The fixture needed to LOAD a store - N chunks
  plus N vectors as Python objects - was 219 MB at 10,000 rows, larger than any store's own
  footprint, so a single process measures mostly the harness. Freeing it first does not help:
  the allocator does not return the pages, so RSS stays high and the store still looks fat.
  So the ingest process exits, and a fresh one opens the store on disk and queries it. That
  split also produces the two numbers a deployment actually needs separately: the memory to
  BUILD an index, and the memory to SERVE it.

Env: SCALES (default 1000,10000,50000), DIM (384), QUERIES (20), STORES, EMBEDDERS, OUT.
Run one cell directly with CELL_KIND=ingest|serve|embedder plus its CELL_* variables; that is how
the driver invokes this same file as its own child.
"""

# pyright: basic
# One-off benchmark harness; the child-process protocol is JSON on stdout by design.

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _bench_memory import Measurement, current_rss_mb, interpreter_baseline_mb, peak_rss_mb, run_cell

DEFAULT_SCALES = (1_000, 10_000, 50_000)
DEFAULT_STORES = ("json", "sqlite_vec", "lancedb")
DEFAULT_EMBEDDERS = ("placeholder", "model2vec", "fastembed", "sentence_transformers")


# --------------------------------------------------------------------------- the child cells


def _synthetic(n: int, dim: int) -> tuple[list[Any], list[Any]]:
    """Deterministic vectors and chunks. Content is irrelevant to memory; SIZE is the variable."""
    import numpy as np

    from semdex.domain.models import Chunk, SourceRef

    rng = np.random.default_rng(20260811)
    vectors = rng.random((n, dim), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    text = "filler " * 20
    chunks = [
        Chunk(
            text=f"chunk {i} {text}",
            source=SourceRef(uri=f"/doc{i}", label="bench", content_hash=f"h{i}", mtime=0.0),
            ordinal=0,
            token_count=21,
        )
        for i in range(n)
    ]
    return chunks, [tuple(row) for row in vectors]


COLLECTION = "membench"


def ingest_cell() -> None:
    """Build ONE store at ONE scale into CELL_STORE_DIR, then exit so its memory goes with it.

    Reports the peak reached while building, which is the ingest-side answer: a machine that can
    serve an index may still not have the memory to create one.
    """
    from semdex.composition import build_vector_store
    from semdex.domain.enums import StoreBackend
    from semdex.domain.models import Collection

    backend = StoreBackend(os.environ["CELL_BACKEND"])
    n, dim = int(os.environ["CELL_N"]), int(os.environ["CELL_DIM"])
    chunks, vectors = _synthetic(n, dim)
    fixture_peak = peak_rss_mb()

    store = build_vector_store(backend, Path(os.environ["CELL_STORE_DIR"]))
    store.ensure_collection(Collection(name=COLLECTION, model_id="bench", dim=dim))
    for i in range(0, n, 1000):
        store.upsert(collection=COLLECTION, chunks=chunks[i : i + 1000], vectors=vectors[i : i + 1000])
    store.compact(collection=COLLECTION)
    result = Measurement(
        peak_mb=peak_rss_mb(),
        resident_mb=current_rss_mb(),
        baseline_mb=float(os.environ["CELL_BASELINE"]),
        extra={
            "backend": backend.value,
            "rows": n,
            "dim": dim,
            # The harness's own cost, reported so a reader can see how much of the ingest peak is
            # the store and how much is the fixture feeding it.
            "harness_fixture_mb": round(fixture_peak - float(os.environ["CELL_BASELINE"]), 1),
        },
    )
    store.close()
    print(json.dumps(result.as_row()), flush=True)


def serve_cell() -> None:
    """Open an EXISTING store and query it. This process never builds the fixture.

    That is the point: whatever this reaches is the store plus its library, with none of the
    harness that loaded it, which is what a serving container has to hold.
    """
    from semdex.composition import build_vector_store
    from semdex.domain.enums import StoreBackend

    backend = StoreBackend(os.environ["CELL_BACKEND"])
    dim, n_queries = int(os.environ["CELL_DIM"]), int(os.environ.get("CELL_QUERIES", "20"))
    import numpy as np

    rng = np.random.default_rng(11)
    probes = rng.random((n_queries, dim), dtype=np.float32)
    probes /= np.linalg.norm(probes, axis=1, keepdims=True) + 1e-12

    store = build_vector_store(backend, Path(os.environ["CELL_STORE_DIR"]))
    opened_peak = peak_rss_mb()
    started = time.perf_counter()
    for probe in probes:
        store.query(collection=COLLECTION, vector=tuple(probe), k=10)
    query_s = (time.perf_counter() - started) / max(1, n_queries)
    result = Measurement(
        peak_mb=peak_rss_mb(),
        resident_mb=current_rss_mb(),
        baseline_mb=float(os.environ["CELL_BASELINE"]),
        extra={
            "backend": backend.value,
            "rows": int(os.environ["CELL_N"]),
            "dim": dim,
            "open_peak_mb_gross": opened_peak,
            "query_ms": round(query_s * 1000, 2),
            "counted": store.count(collection=COLLECTION),
        },
    )
    store.close()
    print(json.dumps(result.as_row()), flush=True)


def embedder_cell() -> None:
    """Load ONE embedding provider, embed a batch, and report peak memory."""
    from semdex.composition import build_embedding
    from semdex.domain.enums import EmbeddingBackend

    provider = EmbeddingBackend(os.environ["CELL_PROVIDER"])
    texts = [f"passage number {i} about vector search and retrieval quality" for i in range(64)]
    embedding = build_embedding(provider, allow_fallback=False)
    loaded_peak = peak_rss_mb()
    vectors = embedding.embed_passages(texts)
    result = Measurement(
        peak_mb=peak_rss_mb(),
        resident_mb=current_rss_mb(),
        baseline_mb=float(os.environ["CELL_BASELINE"]),
        extra={
            "provider": provider.value,
            "model_id": embedding.model_id,
            "dim": embedding.dim,
            "load_peak_mb_gross": loaded_peak,
            "batch": len(vectors),
        },
    )
    print(json.dumps(result.as_row()), flush=True)


# --------------------------------------------------------------------------- the parent driver


def main() -> None:
    if os.environ.get("CELL_KIND") == "ingest":
        return ingest_cell()
    if os.environ.get("CELL_KIND") == "serve":
        return serve_cell()
    if os.environ.get("CELL_KIND") == "embedder":
        return embedder_cell()

    me = Path(__file__).resolve()
    baseline = interpreter_baseline_mb()
    print(f"interpreter baseline: {baseline} MB (subtracted from every row)", flush=True)
    dim = os.environ.get("DIM", "384")
    scales = [int(s) for s in os.environ.get("SCALES", ",".join(str(s) for s in DEFAULT_SCALES)).split(",")]
    stores = os.environ.get("STORES", ",".join(DEFAULT_STORES)).split(",")
    embedders = os.environ.get("EMBEDDERS", ",".join(DEFAULT_EMBEDDERS)).split(",")

    store_rows: list[dict[str, Any]] = []
    for backend in stores:
        for n in scales:
            print(f"store {backend} @ {n:,} rows", flush=True)
            with tempfile.TemporaryDirectory(prefix="membench-") as store_dir:
                shared = {
                    "CELL_BACKEND": backend,
                    "CELL_N": str(n),
                    "CELL_DIM": dim,
                    "CELL_STORE_DIR": store_dir,
                    "CELL_BASELINE": str(baseline),
                }
                built = run_cell(me, {**shared, "CELL_KIND": "ingest"})
                if not built:
                    continue
                served = run_cell(me, {**shared, "CELL_KIND": "serve", "CELL_QUERIES": os.environ.get("QUERIES", "20")})
                if not served:
                    continue
            row = {
                **served,
                "ingest_peak_mb": built["peak_mb"],
                "harness_fixture_mb": built["harness_fixture_mb"],
                "disk_mb": built.get("disk_mb"),
            }
            print(
                f"    serve {row['resident_mb']} MB resident, ingest peak {row['ingest_peak_mb']} MB, "
                f"query {row['query_ms']} ms, rows counted {row['counted']:,}",
                flush=True,
            )
            store_rows.append(row)

    embedder_rows: list[dict[str, Any]] = []
    for provider in embedders:
        print(f"embedder {provider}", flush=True)
        row = run_cell(me, {"CELL_KIND": "embedder", "CELL_PROVIDER": provider, "CELL_BASELINE": str(baseline)})
        if row:
            print(f"    peak {row['peak_mb']} MB net, dim {row['dim']}", flush=True)
            embedder_rows.append(row)

    payload = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "interpreter_baseline_mb": baseline,
        "dim": int(dim),
        "stores": store_rows,
        "embedders": embedder_rows,
    }
    out_path = Path(os.environ.get("OUT", "memory.json"))
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
