#!/usr/bin/env python
"""Pure chunking throughput: docs/s and chunks/s per strategy, chunking ONLY.

No embedding provider, no store, no search - isolates the chunker. late/semantic do embed
internally (that IS their algorithm), so their rate honestly includes it. Uses real corpus
documents (NFCorpus long docs + CQADupStack short posts) so rates map to the quality tables.

Every rate is a MEDIAN of repeated samples with its spread and the machine's load beside it. The
earlier version of this script timed one pass: the whitespace chunker's 50,620 docs/s came from a
0.01-second measurement, which is the clock's resolution and one scheduler slice rather than a
throughput. Samples are now batched to a floor duration and repeated, so the spread shows how
quiet the box actually was.

It also sweeps WORKER concurrency, because "how many documents per second" is not a deployment
answer without knowing whether a second worker buys anything. Each worker gets its own chunker,
which is both what a real server does and what a HuggingFace tokenizer requires. The sweep covers
the four strategies whose cost is chunking; semantic and late spend theirs inside an embedding
model whose own thread pool is capped here, so sweeping workers over them would measure torch.

Env: DOCS (per corpus, default 300), REPEATS (default 7), THREADS (default "1", e.g. "1,2,4"),
NATIVE_THREADS (BLAS/torch pool, default 4 - uncapped torch takes every core and makes the rate
depend on who else is running), MERGE=1 to add rows to an existing OUT rather than replace it, OUT.
"""

# pyright: basic
# One-off benchmark harness on untyped deps (ir_datasets); strict mode would only add
# reportUnknown* noise (matches the other bench scripts).

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bench_timing import DEFAULT_REPEATS, MIN_SAMPLE_SECONDS, load_snapshot, measure

from semdex.composition import build_chunker
from semdex.domain.enums import ChunkStrategy
from semdex.domain.models import ExtractedDocument, SourceRef

STRATEGIES = [
    ChunkStrategy.WHITESPACE,
    ChunkStrategy.RECURSIVE,
    ChunkStrategy.MARKDOWN,
    ChunkStrategy.FAST,
    ChunkStrategy.SEMANTIC,
    ChunkStrategy.LATE,
]

# Strategies whose cost is chunking rather than embedding, and so the ones a worker sweep says
# anything about. semantic and late run an embedding model per document; their throughput is that
# model's, and adding chunking workers to it would measure the torch pool NATIVE_THREADS caps.
CONCURRENCY_STRATEGIES = {
    ChunkStrategy.WHITESPACE,
    ChunkStrategy.RECURSIVE,
    ChunkStrategy.MARKDOWN,
    ChunkStrategy.FAST,
}

CHUNK_MAX_TOKENS = 256


def load_docs(dataset_id: str, n: int) -> list[str]:
    import ir_datasets

    out = []
    for d in ir_datasets.load(dataset_id).docs_iter():
        out.append(d.default_text())
        if len(out) >= n:
            break
    return out


def chunk_all(chunkers: list[Any], docs: list[ExtractedDocument]) -> int:
    """Chunk every document once across ``len(chunkers)`` worker threads; return the chunk count.

    One chunker PER worker, not one shared between them. Sharing raises "Already borrowed" from
    HuggingFace tokenizers, whose Rust tokenizer has interior mutability and cannot be called
    concurrently - and a per-worker chunker is what a real server holds anyway, so this is the
    realistic configuration rather than a workaround. Threads rather than processes because that
    is what a server reaches for first, and whether it buys anything is the question: a
    pure-Python chunker holds the GIL and will not scale, a Rust-backed one releases it and should.
    """
    if len(chunkers) == 1:
        return _chunk_shard(chunkers[0], docs)
    # Round-robin rather than contiguous blocks: documents are size-ordered often enough that
    # contiguous shards hand one worker all the long ones and measure the straggler, not the scaling.
    shards = [docs[i :: len(chunkers)] for i in range(len(chunkers))]
    with ThreadPoolExecutor(max_workers=len(chunkers)) as pool:
        return sum(pool.map(_chunk_shard, chunkers, shards))


def _chunk_shard(chunker: Any, shard: list[ExtractedDocument]) -> int:
    return sum(len(chunker(doc, max_tokens=CHUNK_MAX_TOKENS)) for doc in shard)


def cap_native_threads(threads: int) -> None:
    """Bound the BLAS/torch thread pools before any of them is created.

    semantic and late chunk by embedding, and torch will otherwise take every core it can see.
    On a shared box that is antisocial and it also makes the rate unreproducible: the figure then
    depends on how many cores happened to be idle. Capping it makes the number a property of the
    configuration, which is why the cap is recorded beside the results.
    """
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TORCH_NUM_THREADS"):
        os.environ.setdefault(name, str(threads))
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:
        pass


def existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("rows", [])


def merge_rows(old: list[dict[str, Any]], fresh: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold a partial re-measurement into an existing set, keyed on what identifies a cell.

    The worker sweep is cheap and the embedding strategies are not, so re-measuring concurrency
    must not require re-measuring `late`. Without this, the second run would publish a table
    holding four of the six strategies and read as though the other two had never been measured.
    """
    by_key = {(r["corpus"], r["strategy"], r["threads"]): r for r in old}
    by_key.update({(r["corpus"], r["strategy"], r["threads"]): r for r in fresh})
    return sorted(by_key.values(), key=lambda r: (r["corpus"], r["strategy"], r["threads"]))


def main() -> None:
    native_threads = int(os.environ.get("NATIVE_THREADS", "4"))
    cap_native_threads(native_threads)
    n = int(os.environ.get("DOCS", "300"))
    repeats = int(os.environ.get("REPEATS", str(DEFAULT_REPEATS)))
    # default 1 only: see chunk_all - the thread sweep is blocked on the tokenizer issue
    thread_counts = [int(t) for t in os.environ.get("THREADS", "1").split(",") if t.strip()]
    out_path = Path(os.environ.get("OUT", "chunker-throughput.json"))
    corpora = {
        "nfcorpus (long docs)": load_docs("beir/nfcorpus/test", n),
        "cqadupstack (short posts)": load_docs("beir/cqadupstack/programmers", n),
    }
    before = load_snapshot()
    rows = []
    for corpus_name, texts in corpora.items():
        docs = [
            ExtractedDocument(source=SourceRef(uri=str(Path(f"/d{i}")), label="", content_hash="", mtime=0.0), text=t)
            for i, t in enumerate(texts)
        ]
        for strategy in STRATEGIES:
            for threads in thread_counts:
                if threads > 1 and strategy not in CONCURRENCY_STRATEGIES:
                    continue
                chunkers = [build_chunker(strategy, recipe="") for _ in range(threads)]
                n_chunks = chunk_all(chunkers, docs)
                # partial, not a lambda: a closure over the loop variables is a late-binding trap
                timing = measure(partial(chunk_all, chunkers, docs), repeats=repeats)
                row = {
                    "corpus": corpus_name,
                    "strategy": strategy.value,
                    "threads": threads,
                    "docs": len(docs),
                    "chunks": n_chunks,
                    "seconds": round(timing.median_s, 4),
                    "docs_per_s": round(len(docs) / timing.median_s, 1),
                    "chunks_per_s": round(n_chunks / timing.median_s, 1),
                    "spread_pct": timing.spread_pct,
                    "repeats": timing.repeats,
                    "iterations_per_sample": timing.iterations_per_sample,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    payload = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "load_before": before,
        "load_after": load_snapshot(),
        "min_sample_seconds": MIN_SAMPLE_SECONDS,
        "native_threads": native_threads,
        "rows": merge_rows(existing_rows(out_path) if os.environ.get("MERGE") else [], rows),
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
