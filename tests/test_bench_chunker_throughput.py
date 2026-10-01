"""Sharding and merging in scripts/bench_chunker_throughput.py.

The published throughput figures rest on two pieces of plumbing that fail silently rather than
loudly. If the worker sharding drops or duplicates documents, the rate is computed over the wrong
denominator and reads as a speedup. If the merge is wrong, a cheap re-measurement of the worker
sweep quietly deletes the expensive `late` and `semantic` rows and the table just gets shorter.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench_chunker_throughput.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("bench_chunker_throughput", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench() -> Any:
    return _load()


class _CountingChunker:
    """Stands in for a real chunker: returns one chunk per document and records what it saw."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def __call__(self, doc: Any, *, max_tokens: int) -> list[str]:
        self.seen.append(doc)
        return ["chunk"]


# --- sharding across workers -------------------------------------------------------------------


def test_every_document_is_chunked_exactly_once_across_workers(bench: Any) -> None:
    """The denominator of every rate in the table.

    A shard split that drops the tail chunks fewer documents in less time, which is precisely the
    shape of a speedup - so this failure publishes as good news rather than as an error.
    """
    docs = [f"doc{i}" for i in range(23)]  # deliberately not divisible by the worker count
    chunkers = [_CountingChunker() for _ in range(4)]

    total = bench.chunk_all(chunkers, docs)

    assert total == 23
    assert sorted(d for c in chunkers for d in c.seen) == sorted(docs)


def test_a_single_worker_still_chunks_everything(bench: Any) -> None:
    """The one-worker path is the baseline every speedup is measured against."""
    docs = [f"doc{i}" for i in range(5)]
    chunkers = [_CountingChunker()]

    assert bench.chunk_all(chunkers, docs) == 5
    assert chunkers[0].seen == docs


def test_adjacent_documents_go_to_different_workers(bench: Any) -> None:
    """Round-robin, not contiguous blocks.

    Corpora arrive in size order often enough that a contiguous split hands one worker every long
    document. The run then measures that straggler and reports concurrency as useless.

    Asserting only that the shards are equal in COUNT does not test this: a contiguous split of an
    evenly divisible corpus is equally balanced and still gives worker 0 the whole head of the
    size order. So this pins which documents each worker actually got.
    """
    docs = [f"doc{i}" for i in range(20)]
    chunkers = [_CountingChunker() for _ in range(4)]

    bench.chunk_all(chunkers, docs)

    assert chunkers[0].seen == ["doc0", "doc4", "doc8", "doc12", "doc16"]
    assert chunkers[1].seen == ["doc1", "doc5", "doc9", "doc13", "doc17"]


# --- merging a partial re-measurement ----------------------------------------------------------


def _row(strategy: str, threads: int, rate: float, corpus: str = "nfcorpus") -> dict[str, Any]:
    return {"corpus": corpus, "strategy": strategy, "threads": threads, "docs_per_s": rate}


def test_merging_keeps_cells_the_new_run_did_not_measure(bench: Any) -> None:
    """`late` takes tens of minutes and the worker sweep takes seconds.

    Re-running the cheap half must not require re-running the expensive one, or it never gets
    re-run and the table silently loses its slowest two strategies.
    """
    old = [_row("late", 1, 5.0), _row("semantic", 1, 545.0), _row("fast", 1, 25800.0)]
    fresh = [_row("fast", 1, 26000.0), _row("fast", 2, 41000.0)]

    merged = bench.merge_rows(old, fresh)

    assert {r["strategy"] for r in merged} == {"late", "semantic", "fast"}


def test_re_measuring_a_cell_replaces_it(bench: Any) -> None:
    """Two rows for one cell would be silently averaged, or plotted twice, by any reader."""
    merged = bench.merge_rows([_row("fast", 1, 25800.0)], [_row("fast", 1, 26000.0)])

    assert len(merged) == 1
    assert merged[0]["docs_per_s"] == 26000.0


def test_worker_count_is_part_of_a_cell_identity(bench: Any) -> None:
    """One worker and four workers are different measurements of the same strategy.

    Keying on strategy alone would make each worker count overwrite the last, leaving exactly one
    row per strategy and no sweep at all.
    """
    merged = bench.merge_rows([_row("fast", 1, 25800.0)], [_row("fast", 2, 41000.0)])

    assert sorted(r["threads"] for r in merged) == [1, 2]


def test_the_same_strategy_in_two_corpora_stays_two_rows(bench: Any) -> None:
    merged = bench.merge_rows(
        [_row("fast", 1, 25800.0, corpus="nfcorpus")], [_row("fast", 1, 31000.0, corpus="cqadupstack")]
    )

    assert len(merged) == 2
