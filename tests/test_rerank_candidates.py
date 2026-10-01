"""Candidate selection and reordering for scripts/score_rerank_sweep.py.

The reranker is expensive, so what it is asked to judge, and how its verdict is applied, decides
both the cost and the answer. These cover the two pure pieces: choosing one representative passage
per document, and reordering by score.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_rerank_sweep.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("score_rerank_sweep", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rerank() -> Any:
    return _load()


def test_a_document_is_represented_by_its_highest_ranked_chunk(rerank: Any) -> None:
    """The passage the retrieval stage actually matched is the one the reranker should judge.

    Document "a" appears at rows 5 and 1; row 5 ranked higher, so that is its representative.
    Taking the lowest row number instead would hand the reranker a passage the retrieval stage
    did not rank on.
    """
    uris = {5: "a", 3: "b", 1: "a", 7: "c"}
    order = [5, 3, 1, 7]

    docs = rerank._documents_with_source(order, [uris.get(i, f"x{i}") for i in range(8)], limit=10)

    assert docs == [("a", 5), ("b", 3), ("c", 7)]


def test_the_depth_limit_counts_documents_not_chunks(rerank: Any) -> None:
    """Cost scales with documents reranked, so the cap must be on distinct documents."""
    uris = ["a", "a", "a", "b", "b", "c", "d"]

    docs = rerank._documents_with_source(range(len(uris)), uris, limit=2)

    assert [uri for uri, _row in docs] == ["a", "b"]


def test_reordering_is_by_descending_score(rerank: Any) -> None:
    cache = rerank.ScoreCache.__new__(rerank.ScoreCache)
    cache._scores = {("q", 1): 0.1, ("q", 2): 0.9, ("q", 3): 0.5}

    assert cache.order("q", [("low", 1), ("high", 2), ("mid", 3)]) == ["high", "mid", "low"]


def test_reordering_keeps_every_candidate(rerank: Any) -> None:
    """Reranking reorders a shortlist; it must never drop a document from it.

    Losing a candidate here would show up as a recall drop that looks like a reranker weakness
    rather than a bug in applying it.
    """
    cache = rerank.ScoreCache.__new__(rerank.ScoreCache)
    cache._scores = {("q", i): float(i % 3) for i in range(6)}
    docs = [(f"d{i}", i) for i in range(6)]

    assert sorted(cache.order("q", docs)) == sorted(uri for uri, _ in docs)


def test_a_missing_score_is_not_silently_ranked_last(rerank: Any) -> None:
    """An unscored pair means the cache was not filled; that must fail, not quietly rank it.

    Defaulting a missing score to zero would bury the document and read as a reranker judgement
    rather than as the bug it is.
    """
    cache = rerank.ScoreCache.__new__(rerank.ScoreCache)
    cache._scores = {("q", 1): 0.5}

    with pytest.raises(KeyError):
        cache.order("q", [("scored", 1), ("never_scored", 2)])
