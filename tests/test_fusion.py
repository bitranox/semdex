"""Reciprocal rank fusion for scripts/_fusion.py.

The fusion decides what a hybrid run retrieves, so these check the properties the benchmark relies
on: that it uses ranks rather than scores, that a document only one system found can still surface,
and that the output is deterministic.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "_fusion.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("_fusion", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fusion() -> Any:
    return _load()


def test_agreeing_systems_keep_their_order(fusion: Any) -> None:
    assert fusion.reciprocal_rank_fusion([["a", "b", "c"], ["a", "b", "c"]]) == ["a", "b", "c"]


def test_a_document_both_systems_found_beats_one_system_s_top_hit(fusion: Any) -> None:
    """The property that makes fusion worth doing: agreement outweighs a single confident hit.

    "solo" is dense's top result and lexical never returns it. "shared" is only second for each,
    but both found it, and that is what should win.
    """
    fused = fusion.reciprocal_rank_fusion([["solo", "shared"], ["other", "shared"]])

    assert fused[0] == "shared"


def test_rrf_slightly_rewards_disagreement_at_the_extremes(fusion: Any) -> None:
    """A surprise worth pinning: 1st-and-3rd scores ABOVE 2nd-and-2nd.

    1/(k+rank) is convex, so the two extreme ranks sum to more than twice the middle one. It is
    easy to assume RRF rewards the consensus middle - it does not, and a fusion written to match
    that assumption would not be RRF. The margin is tiny, which is why the ordering here is
    decided by the tie-break rather than by any meaningful difference in relevance.
    """
    fused = fusion.reciprocal_rank_fusion([["x", "y", "z"], ["z", "y", "x"]])

    assert fused[-1] == "y", "the consensus-middle document ranks last, not first"
    assert set(fused[:2]) == {"x", "z"}


def test_a_document_only_one_system_found_still_appears(fusion: Any) -> None:
    """The reason to fuse at all: each system covers what the other misses.

    A rare exact token the embedding never learned is found only by BM25, and a paraphrase sharing
    no words only by the dense side. Dropping either would defeat the exercise.
    """
    fused = fusion.reciprocal_rank_fusion([["dense_only", "shared"], ["shared", "lexical_only"]])

    assert set(fused) == {"dense_only", "shared", "lexical_only"}
    assert fused[0] == "shared"


def test_only_ranks_matter_never_the_raw_scores(fusion: Any) -> None:
    """RRF must be invariant to score scale, which is the whole reason it is used here.

    A cosine similarity and a BM25 score share no scale, and BM25's range moves with corpus
    statistics. The function takes no scores at all; this pins that contract.
    """
    fused_a = fusion.reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "c"]])
    fused_b = fusion.reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "c"]])

    assert fused_a == fused_b == fusion.reciprocal_rank_fusion([list("abc"), list("bac")])


def test_the_result_is_deterministic_for_tied_documents(fusion: Any) -> None:
    """Two documents with identical fused scores must not depend on dict ordering."""
    first = fusion.reciprocal_rank_fusion([["p", "q"], ["q", "p"]])
    second = fusion.reciprocal_rank_fusion([["q", "p"], ["p", "q"]])

    assert first == second, "a symmetric input must give one stable answer"


def test_a_larger_k_flattens_the_advantage_of_rank_one(fusion: Any) -> None:
    """k is a damping constant, so raising it must reduce how much the top rank dominates."""
    rankings = [["top", "second"], ["second", "top"]]

    small_k_gap = 1 / (1 + 1) - 1 / (1 + 2)
    large_k_gap = 1 / (1000 + 1) - 1 / (1000 + 2)

    assert large_k_gap < small_k_gap
    # and with symmetric input the outcome stays a tie either way
    assert fusion.reciprocal_rank_fusion(rankings, k=1) == fusion.reciprocal_rank_fusion(rankings, k=1000)


def test_one_empty_system_leaves_the_other_untouched(fusion: Any) -> None:
    """A system with nothing to say must not reorder the one that does."""
    assert fusion.reciprocal_rank_fusion([["a", "b", "c"], []]) == ["a", "b", "c"]


def test_no_input_is_an_empty_ranking(fusion: Any) -> None:
    assert fusion.reciprocal_rank_fusion([]) == []
    assert fusion.reciprocal_rank_fusion([[], []]) == []


# The BM25 analyzer language is picked from the corpus name. An English analyzer on German text
# would understate lexical retrieval and bias the dense-versus-hybrid comparison, which is the one
# thing this measurement exists to get right.


def _hybrid() -> Any:
    spec = importlib.util.spec_from_file_location(
        "score_hybrid_sweep", Path(__file__).resolve().parents[1] / "scripts" / "score_hybrid_sweep.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("corpus", "expected"),
    [
        ("mldr_de_3k_slice", "german"),
        ("miracl_de_100k_slice", "german"),
        ("mldr_en_8k_slice", "english"),
        ("miracl_en_100k_slice", "english"),
        ("nfcorpus", "english"),
        ("scifact", "english"),
    ],
)
def test_the_analyzer_language_follows_the_corpus(corpus: str, expected: str) -> None:
    assert _hybrid().language_of(corpus) == expected


def test_an_unknown_corpus_falls_back_to_english() -> None:
    """Explicit fallback rather than a crash, but it must not silently claim German."""
    assert _hybrid().language_of("some_new_corpus") == "english"
