#!/usr/bin/env python
# pyright: basic
"""Combine two rankings of the same documents into one.

Dense retrieval and BM25 fail differently: the embedding misses a rare exact token it never
learned, and the lexical index misses a paraphrase that shares no words. Fusing them is the
standard way to get both, and it is the largest measurement missing from this benchmark set.

Reciprocal Rank Fusion is used rather than a weighted sum of scores. A cosine similarity and a
BM25 score are not on the same scale, are not comparable between queries, and BM25's range moves
with corpus statistics, so summing them requires a normalisation that is itself a tuning knob and
a source of quiet error. RRF ignores the scores entirely and uses only the RANK each system gave a
document, which is why it needs no per-system calibration and why it is the usual default.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["RRF_K", "reciprocal_rank_fusion"]

# The conventional constant from the original RRF paper. It damps the influence of the very top
# ranks so one system cannot dominate the fusion on a single confident hit; larger values flatten
# the contribution further. Exposed so the sweep can vary it rather than treat it as a fact.
RRF_K = 60


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], *, k: int = RRF_K) -> list[str]:
    """Fuse ranked document-id lists into one ranking, best first.

    Args:
        rankings: one ranked list of document ids per system. Lists may differ in length and need
            not cover the same documents; a document missing from a system simply earns nothing
            from it, which is the behaviour that lets a system contribute only where it has an
            opinion.
        k: the RRF damping constant.

    Returns:
        Document ids ordered by descending fused score. Ties break on the document's best rank
        across systems, then on the id, so the output is deterministic rather than dependent on
        dict ordering.

    Examples:
        >>> reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
        ['a', 'b']
        >>> reciprocal_rank_fusion([["a", "b", "c"], ["c", "b", "a"]])[0]
        'a'
    """
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
            best_rank[doc] = min(best_rank.get(doc, rank), rank)
    return sorted(scores, key=lambda doc: (-scores[doc], best_rank[doc], doc))
