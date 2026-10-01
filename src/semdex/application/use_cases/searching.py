"""Search use case: embed the query and read the closest chunks.

Depends only on the embedding provider and the vector store READER slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...domain.models import FusedHit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...domain.models import Hit, Vector
    from ..ports import EmbeddingProvider, VectorStoreReader

_DEFAULT_K = 5
_DEFAULT_RRF_K = 60


def search(
    *,
    embedding: EmbeddingProvider,
    store: VectorStoreReader,
    collection: str,
    query: str,
    k: int = _DEFAULT_K,
) -> list[Hit]:
    """Embed ``query`` with the collection's model and return the top-k hits."""
    vector = embedding.embed_query(query)
    return store.query(collection=collection, vector=vector, k=k)


@dataclass(frozen=True, slots=True)
class DatasetSearchTarget:
    """One dataset to search in a fan-out.

    ``model_key`` is a stable identity for the embedding model (provider + model +
    endpoint) so the query is embedded ONCE per distinct model across the selected
    datasets; ``embedding`` and ``store`` are the read ports, ``collection`` the
    dataset's collection.
    """

    name: str
    model_key: str
    embedding: EmbeddingProvider
    store: VectorStoreReader
    collection: str


def search_across_datasets(
    *,
    targets: Sequence[DatasetSearchTarget],
    query: str,
    k: int = _DEFAULT_K,
    rrf_k: int = _DEFAULT_RRF_K,
) -> list[FusedHit]:
    """Fan ``query`` out across datasets and fuse the results by RRF.

    Embeds the query ONCE per distinct ``model_key`` (datasets sharing a model
    reuse the vector), queries each dataset's store for its top-``k`` hits, then
    fuses every per-dataset list into one top-``k`` ranking via
    :func:`reciprocal_rank_fusion`. Similarity scores are never compared across
    datasets.
    """
    vectors: dict[str, Vector] = {}
    for target in targets:
        if target.model_key not in vectors:
            vectors[target.model_key] = target.embedding.embed_query(query)
    ranked: list[tuple[str, Sequence[Hit]]] = [
        (target.name, target.store.query(collection=target.collection, vector=vectors[target.model_key], k=k))
        for target in targets
    ]
    return reciprocal_rank_fusion(ranked, k=rrf_k, limit=k)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[tuple[str, Sequence[Hit]]],
    *,
    k: int = _DEFAULT_RRF_K,
    limit: int,
) -> list[FusedHit]:
    """Fuse per-dataset ranked hit lists into one ranking by Reciprocal Rank Fusion.

    Each hit contributes ``1 / (k + rank)`` (rank 1-based) to its identity; the
    fused ranking uses those summed contributions ONLY, never the raw similarity
    ``Hit.score`` (which is not comparable across different embedding models).
    Hits are identified across lists by ``(uri, ordinal)``; a hit surfaced by
    several datasets sums their contributions. The ``dataset`` on a result is the
    dataset whose single contribution to it was largest. Ties in the fused score
    break deterministically by ``(uri, ordinal)``. Returns the top ``limit``.

    Example:
        >>> from semdex.domain.models import Hit
        >>> h = Hit(chunk_text="x", score=0.0, uri="u", ordinal=0, label="", collection="c")
        >>> reciprocal_rank_fusion([("d", [h])], limit=1)[0].dataset
        'd'
    """
    scores: dict[tuple[str, int], float] = {}
    best: dict[tuple[str, int], tuple[float, str, Hit]] = {}
    for dataset, hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            key = (hit.uri, hit.ordinal)
            contribution = 1.0 / (k + rank)
            scores[key] = scores.get(key, 0.0) + contribution
            current = best.get(key)
            if current is None or contribution > current[0]:
                best[key] = (contribution, dataset, hit)
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    fused = [FusedHit(hit=best[key][2], dataset=best[key][1], rrf_score=scores[key]) for key in ordered]
    return fused[:limit]


__all__ = [
    "DatasetSearchTarget",
    "reciprocal_rank_fusion",
    "search",
    "search_across_datasets",
]
