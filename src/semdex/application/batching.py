"""Batching wrapper for an embedding provider.

``[embedding].batch`` caps the passages per ``embed_passages`` call at index time - useful to
bound peak memory / request size on a constrained embed server. ``embed_query`` and the provider
identity (``model_id`` / ``dim``) pass straight through. A batch that is None/0 or >= the input
size degrades to a single call, so it is non-breaking by default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..domain.models import Vector
    from .ports import EmbeddingProvider


class BatchingEmbedding:
    """Wrap a provider so ``embed_passages`` runs in batches of at most ``batch`` texts."""

    def __init__(self, inner: EmbeddingProvider, *, batch: int) -> None:
        # Bind private fields FIRST so an early attribute access cannot recurse through __getattr__.
        self._inner = inner
        self._batch = batch

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def dim(self) -> int:
        return self._inner.dim

    def embed_passages(self, texts: Sequence[str]) -> list[Vector]:
        if self._batch <= 0 or len(texts) <= self._batch:
            return self._inner.embed_passages(texts)
        vectors: list[Vector] = []
        for start in range(0, len(texts), self._batch):
            vectors.extend(self._inner.embed_passages(texts[start : start + self._batch]))
        return vectors

    def embed_query(self, text: str) -> Vector:
        return self._inner.embed_query(text)

    def __getattr__(self, name: str) -> Any:  # delegate any other provider attribute
        return getattr(self._inner, name)


__all__ = ["BatchingEmbedding"]
