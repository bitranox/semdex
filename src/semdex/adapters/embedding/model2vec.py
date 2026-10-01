"""model2vec embedding provider (static embeddings, no torch; the lightest real one).

Wraps a model2vec ``StaticModel`` (``minishlab/potion-base-8M`` by default): tiny,
CPU-instant, offline after the first download. Static embeddings have no
query/passage asymmetry, so both use ``encode``. Opt-in via ``semdex[model2vec]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...domain.errors import EmbeddingError
from ._base import CallableEmbedding, to_vector, to_vectors

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...application.ports import EmbeddingProvider
    from ...domain.models import Vector

_DEFAULT_MODEL = "minishlab/potion-base-8M"


def load_model2vec_embedding(model_id: str | None = None) -> EmbeddingProvider:
    """Load a model2vec static model and wrap it as an EmbeddingProvider.

    Raises :class:`~semdex.domain.errors.EmbeddingError` if model2vec is missing
    or the model cannot be loaded.
    """
    resolved = model_id or _DEFAULT_MODEL
    try:
        from model2vec import StaticModel  # type: ignore  # optional dep; see module docstring
    except ImportError as exc:  # pragma: no cover - only without semdex[model2vec]
        raise EmbeddingError("model2vec is not installed; install semdex[model2vec]") from exc
    try:
        model = StaticModel.from_pretrained(resolved)  # pyright: ignore[reportUnknownMemberType]
    except Exception as exc:
        raise EmbeddingError(f"model2vec could not load model '{resolved}': {exc}") from exc

    def passages_fn(texts: Sequence[str]) -> list[Vector]:
        return to_vectors(model.encode(list(texts)))  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]

    def query_fn(text: str) -> Vector:
        return to_vector(model.encode([text])[0])  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]

    return CallableEmbedding(model_id=resolved, dim=int(model.dim), passages_fn=passages_fn, query_fn=query_fn)


__all__ = [
    "load_model2vec_embedding",
]
