"""Local, offline semantic embeddings via sentence-transformers.

The real EmbeddingProvider: it wraps a sentence-transformers model and applies
per-model query/passage prefixes. The heavy dependency is optional
(``semdex[embed]``) and imported lazily in the factory, so this module imports
cleanly without it. The class takes an injected encoder (a small structural
Protocol), so its logic is unit-tested with a fake and never needs a model
download. ``sentence_transformers`` is the single untyped/optional surface; its
one quarantined import and casts live in the factory only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from ...domain.errors import EmbeddingError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...domain.models import Vector

# Multilingual, offline default (EN + DE); overridable per deployment.
_DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class _Encoder(Protocol):
    """The slice of a SentenceTransformer this adapter uses."""

    def encode(self, sentences: Sequence[str], *, normalize_embeddings: bool = ...) -> Any: ...

    def get_sentence_embedding_dimension(self) -> int | None: ...


class SentenceTransformerEmbedding:
    """EmbeddingProvider backed by an injected sentence-transformers encoder."""

    def __init__(
        self,
        *,
        model_id: str,
        encoder: _Encoder,
        query_prefix: str = "",
        passage_prefix: str = "",
        normalize: bool = True,
    ) -> None:
        dim = encoder.get_sentence_embedding_dimension()
        if dim is None:
            raise EmbeddingError(f"model '{model_id}' did not report an embedding dimension")
        self._model_id = model_id
        self._encoder = encoder
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._normalize = normalize
        self._dim = dim

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return self._dim

    def embed_passages(self, texts: Sequence[str]) -> list[Vector]:
        return self._encode([self._passage_prefix + text for text in texts])

    def embed_query(self, text: str) -> Vector:
        return self._encode([self._query_prefix + text])[0]

    def _encode(self, texts: list[str]) -> list[Vector]:
        raw: Any = self._encoder.encode(texts, normalize_embeddings=self._normalize)
        rows = raw.tolist() if hasattr(raw, "tolist") else raw
        return [tuple(float(value) for value in row) for row in rows]


def load_sentence_transformer_embedding(
    model_id: str = _DEFAULT_MODEL,
    *,
    query_prefix: str = "",
    passage_prefix: str = "",
    normalize: bool = True,
) -> SentenceTransformerEmbedding:
    """Load a sentence-transformers model and wrap it as an EmbeddingProvider.

    Raises :class:`~semdex.domain.errors.EmbeddingError` if the optional
    ``semdex[embed]`` dependency is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore  # optional dep; see module docstring
    except ImportError as exc:
        raise EmbeddingError("sentence-transformers is not installed; install semdex[embed]") from exc

    # Quarantined boundary to the untyped/optional library (see module docstring):
    # the SentenceTransformer instance structurally satisfies the _Encoder Protocol.
    return SentenceTransformerEmbedding(
        model_id=model_id,
        # reportArgumentType: newer sentence-transformers ship stubs whose overloaded `encode`
        # does not structurally match the minimal `_Encoder` protocol; the boundary is fine at runtime.
        encoder=SentenceTransformer(model_id),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportArgumentType]
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
        normalize=normalize,
    )


# Static conformance assertion -- SentenceTransformerEmbedding satisfies EmbeddingProvider.
if TYPE_CHECKING:
    from ...application.ports import EmbeddingProvider

    _assert_embedding: EmbeddingProvider = load_sentence_transformer_embedding()


__all__ = [
    "SentenceTransformerEmbedding",
    "load_sentence_transformer_embedding",
]
