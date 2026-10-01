"""Sentence-transformers embedding adapter (logic tested with a fake encoder).

The real model path is covered by a local_only integration test that skips when
sentence-transformers is not installed.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from semdex.adapters.embedding import SentenceTransformerEmbedding
from semdex.domain.errors import EmbeddingError


class FakeEncoder:
    """Deterministic stand-in for a SentenceTransformer, recording inputs."""

    def __init__(self, *, dim: int = 4, reported_dim: int | None = 4) -> None:
        self._dim = dim
        self._reported_dim = reported_dim
        self.seen: list[str] = []

    def encode(self, sentences: Sequence[str], *, normalize_embeddings: bool = False) -> list[list[float]]:
        self.seen.extend(sentences)
        return [self._vector(text) for text in sentences]

    def get_sentence_embedding_dimension(self) -> int | None:
        return self._reported_dim

    def _vector(self, text: str) -> list[float]:
        base = [float(len(text)), float(text.count("a")), float(text.count("e")), 1.0]
        return base[: self._dim]


@pytest.mark.os_agnostic
def test_exposes_model_id_and_dimension() -> None:
    """The adapter reports its model id and the encoder's dimension."""
    embed = SentenceTransformerEmbedding(model_id="fake/model", encoder=FakeEncoder(dim=4))
    assert embed.model_id == "fake/model"
    assert embed.dim == 4


@pytest.mark.os_agnostic
def test_embed_passages_applies_passage_prefix() -> None:
    """Passage embedding prepends the configured passage prefix per text."""
    encoder = FakeEncoder()
    embed = SentenceTransformerEmbedding(model_id="m", encoder=encoder, passage_prefix="passage: ")

    embed.embed_passages(["alpha", "beta"])

    assert encoder.seen == ["passage: alpha", "passage: beta"]


@pytest.mark.os_agnostic
def test_embed_query_applies_query_prefix_and_returns_one_vector() -> None:
    """Query embedding prepends the query prefix and returns a single vector."""
    encoder = FakeEncoder(dim=4)
    embed = SentenceTransformerEmbedding(model_id="m", encoder=encoder, query_prefix="query: ")

    vector = embed.embed_query("hello")

    assert encoder.seen == ["query: hello"]
    assert isinstance(vector, tuple)
    assert len(vector) == 4
    assert all(isinstance(value, float) for value in vector)


@pytest.mark.os_agnostic
def test_embed_passages_returns_a_vector_per_text() -> None:
    """Passage embedding returns one float-vector per input text."""
    embed = SentenceTransformerEmbedding(model_id="m", encoder=FakeEncoder(dim=4))
    vectors = embed.embed_passages(["aaa", "eee", ""])
    assert len(vectors) == 3
    assert all(len(vector) == 4 for vector in vectors)


@pytest.mark.os_agnostic
def test_missing_reported_dimension_is_an_embedding_error() -> None:
    """An encoder that does not report a dimension is rejected at construction."""
    with pytest.raises(EmbeddingError):
        SentenceTransformerEmbedding(model_id="m", encoder=FakeEncoder(reported_dim=None))


@pytest.mark.local_only
@pytest.mark.os_agnostic
def test_real_model_gives_semantic_ranking() -> None:
    """A real small model embeds text and ranks a paraphrase above an unrelated line."""
    pytest.importorskip("sentence_transformers")
    import math

    from semdex.adapters.embedding import load_sentence_transformer_embedding

    embed = load_sentence_transformer_embedding("sentence-transformers/all-MiniLM-L6-v2")

    def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb)

    query = embed.embed_query("how do I release the package")
    (related, unrelated) = embed.embed_passages(["publish a new version to PyPI", "the cat sat on the mat"])
    assert embed.dim > 0
    assert cosine(query, related) > cosine(query, unrelated)
