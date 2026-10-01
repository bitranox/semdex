"""Unit tests for the BatchingEmbedding wrapper (#44 embed batch)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from semdex.application.batching import BatchingEmbedding
from semdex.domain.models import Vector

pytestmark = pytest.mark.os_agnostic


class _FakeEmbedding:
    """Records each embed_passages call so batching is observable. Satisfies EmbeddingProvider
    structurally (a Vector is tuple[float, ...]), so it type-checks as the wrapper's inner."""

    model_id = "m"
    dim = 2

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_passages(self, texts: Sequence[str]) -> list[Vector]:
        self.calls.append(list(texts))
        return [(float(i), 0.0) for i in range(len(texts))]

    def embed_query(self, text: str) -> Vector:
        return (1.0, 0.0)


def test_splits_into_batches() -> None:
    inner = _FakeEmbedding()
    out = BatchingEmbedding(inner, batch=2).embed_passages(["a", "b", "c", "d", "e"])
    assert [len(c) for c in inner.calls] == [2, 2, 1]  # 5 texts -> batches of 2, 2, 1
    assert len(out) == 5  # all vectors returned, in order


def test_batch_at_or_above_input_is_a_single_call() -> None:
    inner = _FakeEmbedding()
    BatchingEmbedding(inner, batch=10).embed_passages(["a", "b", "c"])
    assert len(inner.calls) == 1  # 3 <= 10 -> one call (non-breaking)


def test_zero_batch_is_a_single_call() -> None:
    inner = _FakeEmbedding()
    BatchingEmbedding(inner, batch=0).embed_passages(["a", "b", "c"])
    assert len(inner.calls) == 1


def test_identity_and_query_pass_through() -> None:
    wrapped = BatchingEmbedding(_FakeEmbedding(), batch=1)
    assert wrapped.model_id == "m"
    assert wrapped.dim == 2
    assert wrapped.embed_query("q") == (1.0, 0.0)
