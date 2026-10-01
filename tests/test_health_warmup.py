"""Unit tests for the best-effort startup warm-up.

Fake providers record their calls: a warm-up issues exactly one throwaway call,
and a provider that raises makes warm-up return False WITHOUT letting the
exception escape (warm-up must never fail startup).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from semdex.adapters.health import warmup_embedding, warmup_summary
from semdex.domain.errors import EmbeddingError, SummaryError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from semdex.domain.models import Vector

pytestmark = pytest.mark.os_agnostic


class _FakeEmbedding:
    """Records embed_query calls; raises on demand to model a down backend."""

    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail
        self.query_calls: list[str] = []

    @property
    def model_id(self) -> str:
        return "fake"

    @property
    def dim(self) -> int:
        return 3

    def embed_passages(self, texts: Sequence[str]) -> list[Vector]:
        return [(0.0, 0.0, 0.0) for _ in texts]

    def embed_query(self, text: str) -> Vector:
        self.query_calls.append(text)
        if self._fail:
            raise EmbeddingError("backend down")
        return (0.0, 0.0, 0.0)


class _FakeSummary:
    """Records summarize calls; raises on demand to model a down backend."""

    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail
        self.calls: list[str] = []

    @property
    def model_id(self) -> str:
        return "fake"

    def summarize(self, text: str) -> str:
        self.calls.append(text)
        if self._fail:
            raise SummaryError("backend down")
        return "ok"


def test_warmup_embedding_calls_provider_once() -> None:
    """A healthy provider is warmed with exactly one embed_query and returns True."""
    provider = _FakeEmbedding()
    assert warmup_embedding(provider) is True
    assert len(provider.query_calls) == 1


def test_warmup_embedding_swallows_failure() -> None:
    """A raising provider yields False and lets no exception escape."""
    provider = _FakeEmbedding(fail=True)
    assert warmup_embedding(provider) is False
    assert len(provider.query_calls) == 1


def test_warmup_summary_calls_provider_once() -> None:
    """A healthy summarizer is warmed with exactly one summarize and returns True."""
    provider = _FakeSummary()
    assert warmup_summary(provider) is True
    assert len(provider.calls) == 1


def test_warmup_summary_swallows_failure() -> None:
    """A raising summarizer yields False and lets no exception escape."""
    provider = _FakeSummary(fail=True)
    assert warmup_summary(provider) is False
    assert len(provider.calls) == 1
