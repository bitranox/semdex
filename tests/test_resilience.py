"""Unit tests for the restart-on-exhaustion self-heal wrappers.

Fakes injected at the edge (a failing inner provider, a recording restart hook, a
healthy probe, a no-op sleep) so the wrapper's restart-wait-retry-once path runs
offline and deterministically. Covers the happy passthrough, a single restart +
one retry, propagation after the retry also fails, attribute delegation, and the
no-restart shortcut.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from semdex.application.resilience import ResilientEmbedding, ResilientSummary
from semdex.domain.errors import EmbeddingError, SummaryError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from semdex.domain.models import Vector

pytestmark = pytest.mark.os_agnostic

_VALUE: Vector = (1.0, 2.0)


class _InnerEmbedding:
    """Fails its first ``fail_times`` embed_query calls, then returns a fixed vector."""

    def __init__(self, *, fail_times: int = 0) -> None:
        self._fail_times = fail_times
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "inner-model"

    @property
    def dim(self) -> int:
        return 7

    def embed_passages(self, texts: Sequence[str]) -> list[Vector]:
        return [_VALUE for _ in texts]

    def embed_query(self, text: str) -> Vector:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise EmbeddingError("backend down")
        return _VALUE


class _InnerSummary:
    """Fails its first ``fail_times`` summarize calls, then returns a fixed string."""

    def __init__(self, *, fail_times: int = 0) -> None:
        self._fail_times = fail_times
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "sum-model"

    def summarize(self, text: str) -> str:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise SummaryError("backend down")
        return "summary"


class _FakeRestart:
    """Records restart() calls."""

    def __init__(self) -> None:
        self.calls = 0

    def restart(self) -> bool:
        self.calls += 1
        return True


class _FakeProbe:
    """Reports a fixed health, counting checks."""

    def __init__(self, *, healthy: bool = True) -> None:
        self._healthy = healthy
        self.calls = 0

    def check(self) -> bool:
        self.calls += 1
        return self._healthy


def _noop_sleep(_seconds: float) -> None:
    return None


# -- ResilientEmbedding ------------------------------------------------------


def test_embedding_happy_path_passes_through_and_delegates_attrs() -> None:
    """No failure: the call passes through and .model_id/.dim delegate to inner."""
    inner = _InnerEmbedding()
    restart = _FakeRestart()
    wrapped = ResilientEmbedding(inner, restart=restart, probe=_FakeProbe(), sleep=_noop_sleep)

    assert wrapped.embed_query("x") == _VALUE
    assert wrapped.embed_passages(["a", "b"]) == [_VALUE, _VALUE]
    assert wrapped.model_id == "inner-model"
    assert wrapped.dim == 7
    assert restart.calls == 0


def test_embedding_restarts_waits_then_retries_once() -> None:
    """One failure triggers a restart, a health wait, and one successful retry."""
    inner = _InnerEmbedding(fail_times=1)
    restart = _FakeRestart()
    probe = _FakeProbe(healthy=True)
    wrapped = ResilientEmbedding(inner, restart=restart, probe=probe, recover_timeout=5.0, sleep=_noop_sleep)

    assert wrapped.embed_query("x") == _VALUE
    assert restart.calls == 1
    assert probe.calls >= 1  # waited for the endpoint to report healthy
    assert inner.calls == 2  # original failure + one retry


def test_embedding_propagates_when_retry_also_fails() -> None:
    """If the single retry fails too, the error propagates after one restart."""
    inner = _InnerEmbedding(fail_times=5)
    restart = _FakeRestart()
    wrapped = ResilientEmbedding(inner, restart=restart, probe=_FakeProbe(), sleep=_noop_sleep)

    with pytest.raises(EmbeddingError):
        wrapped.embed_query("x")
    assert restart.calls == 1
    assert inner.calls == 2  # original + exactly one retry, then propagate


def test_embedding_without_restart_propagates_immediately() -> None:
    """A None restart hook means the error propagates at once, with no retry."""
    inner = _InnerEmbedding(fail_times=1)
    wrapped = ResilientEmbedding(inner, restart=None, probe=None, sleep=_noop_sleep)

    with pytest.raises(EmbeddingError):
        wrapped.embed_query("x")
    assert inner.calls == 1  # no retry


# -- ResilientSummary --------------------------------------------------------


def test_summary_happy_path_passes_through_and_delegates_model_id() -> None:
    """No failure: summarize passes through and .model_id delegates to inner."""
    inner = _InnerSummary()
    restart = _FakeRestart()
    wrapped = ResilientSummary(inner, restart=restart, probe=_FakeProbe(), sleep=_noop_sleep)

    assert wrapped.summarize("doc") == "summary"
    assert wrapped.model_id == "sum-model"
    assert restart.calls == 0


def test_summary_restarts_waits_then_retries_once() -> None:
    """One failure triggers a restart, a health wait, and one successful retry."""
    inner = _InnerSummary(fail_times=1)
    restart = _FakeRestart()
    probe = _FakeProbe(healthy=True)
    wrapped = ResilientSummary(inner, restart=restart, probe=probe, recover_timeout=5.0, sleep=_noop_sleep)

    assert wrapped.summarize("doc") == "summary"
    assert restart.calls == 1
    assert probe.calls >= 1
    assert inner.calls == 2


def test_summary_without_restart_propagates_immediately() -> None:
    """A None restart hook means the error propagates at once, with no retry."""
    inner = _InnerSummary(fail_times=1)
    wrapped = ResilientSummary(inner, restart=None, probe=None, sleep=_noop_sleep)

    with pytest.raises(SummaryError):
        wrapped.summarize("doc")
    assert inner.calls == 1
