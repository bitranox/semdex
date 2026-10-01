"""Self-healing wrappers for the server-backed embedding and summary providers.

These decorators make a provider recover from a HARD failure - the backend model
server is down - which is distinct from part-1's transport retry (``_http_retry``)
that rides out a transient blip inside one request. When an inner call raises the
provider's domain error, the wrapper runs the operator's restart hook, waits for
the endpoint to come back healthy, and retries the call ONCE; if it fails again
the error propagates unchanged.

Layer: application. Depends only on the application ports (:class:`RestartHook`,
:class:`HealthProbe`) and the domain errors, never on an adapter - the concrete
hook/probe are injected by the composition root.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, TypeVar

from ..domain.errors import EmbeddingError, SummaryError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ..domain.models import Vector
    from .ports import EmbeddingProvider, HealthProbe, RestartHook, SummaryProvider

_T = TypeVar("_T")

# The default seconds to wait for the endpoint to recover after a restart, and the
# gap between recovery polls. Kept small so a real recovery is noticed promptly.
_DEFAULT_RECOVER_TIMEOUT = 120.0
_RECOVER_POLL_SECONDS = 1.0


def _wait_healthy(probe: HealthProbe | None, *, recover_timeout: float, sleep: Callable[[float], None]) -> None:
    """Block until ``probe`` reports healthy or ``recover_timeout`` seconds pass.

    A ``None`` probe returns at once (no way to confirm recovery, so the caller
    simply retries). Uses a monotonic deadline so a wall-clock change cannot
    stretch or cut the wait; ``sleep`` is injected so tests drive it without delay.
    """
    if probe is None:
        return
    deadline = time.monotonic() + recover_timeout
    while time.monotonic() < deadline:
        if probe.check():
            return
        sleep(_RECOVER_POLL_SECONDS)


class ResilientEmbedding:
    """Wrap an EmbeddingProvider so a down backend triggers restart-and-retry.

    Structurally satisfies :class:`~semdex.application.ports.EmbeddingProvider`:
    ``model_id`` and ``dim`` (and any future attribute) delegate to the inner
    provider, while ``embed_passages`` / ``embed_query`` route through the guard.
    """

    def __init__(
        self,
        inner: EmbeddingProvider,
        *,
        restart: RestartHook | None,
        probe: HealthProbe | None,
        recover_timeout: float = _DEFAULT_RECOVER_TIMEOUT,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        # Set the private fields FIRST: any access before this runs would hit
        # __getattr__ and recurse on the not-yet-bound _inner.
        self._inner = inner
        self._restart = restart
        self._probe = probe
        self._recover_timeout = recover_timeout
        self._sleep = sleep

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def dim(self) -> int:
        return self._inner.dim

    def embed_passages(self, texts: Sequence[str]) -> list[Vector]:
        return self._guard(lambda: self._inner.embed_passages(texts))

    def embed_query(self, text: str) -> Vector:
        return self._guard(lambda: self._inner.embed_query(text))

    def __getattr__(self, name: str) -> Any:
        # Delegate any future public attr to the inner provider (model_id/dim are
        # explicit above). Guard private names (never delegated) so a missing
        # private attribute raises instead of recursing through here.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    def _guard(self, fn: Callable[[], _T]) -> _T:
        try:
            return fn()
        except EmbeddingError:
            if self._restart is None:
                raise
            self._restart.restart()
            _wait_healthy(self._probe, recover_timeout=self._recover_timeout, sleep=self._sleep)
            # Retry ONCE; a second failure is a real outage and propagates.
            return fn()


class ResilientSummary:
    """Wrap a SummaryProvider so a down backend triggers restart-and-retry.

    The summary-tier twin of :class:`ResilientEmbedding`: ``model_id`` delegates to
    the inner provider and ``summarize`` routes through the guard, which catches
    :class:`~semdex.domain.errors.SummaryError`.
    """

    def __init__(
        self,
        inner: SummaryProvider,
        *,
        restart: RestartHook | None,
        probe: HealthProbe | None,
        recover_timeout: float = _DEFAULT_RECOVER_TIMEOUT,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._inner = inner
        self._restart = restart
        self._probe = probe
        self._recover_timeout = recover_timeout
        self._sleep = sleep

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    def summarize(self, text: str) -> str:
        return self._guard(lambda: self._inner.summarize(text))

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    def _guard(self, fn: Callable[[], _T]) -> _T:
        try:
            return fn()
        except SummaryError:
            if self._restart is None:
                raise
            self._restart.restart()
            _wait_healthy(self._probe, recover_timeout=self._recover_timeout, sleep=self._sleep)
            return fn()


__all__ = [
    "ResilientEmbedding",
    "ResilientSummary",
]
