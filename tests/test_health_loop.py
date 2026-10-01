"""Unit tests for the active health-check loop.

Drives the real background thread with fakes: a probe that reports down once then
up, a recording hook, and an injected no-op ``sleep``. Asserts the loop restarts
the down backend and that ``stop()`` halts it promptly (no more probing after).
"""

from __future__ import annotations

import threading
import time

import pytest

from semdex.adapters.health import HealthLoop

pytestmark = pytest.mark.os_agnostic


class _FakeProbe:
    """Reports down on the first check, healthy on every check after."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._checks = 0

    def check(self) -> bool:
        with self._lock:
            self._checks += 1
            first = self._checks == 1
        return not first

    @property
    def checks(self) -> int:
        with self._lock:
            return self._checks


class _FakeHook:
    """Records how many times restart() was called."""

    def __init__(self) -> None:
        self.calls = 0

    def restart(self) -> bool:
        self.calls += 1
        return True


def test_loop_restarts_down_backend_then_stops_promptly() -> None:
    """A down probe triggers one restart; stop() halts the loop (no more probes)."""
    probe = _FakeProbe()
    hook = _FakeHook()
    slept: list[float] = []
    loop = HealthLoop([(probe, hook)], interval=0.001, recover_timeout=1.0, sleep=slept.append)

    loop.start()
    deadline = time.monotonic() + 2.0
    while hook.calls == 0 and time.monotonic() < deadline:
        time.sleep(0.005)

    assert hook.calls == 1  # the down backend was restarted exactly once
    loop.stop()

    settled = probe.checks
    time.sleep(0.02)
    assert probe.checks == settled  # loop is stopped: no further probing


def test_loop_without_hook_only_detects() -> None:
    """A None hook means detect-only: the loop runs and stops without a restart."""
    probe = _FakeProbe()
    slept: list[float] = []
    loop = HealthLoop([(probe, None)], interval=0.001, recover_timeout=1.0, sleep=slept.append)

    loop.start()
    deadline = time.monotonic() + 2.0
    while probe.checks == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
    loop.stop()

    settled = probe.checks
    time.sleep(0.02)
    assert probe.checks == settled  # stopped cleanly with no hook to call
