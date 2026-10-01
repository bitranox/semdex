"""Opt-in active health-check loop for the configured backend endpoints.

``HealthLoop`` runs a background daemon thread that, every ``interval`` seconds,
probes each configured endpoint; when one is down it runs that endpoint's restart
hook (if any) and waits up to ``recover_timeout`` for it to come back. This is the
PROACTIVE counterpart to the reactive per-call self-heal in
``application/resilience.py`` - it notices an idle backend has died before the
next query does. Off by default; enabled by ``[health].check_enabled``.

All waiting goes through the injected ``sleep`` and a ``threading.Event`` stop
flag, so the loop shuts down promptly and tests drive it deterministically without
real delay.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ...application.ports import HealthProbe, RestartHook

logger = logging.getLogger(__name__)

# Poll granularity: the loop sleeps in slices no longer than this so a stop() is
# noticed within roughly one slice even when interval/recover_timeout are large.
_POLL_SECONDS = 1.0
# Extra seconds added to the join timeout so a shutdown cannot block forever if a
# probe/hook is mid-call when stop() is requested.
_JOIN_GRACE_SECONDS = 5.0


class HealthLoop:
    """Background thread that probes endpoints and restarts a down one.

    ``checks`` pairs each :class:`~semdex.application.ports.HealthProbe` with the
    :class:`~semdex.application.ports.RestartHook` to run when it reports down (or
    ``None`` to only detect, not restart). :meth:`start` spawns the daemon thread;
    :meth:`stop` signals it and joins (bounded).
    """

    def __init__(
        self,
        checks: Sequence[tuple[HealthProbe, RestartHook | None]],
        *,
        interval: float,
        recover_timeout: float,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._checks = list(checks)
        self._interval = interval
        self._recover_timeout = recover_timeout
        self._sleep = sleep
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Spawn the daemon thread running the check loop (idempotent per instance)."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="semdex-health-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the loop to stop and join the thread (bounded wait)."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self._interval + _JOIN_GRACE_SECONDS)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._check_once()
            self._wait(self._interval)

    def _check_once(self) -> None:
        """Probe every endpoint once; restart and wait for recovery on a down one."""
        for probe, hook in self._checks:
            if self._stop.is_set():
                return
            if probe.check():
                continue
            logger.warning("health check reported a down backend")
            if hook is not None:
                hook.restart()
            self._wait_recover(probe)

    def _wait_recover(self, probe: HealthProbe) -> None:
        """Poll ``probe`` until healthy or ``recover_timeout`` elapses (stop-aware)."""
        deadline = time.monotonic() + self._recover_timeout
        while time.monotonic() < deadline:
            if self._stop.is_set() or probe.check():
                return
            self._sleep(_POLL_SECONDS)

    def _wait(self, seconds: float) -> None:
        """Sleep ``seconds`` in small stop-aware slices between check cycles."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self._stop.is_set():
                return
            self._sleep(min(_POLL_SECONDS, seconds))


__all__ = [
    "HealthLoop",
]
