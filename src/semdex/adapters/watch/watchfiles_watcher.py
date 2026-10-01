"""Cross-platform filesystem watcher backed by the ``watchfiles`` library.

The production :class:`~semdex.application.ports.Watcher`: runs ``watchfiles.watch``
in a daemon thread and delivers one :class:`~semdex.domain.models.ChangeEvent`
per relevant file change to the injected callback. ``watchfiles`` uses native OS
events (Rust ``notify``) with a built-in polling fallback; ``WatchMode.POLL``
forces polling, which is the safe choice for network mounts where native events
are unreliable. Opt-in via ``semdex[watch]``.

The watcher never reads or mutates a source - it only reports *that* a path
changed; the reconcile use case (driven by the ``serve`` command's callback) does
the re-index. A failure in the watch thread is logged and stops watching for that
root, never crashes the server (self-healing).
"""

from __future__ import annotations

import importlib.util
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from ...domain.enums import ChangeKind, WatchMode
from ...domain.errors import WatchError
from ...domain.models import ChangeEvent

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)

_DEFAULT_EXTENSIONS = (".md", ".txt")
_DEFAULT_DEBOUNCE_MS = 400
_STOP_JOIN_TIMEOUT_S = 5.0


class WatchfilesWatcher:
    """Watcher over ``watchfiles``: native OS events (or polling) -> callback.

    ``mode`` selects the strategy (``POLL`` forces the polling observer);
    ``debounce_ms`` is how long ``watchfiles`` batches a burst of changes before
    delivering them (so a save that touches several files fires one reconcile);
    ``extensions`` filters to the source file types the connector indexes.
    """

    def __init__(
        self,
        *,
        mode: WatchMode = WatchMode.AUTO,
        debounce_ms: int = _DEFAULT_DEBOUNCE_MS,
        extensions: Sequence[str] = _DEFAULT_EXTENSIONS,
    ) -> None:
        self._mode = mode
        self._debounce_ms = debounce_ms
        self._extensions = tuple(ext.lower() for ext in extensions)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, *, roots: Sequence[Path], on_change: Callable[[ChangeEvent], None]) -> None:
        """Begin watching ``roots``; call ``on_change`` for each matching change.

        Raises :class:`~semdex.domain.errors.WatchError` if ``watchfiles`` is not
        installed or a root does not exist.
        """
        resolved = [Path(root) for root in roots]
        for root in resolved:
            if not root.exists():
                raise WatchError(f"watch root does not exist: {root}")
        if importlib.util.find_spec("watchfiles") is None:  # optional dep; fail before spawning the thread
            raise WatchError("watchfiles is not installed; install semdex[watch]")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(resolved, on_change), name="semdex-watch", daemon=True)
        self._thread.start()

    def _run(self, roots: Sequence[Path], on_change: Callable[[ChangeEvent], None]) -> None:
        import watchfiles

        kind_of = {
            watchfiles.Change.added: ChangeKind.CREATED,
            watchfiles.Change.modified: ChangeKind.MODIFIED,
            watchfiles.Change.deleted: ChangeKind.DELETED,
        }
        try:
            for batch in watchfiles.watch(
                *(str(root) for root in roots),
                stop_event=self._stop,
                force_polling=self._mode is WatchMode.POLL,
                debounce=self._debounce_ms,
                recursive=True,
            ):
                for change, raw_path in batch:
                    path = Path(raw_path)
                    if path.suffix.lower() in self._extensions:
                        on_change(ChangeEvent(path=path, kind=kind_of[change]))
        except Exception:  # self-healing: a watch failure must not crash serve; log + exit the thread
            logger.exception("filesystem watcher thread failed; watching stopped for roots %s", roots)

    def stop(self) -> None:
        """Signal the watch thread to stop and join it (bounded wait)."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=_STOP_JOIN_TIMEOUT_S)
        self._thread = None


# Static conformance assertion -- WatchfilesWatcher satisfies the Watcher port.
if TYPE_CHECKING:
    from ...application.ports import Watcher

    _assert_watcher: Watcher = WatchfilesWatcher()


__all__ = ["WatchfilesWatcher"]
