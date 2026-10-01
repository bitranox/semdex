"""Filesystem watcher adapters (behind the ``Watcher`` port).

    * :class:`.watchfiles_watcher.WatchfilesWatcher` - the production watcher
      (native OS events with a polling fallback, opt-in via ``semdex[watch]``).

The deterministic, test-driven ``InMemoryWatcher`` lives in :mod:`..memory.index`.
"""

from __future__ import annotations

from .watchfiles_watcher import WatchfilesWatcher

__all__ = ["WatchfilesWatcher"]
