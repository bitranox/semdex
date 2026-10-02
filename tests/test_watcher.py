"""Unit tests for the watchfiles-backed filesystem watcher.

Drives the REAL ``WatchfilesWatcher`` against a tmp dir (native OS events on the
runner) and asserts change delivery + extension filtering + stop + the missing-root
and missing-``watchfiles`` error paths. Waits are condition-based: they re-touch the
file each poll iteration (so the wait survives the inotify-watch establish delay
without a fixed startup sleep) and assert on the delivered events.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from semdex.adapters.watch import WatchfilesWatcher
from semdex.domain.enums import ChangeKind, WatchMode
from semdex.domain.errors import WatchError
from semdex.domain.models import ChangeEvent

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.os_agnostic

_HAS_WATCHFILES = importlib.util.find_spec("watchfiles") is not None
_needs_watchfiles = pytest.mark.skipif(not _HAS_WATCHFILES, reason="needs semdex[watch] (watchfiles)")


def _poke_until(mutate: Callable[[], None], predicate: Callable[[], bool], *, timeout: float = 8.0) -> bool:
    """Re-run ``mutate`` (a file write) each poll until ``predicate`` holds or timeout.

    Re-touching each iteration bridges the gap before watchfiles has established its
    watch, without a fixed sleep - once the watch is up, the next touch is delivered.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        mutate()
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


@_needs_watchfiles
def test_created_and_modified_events_are_delivered(tmp_path: Path) -> None:
    """Writing a .md file delivers a CREATED/MODIFIED event; a later write delivers MODIFIED.

    macOS FSEvents coalesces a path's flags, so a re-write shortly after creation can still carry
    the created flag and arrive as CREATED. serve reconciles on any event whatever its kind, so
    both satisfy the contract there; elsewhere the OS reports the modification and it is required.
    """
    rewrite_kinds = {ChangeKind.MODIFIED, ChangeKind.CREATED} if sys.platform == "darwin" else {ChangeKind.MODIFIED}
    events: list[ChangeEvent] = []
    watcher = WatchfilesWatcher(mode=WatchMode.AUTO, debounce_ms=50)
    watcher.start(roots=[tmp_path], on_change=events.append)
    try:
        target = tmp_path / "note.md"
        seq = [0]

        def _write() -> None:
            seq[0] += 1
            target.write_text(f"hello {seq[0]}", encoding="utf-8")

        assert _poke_until(_write, lambda: any(e.path == target for e in events)), "no event for the file"
        assert {e.kind for e in events if e.path == target} & {ChangeKind.CREATED, ChangeKind.MODIFIED}

        events.clear()
        assert _poke_until(_write, lambda: any(e.kind in rewrite_kinds for e in events if e.path == target)), (
            f"no {sorted(k.value for k in rewrite_kinds)} event for the re-written file; "
            f"saw {[(e.kind.value, e.path.name) for e in events]}"
        )
    finally:
        watcher.stop()


@_needs_watchfiles
def test_non_source_extension_is_filtered_out(tmp_path: Path) -> None:
    """A change to a non-.md/.txt file is not delivered; a .md alongside it still is."""
    events: list[ChangeEvent] = []
    watcher = WatchfilesWatcher(debounce_ms=50)
    watcher.start(roots=[tmp_path], on_change=events.append)
    try:
        noise = tmp_path / "ignore.log"
        md = tmp_path / "keep.md"
        seq = [0]

        def _write_both() -> None:
            seq[0] += 1
            noise.write_text(f"noise {seq[0]}", encoding="utf-8")
            md.write_text(f"kept {seq[0]}", encoding="utf-8")

        assert _poke_until(_write_both, lambda: any(e.path == md for e in events)), "the .md change was never delivered"
        assert all(e.path.suffix in (".md", ".txt") for e in events)
        assert not any(e.path.name == "ignore.log" for e in events)
    finally:
        watcher.stop()


@_needs_watchfiles
def test_stop_halts_delivery(tmp_path: Path) -> None:
    """After stop(), later changes are not delivered."""
    events: list[ChangeEvent] = []
    watcher = WatchfilesWatcher(debounce_ms=50)
    watcher.start(roots=[tmp_path], on_change=events.append)
    first = tmp_path / "first.md"
    seq = [0]

    def _write_first() -> None:
        seq[0] += 1
        first.write_text(f"a {seq[0]}", encoding="utf-8")

    assert _poke_until(_write_first, lambda: len(events) > 0), "watcher never delivered an event"
    watcher.stop()
    events.clear()
    (tmp_path / "second.md").write_text("b", encoding="utf-8")
    time.sleep(0.6)  # bounded negative wait: assert nothing arrives after stop
    assert events == []


@_needs_watchfiles
def test_missing_root_raises_watch_error(tmp_path: Path) -> None:
    """Starting on a nonexistent root is a clean WatchError."""
    watcher = WatchfilesWatcher()
    with pytest.raises(WatchError):
        watcher.start(roots=[tmp_path / "does-not-exist"], on_change=lambda _e: None)


def test_missing_watchfiles_raises_watch_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the watchfiles extra, start() fails fast with an install hint."""
    real_find_spec = importlib.util.find_spec

    def _fake_find_spec(name: str, package: str | None = None) -> object:
        return None if name == "watchfiles" else real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)
    watcher = WatchfilesWatcher()
    with pytest.raises(WatchError, match="watchfiles"):
        watcher.start(roots=[tmp_path], on_change=lambda _e: None)
