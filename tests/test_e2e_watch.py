"""End-to-end live watch: edit a file -> the real watcher auto-reindexes -> search sees it.

Uses the REAL WatchfilesWatcher wired to the serve command's reconcile callback,
but the offline placeholder embedding + a json store, so the only real dependency
is native filesystem events (hence local_only - CI runners' inotify timing is
unreliable). All waits poll a condition with a timeout; a non-converging watch
fails loud, never silently passes.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from semdex.adapters.cli.commands.serve import _reconcile_callback  # pyright: ignore[reportPrivateUsage]
from semdex.adapters.config.dataset import DatasetConfig
from semdex.adapters.watch import WatchfilesWatcher
from semdex.application.use_cases.searching import search
from semdex.composition import build_dataset_services
from semdex.domain.enums import EmbeddingBackend, Partition, StoreBackend, WatchMode

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = [
    pytest.mark.local_only,
    pytest.mark.os_agnostic,
    pytest.mark.skipif(importlib.util.find_spec("watchfiles") is None, reason="needs semdex[watch]"),
]


def _wait_until(predicate: Callable[[], bool], *, timeout: float = 10.0, poll: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    return predicate()


def _hits_for(svc: object, query: str) -> list[str]:  # svc: DatasetServices
    from semdex.composition import DatasetServices
    from semdex.domain.errors import VectorStoreError

    assert isinstance(svc, DatasetServices)
    try:
        hits = search(embedding=svc.embedding, store=svc.store, collection=svc.collection, query=query, k=5)
    except VectorStoreError:
        return []  # the collection does not exist until the first reconcile creates it
    return [hit.uri for hit in hits]


def test_live_watch_auto_reindexes_on_create_modify_delete(tmp_path: Path) -> None:
    """A created .md becomes searchable; a delete removes it - all via the live watcher."""
    src = tmp_path / "docs"
    src.mkdir()
    dataset = DatasetConfig(
        name="docs",
        backend=StoreBackend.JSON,
        store_dir=str(tmp_path / "store"),
        collection="docs",
        embedding_provider=EmbeddingBackend.PLACEHOLDER,
        sources=(str(src),),
    )
    svc = build_dataset_services(dataset, default_partition=Partition.TABLE)
    watcher = WatchfilesWatcher(mode=WatchMode.AUTO, debounce_ms=50)
    watcher.start(roots=[src], on_change=_reconcile_callback(svc))
    try:
        note = src / "note.md"
        seq = [0]

        def _write() -> None:
            seq[0] += 1
            note.write_text(f"banana cherry apple {seq[0]}", encoding="utf-8")

        # Re-touch each poll so the create survives the inotify-establish delay.
        assert _poke_until(_write, lambda: any(u.endswith("note.md") for u in _hits_for(svc, "banana apple"))), (
            "watcher never auto-indexed the created file"
        )

        note.unlink()  # watch is established by now; a single delete event suffices
        assert _wait_until(lambda: all(not u.endswith("note.md") for u in _hits_for(svc, "banana apple"))), (
            "watcher never pruned the deleted file"
        )
    finally:
        watcher.stop()


def _poke_until(mutate: Callable[[], None], predicate: Callable[[], bool], *, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        mutate()
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()
