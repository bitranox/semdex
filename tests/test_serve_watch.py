"""The serve command's live-watch wiring: callback reconciles; selection is correct.

Offline + deterministic - no real watcher, no embedding server. Proves the
on-change callback actually re-indexes under the lock, and that ``_start_watchers``
watches only filesystem source datasets (skips writable / read-only).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from semdex.adapters.cli.commands.serve import (
    _reconcile_callback,  # pyright: ignore[reportPrivateUsage]
    _start_watchers,  # pyright: ignore[reportPrivateUsage]
)
from semdex.adapters.config.dataset import DatasetConfig
from semdex.composition import build_dataset_services
from semdex.domain.enums import ChangeKind, EmbeddingBackend, Partition, StoreBackend, WatchMode
from semdex.domain.models import ChangeEvent

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from lib_layered_config import Config

    from semdex.adapters.cli.context import CLIContext
    from semdex.application.ports import Watcher
    from semdex.composition import DatasetServices

pytestmark = pytest.mark.os_agnostic


def _fs_dataset(name: str, *, source: str, store_dir: str, **kw: Any) -> DatasetConfig:
    return DatasetConfig(
        name=name,
        backend=StoreBackend.JSON,
        store_dir=store_dir,
        collection=name,
        embedding_provider=EmbeddingBackend.PLACEHOLDER,
        sources=(source,),
        **kw,
    )


def _services(dataset: DatasetConfig) -> DatasetServices:
    return build_dataset_services(dataset, default_partition=Partition.TABLE)


def test_reconcile_callback_indexes_on_change(tmp_path: Path) -> None:
    """The on-change callback runs reconcile: a new source file becomes searchable."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "note.md").write_text("banana cherry apple", encoding="utf-8")
    svc = _services(_fs_dataset("notes", source=str(src), store_dir=str(tmp_path / "store")))

    _reconcile_callback(svc)(ChangeEvent(path=src / "note.md", kind=ChangeKind.CREATED))

    assert svc.store.count(collection="notes") >= 1
    hashes = svc.store.source_hashes(collection="notes")
    assert any(uri.endswith("note.md") for uri in hashes)


class _FakeWatcher:
    """Records start/stop instead of touching the filesystem."""

    def __init__(self) -> None:
        self.roots: tuple[object, ...] = ()
        self.started = False
        self.stopped = False

    def start(self, *, roots: Sequence[object], on_change: Callable[[ChangeEvent], None]) -> None:
        self.roots = tuple(roots)
        self.started = True
        _ = on_change

    def stop(self) -> None:
        self.stopped = True


def test_start_watchers_watches_only_filesystem_source_datasets(
    tmp_path: Path, config_factory: Callable[[dict[str, Any]], Config]
) -> None:
    """A filesystem dataset is watched; a writable and a read-only dataset are skipped."""
    src = tmp_path / "docs"
    src.mkdir()
    # A writable dataset has no sources (config-enforced); it must not be watched.
    kb = DatasetConfig(
        name="kb",
        backend=StoreBackend.JSON,
        store_dir=str(tmp_path / "k"),
        collection="kb",
        embedding_provider=EmbeddingBackend.PLACEHOLDER,
        writable=True,
    )
    datasets = [
        _fs_dataset("docs", source=str(src), store_dir=str(tmp_path / "d")),
        kb,
        _fs_dataset("ro", source=str(src), store_dir=str(tmp_path / "r"), read_only=True),
    ]
    config = config_factory({"dataset": [d.model_dump(mode="json") for d in datasets], "watch": {"enabled": True}})
    services = [_services(d) for d in datasets]

    made: list[_FakeWatcher] = []

    def _factory(mode: WatchMode = WatchMode.AUTO, *, debounce_ms: int = 400) -> Watcher:
        _ = (mode, debounce_ms)
        watcher = _FakeWatcher()
        made.append(watcher)
        return cast("Watcher", watcher)

    cli_ctx = cast("CLIContext", SimpleNamespace(watcher_factory=_factory))
    watchers = _start_watchers(cli_ctx, config, services)

    assert len(watchers) == 1  # only the plain filesystem dataset
    assert len(made) == 1
    assert made[0].started
    assert made[0].roots == (src,)


def test_start_watchers_disabled_returns_none(
    tmp_path: Path, config_factory: Callable[[dict[str, Any]], Config]
) -> None:
    """[watch].enabled = false starts no watchers."""
    src = tmp_path / "docs"
    src.mkdir()
    dataset = _fs_dataset("docs", source=str(src), store_dir=str(tmp_path / "d"))
    config = config_factory({"dataset": [dataset.model_dump(mode="json")], "watch": {"enabled": False}})

    def _factory(mode: WatchMode = WatchMode.AUTO, *, debounce_ms: int = 400) -> Watcher:
        raise AssertionError("factory must not be called when watching is disabled")

    cli_ctx = cast("CLIContext", SimpleNamespace(watcher_factory=_factory))
    assert _start_watchers(cli_ctx, config, [_services(dataset)]) == []
