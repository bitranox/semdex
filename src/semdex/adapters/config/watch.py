"""Filesystem-watch configuration parsed from the ``[watch]`` section.

Controls the live file watcher the ``serve`` command starts for each
filesystem-backed source dataset: whether to watch at all, the watch strategy
(native OS events vs polling), and how long to batch a burst of changes before
re-indexing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from ...domain.enums import WatchMode

if TYPE_CHECKING:
    from lib_layered_config import Config


class WatchConfig(BaseModel):
    """Validated, immutable filesystem-watch settings.

    Example:
        >>> WatchConfig().mode.value
        'auto'
    """

    model_config = ConfigDict(frozen=True)

    # Master switch for live watching in ``serve``. True = watch filesystem source
    # datasets and auto-reindex on change; False = index only on an explicit reindex.
    enabled: bool = True
    # Watch strategy: auto (native events + polling fallback), native (force OS
    # events), poll (force polling; use on network mounts where native is flaky).
    mode: WatchMode = WatchMode.AUTO
    # Milliseconds to batch a burst of changes before one reconcile (a multi-file
    # save fires a single re-index). Higher = fewer reindexes, more latency.
    debounce_ms: int = Field(default=400, gt=0)


def get_watch_config(config: Config) -> WatchConfig:
    """Parse the ``[watch]`` section into a WatchConfig (watching on by default).

    Example:
        >>> from lib_layered_config import Config
        >>> get_watch_config(Config({}, {})).enabled
        True
    """
    return WatchConfig.model_validate(config.get("watch", {}))


__all__ = ["WatchConfig", "get_watch_config"]
