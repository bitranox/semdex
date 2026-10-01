"""Source discovery adapters.

Contents:
    * :mod:`.filesystem` - discover_sources + FilesystemConnector (walk paths for
      text/markdown files)
"""

from __future__ import annotations

from .filesystem import FilesystemConnector, discover_sources

__all__ = [
    "FilesystemConnector",
    "discover_sources",
]
