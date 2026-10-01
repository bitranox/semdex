"""Discover indexable source files under a set of paths.

Walks each path for text/markdown files and builds a SourceRef per file
(content hash + mtime for later incremental reindexing). This is the lean
phase-1 selector; gitignore-style include/exclude via ``igittigitt`` (per
``bitranox:coding-python-gitignore``) is the planned upgrade.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from ...domain.models import SourceRef
from .location import to_uri

_DEFAULT_EXTENSIONS = (".md", ".txt")
_HASH_CHUNK_SIZE = 65536  # hash in fixed chunks so memory stays bounded on huge files


def discover_sources(
    paths: Sequence[Path],
    *,
    label: str = "",
    extensions: Sequence[str] = _DEFAULT_EXTENSIONS,
    hash_chunk_size: int = _HASH_CHUNK_SIZE,
) -> list[SourceRef]:
    """Return a SourceRef for every text/markdown file under ``paths``.

    A path that is a file is taken directly (if its suffix matches); a path
    that is a directory is walked recursively. Each source is tagged with
    ``label`` (a free-form provenance tag). ``extensions`` and ``hash_chunk_size``
    are tunable config values.
    """
    allowed = tuple(extensions)
    return [_source_ref(file, label, hash_chunk_size) for path in paths for file in _iter_text_files(path, allowed)]


class FilesystemConnector:
    """SourceConnector over the filesystem: lists text/markdown files under roots.

    The list side of a filesystem source connector - it binds a set of roots and
    discovery options and yields a fresh SourceRef listing each time (so a
    reconcile picks up created/deleted files). Content fetching stays with the
    Extract port; an email connector implements the same ``sources()`` for mail.
    """

    def __init__(
        self,
        roots: Sequence[Path],
        *,
        label: str = "",
        extensions: Sequence[str] = _DEFAULT_EXTENSIONS,
        hash_chunk_size: int = _HASH_CHUNK_SIZE,
    ) -> None:
        self._roots = tuple(roots)
        self._label = label
        self._extensions = tuple(extensions)
        self._hash_chunk_size = hash_chunk_size

    def sources(self) -> list[SourceRef]:
        return discover_sources(
            self._roots,
            label=self._label,
            extensions=self._extensions,
            hash_chunk_size=self._hash_chunk_size,
        )


def _iter_text_files(path: Path, extensions: tuple[str, ...]) -> Iterator[Path]:
    if path.is_file():
        if path.suffix in extensions:
            yield path
        return
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and candidate.suffix in extensions:
            yield candidate


def _hash_file(path: Path, chunk_size: int) -> str:
    """Stream a file's sha256 in fixed chunks (bounded memory, any file size)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _source_ref(path: Path, label: str, hash_chunk_size: int) -> SourceRef:
    content_hash = _hash_file(path, hash_chunk_size)
    return SourceRef(uri=to_uri(path), label=label, content_hash=content_hash, mtime=path.stat().st_mtime)


# Static conformance assertion -- pyright verifies FilesystemConnector satisfies SourceConnector.
if TYPE_CHECKING:
    from ...application.ports import SourceConnector

    _assert_connector: SourceConnector = FilesystemConnector(())


__all__ = [
    "FilesystemConnector",
    "discover_sources",
]
