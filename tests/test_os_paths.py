"""OS-agnostic tests: source locations round-trip through discovery and the store.

Marked ``os_agnostic`` so the CI matrix runs this on ubuntu/windows/macos,
proving a real, OS-native path (nested subdirectories, native separator)
survives discovery (as its ``file://`` URI) and the store's write/read
unchanged. Uses the offline placeholder embedding + whitespace chunker
(``build_index_production``'s defaults), so no network or model download.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from semdex.adapters.discovery import discover_sources
from semdex.adapters.discovery.location import to_uri
from semdex.application.use_cases import index_sources, search
from semdex.composition import build_index_production

pytestmark = pytest.mark.os_agnostic


def test_source_path_round_trips_through_index_and_search(tmp_path: Path) -> None:
    """A real file's path survives discovery -> index -> search unchanged."""
    doc = tmp_path / "fruit.md"
    doc.write_text("banana cherry apple", encoding="utf-8")

    services = build_index_production(tmp_path / "store")
    sources = discover_sources([tmp_path])
    assert [source.uri for source in sources] == [to_uri(doc)]

    index_sources(
        extract=services.extract,
        chunk=services.chunk,
        embedding=services.embedding,
        store=services.store_writer,
        collection="c",
        sources=sources,
        max_tokens=64,
    )
    hits = search(embedding=services.embedding, store=services.store_reader, collection="c", query="banana apple", k=3)

    assert hits
    assert hits[0].uri == to_uri(doc)


def test_discover_sources_finds_file_in_nested_subdirectory(tmp_path: Path) -> None:
    """A file nested under subdirectories is found with its OS-native path."""
    nested_dir = tmp_path / "a" / "b"
    nested_dir.mkdir(parents=True)
    note = nested_dir / "note.md"
    note.write_text("alpha beta gamma", encoding="utf-8")

    sources = discover_sources([tmp_path])

    assert [source.uri for source in sources] == [to_uri(note)]
