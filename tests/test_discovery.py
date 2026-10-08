"""Filesystem source discovery: walk paths for text/markdown files."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from semdex.adapters.discovery import FilesystemConnector, discover_sources
from semdex.adapters.discovery.location import from_uri, to_uri
from semdex.adapters.extractor._http import guess_mime, source_name


@pytest.mark.os_agnostic
def test_discovers_text_and_markdown_recursively(tmp_path: Path) -> None:
    """Discovery finds .md/.txt files (recursively) and skips other types."""
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    (tmp_path / "c.png").write_bytes(b"\x89PNG")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.md").write_text("delta", encoding="utf-8")

    sources = discover_sources([tmp_path], label="curated")

    assert sorted(from_uri(s.uri).name for s in sources) == ["a.md", "b.txt", "d.md"]
    assert all(s.label == "curated" for s in sources)
    assert all(s.content_hash and s.mtime > 0 for s in sources)


@pytest.mark.os_agnostic
def test_accepts_a_single_file_path(tmp_path: Path) -> None:
    """A file path (not a directory) is discovered directly."""
    file = tmp_path / "note.md"
    file.write_text("x", encoding="utf-8")
    assert [s.uri for s in discover_sources([file])] == [to_uri(file)]


@pytest.mark.os_agnostic
def test_a_symlinked_document_reaches_extractors_under_its_own_name(tmp_path: Path) -> None:
    """A doc linked into a content-addressed store keeps the name and type it was selected by.

    Discovery picks ``report.pdf`` by its suffix; the HTTP extractors upload it under
    ``source_name`` with ``guess_mime``. Naming it by the extension-less blob made Xberg
    answer 500 ``UnsupportedFormatError: application/octet-stream`` on every such file.
    """
    store = tmp_path / "store"
    store.mkdir()
    blob = store / "abc123"
    blob.write_bytes(b"%PDF-1.3\n")
    docs = tmp_path / "docs"
    docs.mkdir()
    try:
        (docs / "report.pdf").symlink_to(blob)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable here: {exc}")

    [source] = discover_sources([docs], extensions=(".pdf",))

    assert source_name(source) == "report.pdf"
    assert guess_mime(source) == "application/pdf"


@pytest.mark.os_agnostic
def test_content_hash_is_sha256_of_bytes(tmp_path: Path) -> None:
    """The SourceRef content_hash is the sha256 of the file's bytes."""
    file = tmp_path / "a.md"
    file.write_text("hello", encoding="utf-8")
    assert discover_sources([file])[0].content_hash == hashlib.sha256(b"hello").hexdigest()


@pytest.mark.os_agnostic
def test_content_hash_streams_files_larger_than_one_chunk(tmp_path: Path) -> None:
    """A file bigger than the read chunk hashes correctly (streamed, not slurped)."""
    data = b"x" * 200_000  # larger than the 64 KiB hashing chunk
    file = tmp_path / "big.txt"
    file.write_bytes(data)
    assert discover_sources([file])[0].content_hash == hashlib.sha256(data).hexdigest()


@pytest.mark.os_agnostic
def test_default_label_is_empty(tmp_path: Path) -> None:
    """Without a label, discovered sources carry the empty label."""
    file = tmp_path / "a.md"
    file.write_text("x", encoding="utf-8")
    assert discover_sources([file])[0].label == ""


@pytest.mark.os_agnostic
def test_filesystem_connector_lists_discovered_sources(tmp_path: Path) -> None:
    """FilesystemConnector.sources() is the SourceConnector view over discovery."""
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "c.png").write_bytes(b"\x89PNG")  # non-text: skipped

    connector = FilesystemConnector([tmp_path], label="curated")
    sources = connector.sources()

    assert [from_uri(s.uri).name for s in sources] == ["a.md"]
    assert all(s.label == "curated" for s in sources)
    assert all(s.content_hash and s.mtime > 0 for s in sources)
