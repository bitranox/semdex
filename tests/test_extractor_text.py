"""Filesystem text/markdown Extract adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from semdex.adapters.extractor import TextExtractor
from semdex.domain.errors import ExtractionError
from semdex.domain.models import ExtractedDocument, SourceRef


def _source(path: Path) -> SourceRef:
    return SourceRef(uri=str(path), label="", content_hash="h", mtime=0.0)


@pytest.mark.os_agnostic
def test_reads_utf8_file_text(tmp_path: Path) -> None:
    """The extractor returns the file's decoded text with its source ref."""
    path = tmp_path / "a.md"
    path.write_text("# Title\nbody", encoding="utf-8")
    source = _source(path)

    assert TextExtractor()(source) == ExtractedDocument(source=source, text="# Title\nbody")


@pytest.mark.os_agnostic
def test_missing_file_raises_extraction_error(tmp_path: Path) -> None:
    """A missing file is an extraction error, not a crash."""
    with pytest.raises(ExtractionError):
        TextExtractor()(_source(tmp_path / "nope.md"))


@pytest.mark.os_agnostic
def test_oversize_file_is_rejected(tmp_path: Path) -> None:
    """A file larger than max_bytes is rejected before being read into memory."""
    path = tmp_path / "big.md"
    path.write_text("x" * 100, encoding="utf-8")
    with pytest.raises(ExtractionError):
        TextExtractor(max_bytes=10)(_source(path))


@pytest.mark.os_agnostic
def test_undecodable_bytes_raise_extraction_error(tmp_path: Path) -> None:
    """Bytes that are not valid in the configured encoding are an error."""
    path = tmp_path / "bin.md"
    path.write_bytes(b"\xff\xfe\x00not utf8")
    with pytest.raises(ExtractionError):
        TextExtractor()(_source(path))
