"""Filesystem Extract adapter for text and markdown files.

Reads a source file's text, bounding the read (size cap) and mapping every
filesystem or decoding failure to a domain ``ExtractionError`` so callers never
see a raw OSError. This is the lean phase-1 extractor; the optional
``markitdown`` adapter (PDF/DOCX/... -> markdown) plugs in behind the same port.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...domain.errors import ExtractionError
from ...domain.models import ExtractedDocument
from ..discovery.location import from_uri

if TYPE_CHECKING:
    from ...domain.models import SourceRef

# 5 MiB: text/markdown notes are far smaller; a bigger file is almost certainly
# binary or unintended, so reject it rather than load it into memory.
_DEFAULT_MAX_BYTES = 5 * 1024 * 1024


class TextExtractor:
    """Read a UTF-8 (configurable) text file into an ExtractedDocument."""

    def __init__(self, *, max_bytes: int = _DEFAULT_MAX_BYTES, encoding: str = "utf-8") -> None:
        self._max_bytes = max_bytes
        self._encoding = encoding

    def __call__(self, source: SourceRef) -> ExtractedDocument:
        path = from_uri(source.uri)
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise ExtractionError(f"cannot stat {path}: {exc}") from exc
        if self._max_bytes and size > self._max_bytes:
            raise ExtractionError(f"{path} is {size} bytes, over the {self._max_bytes}-byte limit")
        try:
            text = path.read_text(encoding=self._encoding)
        except UnicodeDecodeError as exc:
            raise ExtractionError(f"cannot decode {path} as {self._encoding}: {exc}") from exc
        except OSError as exc:
            raise ExtractionError(f"cannot read {path}: {exc}") from exc
        return ExtractedDocument(source=source, text=text)


# Static conformance assertion -- pyright verifies TextExtractor satisfies Extract.
if TYPE_CHECKING:
    from ...application.ports import Extract

    _assert_extract: Extract = TextExtractor()


__all__ = [
    "TextExtractor",
]
