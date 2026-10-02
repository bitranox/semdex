"""Unit tests for the markitdown-mcp Extract adapter.

No container required: a fake ``convert`` callable is injected so the adapter's
file reading, ``data:`` URI encoding, size cap, and endpoint handling are tested
without an MCP server. The real container is covered by the ``local_only`` Docker
test. The MCP-result parsing helpers are tested directly against the real
``mcp.types.CallToolResult`` / ``TextContent`` (not stubs), so a field-name
mismatch between the adapter and the installed ``mcp`` package cannot hide
behind a stub that mirrors the adapter's own (wrong) expectations.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from mcp.types import CallToolResult, TextContent

from semdex.adapters.extractor.markitdown import (
    MarkitdownExtractor,
    _mcp_url,  # pyright: ignore[reportPrivateUsage]
    _text_of,  # pyright: ignore[reportPrivateUsage]
)
from semdex.adapters.discovery.location import to_uri
from semdex.domain.errors import ExtractionError
from semdex.domain.models import SourceRef

pytestmark = pytest.mark.os_agnostic


def _source(tmp_path: Path, *, name: str = "report.docx", data: bytes = b"payload") -> SourceRef:
    path = tmp_path / name
    path.write_bytes(data)
    return SourceRef(uri=to_uri(path), label="lbl", content_hash="h", mtime=1.0)


def test_happy_path_encodes_data_uri_and_returns_markdown(tmp_path: Path) -> None:
    """The file is sent as a base64 data: URI and the converter's text returned."""
    seen: list[str] = []

    def fake_convert(uri: str) -> str:
        seen.append(uri)
        return "# converted"

    extractor = MarkitdownExtractor("http://host:3001", convert=fake_convert)
    doc = extractor(_source(tmp_path, data=b"hello"))

    assert doc.text == "# converted"
    assert seen[0].startswith("data:")
    assert base64.b64encode(b"hello").decode() in seen[0]


def test_oversized_file_rejected(tmp_path: Path) -> None:
    """A file over the cap is rejected before the converter is called."""

    def fake_convert(_uri: str) -> str:  # pragma: no cover - must not run
        raise AssertionError("convert should not be called for an oversized file")

    extractor = MarkitdownExtractor("http://host:3001", max_bytes=4, convert=fake_convert)
    with pytest.raises(ExtractionError, match="over the"):
        extractor(_source(tmp_path, data=b"too-large"))


def test_missing_endpoint_rejected() -> None:
    """No endpoint fails clearly at construction."""
    with pytest.raises(ExtractionError, match="requires"):
        MarkitdownExtractor(None)


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("http://h:3001", "http://h:3001/mcp/"),
        ("http://h:3001/", "http://h:3001/mcp/"),
        ("http://h:3001/mcp", "http://h:3001/mcp/"),
    ],
)
def test_mcp_url_normalization(endpoint: str, expected: str) -> None:
    """The MCP URL targets the /mcp/ mount (trailing slash) exactly once."""
    assert _mcp_url(endpoint) == expected


def test_text_of_reads_first_text_block() -> None:
    """_text_of returns the first text content block."""
    result = CallToolResult(
        content=[TextContent(type="text", text="markdown")],
    )
    assert _text_of(result, "http://h/mcp") == "markdown"


def test_an_error_result_raises_instead_of_returning_the_error_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """An error result maps to ExtractionError, never the error text as if it were content.

    ``mcp.types.CallToolResult`` names this field ``is_error`` (not the camelCase
    ``isError`` only ``fastmcp``'s deprecated bridge exposes), so ``_text_of`` must
    read the real field - else a failed conversion's error text is indexed as if it
    were the document.

    ``import fastmcp`` (which some other test module in the full suite always does
    at collection time, e.g. ``tests/test_mcp_server.py``) installs a camelCase
    ``isError`` compatibility PROPERTY directly on ``mcp.types.CallToolResult``. A
    stale ``getattr(result, "isError", False)`` implementation would then also read
    ``True`` here and this test would pass against the very defect it exists to
    catch - green only because of another module's import, not because the fix is
    real. Deleting the property (a third-party global; the allowed monkeypatch
    edge - our own code has no seam to inject here) makes this test's verdict hold
    regardless of import order or which other test modules already ran.
    """
    for cls in CallToolResult.__mro__:
        if "isError" in vars(cls):
            monkeypatch.delattr(cls, "isError", raising=False)
            break

    result = CallToolResult(
        content=[TextContent(type="text", text="conversion failed")],
        is_error=True,
    )
    with pytest.raises(ExtractionError, match="http://h/mcp"):
        _text_of(result, "http://h/mcp")


def test_text_of_raises_when_no_text() -> None:
    """A result with no text content maps to ExtractionError."""
    with pytest.raises(ExtractionError, match="no text"):
        _text_of(CallToolResult(content=[]), "http://h/mcp")
