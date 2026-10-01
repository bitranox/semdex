"""markitdown-mcp Extract adapter (Office/PDF/... to markdown over MCP).

markitdown ships no plain REST server - its container (``mcp/markitdown``) is an
MCP server exposing one tool, ``convert_to_markdown(uri)``, over Streamable HTTP
at ``{endpoint}/mcp``. This adapter reads the file, encodes it as a ``data:`` URI
(so the container needs no shared volume), calls the tool via the ``mcp`` client
SDK, and returns the markdown. markitdown is lightweight (no torch), so its
container is CPU-only. Needs the optional ``semdex[markitdown]`` extra
(``mcp`` + ``httpx2``, the HTTP client mcp 2.x is built on) and
``[extractor].endpoint`` pointing at the container.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from typing import TYPE_CHECKING

from ...domain.errors import ExtractionError
from ...domain.models import ExtractedDocument
from ._http import DEFAULT_MAX_BYTES, guess_mime, read_source_bytes, require_endpoint

if TYPE_CHECKING:
    from mcp.types import CallToolResult

    from ...domain.models import SourceRef

# A callable turning a document ``data:`` URI into markdown. The default calls
# the real MCP server; tests inject a fake so the adapter's file handling is
# exercised without a container.
ConvertFn = Callable[[str], str]


class MarkitdownExtractor:
    """Extract a document to markdown via a markitdown-mcp container."""

    def __init__(
        self,
        endpoint: str | None,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        timeout: float = 120.0,
        convert: ConvertFn | None = None,
    ) -> None:
        resolved = require_endpoint(endpoint, backend="markitdown")
        self._max_bytes = max_bytes
        self._convert = convert if convert is not None else _mcp_converter(_mcp_url(resolved), timeout)

    def __call__(self, source: SourceRef) -> ExtractedDocument:
        content = read_source_bytes(source, max_bytes=self._max_bytes)
        encoded = base64.b64encode(content).decode("ascii")
        data_uri = f"data:{guess_mime(source)};base64,{encoded}"
        return ExtractedDocument(source=source, text=self._convert(data_uri))


def _mcp_url(endpoint: str) -> str:
    """Return the Streamable-HTTP MCP URL for a base endpoint.

    markitdown-mcp mounts the MCP app at ``/mcp/`` (trailing slash) and 307-redirects
    ``/mcp`` to it, so target the slash form directly to avoid the redirect round-trip.
    """
    trimmed = endpoint.rstrip("/")
    if not trimmed.endswith("/mcp"):
        trimmed = f"{trimmed}/mcp"
    return trimmed + "/"


def _mcp_converter(url: str, timeout: float) -> ConvertFn:
    """Build the production converter that calls the markitdown MCP server."""

    def convert(data_uri: str) -> str:
        import asyncio

        return asyncio.run(_call_convert_tool(url, data_uri, timeout))

    return convert


async def _call_convert_tool(url: str, data_uri: str, timeout: float) -> str:
    """Open an MCP session, call ``convert_to_markdown``, and return its text."""
    try:
        # httpx2, not httpx: mcp 2.x is built on httpx2 and its client types are NOT
        # interchangeable with httpx's, so an httpx.AsyncClient here is rejected by
        # streamable_http_client. Every other adapter in this package talks plain REST and
        # stays on httpx; only this one follows the SDK.
        import httpx2
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as exc:  # pragma: no cover - only hit without the extra installed
        raise ExtractionError("mcp is not installed; install semdex[markitdown]") from exc
    try:
        async with (
            httpx2.AsyncClient(timeout=timeout, follow_redirects=True) as http_client,
            # Two streams, not three: mcp 2.x dropped the trailing get-session-id callable
            # from what the transport yields, and this adapter never used it.
            streamable_http_client(url, http_client=http_client) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool("convert_to_markdown", {"uri": data_uri})
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(f"markitdown MCP call to {url} failed: {exc}") from exc
    return _text_of(result, url)


def _text_of(result: CallToolResult, url: str) -> str:
    """Read the markdown text out of an MCP ``CallToolResult``.

    Reads ``result.is_error``, the field's real name on ``mcp.types.CallToolResult``
    (2.x); the camelCase ``isError`` only appears once ``fastmcp``'s deprecated
    compatibility bridge has patched the class, which this adapter never imports.
    Only ``TextContent`` blocks are read for text, so an image/audio block is
    skipped rather than silently read as empty text.
    """
    from mcp.types import TextContent

    if result.is_error:
        raise ExtractionError(f"markitdown MCP at {url} reported an error")
    for block in result.content:
        if isinstance(block, TextContent):
            return block.text
    raise ExtractionError(f"markitdown MCP at {url} returned no text content")


# Static conformance assertion -- pyright verifies MarkitdownExtractor satisfies Extract.
if TYPE_CHECKING:
    from ...application.ports import Extract

    _assert_extract: Extract = MarkitdownExtractor("http://x")


__all__ = [
    "MarkitdownExtractor",
]
