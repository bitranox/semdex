"""Shared mechanics for the HTTP-based Extract adapters.

The four doc-converter backends (markitdown / xberg / docling / mineru) each
run as their own container and speak their own wire contract, so their request
building and response parsing stay in their own module - they are deliberately
NOT unified (coupling divergent APIs is the accidental-duplication trap called
out in CLAUDE.md for the SQL vector stores). Only the cross-cutting mechanics
live here: the size-bounded file read, the optional-dependency import guard, the
MIME guess, and the POST-with-error-mapping to the domain ``ExtractionError``.
"""

from __future__ import annotations

import mimetypes
import time
from typing import TYPE_CHECKING, Any

from ...domain.errors import ExtractionError
from .._http_retry import DEFAULT_RETRIES, retry_http
from ..discovery.location import from_uri

if TYPE_CHECKING:
    from collections.abc import Callable

    import httpx

    from ...domain.models import SourceRef

# 25 MiB: doc files (PDF/DOCX) run larger than notes, but a bigger upload is
# almost certainly unintended. The cap is checked against stat() before the
# file is read, so an oversized file is never loaded into memory (bounded read).
DEFAULT_MAX_BYTES = 25 * 1024 * 1024


def read_source_bytes(source: SourceRef, *, max_bytes: int) -> bytes:
    """Read a source file's bytes, rejecting an oversized file before reading.

    The size cap is checked against ``stat`` first, so a file over the limit is
    never materialized in memory. Filesystem failures map to ``ExtractionError``
    so callers never see a raw ``OSError``.
    """
    path = from_uri(source.uri)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ExtractionError(f"cannot stat {path}: {exc}") from exc
    if max_bytes and size > max_bytes:
        raise ExtractionError(f"{path} is {size} bytes, over the {max_bytes}-byte limit")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ExtractionError(f"cannot read {path}: {exc}") from exc


def guess_mime(source: SourceRef) -> str:
    """Best-effort MIME type from the file name, defaulting to octet-stream."""
    return mimetypes.guess_type(from_uri(source.uri).name)[0] or "application/octet-stream"


def source_name(source: SourceRef) -> str:
    """The source's file name, for a multipart upload field (file scheme)."""
    return from_uri(source.uri).name


def require_httpx() -> Any:
    """Import httpx or raise a clear ExtractionError naming the missing extra."""
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - only hit without the extra installed
        raise ExtractionError("httpx is not installed; install the extractor's optional extra") from exc
    return httpx


def require_endpoint(endpoint: str | None, *, backend: str) -> str:
    """Return the endpoint URL or raise if the HTTP backend has none configured."""
    if not endpoint:
        raise ExtractionError(f"{backend} extractor requires [extractor].endpoint")
    return endpoint


def post_multipart_json(  # noqa: PLR0913 - one shared transport: url/timeout/client/files/data plus the retry knobs
    *,
    url: str,
    timeout: float,
    client: httpx.Client | None,
    files: Any,
    data: dict[str, str] | None = None,
    retries: int = DEFAULT_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """POST a multipart request and return the parsed JSON body.

    When ``client`` is ``None`` a short-lived client is created and closed here
    (right for the CLI). Tests inject an ``httpx.Client`` bound to a
    ``MockTransport`` and own its lifetime. The POST is retried up to ``retries``
    attempts on a transient blip (dropped connection, 5xx, 429) with exponential
    backoff - the file bytes are re-sent each attempt; ``sleep`` is injectable so
    tests skip the real wait. Any transport, HTTP-status, or JSON decode failure
    is mapped to ``ExtractionError``.
    """
    httpx = require_httpx()
    owns_client = client is None
    active = client if client is not None else httpx.Client(timeout=timeout)

    def send() -> httpx.Response:
        response = active.post(url, files=files, data=data)
        response.raise_for_status()
        return response

    try:
        response = retry_http(send, tries=retries, sleep=sleep)
        return response.json()
    except httpx.HTTPError as exc:
        raise ExtractionError(f"extractor request to {url} failed: {exc}") from exc
    except ValueError as exc:
        raise ExtractionError(f"extractor at {url} returned invalid JSON: {exc}") from exc
    finally:
        if owns_client:
            active.close()


__all__ = [
    "DEFAULT_MAX_BYTES",
    "guess_mime",
    "post_multipart_json",
    "read_source_bytes",
    "require_endpoint",
    "require_httpx",
]
