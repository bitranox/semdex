"""docling-serve REST Extract adapter (layout-aware PDF/DOCX/... to markdown).

Talks to a running docling-serve container over HTTP:
``POST {endpoint}/v1/convert/file`` with the file as a multipart ``files`` part
and ``to_formats=md``, then reads ``document.md_content`` from the returned JSON.
docling-serve has a CPU image, so no GPU is required. Needs the optional
``semdex[docling]`` extra (httpx) and ``[extractor].endpoint`` pointing at the
container.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from ...domain.errors import ExtractionError
from ...domain.models import ExtractedDocument
from ._http import DEFAULT_MAX_BYTES, guess_mime, post_multipart_json, read_source_bytes, require_endpoint, source_name

if TYPE_CHECKING:
    import httpx

    from ...domain.models import SourceRef


class _DoclingDocument(BaseModel):
    """The ``document`` object of docling-serve's response (extra fields ignored)."""

    md_content: str


class _DoclingResponse(BaseModel):
    """docling-serve's ``/v1/convert/file`` JSON response (extra fields ignored)."""

    document: _DoclingDocument


class DoclingExtractor:
    """Extract a document to markdown via a docling-serve REST container."""

    def __init__(
        self,
        endpoint: str | None,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._endpoint = require_endpoint(endpoint, backend="docling")
        self._max_bytes = max_bytes
        self._timeout = timeout
        self._client = client

    def __call__(self, source: SourceRef) -> ExtractedDocument:
        content = read_source_bytes(source, max_bytes=self._max_bytes)
        files = {"files": (source_name(source), content, guess_mime(source))}
        body = post_multipart_json(
            url=f"{self._endpoint.rstrip('/')}/v1/convert/file",
            timeout=self._timeout,
            client=self._client,
            files=files,
            data={"to_formats": "md"},
        )
        return ExtractedDocument(source=source, text=_markdown_of(body, source))


def _markdown_of(body: object, source: SourceRef) -> str:
    """Validate docling-serve's response and read ``document.md_content``."""
    try:
        return _DoclingResponse.model_validate(body).document.md_content
    except ValidationError as exc:
        raise ExtractionError(f"docling returned an unexpected response for {source.uri}: {exc}") from exc


# Static conformance assertion -- pyright verifies DoclingExtractor satisfies Extract.
if TYPE_CHECKING:
    from ...application.ports import Extract

    _assert_extract: Extract = DoclingExtractor("http://x")


__all__ = [
    "DoclingExtractor",
]
