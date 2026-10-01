"""Xberg REST Extract adapter (OCR + 18+ document formats).

Talks to a running Xberg (kreuzberg v4) API container over HTTP:
``POST {endpoint}/extract`` with the file as a multipart ``files`` part, and reads
the ``content`` field of the first returned JSON result. To OCR a scanned image or
force OCR on a searchable PDF, an extra multipart ``config`` field carries the JSON
``{"ocr": {"language": ...}, "force_ocr": true}``. Xberg is MIT-licensed and
CPU-friendly, and its image bundles Tesseract, so its container needs no GPU. Needs
the optional ``semdex[xberg]`` extra (httpx) and ``[extractor].endpoint`` pointing at
the container.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, TypeAdapter, ValidationError

from ...domain.errors import ExtractionError
from ...domain.models import ExtractedDocument
from ._http import DEFAULT_MAX_BYTES, guess_mime, post_multipart_json, read_source_bytes, require_endpoint, source_name

if TYPE_CHECKING:
    import httpx

    from ...domain.models import SourceRef


class _XbergResult(BaseModel):
    """One entry of Xberg's ``/extract`` JSON response (extra fields ignored)."""

    content: str


_XBERG_RESULTS = TypeAdapter(list[_XbergResult])


class XbergExtractor:
    """Extract a document to markdown via a Xberg REST container."""

    def __init__(  # noqa: PLR0913 - DI seam: endpoint + transport tunables + OCR config + injected client
        self,
        endpoint: str | None,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        timeout: float = 120.0,
        force_ocr: bool = False,
        ocr_language: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._endpoint = require_endpoint(endpoint, backend="xberg")
        self._max_bytes = max_bytes
        self._timeout = timeout
        self._force_ocr = force_ocr
        self._ocr_language = ocr_language
        self._client = client

    def __call__(self, source: SourceRef) -> ExtractedDocument:
        content = read_source_bytes(source, max_bytes=self._max_bytes)
        # Xberg's REST API takes the upload under the multipart field "files" (plural) and
        # returns a JSON array of results; sending "data" gets a 400 "No files provided".
        files = {"files": (source_name(source), content, guess_mime(source))}
        body = post_multipart_json(
            url=f"{self._endpoint.rstrip('/')}/extract",
            timeout=self._timeout,
            client=self._client,
            files=files,
            data=self._ocr_config(),
        )
        return ExtractedDocument(source=source, text=_content_of(body, source))

    def _ocr_config(self) -> dict[str, str] | None:
        """The ``config`` multipart field (JSON) when OCR is requested, else None.

        Xberg only OCRs when it decides the input needs it (a scanned image, or a PDF
        with no text layer). ``force_ocr`` runs OCR even on searchable PDFs; ``ocr_language``
        picks the Tesseract language. When neither is set, no config is sent and Xberg uses
        its own defaults.
        """
        config: dict[str, object] = {}
        if self._force_ocr:
            config["force_ocr"] = True
        if self._ocr_language:
            config["ocr"] = {"language": self._ocr_language}
        return {"config": json.dumps(config)} if config else None


def _content_of(body: object, source: SourceRef) -> str:
    """Validate Xberg's response (a list of results) and read the first content."""
    try:
        results = _XBERG_RESULTS.validate_python(body)
    except ValidationError as exc:
        raise ExtractionError(f"xberg returned an unexpected response for {source.uri}: {exc}") from exc
    if not results:
        raise ExtractionError(f"xberg returned no results for {source.uri}")
    return results[0].content


# Static conformance assertion -- pyright verifies XbergExtractor satisfies Extract.
if TYPE_CHECKING:
    from ...application.ports import Extract

    _assert_extract: Extract = XbergExtractor("http://x")


__all__ = [
    "XbergExtractor",
]
