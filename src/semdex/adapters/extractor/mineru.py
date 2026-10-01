"""MinerU REST Extract adapter (high-fidelity PDF/OCR to markdown).

Talks to a running MinerU API container over HTTP:
``POST {endpoint}/file_parse`` with the file as a multipart ``files`` part and
``return_md_content=true``, then reads the per-file ``md_content`` from the
returned JSON ``results`` map. MinerU's official image is CUDA/GPU-only, so this
backend targets a GPU host; its integration test is flagged GPU-required. Needs
the optional ``semdex[mineru]`` extra (httpx) and ``[extractor].endpoint``.
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


class _MineruEntry(BaseModel):
    """One per-file result in MinerU's response (extra fields ignored)."""

    md_content: str


class _MineruResponse(BaseModel):
    """MinerU's ``/file_parse`` JSON response: a filename -> result map."""

    results: dict[str, _MineruEntry]


class MineruExtractor:
    """Extract a document to markdown via a MinerU REST container (GPU host)."""

    def __init__(
        self,
        endpoint: str | None,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        timeout: float = 120.0,
        backend: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self._endpoint = require_endpoint(endpoint, backend="mineru")
        self._max_bytes = max_bytes
        self._timeout = timeout
        # MinerU pipeline selector: None uses the server default (classic
        # pipeline); "vlm-transformers" (etc.) runs MinerU's vision-model pipeline.
        self._backend = backend
        self._client = client

    def __call__(self, source: SourceRef) -> ExtractedDocument:
        content = read_source_bytes(source, max_bytes=self._max_bytes)
        files = {"files": (source_name(source), content, guess_mime(source))}
        data = {"return_md_content": "true"}
        if self._backend is not None:
            data["backend"] = self._backend
        body = post_multipart_json(
            url=f"{self._endpoint.rstrip('/')}/file_parse",
            timeout=self._timeout,
            client=self._client,
            files=files,
            data=data,
        )
        return ExtractedDocument(source=source, text=_markdown_of(body, source))


def _markdown_of(body: object, source: SourceRef) -> str:
    """Validate MinerU's response and read the first per-file ``md_content``."""
    try:
        response = _MineruResponse.model_validate(body)
    except ValidationError as exc:
        raise ExtractionError(f"mineru returned an unexpected response for {source.uri}: {exc}") from exc
    for entry in response.results.values():
        return entry.md_content
    raise ExtractionError(f"mineru returned no results for {source.uri}")


# Static conformance assertion -- pyright verifies MineruExtractor satisfies Extract.
if TYPE_CHECKING:
    from ...application.ports import Extract

    _assert_extract: Extract = MineruExtractor("http://x")


__all__ = [
    "MineruExtractor",
]
