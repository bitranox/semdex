"""Unit tests for the REST document-extractor adapters (xberg/docling/mineru).

No container required: an ``httpx.MockTransport`` is injected so the adapter's
request building, response parsing, size cap, and error mapping are exercised in
isolation. The real containers are covered by the ``local_only`` Docker tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import httpx
import pytest

from semdex.adapters.extractor import DoclingExtractor, MineruExtractor, XbergExtractor
from semdex.domain.errors import ExtractionError
from semdex.domain.models import SourceRef

if TYPE_CHECKING:
    from collections.abc import Callable

    from semdex.application.ports import Extract

pytestmark = pytest.mark.os_agnostic


class _RestAdapterCls(Protocol):
    """The shared constructor shape of the REST extractor adapters."""

    def __call__(
        self,
        endpoint: str | None,
        *,
        max_bytes: int = ...,
        timeout: float = ...,
        client: httpx.Client | None = ...,
    ) -> Extract: ...


# (adapter class, expected URL suffix, JSON body carrying "the markdown" text)
_BACKENDS: list[tuple[_RestAdapterCls, str, object]] = [
    (XbergExtractor, "/extract", [{"content": "the markdown"}]),
    (DoclingExtractor, "/v1/convert/file", {"document": {"md_content": "the markdown"}}),
    (MineruExtractor, "/file_parse", {"results": {"doc": {"md_content": "the markdown"}}}),
]
_BAD_BODIES: list[tuple[_RestAdapterCls, object]] = [
    (XbergExtractor, []),
    (DoclingExtractor, {"document": {}}),
    (MineruExtractor, {"results": {}}),
]
_ALL_ADAPTERS: list[_RestAdapterCls] = [XbergExtractor, DoclingExtractor, MineruExtractor]


def _source(tmp_path: Path, *, name: str = "doc.pdf", data: bytes = b"payload") -> SourceRef:
    path = tmp_path / name
    path.write_bytes(data)
    return SourceRef(uri=str(path), label="lbl", content_hash="h", mtime=1.0)


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(("adapter_cls", "url_suffix", "body"), _BACKENDS)
def test_happy_path_parses_markdown(
    tmp_path: Path,
    adapter_cls: _RestAdapterCls,
    url_suffix: str,
    body: object,
) -> None:
    """Each REST adapter posts the file and returns the parsed markdown."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=body)

    extractor = adapter_cls("http://host:9/", client=_client(handler))
    doc = extractor(_source(tmp_path))

    assert doc.text == "the markdown"
    assert seen[0].url.path.endswith(url_suffix)
    assert b"doc.pdf" in seen[0].content  # the file is in the multipart body


@pytest.mark.parametrize(("adapter_cls", "body"), _BAD_BODIES)
def test_missing_markdown_field_raises(tmp_path: Path, adapter_cls: _RestAdapterCls, body: object) -> None:
    """A response without the expected markdown field is a clear ExtractionError."""
    extractor = adapter_cls("http://host:9", client=_client(lambda _req: httpx.Response(200, json=body)))
    with pytest.raises(ExtractionError):
        extractor(_source(tmp_path))


@pytest.mark.parametrize("adapter_cls", _ALL_ADAPTERS)
def test_http_error_status_raises(tmp_path: Path, adapter_cls: _RestAdapterCls) -> None:
    """A 5xx response is mapped to ExtractionError, not a raw httpx error."""
    extractor = adapter_cls("http://host:9", client=_client(lambda _req: httpx.Response(503)))
    with pytest.raises(ExtractionError):
        extractor(_source(tmp_path))


@pytest.mark.parametrize("adapter_cls", _ALL_ADAPTERS)
def test_oversized_file_rejected_before_request(tmp_path: Path, adapter_cls: _RestAdapterCls) -> None:
    """A file over the size cap is rejected without an HTTP call."""

    def handler(_req: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("request should not be sent for an oversized file")

    extractor = adapter_cls("http://host:9", max_bytes=4, client=_client(handler))
    with pytest.raises(ExtractionError, match="over the"):
        extractor(_source(tmp_path, data=b"too-large"))


@pytest.mark.parametrize("adapter_cls", _ALL_ADAPTERS)
def test_missing_endpoint_rejected(adapter_cls: _RestAdapterCls) -> None:
    """An HTTP backend with no endpoint fails clearly at construction."""
    with pytest.raises(ExtractionError, match="requires"):
        adapter_cls(None)


def test_xberg_sends_ocr_config_only_when_requested(tmp_path: Path) -> None:
    """xberg's force_ocr/ocr_language ride in the multipart 'config' JSON field; default sends none."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=[{"content": "ocr text"}])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    XbergExtractor("http://h:8000", client=client, force_ocr=True, ocr_language="eng")(_source(tmp_path))
    body = seen[0].content
    assert b'name="config"' in body
    assert b'"force_ocr": true' in body
    assert b"eng" in body

    seen.clear()
    XbergExtractor("http://h:8000", client=client)(_source(tmp_path))
    assert b'name="config"' not in seen[0].content  # default: no OCR config field sent
