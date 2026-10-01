"""Unit tests for the vision-LLM OCR extractor (olmOCR / OpenAI-vision).

An ``httpx.MockTransport`` stands in for the vision server, so request building
(the base64 data URI, the /v1/chat/completions path, the Bearer header), response
parsing, PDF page rendering, and error mapping are exercised offline.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from semdex.adapters.discovery.location import to_uri
from semdex.adapters.extractor.vision_ocr import VisionOcrExtractor
from semdex.domain.errors import ExtractionError
from semdex.domain.models import SourceRef

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

pytestmark = pytest.mark.os_agnostic


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _src(path: Path) -> SourceRef:
    return SourceRef(uri=to_uri(path), label="", content_hash="", mtime=0.0)


def _ok(text: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def test_image_posts_data_uri_and_parses(tmp_path: Path) -> None:
    """An image is base64'd into an image_url data URI and posted to /v1/chat/completions."""
    img = tmp_path / "scan.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n fake bytes")
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return _ok("TRANSCRIBED TEXT")

    extractor = VisionOcrExtractor("http://vllm:8000/v1", model="olmocr", api_key="sk-x", client=_client(handler))
    doc = extractor(_src(img))

    assert doc.text == "TRANSCRIBED TEXT"
    assert captured["url"] == "http://vllm:8000/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-x"
    assert captured["body"]["model"] == "olmocr"
    image_part = captured["body"]["messages"][1]["content"][0]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_no_api_key_sends_no_auth_header(tmp_path: Path) -> None:
    """A local vLLM/olmOCR server needs no key, so no Authorization header is sent."""
    img = tmp_path / "s.png"
    img.write_bytes(b"x")
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return _ok("ok")

    VisionOcrExtractor("http://v/v1", client=_client(handler))(_src(img))
    assert seen == [None]


def test_pdf_renders_each_page_and_concatenates(tmp_path: Path) -> None:
    """A PDF is rendered page by page (pymupdf) and each page OCR'd, then joined."""
    pymupdf = pytest.importorskip("pymupdf")
    pdf = tmp_path / "doc.pdf"
    doc = pymupdf.open()
    for _ in range(2):
        doc.new_page()
    doc.save(pdf)
    doc.close()

    posts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal posts
        posts += 1
        return _ok(f"PAGE{posts}")

    result = VisionOcrExtractor("http://v/v1", client=_client(handler))(_src(pdf))
    assert posts == 2  # one vision call per page
    assert result.text == "PAGE1\n\nPAGE2"


def test_http_error_maps_to_extraction_error(tmp_path: Path) -> None:
    """A 5xx from the vision server is an ExtractionError, not a raw httpx error."""
    img = tmp_path / "s.png"
    img.write_bytes(b"x")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with pytest.raises(ExtractionError):
        VisionOcrExtractor("http://v/v1", client=_client(handler))(_src(img))


def test_missing_choices_raises(tmp_path: Path) -> None:
    """A response with no choices is a clean ExtractionError."""
    img = tmp_path / "s.png"
    img.write_bytes(b"x")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    with pytest.raises(ExtractionError):
        VisionOcrExtractor("http://v/v1", client=_client(handler))(_src(img))


def test_non_json_response_raises(tmp_path: Path) -> None:
    """A 200 with a non-JSON body maps to ExtractionError, not a raw ValueError."""
    img = tmp_path / "s.png"
    img.write_bytes(b"x")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json {{")

    with pytest.raises(ExtractionError):
        VisionOcrExtractor("http://v/v1", client=_client(handler))(_src(img))


def test_missing_endpoint_raises() -> None:
    """No endpoint is a clear configuration error."""
    with pytest.raises(ExtractionError):
        VisionOcrExtractor(None)
