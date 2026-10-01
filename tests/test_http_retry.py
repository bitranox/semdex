"""Unit tests for the shared HTTP retry-with-backoff helper and its use by the
four adapter transports (embedding, summarizer, extractor multipart, vision OCR).

No server and no real waiting: an ``httpx.MockTransport`` drives each transport
and a recording no-op ``sleep`` is injected, so the retry/backoff path is
exercised instantly. Verifies that a transient blip (dropped connection, 5xx,
429) is retried, that a 4xx or a schema/JSON failure is not, and that an
exhausted retry budget maps to each adapter's own error type. Mirrors the
MockTransport style of tests/test_embedding_ollama.py and
tests/test_extractor_vision_ocr.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from semdex.adapters._http_retry import is_transient, retry_http
from semdex.adapters.discovery.location import to_uri
from semdex.adapters.embedding._base import post_embedding_request
from semdex.adapters.extractor._http import post_multipart_json
from semdex.adapters.extractor.vision_ocr import VisionOcrExtractor
from semdex.adapters.summarizer._base import post_chat_request
from semdex.domain.errors import EmbeddingError, ExtractionError, SummaryError
from semdex.domain.models import SourceRef

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

pytestmark = pytest.mark.os_agnostic


class _Body(BaseModel):
    """Minimal response model so the transport tests exercise validation too."""

    ok: int


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://x")
    return httpx.HTTPStatusError("e", request=request, response=httpx.Response(status, request=request))


# -- is_transient ------------------------------------------------------------


def test_is_transient_true_for_transport_error() -> None:
    """A dropped connection / timeout (any httpx TransportError) is retryable."""
    assert is_transient(httpx.ConnectError("boom")) is True
    assert is_transient(httpx.ReadTimeout("slow")) is True
    assert is_transient(httpx.RemoteProtocolError("half-close")) is True


@pytest.mark.parametrize("status", [500, 502, 503, 504, 429])
def test_is_transient_true_for_5xx_and_429(status: int) -> None:
    """A server-side 5xx and a 429 rate limit are worth retrying."""
    assert is_transient(_status_error(status)) is True


@pytest.mark.parametrize("status", [400, 401, 404, 422])
def test_is_transient_false_for_other_4xx(status: int) -> None:
    """A 4xx other than 429 is the caller's fault; retrying never helps."""
    assert is_transient(_status_error(status)) is False


def test_is_transient_false_for_validation_and_json_errors() -> None:
    """A schema/JSON failure is a real error, not a blip, so it is not retryable."""
    with pytest.raises(ValidationError) as exc_info:
        _Body.model_validate({"wrong": 1})
    assert is_transient(exc_info.value) is False
    assert is_transient(ValueError("invalid json")) is False


# -- retry_http --------------------------------------------------------------


def test_retry_http_retries_transient_then_returns() -> None:
    """Transient failures are retried; the first success is returned."""
    attempts: list[int] = []
    slept: list[float] = []

    def do_request() -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 3:
            raise httpx.ConnectError("blip")
        return httpx.Response(200)

    response = retry_http(do_request, tries=5, base_delay=1.0, max_delay=30.0, sleep=slept.append)

    assert response.status_code == 200
    assert len(attempts) == 3  # two failures, then success
    assert slept == [1.0, 2.0]  # exponential backoff before each retry


def test_retry_http_reraises_after_exhausting_tries() -> None:
    """When every attempt is transient the last exception propagates after `tries`."""
    attempts: list[int] = []
    slept: list[float] = []

    def do_request() -> httpx.Response:
        attempts.append(1)
        raise httpx.ConnectError("down")

    with pytest.raises(httpx.ConnectError):
        retry_http(do_request, tries=4, base_delay=1.0, max_delay=30.0, sleep=slept.append)

    assert len(attempts) == 4  # exactly `tries` attempts
    assert slept == [1.0, 2.0, 4.0]  # 3 backoffs, doubling


def test_retry_http_caps_backoff_at_max_delay() -> None:
    """The per-attempt pause never exceeds max_delay however long the run of failures."""
    slept: list[float] = []

    def do_request() -> httpx.Response:
        raise httpx.ConnectError("down")

    with pytest.raises(httpx.ConnectError):
        retry_http(do_request, tries=6, base_delay=10.0, max_delay=15.0, sleep=slept.append)

    assert slept == [10.0, 15.0, 15.0, 15.0, 15.0]  # doubling clamped at 15


def test_retry_http_does_not_retry_non_transient() -> None:
    """A 4xx status is raised immediately with no retry and no backoff."""
    attempts: list[int] = []
    slept: list[float] = []

    def do_request() -> httpx.Response:
        attempts.append(1)
        raise _status_error(400)

    with pytest.raises(httpx.HTTPStatusError):
        retry_http(do_request, tries=4, sleep=slept.append)

    assert len(attempts) == 1  # no retry
    assert slept == []


# -- post_embedding_request --------------------------------------------------


def _embed(handler: Callable[[httpx.Request], httpx.Response], slept: list[float], *, retries: int) -> _Body:
    return post_embedding_request(
        url="http://h/api/embed",
        payload={"model": "m", "input": ["x"]},
        response_model=_Body,
        timeout=5.0,
        client=_client(handler),
        service="ollama",
        install_extra="ollama",
        retries=retries,
        sleep=slept.append,
    )


def test_embedding_transport_retries_transient_then_succeeds() -> None:
    """post_embedding_request retries a 503 and returns the eventual success."""
    seen: list[httpx.Request] = []
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(503) if len(seen) < 2 else httpx.Response(200, json={"ok": 1})

    body = _embed(handler, slept, retries=3)

    assert body.ok == 1
    assert len(seen) == 2
    assert slept == [1.0]


def test_embedding_transport_does_not_retry_4xx() -> None:
    """A 400 is mapped to EmbeddingError at once, with no retry."""
    seen: list[httpx.Request] = []
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(400)

    with pytest.raises(EmbeddingError):
        _embed(handler, slept, retries=3)
    assert len(seen) == 1
    assert slept == []


def test_embedding_transport_exhausts_then_maps_to_error() -> None:
    """A persistent 503 exhausts the budget and maps to EmbeddingError."""
    seen: list[httpx.Request] = []
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(503)

    with pytest.raises(EmbeddingError):
        _embed(handler, slept, retries=3)
    assert len(seen) == 3
    assert slept == [1.0, 2.0]


# -- post_chat_request -------------------------------------------------------


def _chat(handler: Callable[[httpx.Request], httpx.Response], slept: list[float], *, retries: int) -> _Body:
    return post_chat_request(
        url="http://h/api/chat",
        payload={"model": "m", "messages": []},
        response_model=_Body,
        timeout=5.0,
        client=_client(handler),
        service="ollama",
        install_extra="summary",
        retries=retries,
        sleep=slept.append,
    )


def test_summarizer_transport_retries_transient_then_succeeds() -> None:
    """post_chat_request retries a transient connection error and then returns."""
    seen: list[httpx.Request] = []
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) < 2:
            raise httpx.ConnectError("dropped")
        return httpx.Response(200, json={"ok": 1})

    body = _chat(handler, slept, retries=3)

    assert body.ok == 1
    assert len(seen) == 2
    assert slept == [1.0]


def test_summarizer_transport_does_not_retry_4xx() -> None:
    """A 404 maps to SummaryError with no retry."""
    seen: list[httpx.Request] = []
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(404)

    with pytest.raises(SummaryError):
        _chat(handler, slept, retries=3)
    assert len(seen) == 1
    assert slept == []


def test_summarizer_transport_exhausts_then_maps_to_error() -> None:
    """A persistent 500 exhausts the budget and maps to SummaryError."""
    seen: list[httpx.Request] = []
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(500)

    with pytest.raises(SummaryError):
        _chat(handler, slept, retries=3)
    assert len(seen) == 3
    assert slept == [1.0, 2.0]


# -- post_multipart_json (extractor) -----------------------------------------


def _multipart(handler: Callable[[httpx.Request], httpx.Response], slept: list[float], *, retries: int) -> object:
    return post_multipart_json(
        url="http://h/extract",
        timeout=5.0,
        client=_client(handler),
        files={"files": ("doc.pdf", b"payload", "application/pdf")},
        retries=retries,
        sleep=slept.append,
    )


def test_extractor_transport_retries_transient_then_succeeds() -> None:
    """post_multipart_json retries a 503 and re-sends the file bytes on success."""
    seen: list[httpx.Request] = []
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(503) if len(seen) < 2 else httpx.Response(200, json={"ok": 1})

    result = _multipart(handler, slept, retries=3)

    assert result == {"ok": 1}
    assert len(seen) == 2
    assert b"doc.pdf" in seen[-1].content  # file bytes re-sent on the retry
    assert slept == [1.0]


def test_extractor_transport_does_not_retry_4xx() -> None:
    """A 400 maps to ExtractionError with no retry."""
    seen: list[httpx.Request] = []
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(400)

    with pytest.raises(ExtractionError):
        _multipart(handler, slept, retries=3)
    assert len(seen) == 1
    assert slept == []


def test_extractor_transport_exhausts_then_maps_to_error() -> None:
    """A persistent 503 exhausts the budget and maps to ExtractionError."""
    seen: list[httpx.Request] = []
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(503)

    with pytest.raises(ExtractionError):
        _multipart(handler, slept, retries=3)
    assert len(seen) == 3
    assert slept == [1.0, 2.0]


# -- VisionOcrExtractor._post ------------------------------------------------


def _src(path: Path) -> SourceRef:
    return SourceRef(uri=to_uri(path), label="", content_hash="", mtime=0.0)


def _ocr_ok(text: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def test_vision_transport_retries_transient_then_succeeds(tmp_path: Path) -> None:
    """The vision OCR POST retries a 503 and returns the transcription."""
    img = tmp_path / "scan.png"
    img.write_bytes(b"x")
    seen: list[httpx.Request] = []
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(503) if len(seen) < 2 else _ocr_ok("OK")

    extractor = VisionOcrExtractor("http://v/v1", client=_client(handler), retries=3, sleep=slept.append)
    doc = extractor(_src(img))

    assert doc.text == "OK"
    assert len(seen) == 2
    assert slept == [1.0]


def test_vision_transport_does_not_retry_4xx(tmp_path: Path) -> None:
    """A 400 maps to ExtractionError with no retry."""
    img = tmp_path / "scan.png"
    img.write_bytes(b"x")
    seen: list[httpx.Request] = []
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(400)

    with pytest.raises(ExtractionError):
        VisionOcrExtractor("http://v/v1", client=_client(handler), retries=3, sleep=slept.append)(_src(img))
    assert len(seen) == 1
    assert slept == []


def test_vision_transport_exhausts_then_maps_to_error(tmp_path: Path) -> None:
    """A persistent 503 exhausts the budget and maps to ExtractionError."""
    img = tmp_path / "scan.png"
    img.write_bytes(b"x")
    seen: list[httpx.Request] = []
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(503)

    with pytest.raises(ExtractionError):
        VisionOcrExtractor("http://v/v1", client=_client(handler), retries=3, sleep=slept.append)(_src(img))
    assert len(seen) == 3
    assert slept == [1.0, 2.0]
