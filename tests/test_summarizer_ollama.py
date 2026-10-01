"""Unit tests for the ollama LLM summarizer adapter.

No server required: an ``httpx.MockTransport`` is injected so request building,
response parsing, input clipping, and error mapping are exercised in isolation,
mirroring tests/test_summarizer_openai.py and tests/test_embedding_ollama.py. A
real ollama server is out of scope for these fast, offline CI tests.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from semdex.adapters.summarizer import load_ollama_summarizer
from semdex.domain.errors import SummaryError

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.os_agnostic


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _request_body(request: httpx.Request) -> Any:
    return json.loads(request.content)


def test_happy_path_posts_chat_and_parses_message() -> None:
    """summarize posts a system+user chat to /api/chat and returns the message content."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"message": {"content": "  Topic: a short summary.  "}})

    summarizer = load_ollama_summarizer("test-model", prompt="be brief", client=_client(handler))

    assert summarizer.model_id == "test-model"
    assert summarizer.summarize("the whole document text") == "Topic: a short summary."

    assert requests[0].url.path.endswith("/api/chat")
    body = _request_body(requests[0])
    assert body["model"] == "test-model"
    assert body["stream"] is False
    assert body["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "the whole document text"},
    ]


def test_input_is_clipped_to_max_input_chars() -> None:
    """The document text is clipped to max_input_chars before it is sent."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"message": {"content": "ok"}})

    summarizer = load_ollama_summarizer("m", max_input_chars=5, client=_client(handler))
    summarizer.summarize("abcdefghij")

    assert _request_body(requests[0])["messages"][1]["content"] == "abcde"


def test_invalid_response_missing_message_raises() -> None:
    """A response missing the `message` field fails validation as a SummaryError."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"wrong": {}})

    summarizer = load_ollama_summarizer(client=_client(handler))
    with pytest.raises(SummaryError):
        summarizer.summarize("x")


def test_http_error_status_raises() -> None:
    """A 5xx response is mapped to SummaryError, not a raw httpx error."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    summarizer = load_ollama_summarizer(client=_client(handler))
    with pytest.raises(SummaryError):
        summarizer.summarize("x")


def test_keep_alive_included_when_set() -> None:
    """keep_alive rides in the request body when set (ollama VRAM residency window)."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"message": {"content": "ok"}})

    load_ollama_summarizer("m", keep_alive="30m", client=_client(handler)).summarize("doc")

    assert _request_body(requests[0])["keep_alive"] == "30m"


def test_keep_alive_absent_when_none() -> None:
    """With keep_alive unset the key is omitted, so the default request is unchanged."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"message": {"content": "ok"}})

    load_ollama_summarizer("m", client=_client(handler)).summarize("doc")

    assert "keep_alive" not in _request_body(requests[0])
