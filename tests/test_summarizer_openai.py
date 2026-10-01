"""Unit tests for the OpenAI-compatible LLM summarizer adapter.

No server required: an ``httpx.MockTransport`` is injected so request building,
response parsing, Bearer auth, input clipping, and error mapping are exercised
offline - mirroring tests/test_summarizer_ollama.py and
tests/test_embedding_openai.py. Real OpenAI-compatible servers (ollama ``/v1``,
llama.cpp) are covered by the local_only e2e cells.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from semdex.adapters.summarizer import load_openai_summarizer
from semdex.domain.errors import SummaryError

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.os_agnostic


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _request_body(request: httpx.Request) -> Any:
    return json.loads(request.content)


def _reply(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}]}


def test_happy_path_posts_chat_and_parses_first_choice() -> None:
    """summarize posts to /chat/completions and returns the first choice's message content."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_reply("  Topic: a short summary.  "))

    summarizer = load_openai_summarizer("test-model", prompt="be brief", client=_client(handler))

    assert summarizer.model_id == "test-model"
    assert summarizer.summarize("the whole document text") == "Topic: a short summary."

    assert requests[0].url.path.endswith("/chat/completions")
    body = _request_body(requests[0])
    assert body["model"] == "test-model"
    assert body["temperature"] == 0.0
    assert body["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "the whole document text"},
    ]


def test_endpoint_base_url_gets_chat_completions_appended() -> None:
    """A caller endpoint (ending in /v1) has /chat/completions appended."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_reply("ok"))

    load_openai_summarizer("m", endpoint="http://host:11434/v1", client=_client(handler)).summarize("x")

    assert str(requests[0].url) == "http://host:11434/v1/chat/completions"


def test_bearer_header_sent_only_when_api_key_set() -> None:
    """api_key becomes an Authorization: Bearer header; absent, no auth header is sent."""
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json=_reply("ok"))

    load_openai_summarizer("m", api_key="sk-secret", client=_client(handler)).summarize("x")
    assert seen[0] == "Bearer sk-secret"

    seen.clear()
    load_openai_summarizer("m", client=_client(handler)).summarize("x")
    assert seen[0] is None


def test_input_is_clipped_to_max_input_chars() -> None:
    """The document text is clipped to max_input_chars before it is sent."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_reply("ok"))

    load_openai_summarizer("m", max_input_chars=5, client=_client(handler)).summarize("abcdefghij")
    assert _request_body(requests[0])["messages"][1]["content"] == "abcde"


def test_empty_choices_raises() -> None:
    """A response with no choices is a SummaryError, not an IndexError."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    summarizer = load_openai_summarizer(client=_client(handler))
    with pytest.raises(SummaryError):
        summarizer.summarize("x")


def test_invalid_response_missing_choices_raises() -> None:
    """A response missing the `choices` field fails validation as a SummaryError."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"wrong": []})

    summarizer = load_openai_summarizer(client=_client(handler))
    with pytest.raises(SummaryError):
        summarizer.summarize("x")


def test_non_json_response_raises() -> None:
    """A 200 with a non-JSON body is a clean SummaryError, not a raw ValueError."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json {{")

    summarizer = load_openai_summarizer(client=_client(handler))
    with pytest.raises(SummaryError):
        summarizer.summarize("x")


def test_http_error_status_raises() -> None:
    """A 5xx response maps to SummaryError, not a raw httpx error."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    summarizer = load_openai_summarizer(client=_client(handler))
    with pytest.raises(SummaryError):
        summarizer.summarize("x")
