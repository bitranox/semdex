"""Unit tests for the OpenAI-compatible embedding adapter.

No server required: an ``httpx.MockTransport`` is injected so request building,
response parsing, the construction-time dimension probe, out-of-order ``index``
reordering, Bearer auth, and error mapping are exercised offline - mirroring
tests/test_embedding_ollama.py. Real OpenAI-compatible servers (ollama ``/v1``,
llama.cpp) are covered by the local_only e2e cells.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from semdex.adapters.embedding import load_openai_embedding
from semdex.domain.errors import EmbeddingError

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.os_agnostic


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _request_body(request: httpx.Request) -> Any:
    return json.loads(request.content)


def _data(vectors: list[list[float]]) -> dict[str, Any]:
    """Build a well-formed ``/v1/embeddings`` response for the given rows."""
    return {"data": [{"embedding": vec, "index": i} for i, vec in enumerate(vectors)]}


def test_happy_path_probes_dimension_and_embeds() -> None:
    """Construction probes the dimension; embed_* post to /v1/embeddings with {model,input}."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        count = len(_request_body(request)["input"])
        return httpx.Response(200, json=_data([[0.1, 0.2, 0.3]] * count))

    provider = load_openai_embedding("test-model", client=_client(handler))

    assert provider.dim == 3
    assert provider.embed_query("x") == (0.1, 0.2, 0.3)
    assert provider.embed_passages(["a", "b"]) == [(0.1, 0.2, 0.3), (0.1, 0.2, 0.3)]

    assert requests[0].url.path.endswith("/v1/embeddings")
    assert _request_body(requests[0]) == {"model": "test-model", "input": ["dimension probe"]}
    assert _request_body(requests[-1]) == {"model": "test-model", "input": ["a", "b"]}


def test_endpoint_base_url_gets_embeddings_appended() -> None:
    """A caller endpoint (ending in /v1) has /embeddings appended, not /v1/embeddings twice."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_data([[0.0, 1.0]]))

    load_openai_embedding("m", endpoint="http://host:11434/v1", client=_client(handler))

    assert str(requests[0].url) == "http://host:11434/v1/embeddings"


def test_rows_are_reordered_by_index() -> None:
    """The API may return rows out of order; they are realigned to request position by `index`."""

    def handler(request: httpx.Request) -> httpx.Response:
        count = len(_request_body(request)["input"])
        if count == 1:  # the construction probe
            return httpx.Response(200, json=_data([[0.5, 0.5]]))
        # Two inputs, returned index 1 THEN index 0 - must be realigned.
        return httpx.Response(
            200, json={"data": [{"embedding": [2.0, 2.0], "index": 1}, {"embedding": [1.0, 1.0], "index": 0}]}
        )

    provider = load_openai_embedding(client=_client(handler))
    assert provider.embed_passages(["first", "second"]) == [(1.0, 1.0), (2.0, 2.0)]


def test_bearer_header_sent_only_when_api_key_set() -> None:
    """api_key becomes an Authorization: Bearer header; absent, no auth header is sent."""
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200, json=_data([[0.0]]))

    load_openai_embedding("m", api_key="sk-secret", client=_client(handler))
    assert seen[0] == "Bearer sk-secret"

    seen.clear()
    load_openai_embedding("m", client=_client(handler))
    assert seen[0] is None


def test_large_dimension_is_discovered() -> None:
    """Whatever width the server returns becomes provider.dim (no per-model assumption)."""
    served_dim = 1536  # text-embedding-3-small native width

    def handler(request: httpx.Request) -> httpx.Response:
        count = len(_request_body(request)["input"])
        return httpx.Response(200, json=_data([[0.01] * served_dim] * count))

    provider = load_openai_embedding(client=_client(handler))
    assert provider.dim == served_dim
    assert len(provider.embed_query("x")) == served_dim


def test_passages_length_mismatch_raises() -> None:
    """Fewer embeddings than inputs is an EmbeddingError, not a silent misalignment."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_data([[0.1, 0.2, 0.3]]))  # always one row

    provider = load_openai_embedding(client=_client(handler))
    with pytest.raises(EmbeddingError):
        provider.embed_passages(["a", "b"])


def test_invalid_response_missing_data_field_raises() -> None:
    """A response missing the `data` field fails validation at construction."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"wrong": []})

    with pytest.raises(EmbeddingError):
        load_openai_embedding(client=_client(handler))


def test_non_json_response_raises() -> None:
    """A 200 with a non-JSON body is a clean EmbeddingError, not a raw ValueError."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json {{")

    with pytest.raises(EmbeddingError):
        load_openai_embedding(client=_client(handler))


def test_http_error_status_raises() -> None:
    """A 5xx response from the probe call maps to EmbeddingError, not a raw httpx error."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with pytest.raises(EmbeddingError):
        load_openai_embedding(client=_client(handler))
