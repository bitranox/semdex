"""Unit tests for the Cohere native embedding adapter.

No server required: an ``httpx.MockTransport`` is injected so request building,
response parsing, the passage-vs-query ``input_type`` split, batch splitting at 96
texts per call, the construction-time dimension probe, the ``Authorization:
Bearer`` auth header, and error mapping are exercised offline - mirroring
tests/test_embedding_openai.py. The paid Cohere API is never called.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from semdex.adapters.embedding import load_cohere_embedding
from semdex.domain.errors import EmbeddingError

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.os_agnostic

_API_KEY = "test-cohere-key"


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _request_body(request: httpx.Request) -> Any:
    return json.loads(request.content)


def _resp(vectors: list[list[float]]) -> dict[str, Any]:
    """Build a well-formed ``/v2/embed`` response (embeddings nested under `float`)."""
    return {"embeddings": {"float": vectors}}


def _vec_for(text: str) -> list[float]:
    """A 1-d marker vector encoding the text index (``t<i>``), else ``[0.0]``."""
    return [float(text[1:])] if text.startswith("t") and text[1:].isdigit() else [0.0]


def test_happy_path_probes_dimension_and_embeds() -> None:
    """Construction probes the dimension; passages post the search_document input_type."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        count = len(_request_body(request)["texts"])
        return httpx.Response(200, json=_resp([[0.1, 0.2, 0.3]] * count))

    provider = load_cohere_embedding("embed-v4.0", api_key=_API_KEY, client=_client(handler))

    assert provider.dim == 3
    assert provider.embed_passages(["a", "b"]) == [(0.1, 0.2, 0.3), (0.1, 0.2, 0.3)]

    last = requests[-1]
    assert last.url.path.endswith("/v2/embed")
    assert last.headers.get("authorization") == f"Bearer {_API_KEY}"
    body = _request_body(last)
    assert body["model"] == "embed-v4.0"
    assert body["input_type"] == "search_document"
    assert body["embedding_types"] == ["float"]
    assert body["texts"] == ["a", "b"]


def test_query_sends_query_input_type() -> None:
    """embed_query sends input_type search_query, not the passages' search_document."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        count = len(_request_body(request)["texts"])
        return httpx.Response(200, json=_resp([[0.5, 0.5]] * count))

    provider = load_cohere_embedding(api_key=_API_KEY, client=_client(handler))
    requests.clear()  # drop the construction probe
    provider.embed_query("hello")

    assert _request_body(requests[0])["input_type"] == "search_query"


def test_sub_batches_at_96_and_preserves_order() -> None:
    """>96 texts split into 96+4 calls; the concatenated result keeps input order."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts = _request_body(request)["texts"]
        calls.append(len(texts))
        return httpx.Response(200, json=_resp([_vec_for(text) for text in texts]))

    provider = load_cohere_embedding(api_key=_API_KEY, client=_client(handler))
    calls.clear()  # drop the construction probe

    texts = [f"t{i}" for i in range(100)]
    result = provider.embed_passages(texts)

    assert calls == [96, 4]
    assert result == [(float(i),) for i in range(100)]


def test_dimension_probe_reports_query_dim() -> None:
    """Whatever width the first (query) response returns becomes provider.dim."""
    served_dim = 1536  # embed-v4.0 default width

    def handler(request: httpx.Request) -> httpx.Response:
        count = len(_request_body(request)["texts"])
        return httpx.Response(200, json=_resp([[0.01] * served_dim] * count))

    provider = load_cohere_embedding(api_key=_API_KEY, client=_client(handler))
    assert provider.dim == served_dim
    assert len(provider.embed_query("x")) == served_dim


def test_endpoint_base_gets_embed_appended() -> None:
    """A caller endpoint (a /v2 base) gets /embed appended, not /v2/embed twice."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_resp([[0.0]]))

    load_cohere_embedding("m", endpoint="https://example.test/v2", api_key=_API_KEY, client=_client(handler))
    assert str(requests[0].url) == "https://example.test/v2/embed"


def test_missing_api_key_raises() -> None:
    """A cloud provider requires an api_key; absent, construction fails with EmbeddingError."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_resp([[0.0]]))

    with pytest.raises(EmbeddingError):
        load_cohere_embedding(client=_client(handler))


def test_count_mismatch_raises() -> None:
    """Fewer embeddings than inputs in a slice is an EmbeddingError, not a silent misalignment."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_resp([[0.1, 0.2]]))  # always one row

    provider = load_cohere_embedding(api_key=_API_KEY, client=_client(handler))
    with pytest.raises(EmbeddingError):
        provider.embed_passages(["a", "b"])


def test_invalid_response_missing_embeddings_raises() -> None:
    """A response missing the `embeddings` field fails validation at construction."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"wrong": {}})

    with pytest.raises(EmbeddingError):
        load_cohere_embedding(api_key=_API_KEY, client=_client(handler))


def test_http_error_status_raises() -> None:
    """A 5xx response from the probe call maps to EmbeddingError, not a raw httpx error."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with pytest.raises(EmbeddingError):
        load_cohere_embedding(api_key=_API_KEY, client=_client(handler))
