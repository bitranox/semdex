"""Unit tests for the Gemini native embedding adapter.

No server required: an ``httpx.MockTransport`` is injected so request building,
response parsing, the passage-vs-query ``taskType`` split, batch splitting at 100
requests per call, the construction-time dimension probe, the ``x-goog-api-key``
auth header, and error mapping are exercised offline - mirroring
tests/test_embedding_openai.py. The paid Gemini API is never called.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from semdex.adapters.embedding import load_gemini_embedding
from semdex.domain.errors import EmbeddingError

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.os_agnostic

_API_KEY = "test-gemini-key"


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _request_body(request: httpx.Request) -> Any:
    return json.loads(request.content)


def _resp(vectors: list[list[float]]) -> dict[str, Any]:
    """Build a well-formed ``:batchEmbedContents`` response for the given rows."""
    return {"embeddings": [{"values": vec} for vec in vectors]}


def _vec_for(text: str) -> list[float]:
    """A 1-d marker vector encoding the text index (``t<i>``), else ``[0.0]``."""
    return [float(text[1:])] if text.startswith("t") and text[1:].isdigit() else [0.0]


def test_happy_path_probes_dimension_and_embeds() -> None:
    """Construction probes the dimension; passages post the DOCUMENT taskType and models/ prefix."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        count = len(_request_body(request)["requests"])
        return httpx.Response(200, json=_resp([[0.1, 0.2, 0.3]] * count))

    provider = load_gemini_embedding("gemini-embedding-001", api_key=_API_KEY, client=_client(handler))

    assert provider.dim == 3
    assert provider.embed_passages(["a", "b"]) == [(0.1, 0.2, 0.3), (0.1, 0.2, 0.3)]

    last = requests[-1]
    assert last.url.path.endswith("/models/gemini-embedding-001:batchEmbedContents")
    assert last.headers.get("x-goog-api-key") == _API_KEY
    body = _request_body(last)
    assert body["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"
    assert body["requests"][0]["model"] == "models/gemini-embedding-001"
    assert body["requests"][0]["content"]["parts"][0]["text"] == "a"


def test_query_sends_query_task_type() -> None:
    """embed_query sends RETRIEVAL_QUERY, not the passages' RETRIEVAL_DOCUMENT."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        count = len(_request_body(request)["requests"])
        return httpx.Response(200, json=_resp([[0.5, 0.5]] * count))

    provider = load_gemini_embedding(api_key=_API_KEY, client=_client(handler))
    requests.clear()  # drop the construction probe
    provider.embed_query("hello")

    assert _request_body(requests[0])["requests"][0]["taskType"] == "RETRIEVAL_QUERY"


def test_sub_batches_at_100_and_preserves_order() -> None:
    """>100 texts split into 100+30 calls; the concatenated result keeps input order."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts = [item["content"]["parts"][0]["text"] for item in _request_body(request)["requests"]]
        calls.append(len(texts))
        return httpx.Response(200, json=_resp([_vec_for(text) for text in texts]))

    provider = load_gemini_embedding(api_key=_API_KEY, client=_client(handler))
    calls.clear()  # drop the construction probe

    texts = [f"t{i}" for i in range(130)]
    result = provider.embed_passages(texts)

    assert calls == [100, 30]
    assert result == [(float(i),) for i in range(130)]


def test_dimension_probe_reports_query_dim() -> None:
    """Whatever width the first (query) response returns becomes provider.dim."""
    served_dim = 3072  # gemini-embedding-001 native width

    def handler(request: httpx.Request) -> httpx.Response:
        count = len(_request_body(request)["requests"])
        return httpx.Response(200, json=_resp([[0.01] * served_dim] * count))

    provider = load_gemini_embedding(api_key=_API_KEY, client=_client(handler))
    assert provider.dim == served_dim
    assert len(provider.embed_query("x")) == served_dim


def test_endpoint_base_gets_model_path_appended() -> None:
    """A caller endpoint (a v1beta base) gets /models/<model>:batchEmbedContents appended."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_resp([[0.0]]))

    load_gemini_embedding("m", endpoint="https://example.test/v1beta", api_key=_API_KEY, client=_client(handler))
    assert str(requests[0].url) == "https://example.test/v1beta/models/m:batchEmbedContents"


def test_missing_api_key_raises() -> None:
    """A cloud provider requires an api_key; absent, construction fails with EmbeddingError."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_resp([[0.0]]))

    with pytest.raises(EmbeddingError):
        load_gemini_embedding(client=_client(handler))


def test_count_mismatch_raises() -> None:
    """Fewer embeddings than inputs in a slice is an EmbeddingError, not a silent misalignment."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_resp([[0.1, 0.2]]))  # always one row

    provider = load_gemini_embedding(api_key=_API_KEY, client=_client(handler))
    with pytest.raises(EmbeddingError):
        provider.embed_passages(["a", "b"])


def test_invalid_response_missing_embeddings_raises() -> None:
    """A response missing the `embeddings` field fails validation at construction."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"wrong": []})

    with pytest.raises(EmbeddingError):
        load_gemini_embedding(api_key=_API_KEY, client=_client(handler))


def test_http_error_status_raises() -> None:
    """A 5xx response from the probe call maps to EmbeddingError, not a raw httpx error."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with pytest.raises(EmbeddingError):
        load_gemini_embedding(api_key=_API_KEY, client=_client(handler))
