"""Unit tests for the ollama embedding adapter.

No server required: an ``httpx.MockTransport`` is injected so request building,
response parsing, the construction-time dimension probe, and error mapping are
exercised in isolation, mirroring tests/test_extractor_http.py. A real ollama
server is out of scope for these fast, offline CI tests.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from semdex.adapters.embedding import load_ollama_embedding
from semdex.domain.errors import EmbeddingError

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.os_agnostic


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _request_body(request: httpx.Request) -> Any:
    return json.loads(request.content)


def test_happy_path_probes_dimension_and_embeds() -> None:
    """Construction probes the dimension; embed_query/embed_passages post the expected body."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        count = len(_request_body(request)["input"])
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]] * count})

    provider = load_ollama_embedding("test-model", client=_client(handler))

    assert provider.dim == 3
    assert provider.embed_query("x") == (0.1, 0.2, 0.3)
    assert provider.embed_passages(["a", "b"]) == [(0.1, 0.2, 0.3), (0.1, 0.2, 0.3)]

    assert requests[0].url.path.endswith("/api/embed")
    assert _request_body(requests[0]) == {"model": "test-model", "input": ["dimension probe"]}
    assert _request_body(requests[-1]) == {"model": "test-model", "input": ["a", "b"]}


def test_large_matryoshka_dimension_is_discovered() -> None:
    """A high-dimensional model (e.g. qwen3-embedding) has its served dim probed, not assumed.

    The adapter pins no dimension: whatever width the server returns becomes provider.dim,
    so a large Matryoshka model works without any per-model code. Guards the probe path for
    wide vectors and confirms the exact ollama tag (`qwen3-embedding:4b`) is passed through.
    """
    served_dim = 2560  # qwen3-embedding:4b native width
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        count = len(_request_body(request)["input"])
        return httpx.Response(200, json={"embeddings": [[0.01] * served_dim] * count})

    provider = load_ollama_embedding("qwen3-embedding:4b", client=_client(handler))

    assert provider.dim == served_dim
    assert len(provider.embed_query("x")) == served_dim
    assert _request_body(requests[0]) == {"model": "qwen3-embedding:4b", "input": ["dimension probe"]}


def test_passages_length_mismatch_raises() -> None:
    """Fewer embeddings than inputs is an EmbeddingError, not a silent misalignment."""

    def handler(_request: httpx.Request) -> httpx.Response:
        # Always one row: matches the 1-input construction probe, mismatches the 2-input call below.
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]]})

    provider = load_ollama_embedding(client=_client(handler))
    with pytest.raises(EmbeddingError):
        provider.embed_passages(["a", "b"])


def test_invalid_response_missing_embeddings_field_raises() -> None:
    """A response missing the `embeddings` field fails validation at construction."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"wrong": []})

    with pytest.raises(EmbeddingError):
        load_ollama_embedding(client=_client(handler))


def test_http_error_status_raises() -> None:
    """A 5xx response from the probe call is mapped to EmbeddingError, not a raw httpx error."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with pytest.raises(EmbeddingError):
        load_ollama_embedding(client=_client(handler))


def test_keep_alive_included_when_set() -> None:
    """keep_alive rides in every embed request body when set (VRAM residency window)."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        count = len(_request_body(request)["input"])
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]] * count})

    provider = load_ollama_embedding("m", keep_alive="30m", client=_client(handler))
    provider.embed_query("x")

    # Both the construction dimension probe and the query carry keep_alive.
    assert _request_body(requests[0])["keep_alive"] == "30m"
    assert _request_body(requests[-1])["keep_alive"] == "30m"


def test_keep_alive_absent_when_none() -> None:
    """With keep_alive unset the key is omitted, so the default request is unchanged."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        count = len(_request_body(request)["input"])
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]] * count})

    load_ollama_embedding("m", client=_client(handler)).embed_query("x")

    assert "keep_alive" not in _request_body(requests[0])


def test_num_batch_included_when_set() -> None:
    """num_batch rides in every embed request as an ollama runtime option.

    Without it ollama uses a physical batch of 2048 and silently re-runs any longer input
    truncated to that batch, so a long text would be embedded from its first 2048 tokens only.
    """
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        count = len(_request_body(request)["input"])
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]] * count})

    provider = load_ollama_embedding("m", num_batch=4096, client=_client(handler))
    provider.embed_query("x")

    # Both the construction dimension probe and the query carry the option.
    assert _request_body(requests[0])["options"] == {"num_batch": 4096}
    assert _request_body(requests[-1])["options"] == {"num_batch": 4096}


def test_num_batch_absent_when_none() -> None:
    """With num_batch unset the key is omitted, so the default request is unchanged."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        count = len(_request_body(request)["input"])
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]] * count})

    load_ollama_embedding("m", client=_client(handler)).embed_query("x")

    assert "options" not in _request_body(requests[0])
