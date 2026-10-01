"""The instruction prefix must reach the wire, and must reach it asymmetrically.

Qwen3-Embedding is trained to receive an instruction on the query and none on the passage. Before
this existed, the ollama and openai adapters built a query as ``passages_fn([text])[0]``, so a
query and a passage went to the server as byte-identical text and no prefix could be applied at
all. The `[embedding]` section also already owned a knob that was declared, documented, resolved
from config and then never read (`num_batch`), so these tests assert the whole path end to end
rather than the adapter alone: a prefix that is configured but not sent reads as configured and
behaves as absent.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from semdex.adapters.embedding.ollama import load_ollama_embedding
from semdex.adapters.embedding.openai import load_openai_embedding
from semdex.composition import build_embedding
from semdex.domain.enums import EmbeddingBackend
from semdex.domain.errors import ConfigurationError

pytestmark = pytest.mark.os_agnostic

_QWEN3_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"


class _Recorder:
    """Captures the exact input list each embed request carried."""

    def __init__(self, dim: int = 4) -> None:
        self.inputs: list[list[str]] = []
        self._dim = dim

    def ollama(self, request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content)
        self.inputs.append(list(body["input"]))
        return httpx.Response(200, json={"embeddings": [[0.1] * self._dim for _ in body["input"]]})

    def openai(self, request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content)
        self.inputs.append(list(body["input"]))
        return httpx.Response(
            200,
            json={"data": [{"index": i, "embedding": [0.1] * self._dim} for i, _ in enumerate(body["input"])]},
        )

    @property
    def sent(self) -> list[str]:
        """Every text sent after the dimension probe, which fires during construction."""
        return [text for batch in self.inputs[1:] for text in batch]


def _ollama(recorder: _Recorder, **kwargs: Any) -> Any:
    client = httpx.Client(transport=httpx.MockTransport(recorder.ollama))
    return load_ollama_embedding("qwen3-embedding:4b", client=client, **kwargs)


def _openai(recorder: _Recorder, **kwargs: Any) -> Any:
    client = httpx.Client(transport=httpx.MockTransport(recorder.openai))
    return load_openai_embedding("qwen3-embedding:4b", client=client, **kwargs)


# --- the prefix reaches the wire ---------------------------------------------------------------


def test_ollama_sends_the_query_prefix() -> None:
    recorder = _Recorder()
    provider = _ollama(recorder, query_prefix=_QWEN3_PREFIX)

    provider.embed_query("what is a chunk")

    assert recorder.sent == [f"{_QWEN3_PREFIX}what is a chunk"]


def test_ollama_leaves_passages_unprefixed_when_only_the_query_prefix_is_set() -> None:
    """The asymmetry IS the feature.

    Qwen3's model card says "No need to add instruction for retrieval documents". A prefix that
    leaked onto the passage would embed the whole corpus under an instruction it was never meant
    to carry, and the damage would be invisible: search still returns results, just worse ones.
    """
    recorder = _Recorder()
    provider = _ollama(recorder, query_prefix=_QWEN3_PREFIX)

    provider.embed_passages(["a passage about chunks"])

    assert recorder.sent == ["a passage about chunks"]


def test_ollama_applies_each_prefix_to_its_own_side() -> None:
    """E5-style, where both sides carry a different marker."""
    recorder = _Recorder()
    provider = _ollama(recorder, query_prefix="query: ", passage_prefix="passage: ")

    provider.embed_query("q")
    provider.embed_passages(["p1", "p2"])

    assert recorder.sent == ["query: q", "passage: p1", "passage: p2"]


def test_openai_sends_the_query_prefix() -> None:
    """The openai provider is how llama.cpp and ollama's own /v1 are reached, and it had the
    same defect."""
    recorder = _Recorder()
    provider = _openai(recorder, query_prefix=_QWEN3_PREFIX)

    provider.embed_query("what is a chunk")

    assert recorder.sent == [f"{_QWEN3_PREFIX}what is a chunk"]


def test_openai_leaves_passages_unprefixed_when_only_the_query_prefix_is_set() -> None:
    recorder = _Recorder()
    provider = _openai(recorder, query_prefix=_QWEN3_PREFIX)

    provider.embed_passages(["a passage"])

    assert recorder.sent == ["a passage"]


def test_no_prefix_leaves_the_text_byte_identical() -> None:
    """The default must not change a single existing request."""
    recorder = _Recorder()
    provider = _ollama(recorder)

    provider.embed_query("q")
    provider.embed_passages(["p"])

    assert recorder.sent == ["q", "p"]


# --- and it survives the whole composition path ------------------------------------------------


def test_the_configured_prefix_reaches_the_adapter_through_build_embedding() -> None:
    """The dead-knob test.

    `num_batch` was declared, documented, parsed from config and never read by any adapter. A test
    of the adapter alone would have passed throughout. This one fails if the wiring is cut anywhere
    between build_embedding and the request body.
    """
    recorder = _Recorder()
    monkey = httpx.Client(transport=httpx.MockTransport(recorder.ollama))
    import semdex.adapters.embedding as embedding_pkg

    original = embedding_pkg.load_ollama_embedding
    try:
        embedding_pkg.load_ollama_embedding = lambda model, **kw: original(model, client=monkey, **kw)  # type: ignore[assignment]
        provider = build_embedding(
            EmbeddingBackend.OLLAMA,
            model="qwen3-embedding:4b",
            query_prefix=_QWEN3_PREFIX,
        )
        provider.embed_query("what is a chunk")
    finally:
        embedding_pkg.load_ollama_embedding = original  # type: ignore[assignment]

    assert recorder.sent == [f"{_QWEN3_PREFIX}what is a chunk"]


# --- a provider that cannot honour it must say so ----------------------------------------------


@pytest.mark.parametrize("provider", [EmbeddingBackend.FASTEMBED, EmbeddingBackend.MODEL2VEC])
def test_a_prefix_is_refused_by_providers_that_apply_their_own(provider: EmbeddingBackend) -> None:
    """Silently ignoring it would recreate the exact defect this change fixes.

    fastembed calls the model's own ``query_embed`` and model2vec is a static model; a prefix here
    would double-apply or vanish. Refusing is the only option that cannot be mistaken for working.
    """
    with pytest.raises(ConfigurationError, match="query_prefix"):
        build_embedding(provider, query_prefix="query: ")


def test_the_refusal_names_a_provider_that_does_support_it() -> None:
    """An error that only says no costs the reader a search through the source."""
    with pytest.raises(ConfigurationError, match="ollama"):
        build_embedding(EmbeddingBackend.FASTEMBED, query_prefix="query: ")


def test_an_empty_prefix_is_accepted_everywhere() -> None:
    """Only a non-empty prefix is a request for behaviour nothing can deliver."""
    provider = build_embedding(EmbeddingBackend.PLACEHOLDER, query_prefix="", passage_prefix="")

    assert provider.dim > 0
