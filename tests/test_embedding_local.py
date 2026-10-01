"""Local-only tests for the real embedding providers (download models / need a server).

local_only: excluded from ``make test``; run via ``make ti``. Each exercises a real
provider through the composition factory, mirroring the sentence-transformers stance.
"""

from __future__ import annotations

import pytest

from semdex.composition import build_embedding
from semdex.domain.enums import EmbeddingBackend

pytestmark = [pytest.mark.local_only, pytest.mark.os_agnostic]


def test_fastembed_real_embeds() -> None:
    """The default fastembed provider loads bge-small and embeds (384-dim)."""
    pytest.importorskip("fastembed")
    provider = build_embedding(EmbeddingBackend.FASTEMBED)
    assert provider.dim == 384
    query = provider.embed_query("neural network training")
    passages = provider.embed_passages(["gradient descent optimizes weights", "tomatoes need sun"])
    assert len(query) == 384
    assert len(passages) == 2 and all(len(vec) == 384 for vec in passages)


def test_model2vec_real_embeds() -> None:
    """The model2vec provider loads a static model and embeds."""
    pytest.importorskip("model2vec")
    provider = build_embedding(EmbeddingBackend.MODEL2VEC)
    assert provider.dim > 0
    assert len(provider.embed_query("hello world")) == provider.dim


def test_ollama_real_embeds() -> None:
    """The ollama provider embeds against a running server (skipped if unreachable)."""
    httpx = pytest.importorskip("httpx")
    endpoint = "http://127.0.0.1:11434"
    try:
        httpx.get(f"{endpoint}/api/tags", timeout=2)
    except httpx.HTTPError:
        pytest.skip("no ollama server reachable at 127.0.0.1:11434")
    provider = build_embedding(EmbeddingBackend.OLLAMA, endpoint=endpoint)
    assert provider.dim > 0
    assert len(provider.embed_query("hello world")) == provider.dim
