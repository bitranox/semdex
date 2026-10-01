"""Composition embedding factory: routes to a provider, degrades for the default.

The real providers download models, so provider routing is verified by patching
each ``load_*`` factory to a sentinel (no network). The CallableEmbedding adapter
is unit-tested with injected callables.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import semdex.adapters.embedding as emb
from semdex.adapters.embedding import CallableEmbedding
from semdex.composition import build_embedding
from semdex.domain.enums import EmbeddingBackend
from semdex.domain.errors import EmbeddingError

pytestmark = pytest.mark.os_agnostic


def _sentinel(model_id: str) -> CallableEmbedding:
    return CallableEmbedding(
        model_id=model_id, dim=3, passages_fn=lambda _t: [(1.0, 0.0, 0.0)], query_fn=lambda _t: (1.0, 0.0, 0.0)
    )


def test_callable_embedding_delegates() -> None:
    """CallableEmbedding exposes model_id/dim and delegates to its callables."""
    provider = CallableEmbedding(
        model_id="m",
        dim=2,
        passages_fn=lambda texts: [(float(len(t)), 0.0) for t in texts],
        query_fn=lambda t: (float(len(t)), 0.0),
    )
    assert provider.model_id == "m"
    assert provider.dim == 2
    assert provider.embed_passages(["ab", "abc"]) == [(2.0, 0.0), (3.0, 0.0)]
    assert provider.embed_query("abcd") == (4.0, 0.0)


def test_placeholder_provider_has_tunable_dim() -> None:
    """The placeholder provider honors the configured dim."""
    provider = build_embedding(EmbeddingBackend.PLACEHOLDER, dim=8)
    assert provider.dim == 8
    assert len(provider.embed_query("hello world")) == 8


def test_fastembed_degrades_to_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    """If fastembed cannot load, the default degrades to the placeholder (not a crash)."""

    def _fail(_model: str | None = None, *, threads: int | None = None) -> CallableEmbedding:
        raise EmbeddingError("fastembed unavailable")

    monkeypatch.setattr(emb, "load_fastembed_embedding", _fail)
    provider = build_embedding(EmbeddingBackend.FASTEMBED)
    assert provider.model_id == "in-memory"  # the placeholder fallback


def test_fastembed_reraises_when_fallback_disallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """With allow_fallback=False a fastembed load failure raises, never a silent placeholder.

    A benchmark/bulk-embed caller must NOT unknowingly cache placeholder vectors when the
    real model is missing (e.g. a wiped cache), so it opts out of the graceful degradation.
    """

    def _fail(_model: str | None = None, *, threads: int | None = None) -> CallableEmbedding:
        raise EmbeddingError("fastembed unavailable")

    monkeypatch.setattr(emb, "load_fastembed_embedding", _fail)
    with pytest.raises(EmbeddingError):
        build_embedding(EmbeddingBackend.FASTEMBED, allow_fallback=False)


@pytest.mark.parametrize(
    ("backend", "loader_name"),
    [
        (EmbeddingBackend.FASTEMBED, "load_fastembed_embedding"),
        (EmbeddingBackend.MODEL2VEC, "load_model2vec_embedding"),
        (EmbeddingBackend.OLLAMA, "load_ollama_embedding"),
        (EmbeddingBackend.OPENAI, "load_openai_embedding"),
        (EmbeddingBackend.GEMINI, "load_gemini_embedding"),
        (EmbeddingBackend.COHERE, "load_cohere_embedding"),
        (EmbeddingBackend.SENTENCE_TRANSFORMERS, "load_sentence_transformer_embedding"),
    ],
)
def test_provider_routing(monkeypatch: pytest.MonkeyPatch, backend: EmbeddingBackend, loader_name: str) -> None:
    """Each backend routes to its own loader."""

    def _fake(*_a: object, **_k: object) -> CallableEmbedding:
        return _sentinel(loader_name)

    monkeypatch.setattr(emb, loader_name, _fake)
    provider = build_embedding(backend)
    assert provider.model_id == loader_name


def test_build_embedding_forwards_num_batch_to_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole chain must carry num_batch; a missing link here silently restores ollama's 2048."""
    seen: dict[str, object] = {}

    def _fake_loader(model_id: str | None = None, **kwargs: object) -> CallableEmbedding:
        seen.update(kwargs)
        return _sentinel(model_id or "m")

    monkeypatch.setattr(emb, "load_ollama_embedding", _fake_loader)
    build_embedding(EmbeddingBackend.OLLAMA, model="bge-m3", endpoint="http://o.invalid", num_batch=4096)

    assert seen["num_batch"] == 4096


def test_build_embedding_omits_num_batch_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset stays unset, so the default ollama request is unchanged."""
    seen: dict[str, object] = {}

    def _fake_loader(model_id: str | None = None, **kwargs: object) -> CallableEmbedding:
        seen.update(kwargs)
        return _sentinel(model_id or "m")

    monkeypatch.setattr(emb, "load_ollama_embedding", _fake_loader)
    build_embedding(EmbeddingBackend.OLLAMA, model="bge-m3", endpoint="http://o.invalid")

    assert "num_batch" not in seen


def test_build_index_production_forwards_num_batch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The index wiring must carry num_batch: a config key nothing reads is a dead knob."""
    from semdex.composition import build_index_production

    seen: dict[str, object] = {}

    def _fake_loader(model_id: str | None = None, **kwargs: object) -> CallableEmbedding:
        seen.update(kwargs)
        return _sentinel(model_id or "m")

    monkeypatch.setattr(emb, "load_ollama_embedding", _fake_loader)
    build_index_production(
        tmp_path,
        embedding_provider=EmbeddingBackend.OLLAMA,
        embedding_model="bge-m3",
        embedding_endpoint="http://o.invalid",
        embedding_num_batch=4096,
    )

    assert seen["num_batch"] == 4096
