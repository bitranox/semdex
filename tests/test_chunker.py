"""Unit tests for the chunker adapters that CI can run (whitespace + chonkie + sts).

The embedding-driven strategies (semantic/late) need semdex[chunk-semantic] and a
model, so they are covered by the local_only tests. The recursive test pins an
empty recipe so it stays offline (no chonkie hub fetch).
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from semdex.adapters.chunker import ChonkieChunker, StsChunker, WhitespaceChunker
from semdex.domain.enums import ChunkStrategy
from semdex.domain.models import Chunk, ExtractedDocument, SourceRef

pytestmark = pytest.mark.os_agnostic

_SOURCE = SourceRef(uri="/x/doc.md", label="curated", content_hash="h", mtime=1.0)
_TEXT = (
    "# Heading One\n\n"
    + ("Alpha beta gamma delta epsilon zeta eta theta iota kappa. " * 6)
    + "\n\n## Heading Two\n\n"
    + ("Lorem ipsum dolor sit amet consectetur adipiscing elit sed. " * 6)
)
_DOC = ExtractedDocument(source=_SOURCE, text=_TEXT)
_MAX_TOKENS = 40


def _assert_valid(chunks: list[Chunk], *, max_tokens: int) -> None:
    assert chunks, "expected at least one chunk"
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))  # sequential ordinals
    assert all(c.source == _SOURCE for c in chunks)  # provenance threaded
    assert all(c.text.strip() for c in chunks)  # no empty chunks
    assert all(c.token_count > 0 for c in chunks)
    assert max(c.token_count for c in chunks) <= max_tokens  # chunk size respected (no overlap)


def test_whitespace_windows() -> None:
    chunks = WhitespaceChunker()(_DOC, max_tokens=_MAX_TOKENS)
    _assert_valid(chunks, max_tokens=_MAX_TOKENS)
    assert all(len(c.text.split()) <= _MAX_TOKENS for c in chunks)


def test_recursive_token_accurate() -> None:
    pytest.importorskip("chonkie")
    # recipe="" uses chonkie's default rules (offline, no hub fetch).
    chunks = ChonkieChunker(ChunkStrategy.RECURSIVE, recipe="", tokenizer="gpt2")(_DOC, max_tokens=_MAX_TOKENS)
    _assert_valid(chunks, max_tokens=_MAX_TOKENS)


def test_recursive_overlap_adds_context() -> None:
    pytest.importorskip("chonkie")
    base = ChonkieChunker(ChunkStrategy.RECURSIVE, recipe="", overlap=0, tokenizer="gpt2")(_DOC, max_tokens=_MAX_TOKENS)
    overlapped = ChonkieChunker(ChunkStrategy.RECURSIVE, recipe="", overlap=12, tokenizer="gpt2")(
        _DOC, max_tokens=_MAX_TOKENS
    )
    # Overlap keeps the same chunk count but grows chunks with prepended context.
    assert len(overlapped) == len(base)
    assert sum(c.token_count for c in overlapped) > sum(c.token_count for c in base)


class _RecordingSemanticChunker:
    """Fake chonkie.SemanticChunker that records its init kwargs (real one downloads a model)."""

    last_kwargs: ClassVar[dict[str, object]] = {}

    def __init__(self, **kwargs: object) -> None:
        _RecordingSemanticChunker.last_kwargs = kwargs

    def __call__(self, _text: str) -> list[object]:
        return []


def _fake_chonkie(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    fake = types.ModuleType("chonkie")
    fake.SemanticChunker = _RecordingSemanticChunker  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "chonkie", fake)


def test_semantic_passes_configured_breakpoint_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured semantic_model reaches chonkie's SemanticChunker as embedding_model.

    The breakpoint embedding model decides WHERE semantic boundaries fall; chonkie's default
    is English-distilled (potion-base-32M), so a multilingual corpus needs an explicit model.
    """
    _fake_chonkie(monkeypatch)
    ChonkieChunker(ChunkStrategy.SEMANTIC, semantic_model="minishlab/potion-multilingual-128M")(
        _DOC, max_tokens=_MAX_TOKENS
    )
    assert _RecordingSemanticChunker.last_kwargs.get("embedding_model") == "minishlab/potion-multilingual-128M"


def test_semantic_omits_embedding_model_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no semantic_model, the adapter does not force one - chonkie keeps its own default."""
    _fake_chonkie(monkeypatch)
    _RecordingSemanticChunker.last_kwargs = {}
    ChonkieChunker(ChunkStrategy.SEMANTIC)(_DOC, max_tokens=_MAX_TOKENS)
    assert "embedding_model" not in _RecordingSemanticChunker.last_kwargs


@pytest.mark.parametrize("strategy", [ChunkStrategy.MARKDOWN, ChunkStrategy.FAST])
def test_sts_strategies(strategy: ChunkStrategy) -> None:
    pytest.importorskip("semantic_text_splitter")
    chunks = StsChunker(strategy, tokenizer="gpt2")(_DOC, max_tokens=_MAX_TOKENS)
    _assert_valid(chunks, max_tokens=_MAX_TOKENS)


def test_resolve_tokenizer_builtin_passthrough() -> None:
    from semdex.adapters.chunker.chonkie import resolve_tokenizer

    # chonkie tokenizes these locally, so they pass through as strings (no object / no network).
    assert resolve_tokenizer("character") == "character"
    assert resolve_tokenizer("word") == "word"


def test_resolve_tokenizer_builds_object_via_cached_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    from semdex.adapters.chunker import chonkie as mod

    calls: list[str] = []
    sentinel = object()

    class _FakeTokenizer:
        @staticmethod
        def from_pretrained(repo_id: str) -> object:
            calls.append(repo_id)
            if repo_id == "gpt2":
                raise OSError("bare id not in local cache")  # mirror the offline-cache miss
            return sentinel

    monkeypatch.setattr(tokenizers, "Tokenizer", _FakeTokenizer)
    mod.resolve_tokenizer.cache_clear()
    try:
        resolved = mod.resolve_tokenizer("gpt2")
    finally:
        mod.resolve_tokenizer.cache_clear()

    # A pre-built Tokenizer object (not the "gpt2" string) is what avoids chonkie's tokie hub call.
    assert resolved is sentinel
    assert calls == ["gpt2", "openai-community/gpt2"]  # bare id, then the cached canonical repo


def test_resolve_tokenizer_falls_back_to_string(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    from semdex.adapters.chunker import chonkie as mod

    class _FailingTokenizer:
        @staticmethod
        def from_pretrained(repo_id: str) -> object:
            raise OSError("unavailable")

    monkeypatch.setattr(tokenizers, "Tokenizer", _FailingTokenizer)
    mod.resolve_tokenizer.cache_clear()
    try:
        resolved = mod.resolve_tokenizer("totally-unknown-tokenizer")
    finally:
        mod.resolve_tokenizer.cache_clear()

    # Unresolvable -> fall back to the string so chonkie can still try its own resolution.
    assert resolved == "totally-unknown-tokenizer"
