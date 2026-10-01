"""Composition chunker factory: selects a strategy, degrades when a lib is absent."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from importlib.machinery import ModuleSpec

import pytest

from semdex.adapters.chunker import ChonkieChunker, StsChunker, WhitespaceChunker
from semdex.composition import build_chunker
from semdex.domain.enums import ChunkStrategy

pytestmark = pytest.mark.os_agnostic


def _find_spec_without(missing: str) -> Callable[[str], ModuleSpec | None]:
    """A find_spec replacement that pretends *missing* is not installed."""
    real = importlib.util.find_spec

    def _find(name: str) -> ModuleSpec | None:
        return None if name == missing else real(name)

    return _find


def test_whitespace_strategy() -> None:
    assert isinstance(build_chunker(ChunkStrategy.WHITESPACE), WhitespaceChunker)


@pytest.mark.parametrize("strategy", [ChunkStrategy.RECURSIVE, ChunkStrategy.SEMANTIC, ChunkStrategy.LATE])
def test_chonkie_strategies(strategy: ChunkStrategy) -> None:
    """The chonkie-backed strategies map to ChonkieChunker (needs semdex[chunk])."""
    pytest.importorskip("chonkie")
    assert isinstance(build_chunker(strategy), ChonkieChunker)


@pytest.mark.parametrize("strategy", [ChunkStrategy.MARKDOWN, ChunkStrategy.FAST])
def test_sts_strategies(strategy: ChunkStrategy) -> None:
    """The semantic-text-splitter strategies map to StsChunker (needs semdex[chunk])."""
    pytest.importorskip("semantic_text_splitter")
    assert isinstance(build_chunker(strategy), StsChunker)


def test_falls_back_when_chonkie_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A chonkie strategy degrades to whitespace when chonkie is unavailable."""
    monkeypatch.setattr(importlib.util, "find_spec", _find_spec_without("chonkie"))
    assert isinstance(build_chunker(ChunkStrategy.RECURSIVE), WhitespaceChunker)


def test_falls_back_when_sts_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """An sts strategy degrades to whitespace when semantic-text-splitter is unavailable."""
    monkeypatch.setattr(importlib.util, "find_spec", _find_spec_without("semantic_text_splitter"))
    assert isinstance(build_chunker(ChunkStrategy.MARKDOWN), WhitespaceChunker)
