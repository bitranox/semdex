"""Chunker configuration model parsed from the [chunker] section."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from lib_layered_config import Config
from pydantic import ValidationError

from semdex.adapters.config.chunker import get_chunker_config
from semdex.domain.enums import ChunkStrategy


@pytest.mark.os_agnostic
def test_default_strategy_is_markdown_recursive(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """With no [chunker] section, markdown-aware recursive is the default."""
    cfg = get_chunker_config(config_factory({}))
    assert cfg.strategy is ChunkStrategy.RECURSIVE
    assert cfg.recipe == "markdown"
    assert cfg.chunk_overlap == 0
    assert cfg.tokenizer == "gpt2"
    assert cfg.semantic_model is None  # None keeps chonkie's own default breakpoint model


@pytest.mark.os_agnostic
def test_reads_semantic_model(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """The semantic breakpoint embedding model is read from [chunker].semantic_model."""
    cfg = get_chunker_config(
        config_factory({"chunker": {"strategy": "semantic", "semantic_model": "minishlab/potion-multilingual-128M"}})
    )
    assert cfg.semantic_model == "minishlab/potion-multilingual-128M"


@pytest.mark.os_agnostic
def test_reads_strategy_and_overlap(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """Strategy, overlap, and tokenizer are read from config."""
    cfg = get_chunker_config(
        config_factory({"chunker": {"strategy": "markdown", "chunk_overlap": 16, "tokenizer": "character"}})
    )
    assert cfg.strategy is ChunkStrategy.MARKDOWN
    assert cfg.chunk_overlap == 16
    assert cfg.tokenizer == "character"


@pytest.mark.os_agnostic
def test_unknown_strategy_is_rejected(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """An unknown strategy name is a validation error."""
    with pytest.raises(ValidationError):
        get_chunker_config(config_factory({"chunker": {"strategy": "nonsense"}}))


@pytest.mark.os_agnostic
def test_negative_overlap_is_rejected(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """A negative overlap is rejected at the boundary."""
    with pytest.raises(ValidationError):
        get_chunker_config(config_factory({"chunker": {"chunk_overlap": -1}}))


@pytest.mark.os_agnostic
def test_model_is_frozen(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """ChunkerConfig is immutable once parsed."""
    cfg = get_chunker_config(config_factory({}))
    with pytest.raises(ValidationError):
        cfg.strategy = ChunkStrategy.WHITESPACE  # type: ignore[misc]
