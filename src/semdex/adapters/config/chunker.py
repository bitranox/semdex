"""Chunker configuration model parsed from the ``[chunker]`` section.

Selects the strategy that splits a document into embeddable chunks. The
composition root turns this into a concrete adapter behind the ``ChunkText``
port. The chunk size itself is ``[index].max_tokens`` (already wired); this
section adds the strategy, the recursive recipe, overlap, and the tokenizer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from ...domain.enums import ChunkStrategy

if TYPE_CHECKING:
    from lib_layered_config import Config

# Offline gpt2 subword counting; "character" or an HF model id also work. The
# overlap default is 0 (deterministic); ~1/8 of the chunk size aids retrieval.
_DEFAULT_TOKENIZER = "gpt2"
_DEFAULT_RECIPE = "markdown"


class ChunkerConfig(BaseModel):
    """Validated, immutable chunker selection.

    Example:
        >>> ChunkerConfig().strategy.value
        'recursive'
        >>> ChunkerConfig().recipe
        'markdown'
    """

    model_config = ConfigDict(frozen=True)

    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    # chonkie recursive recipe (rule set); "markdown" makes the default recursive
    # strategy markdown-structure-aware. Ignored by the other strategies.
    recipe: str = _DEFAULT_RECIPE
    # Overlap between adjacent chunks, in tokens (0 = none).
    chunk_overlap: int = Field(default=0, ge=0)
    # Tokenizer used for token-accurate chunk sizes ("gpt2", "character", or an
    # HF model id).
    tokenizer: str = _DEFAULT_TOKENIZER
    # Embedding model the SEMANTIC/LATE strategies cut boundaries on (an HF/model2vec
    # or sentence-transformers id). None keeps chonkie's default (potion-base-32M),
    # which is English-distilled; set a multilingual model (e.g.
    # minishlab/potion-multilingual-128M, BAAI/bge-m3) for non-English corpora, where
    # the default finds boundaries on text it cannot read. Ignored by recursive/whitespace.
    semantic_model: str | None = None
    # Lossless size-guard: split any chunk over [index].max_tokens into token windows. The
    # semantic/late strategies group at sentence granularity, so an un-splittable "sentence" (a
    # minified-JS/web-scrape blob) can overflow far past the target - the embedder would then
    # truncate it and drop the overflow. On by default; a safety net, never a truncate.
    enforce_max_tokens: bool = True


def get_chunker_config(config: Config) -> ChunkerConfig:
    """Parse the ``[chunker]`` section into a ChunkerConfig.

    Falls back to the markdown-aware recursive strategy when the section is absent.

    Example:
        >>> from lib_layered_config import Config
        >>> get_chunker_config(Config({}, {})).strategy.value
        'recursive'
    """
    return ChunkerConfig.model_validate(config.get("chunker", {}))


__all__ = [
    "ChunkerConfig",
    "get_chunker_config",
]
