"""Whitespace fixed-window chunker: the zero-dependency fallback.

Splits on whitespace into ordered windows of ``max_tokens`` words. Not
tokenizer-accurate and structure-blind, but dependency-free and deterministic;
the composition root falls back to it when the selected strategy's library is
unavailable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._base import window_chunks

if TYPE_CHECKING:
    from ...domain.models import Chunk, ExtractedDocument

_DEFAULT_MAX_TOKENS = 256


class WhitespaceChunker:
    """Split a document into whitespace-token-bounded windows."""

    def __call__(self, document: ExtractedDocument, *, max_tokens: int = _DEFAULT_MAX_TOKENS) -> list[Chunk]:
        return window_chunks(document, max_tokens=max_tokens)


# Static conformance assertion -- pyright verifies WhitespaceChunker satisfies ChunkText.
if TYPE_CHECKING:
    from ...application.ports import ChunkText

    _assert_chunk: ChunkText = WhitespaceChunker()


__all__ = [
    "WhitespaceChunker",
]
