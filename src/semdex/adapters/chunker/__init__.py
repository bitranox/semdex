"""Chunker adapters, each behind the ``ChunkText`` port.

    * :class:`.whitespace.WhitespaceChunker` - zero-dependency fixed-window fallback
    * :class:`.chonkie.ChonkieChunker` - recursive (default), semantic (SDPM), late
    * :class:`.semantic_text_splitter.StsChunker` - markdown / fast (CommonMark, Rust)

Selected by ``[chunker].strategy`` via ``composition.build_chunker``.
"""

from __future__ import annotations

from .chonkie import ChonkieChunker
from .semantic_text_splitter import StsChunker
from .whitespace import WhitespaceChunker

__all__ = [
    "ChonkieChunker",
    "StsChunker",
    "WhitespaceChunker",
]
