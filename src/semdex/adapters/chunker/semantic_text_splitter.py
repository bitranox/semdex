"""semantic-text-splitter chunker adapter: markdown and fast recursive.

``MARKDOWN`` uses ``MarkdownSplitter`` (true CommonMark parsing, so it never
splits inside a code block or table); ``FAST`` uses ``TextSplitter`` (plain fast
recursive). Both are Rust-backed with native overlap. Token-accurate sizing uses
chonkie's tokenizer as the length callback (see :func:`._base.token_counter`), so
chunk sizes match the chonkie strategies. Needs ``semdex[chunk]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...domain.enums import ChunkStrategy
from ...domain.errors import ChunkingError
from ._base import chunks_from_pairs, token_counter

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...domain.models import Chunk, ExtractedDocument

_DEFAULT_MAX_TOKENS = 256


class StsChunker:
    """Chunk a document with semantic-text-splitter's Markdown/Text splitter."""

    def __init__(self, strategy: ChunkStrategy, *, overlap: int = 0, tokenizer: str = "gpt2") -> None:
        self._strategy = strategy
        self._overlap = overlap
        self._count = token_counter(tokenizer)
        self._cached: tuple[int, Any] | None = None

    def __call__(self, document: ExtractedDocument, *, max_tokens: int = _DEFAULT_MAX_TOKENS) -> list[Chunk]:
        splitter = self._resolve(max_tokens)
        texts = splitter.chunks(document.text)
        return chunks_from_pairs(((text, self._count(text)) for text in texts), source=document.source)

    def _resolve(self, max_tokens: int) -> Any:
        if self._cached is not None and self._cached[0] == max_tokens:
            return self._cached[1]
        splitter = self._build(max_tokens)
        self._cached = (max_tokens, splitter)
        return splitter

    def _build(self, max_tokens: int) -> Any:
        try:
            from semantic_text_splitter import MarkdownSplitter, TextSplitter
        except ImportError as exc:
            raise ChunkingError("semantic-text-splitter is not installed; install semdex[chunk]") from exc
        counter: Callable[[str], int] = self._count
        if self._strategy is ChunkStrategy.MARKDOWN:
            return MarkdownSplitter.from_callback(counter, max_tokens, overlap=self._overlap)
        return TextSplitter.from_callback(counter, max_tokens, overlap=self._overlap)


# Static conformance assertion -- pyright verifies StsChunker satisfies ChunkText.
if TYPE_CHECKING:
    from ...application.ports import ChunkText

    _assert_chunk: ChunkText = StsChunker(ChunkStrategy.MARKDOWN)


__all__ = [
    "StsChunker",
]
