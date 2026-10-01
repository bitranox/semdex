"""Shared helpers for the chunker adapters.

Turns a library's output into ordered domain :class:`~semdex.domain.models.Chunk`
objects (dropping empties, numbering ordinals), provides the fixed-window
splitter used by the whitespace fallback, and resolves a token-count callable
from chonkie's tokenizer (offline for ``gpt2``/``character``) so the
semantic-text-splitter strategies count tokens the same way the chonkie ones do.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...domain.models import Chunk

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from ...domain.models import ExtractedDocument, SourceRef


def window_chunks(document: ExtractedDocument, *, max_tokens: int) -> list[Chunk]:
    """Split a document into ordered, whitespace-token-bounded windows."""
    tokens = document.text.split()
    return [
        Chunk(
            text=" ".join(tokens[start : start + max_tokens]),
            source=document.source,
            ordinal=ordinal,
            token_count=len(tokens[start : start + max_tokens]),
        )
        for ordinal, start in enumerate(range(0, len(tokens), max_tokens))
    ]


def split_oversized(
    text: str,
    token_count: int,
    *,
    max_tokens: int,
    offsets_of: Callable[[str], list[tuple[int, int]]],
) -> list[tuple[str, int]]:
    """Split a chunk that exceeds ``max_tokens`` into <=max_tokens pieces, LOSSLESSLY.

    The size-guard for the embedding-driven strategies (semantic/late), whose sentence-level
    grouping can emit a chunk far over the target when a single "sentence" is un-splittable (a
    minified-JS/webpack blob with no boundaries). Without this the embedder truncates the chunk to
    its max_seq at embed time and silently drops the overflow; splitting here embeds ALL of it.

    ``offsets_of(text)`` returns one ``(start_char, end_char)`` per token. The pieces TILE the
    original text at token boundaries, so ``"".join(piece_texts) == text`` exactly (each piece is a
    substring) and every piece holds at most ``max_tokens`` tokens. A within-cap chunk is returned
    unchanged.

    ``token_count`` and ``offsets_of`` need not agree on how many tokens the text holds, and for
    the semantic and late strategies they systematically do not: chonkie's ``SemanticChunker``
    accepts no tokenizer argument and counts in its embedding model's tokenizer, while the offsets
    come from ``[chunker].tokenizer``. Measured divergence on the cached corpora is 6 to 22 percent
    over byte-identical text. This used to return the chunk unchanged whenever OUR tokenizer said
    it fit, which let a chunk recorded at 782 tokens through a 256 cap - and the embedder then
    truncated it and dropped the overflow silently, the exact loss this guard exists to prevent.
    So the RECORDED count decides how many pieces are needed, because that is the count the
    embedder will act on, and the offsets decide only where the cuts fall.
    """
    if token_count <= max_tokens:
        return [(text, token_count)]
    offsets = offsets_of(text)
    n = len(offsets)
    if n <= 1:  # a single token, or a tokenizer that yielded nothing: there is no cut to make
        return [(text, token_count)]
    window = max_tokens
    if n <= max_tokens:
        # Our tokenizer says it fits and the recorded count says it does not. Split into as many
        # EVEN pieces as the recorded count needs, so each piece lands under the cap in the terms
        # that matter, rather than skipping the split entirely.
        pieces_needed = -(-token_count // max_tokens)  # ceil
        window = max(1, -(-n // pieces_needed))
    pieces: list[tuple[str, int]] = []
    for i in range(0, n, window):
        start = 0 if i == 0 else offsets[i][0]
        end = len(text) if i + window >= n else offsets[i + window][0]
        pieces.append((text[start:end], len(offsets[i : i + window])))
    return pieces


def chunks_from_pairs(pairs: Iterable[tuple[str, int]], *, source: SourceRef) -> list[Chunk]:
    """Build ordered domain chunks from ``(text, token_count)`` pairs, dropping empties."""
    result: list[Chunk] = []
    for text, token_count in pairs:
        if not text.strip():
            continue
        result.append(Chunk(text=text, source=source, ordinal=len(result), token_count=int(token_count)))
    return result


def token_counter(tokenizer: str) -> Callable[[str], int]:
    """Return a token-count callable from chonkie's tokenizer resolution.

    chonkie is always present with the semantic-text-splitter strategies (both in
    ``semdex[chunk]``), so reusing its tokenizer keeps token counts consistent
    across strategies. Offline for ``gpt2``/``character``.
    """
    from ...domain.errors import ChunkingError

    try:
        from chonkie import RecursiveChunker
    except ImportError as exc:  # pragma: no cover - only without semdex[chunk]
        raise ChunkingError("chonkie is not installed; install semdex[chunk]") from exc
    counter: Callable[[str], int] = RecursiveChunker(tokenizer=tokenizer, chunk_size=2048).tokenizer.count_tokens
    return counter


__all__ = [
    "chunks_from_pairs",
    "split_oversized",
    "token_counter",
    "window_chunks",
]
