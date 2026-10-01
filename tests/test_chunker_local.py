"""Local-only tests for the embedding-driven chunkers (semantic / late).

local_only: they need ``semdex[chunk-semantic]`` and download an embedding model,
so they run via ``make ti``, not ``make test``. They assert the same invariants as
the CI chunker tests.
"""

from __future__ import annotations

import pytest

from semdex.adapters.chunker import ChonkieChunker
from semdex.domain.enums import ChunkStrategy
from semdex.domain.models import Chunk, ExtractedDocument, SourceRef

pytestmark = [pytest.mark.local_only, pytest.mark.os_agnostic]

_SOURCE = SourceRef(uri="/x/doc.md", label="curated", content_hash="h", mtime=1.0)
_DOC = ExtractedDocument(
    source=_SOURCE,
    text=("Alpha beta gamma delta. " * 8) + "\n\n" + ("Lorem ipsum dolor sit amet consectetur. " * 8),
)


def _assert_valid(chunks: list[Chunk]) -> None:
    assert chunks
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(c.source == _SOURCE for c in chunks)
    assert all(c.text.strip() and c.token_count > 0 for c in chunks)


def test_semantic_sdpm_chunks() -> None:
    """chonkie SemanticChunker (SDPM double-pass) splits and merges by similarity."""
    pytest.importorskip("model2vec")  # chonkie[semantic] embeddings backend
    _assert_valid(ChonkieChunker(ChunkStrategy.SEMANTIC)(_DOC, max_tokens=40))


def test_late_chunks() -> None:
    """chonkie LateChunker embeds the whole doc then pools per chunk."""
    pytest.importorskip("sentence_transformers")
    _assert_valid(ChonkieChunker(ChunkStrategy.LATE)(_DOC, max_tokens=40))


def test_late_chunks_of_a_long_document_carry_real_token_counts_and_hold_the_cap() -> None:
    """chonkie's LateChunker reports token_count 1 for every chunk of a LONG document.

    Reproduced on a 30,000-character German court decision: 60 of 60 chunks at 1 token, and 15
    of them over the 256 cap, which the size-guard then let through because 1 fits any cap. The
    adapter recounts late chunks in the configured tokenizer, so the counts are real and the
    guard splits what is oversized.
    """
    pytest.importorskip("sentence_transformers")
    paragraph = (
        "Die Behoerde hat deshalb zunaechst nach Aktenlage die vorgelegten Unterlagen zu pruefen. "
        "Nach dem Ergebnis dieser Ueberpruefung bestimmt sich die Art der weiteren Ermittlungen. "
    )
    long_doc = ExtractedDocument(source=_SOURCE, text="\n\n".join([paragraph * 12] * 12))  # ~26k chars
    chunks = ChonkieChunker(ChunkStrategy.LATE, tokenizer="gpt2")(long_doc, max_tokens=256)
    _assert_valid(chunks)
    assert not any(c.token_count == 1 for c in chunks), "chonkie's count of 1 must not be recorded"
    assert all(c.token_count <= 256 for c in chunks), max(c.token_count for c in chunks)
