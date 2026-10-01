"""Unit tests for the lossless chunk size-guard (splits oversized chunks, never truncates)."""

from __future__ import annotations

import pytest

from semdex.adapters.chunker._base import split_oversized

pytestmark = pytest.mark.os_agnostic


def _word_offsets(text: str) -> list[tuple[int, int]]:
    """Fake tokenizer: each whitespace word is one token, with its char span in `text`."""
    offsets: list[tuple[int, int]] = []
    i = 0
    for word in text.split(" "):
        start = text.index(word, i) if word else i
        offsets.append((start, start + len(word)))
        i = start + len(word)
    return offsets


def test_within_cap_returns_unchanged() -> None:
    out = split_oversized("a b c", 3, max_tokens=5, offsets_of=_word_offsets)
    assert out == [("a b c", 3)]


def test_oversized_splits_into_capped_pieces() -> None:
    text = " ".join(f"w{i}" for i in range(10))  # 10 word-tokens
    out = split_oversized(text, 10, max_tokens=3, offsets_of=_word_offsets)
    # every piece is within the cap
    assert all(count <= 3 for _text, count in out)
    # LOSSLESS: the pieces tile the original text exactly
    assert "".join(t for t, _c in out) == text
    # sizes: 3,3,3,1
    assert [c for _t, c in out] == [3, 3, 3, 1]


def test_lossless_on_a_no_whitespace_blob() -> None:
    # a token that itself is huge (like a minified-JS blob) still tiles losslessly by offsets
    text = "AAAA BBBB CCCC DDDD EEEE"
    out = split_oversized(text, 5, max_tokens=2, offsets_of=_word_offsets)
    assert "".join(t for t, _c in out) == text
    assert all(c <= 2 for _t, c in out)
    assert len(out) == 3  # 2,2,1


class _GiantChunk:
    def __init__(self, text: str, token_count: int) -> None:
        self.text = text
        self.token_count = token_count


class _FakeSemanticChunker:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def __call__(self, text: str) -> list[_GiantChunk]:
        return [_GiantChunk(text, len(text.split(" ")))]  # the whole doc as ONE oversized chunk


def _fake_chonkie(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    fake = types.ModuleType("chonkie")
    fake.SemanticChunker = _FakeSemanticChunker  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "chonkie", fake)


def _doc(text: str):
    from semdex.domain.models import ExtractedDocument, SourceRef

    return ExtractedDocument(source=SourceRef(uri="/x", label="", content_hash="", mtime=0.0), text=text)


def test_guard_splits_oversized_semantic_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_chonkie(monkeypatch)
    from semdex.adapters.chunker import ChonkieChunker
    from semdex.domain.enums import ChunkStrategy

    text = " ".join(f"w{i}" for i in range(10))
    chunks = ChonkieChunker(ChunkStrategy.SEMANTIC, tokenizer="word")(_doc(text), max_tokens=3)
    assert all(c.token_count <= 3 for c in chunks)  # cap respected
    assert "".join(c.text for c in chunks) == text  # LOSSLESS
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))  # continuous ordinals


def test_guard_off_keeps_the_giant(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_chonkie(monkeypatch)
    from semdex.adapters.chunker import ChonkieChunker
    from semdex.domain.enums import ChunkStrategy

    text = " ".join(f"w{i}" for i in range(10))
    chunks = ChonkieChunker(ChunkStrategy.SEMANTIC, tokenizer="word", enforce_max_tokens=False)(
        _doc(text), max_tokens=3
    )
    assert len(chunks) == 1 and chunks[0].token_count == 10  # unguarded -> one oversized chunk


# A chunk's recorded token_count and the tokenizer used to split it need not agree. For the
# semantic and late strategies they systematically do not: chonkie's SemanticChunker accepts no
# tokenizer argument and counts in its embedding model's tokenizer, while the offsets come from
# [chunker].tokenizer. Measured divergence on the cached corpora is 6 to 22 percent over
# byte-identical text, which is enough to put a chunk on the wrong side of the cap.


def test_a_chunk_over_cap_by_its_recorded_count_is_split_even_when_our_tokenizer_disagrees() -> None:
    """The case that shipped oversized chunks: recorded 782 tokens under a 256 cap, skipped.

    The guard returned such a chunk unchanged because OUR tokenizer counted it as within cap, so
    the embedder then truncated it and dropped the overflow with no error. It must split instead:
    the recorded count is the one the embedder will act on.
    """
    text = " ".join(f"w{i}" for i in range(100))  # 100 tokens by _word_offsets

    out = split_oversized(text, 782, max_tokens=256, offsets_of=_word_offsets)

    assert len(out) > 1, "a chunk recorded as 3x its cap must be split, whatever our tokenizer says"
    assert "".join(piece for piece, _ in out) == text, "the split must stay lossless"
    # 782 recorded tokens against a 256 cap needs at least ceil(782/256) = 4 pieces for each piece
    # to land under the cap in the recorded tokenizer's terms.
    assert len(out) >= 4


def test_the_pieces_are_balanced_when_the_tokenizers_disagree() -> None:
    """Splitting into ceil(recorded/cap) even pieces, not one full window plus a remainder."""
    text = " ".join(f"w{i}" for i in range(100))

    out = split_oversized(text, 782, max_tokens=256, offsets_of=_word_offsets)

    sizes = [count for _, count in out]
    # Guard the guard: with a single piece "balanced" is trivially true, so this assertion would
    # pass while measuring nothing.
    assert len(sizes) > 1, "precondition: the chunk must actually have been split"
    assert max(sizes) - min(sizes) <= 1, f"pieces should be even, got {sizes}"


def test_an_agreeing_oversized_chunk_still_splits_at_the_cap() -> None:
    """The existing behaviour must not change when the two counts agree.

    Without this the fix would silently re-window every already-correct split.
    """
    text = " ".join(f"w{i}" for i in range(10))

    out = split_oversized(text, 10, max_tokens=3, offsets_of=_word_offsets)

    assert [count for _, count in out] == [3, 3, 3, 1]


def test_a_genuinely_small_chunk_is_never_split() -> None:
    """Both counts under the cap: nothing to do, and no accidental splitting."""
    text = " ".join(f"w{i}" for i in range(5))

    out = split_oversized(text, 5, max_tokens=256, offsets_of=_word_offsets)

    assert out == [(text, 5)]


# chonkie's LateChunker reports token_count 1 for every chunk of a long document (reproduced on
# 30,000-character German court decisions: 60 of 60 chunks, and 81 percent of the 179,626 chunks
# of a cached German set), while the same chunker counts a short document correctly. A count of 1
# slips under any cap, so the size-guard never split those chunks and the embedder clipped them.
# The adapter therefore never trusts chonkie's count for late chunks; it recounts every one in
# the configured tokenizer, which is also the unit recursive chunks are counted in.


class _MiscountedLateChunk:
    def __init__(self, text: str) -> None:
        self.text = text
        self.token_count = 1  # what chonkie reports on a long document


class _FakeLateChunker:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def __call__(self, text: str) -> list[_MiscountedLateChunk]:
        # Two chunks, one within a 3-token cap and one over it, both reported as 1 token. They
        # tile the text exactly (the guard's losslessness check joins them back).
        cut = text.index(" ", text.index(" ") + 1) + 1
        return [_MiscountedLateChunk(text[:cut]), _MiscountedLateChunk(text[cut:])]


def _fake_chonkie_late(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    fake = types.ModuleType("chonkie")
    fake.LateChunker = _FakeLateChunker  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "chonkie", fake)


def test_a_late_chunk_is_recounted_in_the_configured_tokenizer_not_taken_from_chonkie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_chonkie_late(monkeypatch)
    from semdex.adapters.chunker import ChonkieChunker
    from semdex.domain.enums import ChunkStrategy

    text = " ".join(f"w{i}" for i in range(8))
    chunks = ChonkieChunker(ChunkStrategy.LATE, tokenizer="word", enforce_max_tokens=False)(_doc(text), max_tokens=3)
    # "w0 w1 " and "w2 w3 w4 w5 w6 w7": chonkie said 1 and 1; the "word" tokenizer counts the
    # split words, and the trailing separator of the first piece is a word boundary, not a word.
    assert [len(c.text.split()) for c in chunks] == [2, 6]
    assert [c.token_count for c in chunks] == [len(c.text.split(" ")) for c in chunks]
    assert all(c.token_count > 1 for c in chunks), "chonkie's count of 1 must not survive"


def test_a_late_chunk_chonkie_miscounts_is_still_split_by_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_chonkie_late(monkeypatch)
    from semdex.adapters.chunker import ChonkieChunker
    from semdex.domain.enums import ChunkStrategy

    text = " ".join(f"w{i}" for i in range(8))
    chunks = ChonkieChunker(ChunkStrategy.LATE, tokenizer="word")(_doc(text), max_tokens=3)
    # Judged on the TEXT, not on the recorded count: a count of 1 satisfies any cap by itself.
    assert all(len(c.text.split()) <= 3 for c in chunks), [c.text for c in chunks]
    assert all(0 < c.token_count <= 3 for c in chunks), [c.token_count for c in chunks]
    assert "".join(c.text for c in chunks) == text  # lossless
