"""Unit test for the surgical-repair planner (re-chunk only giants; copy the rest)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from repair_oversized_chunks import (
    backup_path,
    cap_for,
    cells_done_at,
    plan_repair,
    record_cell_done,
    refresh_meta_count,
)

pytestmark = pytest.mark.os_agnostic


def _word_offsets(text: str) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for word in text.split(" "):
        start = text.index(word, cursor) if word else cursor
        offsets.append((start, start + len(word)))
        cursor = start + len(word)
    return offsets


def test_plan_keeps_small_chunks_and_splits_the_giant() -> None:
    giant = " ".join(f"w{i}" for i in range(10))
    chunks = [("small a", 2), (giant, 10), ("small b", 2)]
    new_chunks, carry = plan_repair(chunks, max_tokens=3, offsets_of=_word_offsets)

    # the two small chunks are carried (copy their old vector); the giant became several NEW pieces
    assert carry[0] == 0  # old index of "small a"
    assert carry[-1] == 2  # old index of "small b"
    new_from_giant = [i for i, c in enumerate(carry) if c == -1]
    assert len(new_from_giant) >= 3  # 10 tokens / 3 -> >=4 pieces, all new (-1)
    # every produced chunk is within the cap
    assert all(count <= 3 for _text, count in new_chunks)
    # lossless: the giant's pieces concatenate back to the giant
    assert "".join(new_chunks[i][0] for i in new_from_giant) == giant


def test_plan_is_a_noop_when_nothing_is_oversized() -> None:
    chunks = [("a", 1), ("b", 1)]
    new_chunks, carry = plan_repair(chunks, max_tokens=8, offsets_of=_word_offsets)
    assert new_chunks == chunks
    assert carry == [0, 1]  # everything carried, nothing re-embedded


# The cap the repair holds a set to. Taking one cap from the environment and applying it to every
# set is what left oversized chunks behind: a set built at max_tokens 256 was repaired against a
# flat 512 and is then still over ITS cap, by construction.


def _chunk_set(tmp_path: Path, name: str, meta: dict[str, object]) -> Path:
    import json

    directory = tmp_path / name
    directory.mkdir()
    (directory / "meta.json").write_text(json.dumps(meta))
    return directory


def test_the_cap_comes_from_the_set_not_from_one_global_number(tmp_path: Path) -> None:
    small = _chunk_set(tmp_path, "corpus__recursive-t256-o0-gpt2", {"max_tokens": 256, "overlap": 0})
    large = _chunk_set(tmp_path, "corpus__recursive-t512-o0-gpt2", {"max_tokens": 512, "overlap": 0})

    assert cap_for(small, None) == 256
    assert cap_for(large, None) == 512


def test_overlap_raises_the_cap_because_it_deliberately_overshoots(tmp_path: Path) -> None:
    """Overlap is prepended context, so the true ceiling is max_tokens + overlap.

    Without this the repair would re-split every overlap chunk, which the size-guard explicitly
    declines to do.
    """
    directory = _chunk_set(tmp_path, "corpus__recursive-t256-o15-gpt2", {"max_tokens": 256, "overlap": 15})

    assert cap_for(directory, None) == 271


def test_a_missing_overlap_key_is_treated_as_no_overlap(tmp_path: Path) -> None:
    directory = _chunk_set(tmp_path, "corpus__semantic-t256-o0-gpt2", {"max_tokens": 256})

    assert cap_for(directory, None) == 256


def test_an_explicit_override_still_wins(tmp_path: Path) -> None:
    """The flat cap remains available, but only when asked for by name."""
    directory = _chunk_set(tmp_path, "corpus__recursive-t256-o0-gpt2", {"max_tokens": 256, "overlap": 0})

    assert cap_for(directory, 512) == 512


# Re-running the repair at a tighter cap is a normal operation, and both of these guard it. The
# data it rewrites is a cache that cost weeks of GPU time and has no other copy.


def test_a_backup_never_overwrites_an_earlier_one(tmp_path: Path) -> None:
    """Path.rename overwrites silently, so a second pass would destroy the original backup."""
    target = tmp_path / "chunks.parquet"
    target.write_text("current")

    first = backup_path(target)
    assert first.name == "chunks.parquet.pre-repair"
    first.write_text("original")

    second = backup_path(target)
    assert second != first
    assert not second.exists()
    second.write_text("after first repair")

    assert backup_path(target).name == "chunks.parquet.pre-repair-3"
    assert first.read_text() == "original", "the original backup must survive every later pass"


def test_a_cell_repaired_at_another_cap_is_not_treated_as_done(tmp_path: Path) -> None:
    """The bug this replaces: resume keyed on 'a backup exists', true for ANY earlier pass.

    A second pass at a tighter cap then skipped re-embedding while still rewriting the chunk
    table, leaving the vectors at the old row count and the set permanently misaligned.
    """
    chunk_dir = tmp_path / "corpus__semantic-t256-o0-gpt2"
    chunk_dir.mkdir()
    record_cell_done(chunk_dir, "corpus__semantic-t256-o0-gpt2__fastembed-bge-base", 512)

    assert cells_done_at(chunk_dir, 512) == {"corpus__semantic-t256-o0-gpt2__fastembed-bge-base"}
    assert cells_done_at(chunk_dir, 256) == set(), "a different cap means the cell must be rebuilt"


def test_resume_at_the_same_cap_still_skips_finished_cells(tmp_path: Path) -> None:
    """A genuine crash-resume must not redo work it already paid for."""
    chunk_dir = tmp_path / "corpus__semantic-t256-o0-gpt2"
    chunk_dir.mkdir()
    record_cell_done(chunk_dir, "cell_a", 256)
    record_cell_done(chunk_dir, "cell_b", 256)

    assert cells_done_at(chunk_dir, 256) == {"cell_a", "cell_b"}


def test_no_marker_means_nothing_is_done(tmp_path: Path) -> None:
    chunk_dir = tmp_path / "corpus__semantic-t256-o0-gpt2"
    chunk_dir.mkdir()

    assert cells_done_at(chunk_dir, 256) == set()


def test_the_repair_brings_the_recorded_count_back_in_line(tmp_path: Path) -> None:
    """A set's meta.json count must match its parquet after the repair changes the row count.

    Left stale, anything that reads the count instead of the data disagrees with it by exactly
    the number of pieces the repair created.
    """
    import json

    chunk_dir = tmp_path / "corpus__semantic-t256-o0-gpt2"
    chunk_dir.mkdir()
    (chunk_dir / "meta.json").write_text(json.dumps({"count": 100, "max_tokens": 256, "overlap": 0}))

    refresh_meta_count(chunk_dir, 137)

    meta = json.loads((chunk_dir / "meta.json").read_text())
    assert meta["count"] == 137
    assert meta["max_tokens"] == 256, "the rest of the metadata must survive untouched"


def test_refreshing_a_count_that_is_already_right_changes_nothing(tmp_path: Path) -> None:
    import json

    chunk_dir = tmp_path / "corpus__semantic-t256-o0-gpt2"
    chunk_dir.mkdir()
    (chunk_dir / "meta.json").write_text(json.dumps({"count": 42, "max_tokens": 256}))
    before = (chunk_dir / "meta.json").read_text()

    refresh_meta_count(chunk_dir, 42)

    assert (chunk_dir / "meta.json").read_text() == before


def test_a_set_with_no_meta_is_left_alone(tmp_path: Path) -> None:
    chunk_dir = tmp_path / "corpus__semantic-t256-o0-gpt2"
    chunk_dir.mkdir()

    refresh_meta_count(chunk_dir, 10)  # must not raise

    assert not (chunk_dir / "meta.json").exists()
