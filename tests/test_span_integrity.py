"""Offset recovery and span containment for scripts/score_span_integrity.py.

Every number that measurement produces rests on two pure functions: locating a chunk's characters
in its source, and deciding whether an answer span survives inside one chunk. Both are easy to get
subtly wrong in ways that produce a plausible rate rather than an error - an off-by-one at the
boundary, or a cursor that cannot cope with overlapping chunks - so they are pinned here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_span_integrity.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("score_span_integrity", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before exec: `@dataclass(slots=True)` rebuilds the class and looks its module up in
    # sys.modules to do it, which fails with an opaque NoneType error when loading by path.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def span() -> Any:
    return _load()


# --- locating chunks in their source ----------------------------------------------------------


def test_consecutive_chunks_are_located_in_order(span: Any) -> None:
    context = "alpha beta gamma delta"

    assert span.chunk_offsets(context, ["alpha beta", " gamma delta"]) == [(0, 10), (10, 22)]


def test_overlapping_chunks_are_each_located_at_their_own_start(span: Any) -> None:
    """With overlap the next chunk begins BEFORE the previous one ends."""
    context = "one two three four five"

    offsets = span.chunk_offsets(context, ["one two three", "two three four", "three four five"])

    assert offsets == [(0, 13), (4, 18), (8, 23)]


def test_overlapping_chunks_are_located_correctly_when_a_chunk_text_repeats(span: Any) -> None:
    """The case where a cursor past the previous chunk's END mislocates rather than just failing.

    An end-anchored cursor cannot find an overlapping chunk, so it falls back to searching the
    whole string - which is usually right by luck, because chunk texts are usually unique. It is
    wrong exactly when a chunk's full text occurs earlier too: the final chunk here repeats the
    first, and the fallback would place it at offset 0, marking every answer in the document's
    tail as split. Overlap plus repetition is the combination that has to be pinned.
    """
    context = "alpha beta gamma delta alpha beta gamma"

    offsets = span.chunk_offsets(
        context,
        ["alpha beta gamma", "beta gamma delta", "delta alpha beta", "alpha beta gamma"],
    )

    assert offsets == [(0, 16), (6, 22), (17, 33), (23, 39)]
    assert context[23:39] == "alpha beta gamma"  # the repeat really is at the tail, not at 0


def test_a_repeated_chunk_text_is_located_at_successive_positions(span: Any) -> None:
    """Identical consecutive chunks must map to successive ranges, not all to the first one.

    Collapsing them onto offset 0 would place two thirds of the document's chunks at the start,
    so every answer past that point would read as split and integrity would be understated for
    reasons that have nothing to do with chunking.
    """
    context = "ab ab ab"

    assert span.chunk_offsets(context, ["ab", "ab", "ab"]) == [(0, 2), (3, 5), (6, 8)]


def test_a_chunk_that_is_not_a_substring_is_dropped_rather_than_guessed(span: Any) -> None:
    """A non-verbatim chunk has no honest offset; inventing one would fake containment."""
    context = "alpha beta"

    offsets = span.chunk_offsets(context, ["alpha", "NOT PRESENT", "beta"])

    assert offsets == [(0, 5), (6, 10)]


def test_a_chunk_that_rejoined_whitespace_is_still_located(span: Any) -> None:
    """The whitespace chunker returns text rejoined on single spaces, not source bytes.

    A plain substring search loses every one of its chunks - 3739 of them in the first run of
    this grid - and a lost chunk reads exactly like a chunk that failed to cover the answer, so
    the strategy would have been reported as splitting answers it never split.
    """
    context = "Aufzugsanlage\n\n=== Seilloser Aufzug ===\nAn der RWTH Aachen"
    rejoined = "Aufzugsanlage === Seilloser Aufzug === An der RWTH Aachen"

    assert span.chunk_offsets(context, [rejoined]) == [(0, len(context))]


def test_a_verbatim_chunk_containing_a_line_break_is_still_located(span: Any) -> None:
    """The context is normalized for matching, so the chunk must be normalized the same way.

    Every strategy except whitespace returns source text VERBATIM, newlines included. Normalizing
    only one side would fail to find any of them - the common case, not the exotic one - and each
    loss reads as a chunk that failed to cover its answer.
    """
    context = "erste Zeile\n\nzweite Zeile"

    assert span.chunk_offsets(context, ["erste Zeile\n\nzweite Zeile"]) == [(0, len(context))]


def test_a_chunk_damaged_by_a_token_boundary_cut_is_still_located(span: Any) -> None:
    """chonkie's overlap cuts at a TOKEN boundary, which can land mid-character.

    The chunk then ends in U+FFFD, which is not in the source, so an exact search drops it. Only
    overlap>0 cells were affected, meaning the measurement deleted exactly the chunks overlap
    adds and then judged whether overlap helped.
    """
    context = "das Fenster war geschlossen"
    damaged = "das Fenster war geschl\ufffd"

    offsets = span.chunk_offsets(context, [damaged])

    assert len(offsets) == 1
    assert offsets[0][0] == 0
    assert offsets[0][1] == len("das Fenster war geschl")


def test_normalisation_keeps_offsets_in_the_source_coordinate_system(span: Any) -> None:
    """Answer spans are indices into the ORIGINAL text, so a normalized offset would be wrong.

    Collapsing two newlines to one space shifts every later position; returning the normalized
    index would misalign every span in the document by that drift.
    """
    context = "alpha\n\n\nbeta gamma"

    offsets = span.chunk_offsets(context, ["beta gamma"])

    assert offsets == [(8, 18)]
    assert context[8:18] == "beta gamma"


# --- does the answer survive inside one chunk -------------------------------------------------


def test_a_span_wholly_inside_one_chunk_is_intact(span: Any) -> None:
    assert span.span_is_intact([(0, 10), (10, 20)], 2, 8)


def test_a_span_crossing_a_boundary_is_not_intact(span: Any) -> None:
    """The whole point: two chunks each holding half an answer is a failure, not a success."""
    assert not span.span_is_intact([(0, 10), (10, 20)], 8, 12)


def test_a_span_is_intact_when_an_overlapping_chunk_covers_it(span: Any) -> None:
    """This is the mechanism overlap is for, so it must be scored as the save it is.

    The span 8..12 is cut by the first pair's boundary at 10, and the overlapping middle chunk
    covers it whole. Judging only the chunk that starts nearest the span would miss that.
    """
    assert span.span_is_intact([(0, 10), (5, 15), (10, 20)], 8, 12)


def test_a_long_earlier_chunk_can_be_the_one_that_covers_the_span(span: Any) -> None:
    """The covering chunk is not always the one starting nearest the span.

    Chunk sizes vary, so a long chunk can still be open across several shorter ones that start
    later. Checking only the nearest-starting chunk would call this split while a retrieved chunk
    holds the answer whole, understating integrity wherever chunk lengths are uneven.
    """
    assert span.span_is_intact([(0, 30), (5, 12), (10, 20)], 8, 25)


def test_a_span_exactly_filling_a_chunk_is_intact(span: Any) -> None:
    """Inclusive at both ends: an answer that is exactly one chunk is not a boundary failure."""
    assert span.span_is_intact([(0, 10), (10, 20)], 10, 20)


def test_a_span_one_character_past_the_chunk_end_is_not_intact(span: Any) -> None:
    """Guards the off-by-one that would report every truncated answer as surviving."""
    assert not span.span_is_intact([(0, 10)], 5, 11)


def test_a_span_starting_one_character_before_the_chunk_is_not_intact(span: Any) -> None:
    assert not span.span_is_intact([(5, 15)], 4, 10)


def test_no_chunks_means_no_span_survives(span: Any) -> None:
    assert not span.span_is_intact([], 0, 1)


def test_a_later_chunk_does_not_rescue_a_span_that_starts_before_it(span: Any) -> None:
    """Containment needs one chunk covering the WHOLE span, never two chunks between them."""
    assert not span.span_is_intact([(0, 6), (6, 12)], 4, 9)


# --- the profile grid --------------------------------------------------------------------------


def test_overlap_levels_are_skipped_for_strategies_that_ignore_the_knob(span: Any) -> None:
    """chonkie applies overlap to recursive only; semantic and late silently discard it.

    Emitting those rows would publish a flat line across an axis that was never applied and read
    as "overlap does not help here" rather than "overlap never happened".
    """
    grid = list(span.profiles(["recursive", "semantic", "late", "markdown"], [256], [0, 64]))

    with_overlap = {row["strategy"] for row in grid if row["overlap_tokens"]}
    assert with_overlap == {"recursive", "markdown"}
    assert {row["strategy"] for row in grid if not row["overlap_tokens"]} == {
        "recursive",
        "semantic",
        "late",
        "markdown",
    }


def test_an_answer_offset_that_does_not_reproduce_its_text_is_rejected(span: Any) -> None:
    """Several sets carry variants with a leading space; scoring those measures the wrong chars."""
    context = "the quick brown fox"

    assert span._first_verifiable_span(context, {"text": ["quick"], "answer_start": [4]}) == (4, "quick")
    assert span._first_verifiable_span(context, {"text": ["quick"], "answer_start": [5]}) is None


def test_a_later_answer_variant_is_used_when_the_first_offset_is_wrong(span: Any) -> None:
    """Dropping the question because variant 0 is off by one would discard usable data."""
    context = "the quick brown fox"

    found = span._first_verifiable_span(context, {"text": [" quick", "quick"], "answer_start": [4, 4]})

    assert found == (4, "quick")


def test_an_overlap_at_or_above_the_chunk_size_is_not_a_configuration(span: Any) -> None:
    """semantic-text-splitter raises on it; chonkie accepts it and produces something unusable.

    It is skipped for every strategy rather than only the ones that raise, so a strategy
    comparison never compares grids of different shapes.
    """
    grid = list(span.profiles(["recursive", "markdown"], [64, 128], [0, 32, 64]))

    assert {(row["max_tokens"], row["overlap_tokens"]) for row in grid} == {
        (64, 0),
        (64, 32),
        (128, 0),
        (128, 32),
        (128, 64),
    }


# --- extending a long grid without re-measuring it ---------------------------------------------


def test_merging_a_run_replaces_a_row_rather_than_duplicating_it(span: Any) -> None:
    """A re-measured profile must update in place; two rows for one cell would be averaged."""
    old = {
        "integrity": [
            {"corpus": "x", "strategy": "recursive", "max_tokens": 256, "overlap_tokens": 0, "span_intact": 0.5}
        ]
    }
    new = {
        "integrity": [
            {"corpus": "x", "strategy": "recursive", "max_tokens": 256, "overlap_tokens": 0, "span_intact": 0.9}
        ]
    }

    merged = span.merge_payload(old, new)

    assert len(merged["integrity"]) == 1
    assert merged["integrity"][0]["span_intact"] == 0.9


def test_merging_keeps_rows_the_new_run_did_not_measure(span: Any) -> None:
    """The point of merging: add an overlap level without re-running the slow strategies."""
    old = {
        "integrity": [{"corpus": "x", "strategy": "late", "max_tokens": 256, "overlap_tokens": 0, "span_intact": 0.5}]
    }
    new = {
        "integrity": [
            {"corpus": "x", "strategy": "recursive", "max_tokens": 256, "overlap_tokens": 32, "span_intact": 0.9}
        ]
    }

    merged = span.merge_payload(old, new)

    assert {(r["strategy"], r["overlap_tokens"]) for r in merged["integrity"]} == {("late", 0), ("recursive", 32)}


def test_rows_of_the_same_profile_at_different_overlaps_are_distinct(span: Any) -> None:
    """overlap_tokens is part of a row's identity, or an overlap sweep would collapse to one row."""
    rows = [
        {"corpus": "x", "strategy": "recursive", "max_tokens": 256, "overlap_tokens": overlap}
        for overlap in (0, 10, 15, 32, 64)
    ]

    assert len({span.row_key(row) for row in rows}) == 5


def test_phase_b_rows_of_different_k_are_distinct(span: Any) -> None:
    """k is part of the retrieval row identity; without it two depths would overwrite each other."""
    rows = [{"corpus": "x", "strategy": "recursive", "max_tokens": 256, "overlap_tokens": 0, "k": k} for k in (1, 5)]

    assert len({span.row_key(row) for row in rows}) == 2


# --- per-question outcomes and paired effects --------------------------------------------------


def test_the_bitmap_round_trips_for_awkward_lengths(span: Any) -> None:
    """Question counts are not multiples of 8, so the tail byte carries padding bits.

    Unpacking the padding as real questions would append phantom failures to every cell.
    """
    for count in (0, 1, 7, 8, 9, 2204):
        flags = [float(i % 3 == 0) for i in range(count)]
        assert span.unpack_bits(span.pack_bits(flags), count) == flags


def test_only_cells_differing_on_exactly_one_axis_are_compared(span: Any) -> None:
    """Two knobs moving at once cannot attribute the difference to either."""
    rows = [
        _cell(span, max_tokens=128, overlap_tokens=0),
        _cell(span, max_tokens=256, overlap_tokens=0),
        _cell(span, max_tokens=256, overlap_tokens=32),
        _cell(span, max_tokens=128, overlap_tokens=32),
    ]

    effects = span.axis_effects(rows)

    # Four cells on a 2x2 grid give four one-axis-apart pairs and no diagonals. Identifying them
    # by axis and level alone would collapse the two max_tokens pairs onto each other, so the
    # held-fixed value is part of the identity - that is what proves the diagonals were skipped.
    assert {(e["axis"], e["from_level"], e["to_level"], tuple(sorted(e["held_fixed"].items()))) for e in effects} == {
        ("max_tokens", 128, 256, (("overlap_tokens", 0), ("strategy", "recursive"))),
        ("max_tokens", 128, 256, (("overlap_tokens", 32), ("strategy", "recursive"))),
        ("overlap_tokens", 0, 32, (("max_tokens", 128), ("strategy", "recursive"))),
        ("overlap_tokens", 0, 32, (("max_tokens", 256), ("strategy", "recursive"))),
    }
    assert len(effects) == 4, "a diagonal pair moves two knobs at once and must not be emitted"


def test_a_pair_is_emitted_once_in_a_stable_direction(span: Any) -> None:
    """Both orderings would publish the same finding twice, with opposite signs."""
    rows = [_cell(span, max_tokens=128), _cell(span, max_tokens=256)]

    effects = span.axis_effects(rows)

    assert len(effects) == 1
    assert (effects[0]["from_level"], effects[0]["to_level"]) == (128, 256)


def test_an_unfit_cell_is_never_paired(span: Any) -> None:
    """A context that fits in one chunk has no boundary, so its 1.0 is not a knob effect.

    Pairing it against a fit cell would credit the knob with the difference between measuring
    something and measuring nothing.
    """
    rows = [_cell(span, max_tokens=128), _cell(span, max_tokens=512, fit=False)]

    assert span.axis_effects(rows) == []


def test_the_effect_sign_says_the_higher_level_protects_more(span: Any) -> None:
    """A positive delta must mean moving UP the axis kept more answers intact.

    A flipped sign would read as "bigger chunks split more answers", the exact opposite of the
    truth, and nothing else in the row would contradict it.
    """
    low = _cell(span, max_tokens=128, flags=[0.0] * 100)
    high = _cell(span, max_tokens=256, flags=[1.0] * 100)

    effect = span.axis_effects([low, high])[0]

    assert effect["from_level"] == 128
    assert effect["mean_delta"] == pytest.approx(1.0)


def _cell(
    span: Any,
    *,
    strategy: str = "recursive",
    max_tokens: int = 256,
    overlap_tokens: int = 0,
    fit: bool = True,
    flags: list[float] | None = None,
) -> dict[str, Any]:
    values = flags if flags is not None else [float(i % 2) for i in range(100)]
    return {
        "corpus": "x",
        "strategy": strategy,
        "max_tokens": max_tokens,
        "overlap_tokens": overlap_tokens,
        "fit_for_claim": fit,
        "n_questions": len(values),
        "intact_bits": span.pack_bits(values),
    }
