"""The knob summary prints a chance ceiling and a material count, and judges a ladder end to end.

A per-level tally on a ladder counts PARTNERS: a middle rung beats everything below it and loses
to everything above it, so it collects the most favours while saying nothing about direction.
These tests build a three-rung ladder where the tally favours the middle rung and require the
table to say "more" from the single highest-against-lowest pair instead.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "gen_bench_tables.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("gen_bench_tables_effect_floors", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator() -> Any:
    return _load()


def _effect(
    axis: str,
    low: Any,
    high: Any,
    delta: float,
    *,
    resolved: bool = True,
    corpus: str = "c",
    embedding: str = "e",
    held: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One paired comparison, stored HIGH to low as the exporter does in half its rows."""
    held_fixed = held if held is not None else {"strategy": "recursive", "max_tokens": 256}
    return {
        "axis": axis,
        "metric": "ndcg@10",
        "corpus": corpus,
        "embedding": embedding,
        "dim": 768,
        "held_fixed": held_fixed,
        "from_level": high,
        "to_level": low,
        "from_cell": f"{corpus}__{axis}-{high}__{embedding}",
        "to_cell": f"{corpus}__{axis}-{low}__{embedding}",
        "mean_delta": -delta,
        "ci_lo": -delta - (0.001 if resolved else 0.1),
        "ci_hi": -delta + (0.001 if resolved else 0.1),
        "wins": 5,
        "ties": 0,
        "losses": 3,
        "n_shared": 8,
        "resolved": resolved,
        "material": resolved and abs(delta) >= 0.005,
    }


def _doc(effects: list[dict[str, Any]]) -> dict[str, Any]:
    return {"alpha": 0.05, "material_floor": {"metric": "ndcg@10", "value": 0.005}, "effects": effects}


def _three_rung_ladder() -> list[dict[str, Any]]:
    """0 -> 51 gains 0.010, 51 -> 128 loses 0.002 (resolved, immaterial), 0 -> 128 gains 0.006.

    The tally over resolved rows favours 51 twice and nothing else; top against bottom says more.
    """
    return [
        _effect("overlap_tokens", 0, 51, 0.010),
        _effect("overlap_tokens", 51, 128, -0.002),
        _effect("overlap_tokens", 0, 128, 0.006),
    ]


def _summary_row(generator: Any, doc: dict[str, Any], axis: str) -> list[str]:
    table = generator._knob_summary_table(doc)
    (row,) = [r for r in table["rows"] if r[0] == f"`{axis}`"]
    return row


def test_a_numeric_axis_is_a_ladder_and_a_categorical_one_is_not(generator: Any) -> None:
    effects = [*_three_rung_ladder(), _effect("strategy", "fast", "recursive", 0.02)]
    assert generator._is_ladder_axis(effects, "overlap_tokens") is True
    assert generator._is_ladder_axis(effects, "strategy") is False


def test_a_ladder_is_judged_by_its_top_against_bottom_pair_not_by_partner_tallies(generator: Any) -> None:
    row = _summary_row(generator, _doc(_three_rung_ladder()), "overlap_tokens")
    direction = row[-1]
    assert direction.startswith("1 favour more, 0 favour less, 0 unresolved or immaterial")
    assert "favour 51" not in direction


def test_the_direction_column_names_the_ceiling_table_for_where_the_optimum_sits(generator: Any) -> None:
    table = generator._knob_summary_table(_doc(_three_rung_ladder()))
    assert "top against bottom" in table["note"]
    assert "ceiling" in table["note"].lower()


def test_the_chance_ceiling_is_comparisons_times_alpha_rounded_up(generator: Any) -> None:
    row = _summary_row(generator, _doc(_three_rung_ladder()), "overlap_tokens")
    assert row[1] == "3"
    assert row[3] == "1"  # ceil(3 * 0.05)
    many = [_effect("overlap_tokens", 0, 51, 0.01, corpus=f"c{i}") for i in range(1023)]
    assert _summary_row(generator, _doc(many), "overlap_tokens")[3] == "52"  # ceil(51.15)


def test_the_material_count_excludes_a_resolved_step_under_the_floor(generator: Any) -> None:
    row = _summary_row(generator, _doc(_three_rung_ladder()), "overlap_tokens")
    assert row[2].startswith("3 ")  # three resolved
    assert row[4] == "2"  # two material


def test_a_categorical_axis_tallies_material_resolutions_only(generator: Any) -> None:
    effects = [
        _effect("strategy", "fast", "recursive", 0.020),
        _effect("strategy", "fast", "semantic", 0.002),  # resolved, immaterial: not tallied
        _effect("strategy", "late", "recursive", 0.030, resolved=False),
    ]
    row = _summary_row(generator, _doc(effects), "strategy")
    assert row[-1] == "1 favour recursive"


def test_the_table_reads_alpha_and_the_floor_from_the_file_not_from_constants(generator: Any) -> None:
    doc = _doc(_three_rung_ladder())
    doc["alpha"] = 0.10
    doc["material_floor"]["value"] = 0.02
    table = generator._knob_summary_table(doc)
    (row,) = [r for r in table["rows"] if r[0] == "`overlap_tokens`"]
    assert row[3] == "1"  # ceil(3 * 0.10) is still 1, but the note must carry the file's values
    assert "0.02" in table["note"] and "0.10" in table["note"]


def test_a_ladder_grouping_keys_by_corpus_held_fixed_and_embedder(generator: Any) -> None:
    effects = [*_three_rung_ladder(), _effect("overlap_tokens", 0, 51, 0.01, embedding="f")]
    ladders = generator._ladders(_doc(effects), "overlap_tokens", None)
    assert sorted(key[2] for key in ladders) == ["e", "f"]
    verdicts = generator._ladder_verdicts(ladders)
    assert verdicts == {"more": 2, "less": 0, "undecided": 0, "unspanned": 0}


def test_the_ceiling_table_reports_material_steps_and_tags_an_immaterial_last_step(generator: Any) -> None:
    doc = _doc(_three_rung_ladder())
    table = generator._rung_ceiling_table(doc, None)
    (row,) = table["rows"]
    steps_col = table["columns"].index("Steps resolved (material)")
    assert row[steps_col] == "2/2 (1 material)"
    assert row[table["columns"].index("Highest material step")] == "0 to 51"
    assert row[-1] == "-0.0020 resolved, immaterial"


def test_a_ladder_with_float_levels_orients_and_resolves_not_unspanned(generator: Any) -> None:
    """2.0 and 10.0 are floats, stored HIGH to low as the exporter does in half its rows.

    Int-only ordering falls back to lexical string comparison for a float pair: "10.0" sorts
    BEFORE "2.0" character by character, so the old code reads the pair as already ordered and
    never flips it. The ladder is then keyed (10.0, 2.0) while ``_ladder_verdicts`` looks up
    (2.0, 10.0), misses it, and the ladder is wrongly counted unspanned.
    """
    effects = [_effect("overlap_tokens", 2.0, 10.0, 0.010)]
    ladders = generator._ladders(_doc(effects), "overlap_tokens", None)
    verdicts = generator._ladder_verdicts(ladders)
    assert verdicts == {"more": 1, "less": 0, "undecided": 0, "unspanned": 0}


def test_a_bool_levelled_axis_is_not_a_ladder(generator: Any) -> None:
    """``bool`` is an ``int`` to ``isinstance``, but never a rung; ``_is_rung`` must exclude it."""
    effects = [_effect("some_flag", False, True, 0.01)]
    assert generator._is_ladder_axis(effects, "some_flag") is False


def test_a_ladder_missing_its_end_to_end_pair_is_unspanned_and_named_in_the_direction_cell(generator: Any) -> None:
    """Only the two consecutive steps are measured; the top-against-bottom pair never was."""
    effects = [
        _effect("overlap_tokens", 0, 51, 0.010),
        _effect("overlap_tokens", 51, 128, 0.010),
    ]
    ladders = generator._ladders(_doc(effects), "overlap_tokens", None)
    verdicts = generator._ladder_verdicts(ladders)
    assert verdicts == {"more": 0, "less": 0, "undecided": 0, "unspanned": 1}
    row = _summary_row(generator, _doc(effects), "overlap_tokens")
    assert "without an end-to-end pair" in row[-1]
