"""Turning an unresolved comparison into a query count.

The arithmetic is one line and every way of getting it wrong returns a believable number. Two of
those wrong answers would change what somebody does: dropping the square understates the cost of
more data fourfold at a factor of two, and returning a finite figure for a zero effect invites
collecting a corpus that cannot settle anything.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import pytest

_MODULE = Path(__file__).resolve().parents[1] / "scripts" / "_query_power.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("_query_power", _MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def power() -> Any:
    return _load()


# --- the query count ------------------------------------------------------------------------------


def test_halving_the_interval_costs_four_times_the_queries(power: Any) -> None:
    """The square is the whole point.

    A bootstrap half-width falls as 1/sqrt(n), so closing a 2x gap needs 4x the data. Dropping the
    square would report a quarter of the real requirement and make an impossible corpus look
    routine.
    """
    needed = power.queries_to_resolve(half_width=0.04, mean_delta=0.02, queries=200)

    assert needed == pytest.approx(800.0)


def test_an_effect_already_clear_of_the_interval_needs_no_more_queries(power: Any) -> None:
    """Below the current n: the comparison already resolves and the figure says so."""
    needed = power.queries_to_resolve(half_width=0.01, mean_delta=0.02, queries=200)

    assert needed == pytest.approx(50.0)


def test_a_ten_fold_gap_needs_a_hundred_fold_corpus(power: Any) -> None:
    needed = power.queries_to_resolve(half_width=0.05, mean_delta=0.005, queries=200)

    assert needed == pytest.approx(20_000.0)


def test_a_zero_effect_is_unresolvable_rather_than_merely_expensive(power: Any) -> None:
    """The distinction the word "underpowered" hides.

    No query count separates two configurations that are the same. Returning an enormous finite
    number would read as a target and send somebody after a corpus that settles nothing.
    """
    assert power.queries_to_resolve(half_width=0.03, mean_delta=0.0, queries=200) == math.inf


def test_a_negligible_effect_is_treated_as_zero(power: Any) -> None:
    """1e-5 nDCG is not a finding anyone acts on, even where a corpus could prove it."""
    assert power.queries_to_resolve(half_width=0.03, mean_delta=1e-5, queries=200) == math.inf


def test_the_sign_of_the_effect_does_not_matter(power: Any) -> None:
    """A configuration being worse needs the same evidence as it being better."""
    better = power.queries_to_resolve(half_width=0.04, mean_delta=0.02, queries=200)
    worse = power.queries_to_resolve(half_width=0.04, mean_delta=-0.02, queries=200)

    assert better == worse


def test_a_degenerate_interval_or_query_count_is_unresolvable(power: Any) -> None:
    assert power.queries_to_resolve(half_width=0.0, mean_delta=0.02, queries=200) == math.inf
    assert power.queries_to_resolve(half_width=0.03, mean_delta=0.02, queries=0) == math.inf


# --- summarising a corpus --------------------------------------------------------------------------


def _effect(delta: float, half_width: float, *, n: int = 200, resolved: bool = False) -> dict[str, Any]:
    return {
        "mean_delta": delta,
        "ci_lo": delta - half_width,
        "ci_hi": delta + half_width,
        "n_shared": n,
        "resolved": resolved,
    }


def test_only_unresolved_comparisons_are_costed(power: Any) -> None:
    """A comparison that already separated needs nothing; including it would drag the median
    toward zero and make the corpus look better served than it is."""
    effects = [_effect(0.2, 0.01, resolved=True), _effect(0.01, 0.04)]

    row = power.summarise_corpus("c", effects, ladder=(1000,))

    assert row["comparisons"] == 2
    assert row["unresolved"] == 1


def test_unresolvable_comparisons_are_counted_separately_not_dropped(power: Any) -> None:
    """They are the ones where the answer is already in: the configurations are the same.

    Dropping them silently would shrink the denominator and overstate how much a bigger corpus
    would buy.
    """
    row = power.summarise_corpus("c", [_effect(0.0, 0.04), _effect(0.02, 0.04)], ladder=(1000,))

    assert row["never_resolvable"] == 1
    assert row["median_required_queries"] == 800


def test_the_ladder_counts_how_many_resolve_at_each_size(power: Any) -> None:
    """What a reader decides on: is the next dataset size worth collecting."""
    effects = [_effect(0.02, 0.04), _effect(0.005, 0.04), _effect(0.0, 0.04)]

    row = power.summarise_corpus("c", effects, ladder=(500, 1000, 20000))

    assert row["resolved_at"]["500"] == 0
    assert row["resolved_at"]["1000"] == 1  # the 0.02 effect needs 800
    assert row["resolved_at"]["20000"] == 2  # the 0.005 effect needs 12,800; the zero one never


def test_the_corpus_reports_its_own_effect_size(power: Any) -> None:
    """A small query set with large effects resolves more than a big one with tiny effects, so the
    effect size is what makes a query count adequate - not the count alone."""
    row = power.summarise_corpus("c", [_effect(0.04, 0.01), _effect(0.02, 0.03)], ladder=(1000,))

    assert row["median_effect"] == pytest.approx(0.03)
    assert row["median_half_width"] == pytest.approx(0.02)


def test_a_corpus_with_nothing_unresolved_reports_no_requirement(power: Any) -> None:
    row = power.summarise_corpus("c", [_effect(0.2, 0.01, resolved=True)], ladder=(1000,))

    assert row["unresolved"] == 0
    assert row["median_required_queries"] is None
