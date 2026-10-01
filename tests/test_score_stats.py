"""Bootstrap and paired intervals for scripts/_score_stats.py.

These numbers decide which published verdicts survive, so the tests check calibration against
cases with a known answer rather than only checking that a number comes out. The paired test in
particular has to demonstrate the property it exists for: resolving a difference that the two
separate intervals leave overlapping.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "_score_stats.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("_score_stats", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stats() -> Any:
    return _load()


def test_the_interval_brackets_the_mean_and_narrows_with_more_queries(stats: Any) -> None:
    """The half-width must fall roughly as 1/sqrt(n): that is the whole argument for query count."""
    rng = np.random.default_rng(5)
    small = stats.bootstrap_ci(rng.normal(0.6, 0.30, 50))
    large = stats.bootstrap_ci(rng.normal(0.6, 0.30, 5000))

    assert small["ci_lo"] < small["mean"] < small["ci_hi"]
    assert large["ci_lo"] < large["mean"] < large["ci_hi"]
    ratio = small["half_width"] / large["half_width"]
    assert 6 < ratio < 15, f"expected roughly sqrt(100)=10x narrower, got {ratio:.1f}x"


def test_the_half_width_matches_the_normal_approximation(stats: Any) -> None:
    """Calibration against a known answer: 1.96 * sd / sqrt(n) for a well-behaved sample.

    Without this the bootstrap could be silently mis-specified (resampling the wrong axis, say)
    and still return a tidy-looking interval.
    """
    rng = np.random.default_rng(11)
    values = rng.normal(0.6, 0.30, 800)
    result = stats.bootstrap_ci(values)

    expected = 1.96 * values.std(ddof=1) / np.sqrt(values.size)
    assert result["half_width"] == pytest.approx(expected, rel=0.12)


def test_the_same_input_gives_the_same_interval_twice(stats: Any) -> None:
    """The docs quote these digits; a wandering last digit would read as a measurement change."""
    values = np.linspace(0.0, 1.0, 300)
    assert stats.bootstrap_ci(values) == stats.bootstrap_ci(values)


def test_degenerate_inputs_do_not_raise(stats: Any) -> None:
    empty = stats.bootstrap_ci([])
    single = stats.bootstrap_ci([0.42])

    assert empty["n"] == 0
    assert single["n"] == 1
    assert single["ci_lo"] == single["ci_hi"] == pytest.approx(0.42)
    assert single["sd"] == 0.0


def test_pairing_resolves_a_difference_that_the_separate_intervals_leave_overlapping(stats: Any) -> None:
    """The reason paired_ci exists.

    Query difficulty dominates the spread and is common to both systems, so the two marginal
    intervals overlap heavily while the per-query difference is consistently positive. Comparing
    the marginal intervals by eye would report "no difference" on a difference that is real in
    every single query.
    """
    rng = np.random.default_rng(2)
    difficulty = rng.normal(0.5, 0.30, 400)  # the shared component
    left = {str(i): float(v + 0.02) for i, v in enumerate(difficulty)}
    right = {str(i): float(v) for i, v in enumerate(difficulty)}

    marginal_left = stats.bootstrap_ci(list(left.values()))
    marginal_right = stats.bootstrap_ci(list(right.values()))
    overlap = marginal_left["ci_lo"] < marginal_right["ci_hi"]
    paired = stats.paired_ci(left, right)

    assert overlap, "precondition: the marginal intervals must overlap for this test to mean anything"
    assert paired["resolved"] is True
    assert paired["mean_delta"] == pytest.approx(0.02, abs=1e-6)
    assert paired["wins"] == 400


def test_an_unresolved_comparison_is_reported_as_unresolved(stats: Any) -> None:
    """A tie must come back as a tie, or the ranking tables invent winners."""
    rng = np.random.default_rng(4)
    values = rng.normal(0.6, 0.30, 200)
    left = {str(i): float(v) for i, v in enumerate(values)}
    right = {str(i): float(v + rng.normal(0, 0.30)) for i, v in enumerate(values)}

    paired = stats.paired_ci(left, right)

    assert paired["resolved"] is False
    assert paired["ci_lo"] < 0 < paired["ci_hi"]


def test_only_shared_queries_are_compared(stats: Any) -> None:
    """Two cells scored on different query sets must compare on the intersection, not by position."""
    left = {"a": 0.9, "b": 0.8, "zzz": 0.1}
    right = {"a": 0.4, "b": 0.3, "other": 0.99}

    paired = stats.paired_ci(left, right)

    assert paired["n_shared"] == 2
    assert paired["wins"] == 2
    assert paired["mean_delta"] == pytest.approx(0.5)


def test_no_shared_queries_is_not_a_result(stats: Any) -> None:
    paired = stats.paired_ci({"a": 1.0}, {"b": 0.0})

    assert paired["n_shared"] == 0
    assert paired["resolved"] is False
