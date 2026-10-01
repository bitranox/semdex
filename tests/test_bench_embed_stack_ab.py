"""The onnxruntime cap has to be re-testable, or it silently becomes permanent.

`pyproject.toml` caps onnxruntime below 1.29.0 because that release embeds 8.5x slower on CPU. A cap
like that rots: the day upstream fixes it, the cap keeps the project on 1.28.0 anyway, because
nobody re-runs an evening's worth of A/B by hand. `scripts/bench_embed_stack_ab.py` makes that check
one command, and these tests cover the part that decides the answer.

The venv-building half needs uv, network and a real chunk corpus, so it is exercised by a
`local_only` test; the comparison logic is pure and runs everywhere.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "bench_embed_stack_ab.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("bench_embed_stack_ab", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_embed_stack_ab"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench() -> Any:
    return _load()


def test_summarise_reports_the_mean_and_the_spread_per_arm(bench: Any) -> None:
    """A single number hides an arm whose rounds disagree, which is how noise reads as a result."""
    stats = bench.summarise([("base", 100.0), ("cand", 800.0), ("base", 110.0), ("cand", 900.0)])
    assert stats["base"]["mean_ms"] == pytest.approx(105.0)
    assert stats["base"]["rounds"] == 2
    assert stats["cand"]["mean_ms"] == pytest.approx(850.0)
    assert stats["base"]["min_ms"] == pytest.approx(100.0)
    assert stats["cand"]["max_ms"] == pytest.approx(900.0)


def _arm(*rounds: float) -> dict[str, float]:
    return {"mean_ms": sum(rounds) / len(rounds), "min_ms": min(rounds), "max_ms": max(rounds), "rounds": len(rounds)}


def test_verdict_calls_a_clear_regression_slower(bench: Any) -> None:
    """The measured case: the arms do not come close to touching."""
    assert bench.verdict(baseline=_arm(96.0, 112.0), candidate=_arm(815.0, 957.0), tolerance=0.15) == "SLOWER"


def test_verdict_calls_equal_performance_parity(bench: Any) -> None:
    """A fixed release must be recognised, or the cap can never be re-opened."""
    assert bench.verdict(baseline=_arm(100.0, 108.0), candidate=_arm(104.0, 110.0), tolerance=0.15) == "PARITY"


def test_verdict_treats_a_faster_candidate_as_parity(bench: Any) -> None:
    """Faster is not a failure - the gate is 'not worse', not 'identical'."""
    assert bench.verdict(baseline=_arm(100.0, 108.0), candidate=_arm(78.0, 82.0), tolerance=0.15) == "PARITY"


def test_verdict_is_inconclusive_while_the_arms_overlap(bench: Any) -> None:
    """Noisy rounds that interleave are not a result, whatever the means say."""
    assert bench.verdict(baseline=_arm(100.0, 150.0), candidate=_arm(120.0, 170.0), tolerance=0.15) == "INCONCLUSIVE"


def test_more_rounds_can_resolve_an_overlap(bench: Any) -> None:
    """The whole point of the change: the docstring tells the reader to run more rounds, so more
    rounds has to be able to change the answer. Same means as the overlapping case above, tighter
    spread - which is exactly what more rounds buys - and the verdict must now land."""
    assert bench.verdict(baseline=_arm(100.0, 105.0), candidate=_arm(140.0, 145.0), tolerance=0.15) == "SLOWER"


def test_a_single_round_does_not_over_claim_a_modest_gap(bench: Any) -> None:
    """One round per arm carries no spread, so disjointness is meaningless - a 1.2x reading there
    is not evidence of a regression, and calling it one would make the tool confidently wrong."""
    assert bench.verdict(baseline=_arm(100.0), candidate=_arm(120.0), tolerance=0.15) == "INCONCLUSIVE"


def test_a_single_round_still_catches_a_gross_regression(bench: Any) -> None:
    """The 8.5x control has to keep working at one round, or the cheap smoke test is lost."""
    assert bench.verdict(baseline=_arm(104.0), candidate=_arm(886.0), tolerance=0.15) == "SLOWER"


def test_exit_code_follows_the_verdict(bench: Any) -> None:
    """Format-independent exit codes: 0 safe to raise the cap, 1 not, 2 could not tell."""
    assert bench.exit_code("PARITY") == 0
    assert bench.exit_code("SLOWER") == 1
    assert bench.exit_code("INCONCLUSIVE") == 2


def test_interleaved_plan_alternates_the_arms(bench: Any) -> None:
    """A,B,A,B - never all of A then all of B, which confounds the arm with the wall clock."""
    assert bench.interleaved_plan(["base", "cand"], rounds=3) == [
        "base",
        "cand",
        "base",
        "cand",
        "base",
        "cand",
    ]


def test_summarise_refuses_an_arm_with_no_measurements(bench: Any) -> None:
    """An arm whose runs all failed must not average to a confident number."""
    with pytest.raises(ValueError, match="no measurements"):
        bench.summarise([("base", 100.0)], expected_arms=("base", "cand"))
