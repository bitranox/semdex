"""Repeat-and-spread timing used by the throughput benchmarks.

Every throughput number this repo published was a single timed pass on a shared machine, and the
fastest strategy's was taken over 0.01 seconds - the clock's own resolution. This module is what
replaces that, so the properties it has to hold are: a fast workload gets batched up to a floor
duration before it is timed, the reported figure is a median rather than a lucky sample, and the
spread that reveals contention actually reflects the samples.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_MODULE = Path(__file__).resolve().parents[1] / "scripts" / "_bench_timing.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("_bench_timing", _MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def timing() -> Any:
    return _load()


def test_a_fast_workload_is_batched_up_to_the_floor(timing: Any) -> None:
    """The defect this replaces: a sub-millisecond pass timed once and divided.

    A workload far faster than the clock's useful resolution must be run many times per sample,
    or the reported rate is arithmetic on scheduler noise.
    """
    calls = 0

    def work() -> int:
        nonlocal calls
        calls += 1
        return 1

    result = timing.measure(work, repeats=2, min_seconds=0.05)

    assert result.iterations_per_sample > 1, "a trivial workload must be batched, not timed once"
    assert calls >= result.iterations_per_sample * 2


def test_a_slow_workload_is_not_batched(timing: Any) -> None:
    """One pass already past the floor is its own sample; batching it would just waste time."""
    result = timing.measure(lambda: _burn(0.02), repeats=2, min_seconds=0.001)

    assert result.iterations_per_sample == 1


def test_the_reported_duration_is_the_median_not_the_best_sample(timing: Any) -> None:
    """Quoting the minimum would report the machine's best moment as its throughput.

    On a contended box the fastest sample is the one that happened to miss the contention, which
    is exactly the number a reader must not be given.
    """
    # measure() spends one call on the warmup and one probing the batch size before any sample,
    # so the scripted durations have to outlast those too.
    scripted = [0.01, 0.01, 0.05, 0.01, 0.30, 0.01]
    calls = 0

    def work() -> int:
        nonlocal calls
        _burn(scripted[calls] if calls < len(scripted) else 0.01)
        calls += 1
        return 1

    result = timing.measure(work, repeats=4, min_seconds=0.0)

    assert result.min_s < result.median_s < result.max_s


def test_the_warmup_keeps_a_slow_first_call_out_of_the_batch_sizing(timing: Any) -> None:
    """A model load in the first call must not decide how many iterations a sample gets.

    The batch size is computed from one probe run. If the probe is the call that pays the model
    load, a fast workload looks slow, gets one iteration per sample, and is then timed over the
    sub-millisecond window this module exists to avoid - the original defect, reintroduced through
    the back door. So the warmup has to come first, and this asserts batching survives it.
    """
    calls = 0

    def work() -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            _burn(0.30)  # the model load
        return 1

    result = timing.measure(work, repeats=2, min_seconds=0.05)

    assert result.iterations_per_sample > 1, "the slow first call reached the probe and killed batching"
    assert result.max_s < 0.05, "the slow first call leaked into a sample"


def test_a_slow_workload_stops_sampling_once_its_budget_is_spent(timing: Any) -> None:
    """Otherwise one slow cell costs `repeats` times however long it takes.

    A fixed sample count is affordable at four milliseconds a pass and not at two minutes. The
    benchmark stops being re-runnable, and numbers nobody can re-run stop being measurements.
    """
    result = timing.measure(lambda: _burn(0.05), repeats=20, min_seconds=0.0, max_seconds=0.15)

    assert result.repeats < 20, "the budget did not stop sampling"


def test_the_budget_never_cuts_below_two_samples(timing: Any) -> None:
    """One sample reports spread 0.0, which reads as a perfectly quiet box rather than as unknown.

    Overrunning the budget is the lesser harm: a reader can see repeats=2 and weigh it, but
    cannot see that a 0.0 spread was manufactured by having nothing to compare against.
    """
    result = timing.measure(lambda: _burn(0.05), repeats=20, min_seconds=0.0, max_seconds=0.0)

    assert result.repeats == 2


def test_a_fast_workload_still_gets_every_repeat(timing: Any) -> None:
    """The budget is a ceiling for slow cells, not a cut applied to all of them."""
    result = timing.measure(lambda: 1, repeats=5, min_seconds=0.01, max_seconds=60.0)

    assert result.repeats == 5


def test_the_reported_repeat_count_is_what_was_actually_sampled(timing: Any) -> None:
    """Reporting the requested count instead would make a truncated run look like a full one.

    That is the same class of defect this whole module exists to fix: a number that describes the
    configuration rather than the measurement. Counted from the calls, because the requested and
    the actual count are equal in the untruncated case and only a truncated run tells them apart.
    """
    calls = 0

    def work() -> int:
        nonlocal calls
        calls += 1
        _burn(0.05)
        return 1

    result = timing.measure(work, repeats=20, min_seconds=0.0, max_seconds=0.15)

    warmup_and_probe = 2
    assert calls == warmup_and_probe + result.repeats * result.iterations_per_sample
    assert result.repeats < 20


def test_spread_reports_how_far_the_extremes_sit_from_the_median(timing: Any) -> None:
    """The contention signal. Without it a rate cannot be told apart from a lucky run."""
    steady = timing.Timing(median_s=1.0, min_s=1.0, max_s=1.0, repeats=5, iterations_per_sample=1)
    noisy = timing.Timing(median_s=1.0, min_s=0.5, max_s=1.5, repeats=5, iterations_per_sample=1)

    assert steady.spread_pct == 0.0
    assert noisy.spread_pct == 100.0


def test_spread_of_a_zero_duration_does_not_divide_by_zero(timing: Any) -> None:
    """A degenerate measurement must report no spread, not crash the whole benchmark run."""
    degenerate = timing.Timing(median_s=0.0, min_s=0.0, max_s=0.0, repeats=1, iterations_per_sample=1)

    assert degenerate.spread_pct == 0.0


def test_the_load_snapshot_records_what_else_the_box_was_doing(timing: Any) -> None:
    """A rate from a shared machine is only readable beside its load, and these were published
    without it."""
    snapshot = timing.load_snapshot()

    assert set(snapshot) == {"load_1min", "load_5min", "load_15min", "cores"}
    assert snapshot["cores"] >= 1


def test_a_platform_without_a_load_average_records_none_not_zero(timing: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows has no os.getloadavg; a zero would read as an idle box, so the load is unrecorded."""
    monkeypatch.setattr(timing.sys, "platform", "win32")

    snapshot = timing.load_snapshot()

    assert snapshot["load_1min"] is None
    assert snapshot["cores"] >= 1


def _burn(seconds: float) -> None:
    """Busy-wait, so the elapsed time is real work rather than a sleep the scheduler may extend."""
    import time

    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass
