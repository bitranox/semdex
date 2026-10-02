"""Repeat-and-spread timing for the throughput benchmarks.

Every throughput figure this repo published was one timed pass on a machine that was also doing
other work. Two things were wrong with that. A single sample cannot show contention, so a rate
inflated or deflated by whatever else the box was running is indistinguishable from the real one.
And the fastest strategy was timed over 0.01 seconds, where the clock's own resolution and one
scheduler preemption are a large share of the measurement - 50,620 docs/s was arithmetic on noise.

The fix is the standard one: repeat the work until it has run for a floor duration, repeat that
several times, and report the median with its spread and the sample count, so a reader can see how
noisy the box was rather than having to trust that it was quiet.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

# A single pass shorter than this is re-run back to back until the total exceeds it, and the rate
# is computed over the whole batch. Well above the clock resolution and long enough that one
# scheduler slice cannot dominate.
MIN_SAMPLE_SECONDS = 0.5

# How many independent samples to take. The spread across them is the contention signal; three is
# too few to see an outlier and twenty costs more than the answer is worth.
DEFAULT_REPEATS = 7

# Wall-clock budget for one measurement. Without it, DEFAULT_REPEATS costs the same seven samples
# whether a pass takes four milliseconds or two minutes, and a slow cell alone can outlast the
# whole rest of the benchmark - which is how a suite stops being re-runnable and its numbers
# become "what we got that one time". A cell that runs out of budget reports the samples it did
# take, and Timing.repeats says how many that was.
MAX_MEASURE_SECONDS = 120.0

# Below two samples there is no spread to report, and a lone sample would publish spread 0.0 -
# indistinguishable from a perfectly quiet box. Better to overrun the budget than to claim that.
MIN_REPEATS = 2


@dataclass(frozen=True, slots=True)
class Timing:
    """The distribution of a repeated measurement, not a single number."""

    median_s: float
    min_s: float
    max_s: float
    repeats: int
    iterations_per_sample: int

    @property
    def spread_pct(self) -> float:
        """How far the extremes sit from the median, as a percentage of it.

        This is the number that says whether the box was quiet. A rate quoted without it cannot
        be told apart from one taken while something else was running.
        """
        if self.median_s <= 0:
            return 0.0
        return round(100.0 * (self.max_s - self.min_s) / self.median_s, 1)


def measure(
    work: Callable[[], int],
    *,
    repeats: int = DEFAULT_REPEATS,
    min_seconds: float = MIN_SAMPLE_SECONDS,
    max_seconds: float = MAX_MEASURE_SECONDS,
) -> Timing:
    """Time ``work`` repeatedly, batching it up to ``min_seconds`` per sample.

    Args:
        work: runs the workload once and returns how many units it processed. Called for a warmup
            first, so model loads and lazy imports never land inside a sample.
        repeats: independent samples to take, budget permitting.
        min_seconds: floor for one sample; the workload is repeated back to back until the elapsed
            time passes it, and the per-sample duration is the batch divided by its iterations.
        max_seconds: wall-clock budget for the samples. Sampling stops early once it is spent, so
            a slow workload costs bounded time instead of ``repeats`` times however long it takes.

    Returns:
        The distribution, including how many samples were actually taken and how many iterations
        each needed - both worth recording, because a strategy needing 50 iterations to fill half
        a second is one whose old single-pass figure was the least trustworthy, and one that
        stopped at two samples has a spread a reader should weigh accordingly.
    """
    work()  # warmup: model load, recipe fetch and lazy imports stay out of the samples
    iterations = _iterations_for(work, min_seconds)
    samples: list[float] = []
    deadline = time.perf_counter() + max_seconds
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(iterations):
            work()
        samples.append((time.perf_counter() - started) / iterations)
        if len(samples) >= MIN_REPEATS and time.perf_counter() >= deadline:
            break
    return Timing(
        median_s=statistics.median(samples),
        min_s=min(samples),
        max_s=max(samples),
        repeats=len(samples),
        iterations_per_sample=iterations,
    )


def _iterations_for(work: Callable[[], int], min_seconds: float) -> int:
    """How many back-to-back runs it takes to fill ``min_seconds``, from one probe run."""
    started = time.perf_counter()
    work()
    once = time.perf_counter() - started
    if once <= 0:
        return 1000
    return max(1, int(min_seconds / once) + 1)


def load_snapshot() -> dict[str, float | int | None]:
    """What else the machine was doing, recorded beside every rate.

    A throughput number from a shared box is only readable next to its load, and this repo's were
    published without it. The load fields are None on Windows, which has no load average: an
    unrecorded load, never a zero one.
    """
    # A platform test rather than hasattr: pyright narrows on sys.platform, so the call below
    # stays typed when it checks for Windows.
    if sys.platform == "win32":
        return {"load_1min": None, "load_5min": None, "load_15min": None, "cores": os.cpu_count() or 0}
    one, five, fifteen = os.getloadavg()
    return {
        "load_1min": round(one, 2),
        "load_5min": round(five, 2),
        "load_15min": round(fifteen, 2),
        "cores": os.cpu_count() or 0,
    }
