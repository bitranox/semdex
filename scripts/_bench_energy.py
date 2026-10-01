"""Energy accounting for the embedding providers: the pure arithmetic, separated from the probes.

There is no price on a locally served model and no power figure anywhere in this repo, so cost is
the one dimension of the component choice that has never been measured. It can be, on this
hardware: the CPU package exposes a RAPL energy counter, the GPU reports power, and the node hangs
on a metered wall plug.

Three properties decide every function here, and each of them silently produces a plausible wrong
number if ignored:

* **A counter is not a gauge.** RAPL and the wall plug's kWh total are cumulative counters, so two
  reads give exact energy over the interval and the sampling rate is irrelevant. GPU power is an
  instantaneous gauge and has to be integrated, where the rate is everything. Treating one as the
  other is off by whatever the sampling happened to catch.
* **Counters wrap.** RAPL wraps at ``max_energy_range_uj`` (about 262 kJ here, so roughly an hour
  at idle). A wrapped delta reads NEGATIVE, and taken at face value it turns a measurement into a
  large energy saving.
* **Nothing here is per-process.** Every probe measures the whole package, card or machine, and
  this box is shared: idle alone swings 70 to 97 W. So the interesting number is MARGINAL - what
  the work added over an idle baseline measured around it - and that baseline has to be
  interleaved with the work rather than taken once before it.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

# Reported alongside every figure. A marginal energy smaller than the idle spread it was extracted
# from is arithmetic, not a measurement, and has to say so rather than be quoted to three digits.
MIN_SIGNAL_TO_NOISE = 2.0


def counter_delta_uj(before: int, after: int, *, max_range_uj: int) -> int:
    """Energy between two reads of a wrapping cumulative counter, in microjoules.

    Args:
        before: counter value at the start.
        after: counter value at the end.
        max_range_uj: the value at which the counter wraps to zero.

    Returns:
        The elapsed energy, corrected for at most one wrap. A raw ``after - before`` goes negative
        across a wrap, which reads as the machine having generated power.

    Raises:
        ValueError: if ``max_range_uj`` is not positive, which would make a wrap uncorrectable.
    """
    if max_range_uj <= 0:
        raise ValueError("max_energy_range_uj must be positive to correct a wrap")
    delta = after - before
    return delta if delta >= 0 else delta + max_range_uj


def integrate_power_w(samples: Sequence[tuple[float, float]]) -> float:
    """Energy in joules under a series of ``(timestamp_s, watts)`` gauge readings.

    Trapezoidal, because a power gauge between two samples is better approximated by the ramp
    than by either endpoint held flat: rectangular integration on a rising load systematically
    under- or over-counts depending on which endpoint is chosen.

    Fewer than two samples yields 0.0 rather than an error, which falls out of ``pairwise``
    producing no pairs: a run too short to sample twice has no measured energy, and inventing one
    from a single reading times a guessed duration is how a gauge gets treated as a counter.
    """
    return sum(
        (later_w + earlier_w) / 2.0 * (later_t - earlier_t)
        for (earlier_t, earlier_w), (later_t, later_w) in itertools.pairwise(samples)
    )


@dataclass(frozen=True, slots=True)
class EnergyResult:
    """What one workload cost, separated into the part it added and the part the machine draws."""

    total_j: float
    seconds: float
    idle_w: float
    idle_spread_w: float
    units: int

    @property
    def average_w(self) -> float:
        return self.total_j / self.seconds if self.seconds > 0 else 0.0

    @property
    def marginal_j(self) -> float:
        """Energy the WORK added, over an idle machine running for the same time.

        Clamped at zero: a workload cannot draw less than idle, so a negative result is the idle
        estimate drifting, not a saving, and publishing it as one would be worse than losing it.
        A zero-length run is zero too - there is no interval over which to subtract a baseline, so
        the whole total would be attributed to work that took no time.
        """
        if self.seconds <= 0:
            return 0.0
        return max(0.0, self.total_j - self.idle_w * self.seconds)

    @property
    def marginal_j_per_unit(self) -> float:
        return self.marginal_j / self.units if self.units else 0.0

    @property
    def total_j_per_unit(self) -> float:
        """Including the machine's idle draw - the right figure when the box exists only for this."""
        return self.total_j / self.units if self.units else 0.0

    @property
    def signal_to_noise(self) -> float:
        """Marginal power over the idle spread it had to be extracted from.

        The number that says whether the row is a measurement. Idle on a shared box swings by tens
        of watts, so a workload adding less than that cannot be resolved from it however many
        decimals the arithmetic yields.
        """
        marginal_w = self.marginal_j / self.seconds if self.seconds > 0 else 0.0
        return marginal_w / self.idle_spread_w if self.idle_spread_w > 0 else float("inf")

    @property
    def resolved(self) -> bool:
        return self.signal_to_noise >= MIN_SIGNAL_TO_NOISE


def idle_estimate(samples: Sequence[float]) -> tuple[float, float]:
    """Baseline watts and its spread, from idle windows measured AROUND the work.

    Median rather than mean: on a shared machine one neighbour's burst during a baseline window
    drags a mean upward, which then over-subtracts and understates the workload. The spread is the
    full range, because the question it answers is how far a single window can land from the
    middle - which is what limits resolution - not how tightly they cluster.
    """
    if not samples:
        return 0.0, 0.0
    return median(samples), max(samples) - min(samples)


def window_plan(docs_per_s: float, corpus_size: int, min_seconds: float) -> tuple[int, int]:
    """How many documents per pass, and how many passes, to fill a measurement window.

    Sized from BOTH directions, because the providers span four orders of magnitude: the
    placeholder embeds tens of thousands of documents a second and needs the corpus repeated to
    fill a window at all, while fastembed on a CPU manages single digits and would spend eleven
    minutes on one pass of the same corpus. Truncating for the slow ones is what keeps every row
    measured over a comparable window, so the idle drift underneath them is comparable too - and
    it is the difference between a nine-minute sweep and a two-hour one on a shared machine.

    Args:
        docs_per_s: measured rate from a short probe.
        corpus_size: documents available.
        min_seconds: the window each measurement should fill.

    Returns:
        ``(documents_per_pass, passes)``. Never zero documents: a rate so slow that the target
        rounds to nothing still measures one document, which is honest about being slow rather
        than dividing by zero.
    """
    if docs_per_s <= 0 or min_seconds <= 0:
        return corpus_size, 1
    target = max(1, round(docs_per_s * min_seconds))
    if target <= corpus_size:
        return target, 1
    return corpus_size, max(1, -(-target // corpus_size))  # ceil, so the window is filled not merely approached


def cost_per_million(joules_per_unit: float, *, tariff_eur_per_kwh: float) -> float:
    """Electricity cost of a million units, at a STATED tariff.

    The tariff is the assumption the whole money column rests on, so it is a parameter and is
    printed beside every figure derived from it rather than folded into a constant.
    """
    return joules_per_unit * 1_000_000 / 3_600_000 * tariff_eur_per_kwh


__all__ = [
    "MIN_SIGNAL_TO_NOISE",
    "EnergyResult",
    "cost_per_million",
    "counter_delta_uj",
    "idle_estimate",
    "integrate_power_w",
    "window_plan",
]
