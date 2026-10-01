#!/usr/bin/env python
# pyright: basic
# Benchmark harness on numpy (no strict stubs); strict mode would only add reportUnknown* noise.
"""Uncertainty for retrieval metrics, so a ranking table can say which gaps are real.

Every nDCG in this project is a mean over a query set: 20 queries on the gated slice, 200 on
mldr_de, 799 on miracl_en. Reported bare, two numbers that differ in the third decimal look like
a finding. With per-query nDCG standard deviation around 0.30 the 95 percent half-width is about
0.021 at n=800 and about 0.13 at n=20, so a fair number of published verdicts sit inside their own
noise.

Two instruments here, and the difference between them matters:

``bootstrap_ci`` gives one cell's interval, which is what a ranking column should print.

``paired_ci`` compares two cells over the queries they SHARE, which is what a "winner" claim needs.
The unpaired comparison throws away the fact that both systems answered the same queries, and a
query set contains easy and hard queries, so most of the variance is query difficulty common to
both. Pairing removes it, and routinely resolves a difference the two separate intervals leave
looking ambiguous. Comparing two overlapping confidence intervals by eye is a weaker test than
the paired interval, and it is the mistake this function exists to prevent.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["DEFAULT_ALPHA", "MATERIAL_FLOOR", "MATERIAL_FLOOR_METRIC", "bootstrap_ci", "is_material", "paired_ci"]

_DEFAULT_BOOT = 10000
# Fixed so a re-run of the same scores reproduces the same interval; the docs quote these numbers
# and a wandering last digit reads as a measurement change.
_DEFAULT_SEED = 20260806
_BOOT_BLOCK = 1000  # resamples per batch, so the index array stays bounded for large query sets

# The interval's alpha and the floor under which a RESOLVED difference is not worth acting on.
# Both are written into chunk-knob-effects.json by the exporter, so a table prints what the rows
# were judged against instead of restating the numbers. 0.005 nDCG@10 is half a point: at 12,298
# queries the paired interval resolves a step of 0.0009, which no deployer would change a setting
# for, while every comparison on an 800-query corpus that resolves at all clears 0.005.
DEFAULT_ALPHA = 0.05
MATERIAL_FLOOR = 0.005
MATERIAL_FLOOR_METRIC = "ndcg@10"


def _boot_means(values: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Bootstrap resample means, drawn in blocks so memory stays flat in ``n_boot``."""
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    means = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, _BOOT_BLOCK):
        stop = min(start + _BOOT_BLOCK, n_boot)
        picks = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = values[picks].mean(axis=1)
    return means


def bootstrap_ci(
    values: list[float] | np.ndarray,
    *,
    n_boot: int = _DEFAULT_BOOT,
    seed: int = _DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, float]:
    """Percentile bootstrap interval for the mean of one cell's per-query scores.

    Args:
        values: per-query metric values. One entry per query, not a pre-averaged number.
        n_boot: resample count.
        seed: fixed for reproducibility.
        alpha: 0.05 gives a 95 percent interval.

    Returns:
        ``mean``, ``ci_lo``, ``ci_hi``, ``sd``, ``n``, and ``half_width``. An empty input gives
        zeros and ``n`` 0; a single value gives a degenerate interval at that value, which is
        correct rather than an error, because one query genuinely constrains nothing.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "sd": 0.0, "n": 0, "half_width": 0.0}
    means = _boot_means(array, n_boot, seed)
    lo, hi = np.percentile(means, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return {
        "mean": float(array.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "sd": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "n": int(array.size),
        "half_width": float((hi - lo) / 2),
    }


def paired_ci(
    left: dict[str, float],
    right: dict[str, float],
    *,
    n_boot: int = _DEFAULT_BOOT,
    seed: int = _DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, object]:
    """Bootstrap the per-query DIFFERENCE between two cells over their shared queries.

    Args:
        left: per-query scores keyed by query id.
        right: the same, for the cell being compared against.
        n_boot: resample count.
        seed: fixed for reproducibility.
        alpha: 0.05 gives a 95 percent interval.

    Returns:
        ``mean_delta`` (left minus right) with ``ci_lo``/``ci_hi``, the ``wins``/``ties``/
        ``losses`` counts, ``n_shared``, and ``resolved`` - false when the interval spans zero,
        which is the flag a ranking table needs in order to print a tie instead of a winner.
    """
    shared = sorted(set(left) & set(right))
    if not shared:
        return {
            "mean_delta": 0.0,
            "ci_lo": 0.0,
            "ci_hi": 0.0,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "n_shared": 0,
            "resolved": False,
        }
    deltas = np.asarray([left[q] - right[q] for q in shared], dtype=np.float64)
    stats = bootstrap_ci(deltas, n_boot=n_boot, seed=seed, alpha=alpha)
    return {
        "mean_delta": stats["mean"],
        "ci_lo": stats["ci_lo"],
        "ci_hi": stats["ci_hi"],
        "wins": int((deltas > 0).sum()),
        "ties": int((deltas == 0).sum()),
        "losses": int((deltas < 0).sum()),
        "n_shared": len(shared),
        # A sign change inside the interval means the query set cannot tell these two apart.
        "resolved": bool(stats["ci_lo"] > 0 or stats["ci_hi"] < 0),
    }


def is_material(paired: dict[str, Any], *, floor: float = MATERIAL_FLOOR) -> bool:
    """Whether a paired comparison is resolved AND moves the metric by at least the floor.

    Resolved says the query set can tell the two configurations apart. Material says the
    difference is large enough to act on. A large query set makes the first cheap and leaves
    the second untouched, which is why a count of resolutions alone overstates what was found.

    Args:
        paired: a ``paired_ci`` result, or any row carrying ``mean_delta`` and ``resolved``.
        floor: the smallest absolute difference that counts, in the metric's own units.

    Returns:
        True only for a resolved comparison whose absolute mean difference is at or above the floor.
    """
    return bool(paired["resolved"]) and abs(float(paired["mean_delta"])) >= floor
