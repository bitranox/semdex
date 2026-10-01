"""Turning "underpowered" into a number of queries.

A comparison reported as a tie is ambiguous in a way the word hides. It can mean a real difference
too small to see at this query count, which more data would resolve, or it can mean the two
configurations are the same, which no query count resolves because there is nothing to find. Those
call for opposite actions and the tables here render them identically.

The arithmetic that separates them is one line - a paired bootstrap half-width shrinks as
1/sqrt(n), so resolving an observed effect ``d`` from a half-width ``h`` needs ``n * (h/d)^2`` -
and every way of getting it wrong yields a plausible number rather than an error.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

# Below this the observed effect is indistinguishable from exactly zero at any query count worth
# collecting, so it is reported as unresolvable rather than as a very large number. A delta of
# 1e-4 nDCG is not a finding anybody would act on even if a corpus could prove it.
NEGLIGIBLE_DELTA = 1e-4


def queries_to_resolve(*, half_width: float, mean_delta: float, queries: int) -> float:
    """Queries needed for the interval to clear the observed effect.

    The half-width of a paired bootstrap interval falls as 1/sqrt(n), so shrinking it from ``h`` to
    ``|d|`` needs a factor ``(h/d)^2`` more queries. The SQUARE is the part worth stating: halving
    an interval costs four times the data, so a comparison that is a long way off is much further
    off than it looks.

    The sign of ``mean_delta`` is irrelevant and needs no absolute value here: squaring removes
    it, and a configuration being worse needs exactly as much evidence as it being better.

    Returns ``inf`` when the observed effect is negligible - there is no query count that resolves
    a difference of zero, and returning an enormous finite number would invite someone to chase it.
    """
    if abs(mean_delta) < NEGLIGIBLE_DELTA:
        return math.inf
    if half_width <= 0 or queries <= 0:
        return math.inf
    return queries * (half_width / mean_delta) ** 2


def summarise_corpus(corpus: str, effects: Sequence[dict[str, Any]], *, ladder: Sequence[int]) -> dict[str, Any]:
    """One corpus: how many comparisons are ties, and what it would take to settle them.

    ``effects`` are paired comparisons carrying ``mean_delta``, ``ci_lo``/``ci_hi``, ``n_shared``
    and ``resolved``.
    """
    unresolved = [e for e in effects if not e["resolved"]]
    queries = effects[0]["n_shared"] if effects else 0
    required = [
        queries_to_resolve(
            half_width=(e["ci_hi"] - e["ci_lo"]) / 2,
            mean_delta=e["mean_delta"],
            queries=e["n_shared"],
        )
        for e in unresolved
    ]
    finite = sorted(r for r in required if math.isfinite(r))
    return {
        "corpus": corpus,
        "queries": queries,
        "comparisons": len(effects),
        "unresolved": len(unresolved),
        "unresolved_share": round(len(unresolved) / len(effects), 4) if effects else 0.0,
        # The effects the corpus CAN see, which is what makes a query count adequate or not: a
        # small set with large effects resolves more than a big set with tiny ones.
        "median_effect": round(_median([abs(e["mean_delta"]) for e in effects]), 4),
        "median_half_width": round(_median([(e["ci_hi"] - e["ci_lo"]) / 2 for e in effects]), 4),
        "median_required_queries": round(_median(finite)) if finite else None,
        "never_resolvable": len(required) - len(finite),
        "resolved_at": {str(target): sum(1 for r in required if r <= target) for target in ladder},
    }


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


__all__ = ["NEGLIGIBLE_DELTA", "queries_to_resolve", "summarise_corpus"]
