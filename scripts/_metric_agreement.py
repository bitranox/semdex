"""How much independent information four retrieval metrics actually carry.

nDCG@10, Recall@10, MRR and P@1 are printed side by side on every ranking table here, which
implies four perspectives on a result. On a corpus that ships about one relevant document per
query they are not four: Recall@10 can only be 0 or 1, and nDCG@10 collapses toward MRR because
with a single relevant document the gain vector has one non-zero entry whose position is exactly
what MRR measures.

That is an argument. This module measures it, three ways, because each answers a different
question a reader might have:

* **Rank correlation** - do the metrics order the configurations the same way?
* **Pairwise agreement** - for two configurations, how often do two metrics disagree about which
  one is better? This is the decision-relevant form: a second column earns its place only if it
  ever changes the answer.
* **Effective views** - collapsing the correlation matrix to a single number, so "four columns,
  how many perspectives" has an answer rather than a vibe.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np

# Below this the two metrics are ordering the cells almost identically, which is the threshold at
# which a second column stops informing a decision and starts padding a table.
NEAR_DUPLICATE_RHO = 0.95


def _ranks(values: Sequence[float]) -> np.ndarray:
    """Average ranks, so tied values share a rank rather than taking an arbitrary order.

    Ties are common here: P@1 on a small query set takes few distinct values, and breaking those
    ties by input order would manufacture correlation out of the order the cells were written.
    """
    array = np.asarray(values, dtype=np.float64)
    order = array.argsort()
    ranks = np.empty(len(array), dtype=np.float64)
    ranks[order] = np.arange(len(array), dtype=np.float64)
    # average the ranks within each group of equal values
    for value in np.unique(array):
        mask = array == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Rank correlation between two metrics over the same cells.

    Rank rather than linear correlation: the metrics live on different scales and none of the
    relationships need be linear. What matters is whether they ORDER the configurations alike.

    Returns 0.0 when either series is constant, where correlation is undefined - a constant metric
    orders nothing, and reporting 1.0 there would claim perfect agreement with everything.
    """
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    rx, ry = _ranks(xs), _ranks(ys)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def pairwise_disagreement(xs: Sequence[float], ys: Sequence[float]) -> dict[str, float]:
    """Over every pair of cells, how often the two metrics pick different winners.

    The decision-relevant form of the question. A pair where EITHER metric ties is excluded rather
    than counted as agreement: a tie is the metric declining to choose, and scoring that as
    agreement inflates every comparison involving a coarse metric like P@1.

    Returns the comparable pair count, how many disagreed, and the rate.
    """
    compared = 0
    disagreed = 0
    for i, j in itertools.combinations(range(len(xs)), 2):
        left = xs[i] - xs[j]
        right = ys[i] - ys[j]
        if left == 0 or right == 0:
            continue
        compared += 1
        if (left > 0) != (right > 0):
            disagreed += 1
    return {
        "pairs": compared,
        "disagreements": disagreed,
        "rate": disagreed / compared if compared else 0.0,
    }


def effective_views(correlations: np.ndarray) -> float:
    """How many independent perspectives a correlation matrix really holds.

    The participation ratio of its eigenvalues: ``(sum L)^2 / sum(L^2)``. Four metrics that agree
    perfectly give 1.0, four independent ones give 4.0, and the gap this measures claims the
    answer here is nearer one and a half than four.

    Negative eigenvalues are clipped to zero. A correlation matrix assembled PAIRWISE is not
    guaranteed positive semi-definite, and a negative eigenvalue both lowers the numerator and
    raises the denominator, so it DEFLATES the count: an inconsistent 3x3 measured 1.15 views
    unclipped against 2.0 clipped. That is the direction that would wrongly confirm the redundancy
    this function exists to test, which is the worst direction for it to be wrong in.
    """
    eigenvalues = np.clip(np.linalg.eigvalsh(correlations), 0.0, None)
    total = eigenvalues.sum()
    squared = (eigenvalues**2).sum()
    return float(total**2 / squared) if squared > 0 else 0.0


def correlation_matrix(columns: dict[str, Sequence[float]]) -> tuple[list[str], np.ndarray]:
    """Spearman correlations between every pair of named metric columns."""
    names = list(columns)
    size = len(names)
    matrix = np.eye(size, dtype=np.float64)
    for i, j in itertools.combinations(range(size), 2):
        rho = spearman(columns[names[i]], columns[names[j]])
        matrix[i, j] = matrix[j, i] = rho
    return names, matrix


__all__ = [
    "NEAR_DUPLICATE_RHO",
    "correlation_matrix",
    "effective_views",
    "pairwise_disagreement",
    "spearman",
]
