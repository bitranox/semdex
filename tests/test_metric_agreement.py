"""Agreement arithmetic behind the "four metrics, how many views" measurement.

The claim being tested is that four printed columns are closer to one and a half perspectives.
Every function that decides it can be wrong in a direction that CONFIRMS the claim, which is the
dangerous direction: ties scored as agreement, ranks broken by input order, or an eigenvalue
collapsed by numerical noise all make the metrics look more redundant than they are.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_MODULE = Path(__file__).resolve().parents[1] / "scripts" / "_metric_agreement.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("_metric_agreement", _MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def agree() -> Any:
    return _load()


# --- rank correlation -----------------------------------------------------------------------------


def test_identical_orderings_correlate_perfectly(agree: Any) -> None:
    assert agree.spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(1.0)


def test_reversed_orderings_correlate_negatively(agree: Any) -> None:
    assert agree.spearman([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]) == pytest.approx(-1.0)


def test_it_measures_order_not_linearity(agree: Any) -> None:
    """The metrics live on different scales and need not be linearly related.

    A monotone but wildly non-linear pairing still orders the configurations identically, which is
    the only thing that matters for whether a second column changes a decision.
    """
    assert agree.spearman([1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 900.0, 10000.0]) == pytest.approx(1.0)


def test_tied_values_share_a_rank_rather_than_taking_input_order(agree: Any) -> None:
    """P@1 on a small query set takes few distinct values, so ties are everywhere.

    Breaking them by the order the cells happen to sit in the file would manufacture correlation
    out of the file layout - and would do so in the direction that confirms the redundancy claim.
    """
    tied = [5.0, 5.0, 5.0, 9.0]

    forward = agree.spearman(tied, [1.0, 2.0, 3.0, 4.0])
    shuffled = agree.spearman([5.0, 9.0, 5.0, 5.0], [1.0, 4.0, 2.0, 3.0])

    assert forward == pytest.approx(shuffled)


def test_a_constant_metric_correlates_with_nothing(agree: Any) -> None:
    """Correlation is undefined against a constant. Returning 1.0 would claim a metric that orders
    nothing agrees perfectly with everything."""
    assert agree.spearman([3.0, 3.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_too_few_points_correlate_at_zero(agree: Any) -> None:
    assert agree.spearman([1.0], [2.0]) == 0.0
    assert agree.spearman([], []) == 0.0


# --- pairwise disagreement -------------------------------------------------------------------------


def test_metrics_that_always_pick_the_same_winner_never_disagree(agree: Any) -> None:
    result = agree.pairwise_disagreement([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])

    assert result["pairs"] == 3
    assert result["disagreements"] == 0
    assert result["rate"] == 0.0


def test_a_reversed_metric_disagrees_on_every_pair(agree: Any) -> None:
    result = agree.pairwise_disagreement([1.0, 2.0, 3.0], [30.0, 20.0, 10.0])

    assert result["rate"] == pytest.approx(1.0)


def test_a_tie_in_either_metric_is_excluded_not_counted_as_agreement(agree: Any) -> None:
    """A tie is the metric DECLINING to choose.

    Scoring it as agreement inflates every comparison involving a coarse metric - P@1 ties
    constantly - and would report near-perfect agreement for a column that mostly says nothing.
    """
    # pairs: (0,1) ties in ys, (0,2) and (1,2) are comparable
    result = agree.pairwise_disagreement([1.0, 2.0, 3.0], [10.0, 10.0, 30.0])

    assert result["pairs"] == 2
    assert result["disagreements"] == 0


def test_everything_tied_reports_no_comparable_pairs_rather_than_perfect_agreement(agree: Any) -> None:
    result = agree.pairwise_disagreement([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])

    assert result["pairs"] == 0
    assert result["rate"] == 0.0


def test_a_single_disagreement_is_counted_at_its_true_rate(agree: Any) -> None:
    """Three cells give three pairs; one flipped pair is a third, not a half."""
    result = agree.pairwise_disagreement([1.0, 2.0, 3.0], [10.0, 30.0, 20.0])

    assert result["pairs"] == 3
    assert result["disagreements"] == 1
    assert result["rate"] == pytest.approx(1 / 3)


# --- effective number of views ---------------------------------------------------------------------


def test_four_perfectly_agreeing_metrics_are_one_view(agree: Any) -> None:
    """The claim under test, in its extreme form: four identical columns are one perspective."""
    assert agree.effective_views(np.ones((4, 4))) == pytest.approx(1.0)


def test_four_independent_metrics_are_four_views(agree: Any) -> None:
    assert agree.effective_views(np.eye(4)) == pytest.approx(4.0)


def test_partly_correlated_metrics_land_between(agree: Any) -> None:
    """Two pairs, each pair internally identical and independent of the other pair, is two views."""
    block = np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])

    assert agree.effective_views(block) == pytest.approx(2.0)


def test_a_degenerate_matrix_still_reports_one_view(agree: Any) -> None:
    """Perfectly correlated columns yield eigenvalues that are zero to floating-point precision."""
    views = agree.effective_views(np.ones((4, 4)))

    assert 0.99 <= views <= 1.01


def test_a_negative_eigenvalue_does_not_deflate_the_view_count(agree: Any) -> None:
    """A pairwise-assembled correlation matrix need not be positive semi-definite.

    A negative eigenvalue lowers the numerator AND raises the denominator, so left unclipped it
    reports FEWER independent views than exist - 1.15 against 2.0 on this matrix. That is the
    direction that wrongly confirms the very redundancy this measurement is testing, so the clip
    is load-bearing rather than defensive.
    """
    not_psd = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    assert np.linalg.eigvalsh(not_psd).min() < -0.5  # the matrix really is indefinite

    assert agree.effective_views(not_psd) == pytest.approx(2.0)


# --- the matrix ------------------------------------------------------------------------------------


def test_the_matrix_is_symmetric_with_a_unit_diagonal(agree: Any) -> None:
    names, matrix = agree.correlation_matrix(
        {"ndcg": [1.0, 2.0, 3.0], "recall": [1.0, 3.0, 2.0], "mrr": [3.0, 2.0, 1.0]}
    )

    assert names == ["ndcg", "recall", "mrr"]
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0)


def test_the_matrix_carries_the_pairwise_correlations(agree: Any) -> None:
    _names, matrix = agree.correlation_matrix({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})

    assert matrix[0, 1] == pytest.approx(-1.0)
