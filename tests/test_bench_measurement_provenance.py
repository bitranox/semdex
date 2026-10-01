"""The stamp a scorer writes onto a cell at the moment it measures it.

Every scorer here persists incrementally into one results file and SKIPS a cell it has already
measured, so a sweep that resumes days later on a different machine rewrites the whole file while
re-measuring none of it. That makes one property load-bearing: stamping must touch only the rows
being measured now. A helper that re-stamped the whole results dict on each write would move the
original defect from the exporter into the scorers, where it would be harder to see - the file
would still be re-dated by whoever ran the last resume.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "_provenance.py"

pytestmark = pytest.mark.os_agnostic

_EXPECTED_KEYS = {
    "measured_utc",
    "semdex_git_sha",
    "host",
    "cpu",
    "python",
    "numpy",
    "blas",
    "openblas_num_threads",
}


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("_provenance", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_provenance"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def prov() -> Any:
    return _load()


def test_the_stamp_records_the_machine_and_the_code_that_measured(prov: Any) -> None:
    """A latency number is only comparable against another run on a known box."""
    stamp = prov.measurement_provenance()
    assert set(stamp) == _EXPECTED_KEYS
    empty = sorted(key for key, value in stamp.items() if not str(value).strip())
    assert empty == [], f"provenance fields collected but left empty: {empty}"


def test_a_freshly_measured_row_is_stamped(prov: Any) -> None:
    row = prov.stamped({"ndcg@10": 0.5})
    assert set(row[prov.MEASURED_ON]) == _EXPECTED_KEYS
    assert row["ndcg@10"] == 0.5


def test_a_row_that_already_carries_a_stamp_keeps_it(prov: Any) -> None:
    """The resume case: rewriting the results file must not re-date cells measured earlier."""
    original = {
        "measured_utc": "2026-08-10T02:12:00+00:00",
        "semdex_git_sha": "aaaaaaa",
        "host": "px-semdex-test-embeddings",
        "cpu": "x86_64",
        "python": "3.14.0",
        "numpy": "2.3.1",
        "blas": "scipy-openblas 0.3.33.112.0",
        "openblas_num_threads": "unset",
    }
    row = prov.stamped({"ndcg@10": 0.5, prov.MEASURED_ON: original})
    assert row[prov.MEASURED_ON] == original


def test_stamping_leaves_the_caller_s_row_untouched(prov: Any) -> None:
    """Scorers hold these rows in a dict they rewrite; an in-place stamp would alias into it."""
    row = {"ndcg@10": 0.5}
    prov.stamped(row)
    assert prov.MEASURED_ON not in row
