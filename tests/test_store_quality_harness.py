"""The composed store benchmark must measure what semdex SHIPS, not what each driver defaults to.

`scripts/score_through_store.py` builds a real store, runs the evaluation queries through it, and
publishes the result as the quality a user gets. That claim only holds if the store is configured
the way a stock install configures it. It was not: stores were built with no ANN params at all,
which is each DRIVER's default - a different thing since lancedb's `balanced` preset stopped being
the bare driver default, and the reason the published lancedb rows understated it by 0.0438 nDCG.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from semdex.adapters.config.vectorstore import VectorStoreConfig
from semdex.domain.ann_tuning import AnnParams
from semdex.domain.enums import StoreBackend

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_through_store.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("score_through_store", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harness() -> Any:
    return _load()


def test_the_harness_measures_the_shipped_configuration(harness: Any) -> None:
    """Whatever a stock install resolves is what the benchmark must run.

    Asserted against the config model rather than a literal, so changing a preset moves the
    benchmark with it instead of silently leaving the table describing an older default.
    """
    config = VectorStoreConfig()
    for backend in StoreBackend:
        assert harness.shipped_ann_params(backend) == config.resolved_ann_params(backend)


def test_lancedb_is_not_measured_at_the_bare_driver_default(harness: Any) -> None:
    """The specific regression: empty params mean the driver's default, not the shipped one.

    lancedb's driver default keeps 0.567 of the exact top-10 against 0.986 for what semdex now
    ships, so measuring it would republish the understated figure this test exists to prevent.
    """
    assert harness.shipped_ann_params(StoreBackend.LANCEDB) != AnnParams()


def test_exact_stores_are_measured_with_no_ann_params(harness: Any) -> None:
    """json and sqlite_vec scan every row; an ANN knob there would be meaningless, not neutral."""
    for backend in (StoreBackend.JSON, StoreBackend.SQLITE_VEC):
        assert harness.shipped_ann_params(backend) == AnnParams()


def test_every_row_records_the_setting_it_was_measured_at(harness: Any) -> None:
    """A number without its ANN setting cannot be compared against a later run of the same cell."""
    assert harness._describe(AnnParams()) == "driver default"
    assert harness._describe(AnnParams(nprobes=10, refine_factor=5)) == "nprobes=10 refine_factor=5"
    assert harness._describe(AnnParams(ef_search=200)) == "ef_search=200"


def test_the_store_the_harness_builds_actually_carries_those_params(tmp_path: Path, harness: Any) -> None:
    """The call site, not just the helper.

    Resolving the right params and then calling the factory without them is exactly the bug this
    file exists for, and a test of `shipped_ann_params` alone passes against it. So build a real
    store the way the benchmark does and ask the store what it got.
    """
    pytest.importorskip("lancedb")

    store = harness._load_store(StoreBackend.LANCEDB, tmp_path / "lance", 8)
    try:
        assert store.ann_params == harness.shipped_ann_params(StoreBackend.LANCEDB)
        assert store.ann_params.refine_factor == 5
    finally:
        store.close()
