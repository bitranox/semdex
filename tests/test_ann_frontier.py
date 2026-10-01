"""Recall accounting and knob-grid discipline for scripts/score_ann_frontier.py.

The sweep exists to replace a single default-parameter reading with a frontier, so the two things
that decide whether its numbers mean anything are how recall against the exact answer is counted,
and whether a row moved one knob or two. Both are pure and pinned here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_ann_frontier.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("score_ann_frontier", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def frontier() -> Any:
    return _load()


# --- recall against the exact answer ----------------------------------------------------------


def test_recall_is_one_when_the_store_returns_the_exact_set(frontier: Any) -> None:
    reference = {"q": ["a", "b", "c"]}

    assert frontier.recall_against(reference, {"q": ["a", "b", "c"]}, 3) == 1.0


def test_recall_ignores_the_order_within_the_top_k(frontier: Any) -> None:
    """Recall counts membership; ranking quality is nDCG's job and is reported separately.

    Conflating them would make a store that returns the right documents in a different order look
    like it had missed them.
    """
    reference = {"q": ["a", "b", "c"]}

    assert frontier.recall_against(reference, {"q": ["c", "a", "b"]}, 3) == 1.0


def test_recall_counts_the_missing_share(frontier: Any) -> None:
    reference = {"q": ["a", "b", "c", "d"]}

    assert frontier.recall_against(reference, {"q": ["a", "b", "x", "y"]}, 4) == 0.5


def test_recall_is_averaged_over_queries_not_over_documents(frontier: Any) -> None:
    """A query with many relevant documents must not outvote one with few.

    Pooling documents instead would let a single easy query with a long answer set carry the
    figure for the whole run.
    """
    reference = {"easy": ["a", "b", "c", "d"], "hard": ["z"]}
    candidate = {"easy": ["a", "b", "c", "d"], "hard": []}

    assert frontier.recall_against(reference, candidate, 4) == 0.5


def test_a_query_the_store_never_answered_counts_as_zero_not_as_absent(frontier: Any) -> None:
    """Skipping it would let a store that returned nothing at all score a perfect recall."""
    reference = {"answered": ["a"], "dropped": ["b"]}

    assert frontier.recall_against(reference, {"answered": ["a"]}, 1) == 0.5


def test_recall_is_capped_at_k_on_both_sides(frontier: Any) -> None:
    """Over-fetching past k must not be scored, or a store could buy recall with a longer list."""
    reference = {"q": ["a", "b", "c"]}

    # abs tolerance because the value is rounded to 4 decimals on its way into the JSON
    assert frontier.recall_against(reference, {"q": ["x", "y", "a", "b", "c"]}, 3) == pytest.approx(1 / 3, abs=5e-5)


def test_a_reference_query_with_no_documents_is_skipped_rather_than_scored_zero(frontier: Any) -> None:
    """Nothing to find means nothing to miss; scoring it zero would punish a perfect store."""
    reference = {"empty": [], "real": ["a"]}

    assert frontier.recall_against(reference, {"real": ["a"]}, 3) == 1.0


# --- the knob grids -----------------------------------------------------------------------------


def test_every_grid_measures_the_driver_default_explicitly(frontier: Any) -> None:
    """The default is the setting every previously published number was taken at.

    Without it in the grid the frontier would have no reference point, and the improvement a knob
    buys could not be stated against what the store was actually doing before.
    """
    from semdex.domain.ann_tuning import AnnParams

    for grid in (frontier._QUERY_GRID, frontier._BUILD_GRID):
        for backend, levels in grid.items():
            assert AnnParams() in levels, f"{backend} has no driver-default level"


def test_query_and_build_knobs_are_kept_on_separate_axes(frontier: Any) -> None:
    """A query grid holding a build knob (or the reverse) would move two things in one row."""
    query_only = {"nprobes", "refine_factor", "ef_search"}
    build_only = {"m", "ef_construction", "num_partitions"}

    for levels in frontier._QUERY_GRID.values():
        for params in levels:
            assert not {f for f in build_only if getattr(params, f) is not None}
    for levels in frontier._BUILD_GRID.values():
        for params in levels:
            assert not {f for f in query_only if getattr(params, f) is not None}


def test_a_store_is_only_swept_on_knobs_it_implements(frontier: Any) -> None:
    """mariadb has no ef_construction and lancedb is IVF-PQ, not HNSW.

    Sweeping a parameter the engine ignores publishes a flat line that reads as "this does not
    help here" rather than "this was never applied" - the dead-knob failure this whole sweep was
    blocked on.
    """
    from semdex.domain.enums import StoreBackend

    maria_fields = {
        f
        for level in frontier._BUILD_GRID[StoreBackend.MARIADB]
        for f in ("ef_construction",)
        if getattr(level, f) is not None
    }
    assert not maria_fields, "mariadb has no ef_construction equivalent"

    lance_fields = {
        f
        for level in frontier._BUILD_GRID[StoreBackend.LANCEDB]
        for f in ("m", "ef_construction")
        if getattr(level, f) is not None
    }
    assert not lance_fields, "lancedb builds an IVF-PQ index, not HNSW"

    for backend in (StoreBackend.PGVECTOR, StoreBackend.MARIADB):
        for level in frontier._QUERY_GRID[backend]:
            assert level.nprobes is None and level.refine_factor is None


def test_merging_a_run_replaces_a_point_rather_than_duplicating_it(frontier: Any) -> None:
    """Re-measuring one knob setting must update it; two rows for one point would be averaged."""
    old = [
        {
            "corpus": "c",
            "embedding": "e",
            "store": "lancedb",
            "build_level": "b",
            "query_level": "q",
            "recall_vs_exact": 0.1,
        }
    ]
    new = [
        {
            "corpus": "c",
            "embedding": "e",
            "store": "lancedb",
            "build_level": "b",
            "query_level": "q",
            "recall_vs_exact": 0.9,
        }
    ]

    merged = frontier.merge_rows(old, new)

    assert len(merged) == 1
    assert merged[0]["recall_vs_exact"] == 0.9


def test_merging_keeps_points_the_new_run_did_not_measure(frontier: Any) -> None:
    old = [
        {
            "corpus": "c",
            "embedding": "e",
            "store": "pgvector",
            "build_level": "b",
            "query_level": "q1",
            "recall_vs_exact": 0.1,
        }
    ]
    new = [
        {
            "corpus": "c",
            "embedding": "e",
            "store": "pgvector",
            "build_level": "b",
            "query_level": "q2",
            "recall_vs_exact": 0.9,
        }
    ]

    assert len(frontier.merge_rows(old, new)) == 2


def test_the_knob_label_names_the_default_rather_than_printing_nothing(frontier: Any) -> None:
    """An empty label would render as a blank table cell and read as a missing measurement."""
    from semdex.domain.ann_tuning import AnnParams

    assert frontier._label(AnnParams()) == "driver default"
    assert frontier._label(AnnParams(nprobes=10, refine_factor=5)) == "nprobes=10 refine=5"


def test_build_and_query_knobs_combine_without_overwriting_each_other(frontier: Any) -> None:
    """The factory takes one object, so the merge must not drop either axis."""
    from semdex.domain.ann_tuning import AnnParams

    build = AnnParams(m=32, ef_construction=200, num_partitions=64)
    query = AnnParams(nprobes=20, refine_factor=5, ef_search=100)

    merged = frontier._merge(build, query)

    # every field of both sides, so dropping any one of them fails here rather than showing up
    # later as a knob level that silently measured the default
    assert (merged.m, merged.ef_construction, merged.num_partitions) == (32, 200, 64)
    assert (merged.nprobes, merged.refine_factor, merged.ef_search) == (20, 5, 100)
