"""Unit tests for the ANN-recall preset resolver (domain/ann_tuning.py).

The resolver maps a portable ``AnnRecall`` preset to each backend's native query-time params,
with optional raw overrides winning field-by-field. ``BALANCED`` means "driver default" for every
backend EXCEPT lancedb, whose driver default the recall/latency sweep measured as the worst point
on its own frontier; exact stores (json / sqlite_vec) never carry ANN params.
"""

from __future__ import annotations

import pytest

from semdex.domain.ann_tuning import AnnParams, resolve_ann_params
from semdex.domain.enums import AnnRecall, StoreBackend

pytestmark = pytest.mark.os_agnostic


def test_balanced_is_the_driver_default_except_where_that_default_is_bad() -> None:
    """BALANCED leaves each driver alone, which is non-breaking - lancedb is the one exception.

    Its own default keeps 0.567 of the exact top-10 and costs 0.0474 nDCG, so BALANCED sets a
    refine factor there instead (0.986 recall for 9 percent more latency). This test exists to
    make that carve-out deliberate: a backend silently drifting away from its driver default is a
    behaviour change users did not ask for, and every one of them belongs on this list.
    """
    for backend in StoreBackend:
        if backend is StoreBackend.LANCEDB:
            continue
        assert resolve_ann_params(backend, AnnRecall.BALANCED) == AnnParams()


def test_balanced_sets_a_refine_factor_on_lancedb() -> None:
    """The measured carve-out. FAST is the escape hatch back to the bare driver behaviour."""
    balanced = resolve_ann_params(StoreBackend.LANCEDB, AnnRecall.BALANCED)

    assert (balanced.nprobes, balanced.refine_factor) == (10, 5)
    assert resolve_ann_params(StoreBackend.LANCEDB, AnnRecall.FAST).refine_factor is None


def test_pgvector_and_mariadb_map_ef_search() -> None:
    for backend in (StoreBackend.PGVECTOR, StoreBackend.MARIADB):
        assert resolve_ann_params(backend, AnnRecall.FAST).ef_search == 20
        assert resolve_ann_params(backend, AnnRecall.ACCURATE).ef_search == 200
        # only ef_search moves; unrelated fields stay unset
        assert resolve_ann_params(backend, AnnRecall.ACCURATE).nprobes is None


def test_lancedb_presets_are_an_increasing_ladder() -> None:
    """Three rungs that measurably differ: 0.566, 0.986 and 0.998 recall on the reference cell.

    Before the sweep FAST and BALANCED were the same setting under two names.
    """
    assert resolve_ann_params(StoreBackend.LANCEDB, AnnRecall.FAST).nprobes == 10
    accurate = resolve_ann_params(StoreBackend.LANCEDB, AnnRecall.ACCURATE)
    assert accurate.nprobes == 40
    assert accurate.refine_factor == 10


def test_exact_stores_never_carry_ann_params() -> None:
    # json / sqlite_vec are brute-force exact; no ANN knob applies at any preset.
    for backend in (StoreBackend.JSON, StoreBackend.SQLITE_VEC):
        for recall in AnnRecall:
            assert resolve_ann_params(backend, recall) == AnnParams()


def test_raw_override_wins_field_by_field() -> None:
    # An explicit override beats the preset for that field, leaving the rest of the preset intact.
    resolved = resolve_ann_params(StoreBackend.PGVECTOR, AnnRecall.ACCURATE, AnnParams(ef_search=99, m=32))
    assert resolved.ef_search == 99  # override wins over the preset's 200
    assert resolved.m == 32  # build-time override carried through

    # An override left unset (None) does NOT clobber a preset value.
    only_m = resolve_ann_params(StoreBackend.LANCEDB, AnnRecall.ACCURATE, AnnParams(m=8))
    assert only_m.nprobes == 40  # preset value preserved
    assert only_m.m == 8


def test_ann_recall_is_str_enum() -> None:
    assert {r.value for r in AnnRecall} == {"fast", "balanced", "accurate"}
    assert AnnRecall.BALANCED == "balanced"  # StrEnum: compares/serializes as its value
