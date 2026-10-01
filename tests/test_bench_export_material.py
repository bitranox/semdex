"""The exporter stamps every paired comparison with whether it is MATERIAL, not only resolved.

A 12,298-query corpus resolves a step of 0.0009 nDCG@10. Resolved says the query set can tell
the two configurations apart; material says the difference is worth acting on. The floor and
the alpha live in _score_stats and are written into the file so a table can print what the rows
were judged against, rather than restating the numbers.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "export_bench_raw.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("export_bench_raw_material", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_bench_raw_material"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exporter() -> Any:
    return _load()


def _cell(overlap: int) -> dict[str, Any]:
    axes = {"strategy": "recursive", "max_tokens": 256, "overlap_tokens": overlap, "breakpoint_model": None}
    return {"cell": f"c__recursive-t256-o{overlap}-gpt2__e", "corpus": "c", "embedding": "e", "axes": axes}


def _write_per_query(directory: Path, cell: str, scores: list[float], exporter: Any) -> None:
    np = exporter.np
    qids = np.array([f"q{i}" for i in range(len(scores))])
    np.savez(directory / f"{cell}.npz", qids=qids, **{exporter.NPZ_KEYS["ndcg@10"]: np.array(scores)})


def _effects_for_shift(exporter: Any, directory: Path, shift: float) -> list[dict[str, Any]]:
    """Two cells whose per-query scores differ by exactly ``shift`` on every query.

    A constant shift gives a degenerate bootstrap interval at the shift itself, so the pair is
    resolved whatever the shift's size; only the floor can tell 0.002 from 0.020.
    """
    base = [0.1 * (i % 7) for i in range(24)]
    cells = [_cell(0), _cell(51)]
    _write_per_query(directory, cells[0]["cell"], base, exporter)
    _write_per_query(directory, cells[1]["cell"], [v + shift for v in base], exporter)
    return exporter.knob_effects(cells)


def test_a_resolved_step_under_the_floor_is_stamped_immaterial(
    exporter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEMDEX_SCORE_PERQUERY", str(tmp_path))
    (effect,) = _effects_for_shift(exporter, tmp_path, 0.002)
    assert effect["resolved"] is True
    assert effect["material"] is False


def test_a_resolved_step_at_or_above_the_floor_is_stamped_material(
    exporter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEMDEX_SCORE_PERQUERY", str(tmp_path))
    (effect,) = _effects_for_shift(exporter, tmp_path, 0.020)
    assert effect["resolved"] is True
    assert effect["material"] is True


def test_a_shift_rounding_up_to_the_floor_is_stamped_material(
    exporter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """0.00498 is BELOW the floor at full precision but rounds to 0.0050 in the published row.

    The stamp must be judged on the same rounded value the row publishes, so a reader can
    reproduce ``material`` from ``mean_delta`` and ``material_floor`` without re-deriving the
    unrounded delta, which the file does not carry.
    """
    monkeypatch.setenv("SEMDEX_SCORE_PERQUERY", str(tmp_path))
    (effect,) = _effects_for_shift(exporter, tmp_path, 0.00498)
    assert effect["resolved"] is True
    assert abs(effect["mean_delta"]) == 0.005
    assert effect["material"] is True


def test_a_shift_rounding_down_short_of_the_floor_is_stamped_immaterial(
    exporter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """0.00494 is the mirror of the 0.00498 case just below the rounding edge: it rounds to 0.0049.

    The published row would print +0.0049, itself under the 0.005 floor, so this one must stay
    immaterial rather than round up the way 0.00498 rounds up to 0.0050.
    """
    monkeypatch.setenv("SEMDEX_SCORE_PERQUERY", str(tmp_path))
    (effect,) = _effects_for_shift(exporter, tmp_path, 0.00494)
    assert effect["resolved"] is True
    assert abs(effect["mean_delta"]) == 0.0049
    assert effect["material"] is False


def test_an_unresolved_comparison_is_never_material(exporter: Any) -> None:
    paired = {"mean_delta": 0.5, "resolved": False}
    assert exporter.is_material(paired) is False


def test_the_effects_file_records_alpha_and_the_floor_it_was_judged_against(
    exporter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    perquery = tmp_path / "perquery"
    perquery.mkdir()
    monkeypatch.setenv("SEMDEX_SCORE_PERQUERY", str(perquery))
    out_dir = tmp_path / "raw"
    out_dir.mkdir()
    cells = [_cell(0), _cell(51)]
    base = [0.1 * (i % 7) for i in range(24)]
    _write_per_query(perquery, cells[0]["cell"], base, exporter)
    _write_per_query(perquery, cells[1]["cell"], [v + 0.02 for v in base], exporter)
    (out_dir / next(iter(exporter._EXPORTS))).write_text(json.dumps({"cells": cells}))
    exporter._write_knob_effects(out_dir)
    payload = json.loads((out_dir / "chunk-knob-effects.json").read_text())
    assert payload["alpha"] == 0.05
    assert payload["material_floor"] == {"metric": "ndcg@10", "value": 0.005}
    assert [e["material"] for e in payload["effects"]] == [True]
