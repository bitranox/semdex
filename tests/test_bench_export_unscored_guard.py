"""The export must name every vector cell nobody scored, and fail on one unless it is withheld.

Four times a cell set has sat embedded in the vector cache with no score anywhere, while the
pages read as if the sweep were complete: 16 cells for four weeks, three of six embedders on
GerDaLIR, the qwen3 cap512 baselines, then 124 MLDR cells. Nothing counted them, because every
stage reports the effort it made (cells scored, rows exported), never the corpus it was handed.

So the export diffs the vector cache against the score files it publishes, on both axes, and
prints the unscored cells by name. One that exists is a failure, not a warning - unless a
committed ``withheld-cells.json`` beside the raw files lists it with a reason, which is how a
deliberate omission travels with the data instead of living in an operator's memory.
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

_CORPUS = "mldr_en_8k_slice"
_SCORED = f"{_CORPUS}__recursive-t256-o0-gpt2__ollama-bge-m3"
_PLANTED = f"{_CORPUS}__fast-t256-o51-gpt2__ollama-bge-m3"
_DENSE_ONLY = f"{_CORPUS}__recursive-t256-o0-gpt2__fastembed-bge-base"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("export_bench_raw", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_bench_raw"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exporter() -> Any:
    return _load()


def _row(cell: str, method: str | None = None) -> dict[str, Any]:
    corpus, profile, _ = cell.split("__")
    row: dict[str, Any] = {
        "corpus": corpus,
        "profile": profile,
        "embedding": "ollama:bge-m3",
        "dim": 1024,
        "n_queries": 100,
        "ndcg@10": 0.5,
    }
    if method:
        row["method"] = method
    return row


def _plant_vectors(cache_root: Path, *cells: str) -> None:
    for cell in cells:
        cell_dir = cache_root / "vectors" / cell
        cell_dir.mkdir(parents=True)
        (cell_dir / "meta.json").write_text(json.dumps({"cell": cell}))


def _write_scores(cache_root: Path, name: str, cells: dict[str, dict[str, Any]]) -> None:
    scores = cache_root / "scores"
    scores.mkdir(parents=True, exist_ok=True)
    (scores / name).write_text(json.dumps(cells))


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A vector cache holding one scored cell, one dense-only cell and one planted unscored one."""
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    _plant_vectors(tmp_path, _SCORED, _DENSE_ONLY, _PLANTED)
    _write_scores(tmp_path, "mldr_chunk_scores.json", {_SCORED: _row(_SCORED)})
    _write_scores(
        tmp_path,
        "hybrid_scores.json",
        {f"{_DENSE_ONLY}__dense": _row(_DENSE_ONLY, "dense"), f"{_DENSE_ONLY}__bm25": _row(_DENSE_ONLY, "bm25")},
    )
    return tmp_path


def test_unscored_cells_names_the_planted_cell_and_nothing_else(exporter: Any, cache: Path) -> None:
    """The census: vectors present, no score in any file that publishes the corpus."""
    census = exporter.unscored_cells()
    assert census == {_CORPUS: [_PLANTED]}, census


def test_a_hybrid_dense_arm_counts_as_a_score(exporter: Any, cache: Path) -> None:
    """The export stands a dense arm in for a plain cell, so the census must not report it twice."""
    census = exporter.unscored_cells()
    assert _DENSE_ONLY not in census.get(_CORPUS, []), census


def test_a_vector_cell_of_a_corpus_no_export_publishes_is_not_the_exporter_s_business(
    exporter: Any, cache: Path
) -> None:
    """msmarco cells feed the scale bench; no chunk export publishes them, so they are not unscored."""
    _plant_vectors(cache, "msmarco_50000__recursive-t256-o0-gpt2__fastembed-bge-small")
    census = exporter.unscored_cells()
    assert all(not cell.startswith("msmarco") for cells in census.values() for cell in cells), census


def test_main_fails_naming_the_unscored_cell(
    exporter: Any, cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The failure is by name, so the operator scores that cell rather than hunting for it."""
    out_dir = tmp_path / "raw"
    monkeypatch.setenv("OUT_DIR", str(out_dir))
    with pytest.raises(SystemExit) as excinfo:
        exporter.main()
    assert _PLANTED in str(excinfo.value), excinfo.value
    assert _PLANTED in capsys.readouterr().out


def test_a_withheld_cell_with_a_reason_does_not_fail_the_export(
    exporter: Any, cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A deliberate omission is recorded beside the data, with why, and is still printed."""
    out_dir = tmp_path / "raw"
    out_dir.mkdir()
    monkeypatch.setenv("OUT_DIR", str(out_dir))
    (out_dir / "withheld-cells.json").write_text(json.dumps({_PLANTED: "GPU queue; scored next week"}))
    exporter.main()
    out = capsys.readouterr().out
    assert _PLANTED in out and "GPU queue" in out, out


def test_a_withheld_entry_without_a_reason_is_refused(
    exporter: Any, cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty reason is the operator's memory again, which is what this file exists to replace."""
    out_dir = tmp_path / "raw"
    out_dir.mkdir()
    monkeypatch.setenv("OUT_DIR", str(out_dir))
    (out_dir / "withheld-cells.json").write_text(json.dumps({_PLANTED: ""}))
    with pytest.raises(SystemExit) as excinfo:
        exporter.main()
    assert "reason" in str(excinfo.value), excinfo.value
