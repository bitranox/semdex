"""Provenance must describe the machine that MEASURED a cell, not the one that exported it.

``export_bench_raw.py`` says in its own docstring that it "stamps provenance (host, git sha, numpy
and BLAS build, thread count), because a latency or throughput number without the box it ran on is
not a measurement". It then collected all of it at EXPORT time, so every committed file described
whatever machine last ran the exporter.

The numbers it describes are weeks older than that. ``/embeddings/scores/store_quality.json`` was
last written on 10 August; the committed ``store-quality.json`` carried a BLAS build and a git sha
from an export run on 2 September, attached to ``search_p50_ms``, ``search_p95_ms`` and
``upsert_s``. A reader comparing those latencies against a later run would be holding the wrong
box entirely.

A single file-level stamp cannot even represent the truth here. Cells inside one file were
measured on different machines: the lancedb rows cannot have run on the Ivy-Bridge cluster nodes
at all, because that wheel needs AVX2. So provenance is per cell, recorded when the cell is
measured, and carried through the export untouched.

Two consequences this pins down:

* the exporting machine must not appear in the file at all, so a re-export of unchanged sources
  produces byte-identical output instead of a provenance-only diff to revert by hand;
* a cell measured before any of this was recorded is reported as NOT RECORDED. There is no
  honest way to recover the box it ran on, and substituting today's is the defect, not the fix.
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

# The keys that describe a machine. None of them may be collected at export time.
_MACHINE_KEYS = ("host", "cpu", "python", "numpy", "blas", "openblas_num_threads")


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


def _write_source(cache_root: Path, name: str, cells: dict[str, dict[str, Any]]) -> None:
    scores = cache_root / "scores"
    scores.mkdir(parents=True, exist_ok=True)
    (scores / name).write_text(json.dumps(cells))


def _measured_on(host: str, *, when: str = "2026-08-10T02:12:00+00:00") -> dict[str, Any]:
    """What a producer records at the moment it measures a cell."""
    return {
        "measured_utc": when,
        "semdex_git_sha": "aaaaaaa",
        "host": host,
        "cpu": "x86_64",
        "python": "3.14.0",
        "numpy": "2.3.1",
        "blas": "scipy-openblas 0.3.33.112.0",
        "openblas_num_threads": "unset",
    }


def _cell(corpus: str, measured_on: dict[str, Any] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "corpus": corpus,
        "profile": "recursive-t256-o0-gpt2",
        "embedding": "ollama:bge-m3",
        "dim": 1024,
        "n_queries": 100,
        "ndcg@10": 0.5,
        "search_p50_ms": 4.4,
    }
    if measured_on is not None:
        row["measured_on"] = measured_on
    return row


def _spec(*, corpora: list[str]) -> dict[str, Any]:
    return {
        "sources": ["src.json"],
        "note": "n",
        "fit_for_chunk_claims": True,
        "corpora": corpora,
    }


def _build(exporter: Any, tmp_path: Path, cells: dict[str, dict[str, Any]], corpora: list[str]) -> dict[str, Any]:
    _write_source(tmp_path, "src.json", cells)
    payload = exporter.build_export("chunk-sweep-mldr.json", _spec(corpora=corpora))
    assert payload is not None
    return payload


def test_the_exporting_machine_is_not_stamped_onto_the_file(
    exporter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No key at file level may describe the box that happened to run the export."""
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    payload = _build(
        exporter,
        tmp_path,
        {"mldr_de_3k_slice__recursive-t256-o0-gpt2__ollama-bge-m3": _cell("mldr_de_3k_slice")},
        ["mldr_de_3k_slice"],
    )
    leaked = [key for key in _MACHINE_KEYS if key in payload]
    assert leaked == [], f"export-time machine facts stamped onto the file: {leaked}"


def test_the_export_run_does_not_date_or_sign_the_file(
    exporter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``generated_utc`` and the exporting repo's sha are what re-churned every unchanged file.

    Both are facts about the commit that adds the file, which the file cannot honestly state, and
    git records both already.
    """
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    payload = _build(
        exporter,
        tmp_path,
        {"mldr_de_3k_slice__recursive-t256-o0-gpt2__ollama-bge-m3": _cell("mldr_de_3k_slice")},
        ["mldr_de_3k_slice"],
    )
    assert "generated_utc" not in payload
    assert "semdex_git_sha" not in payload
    assert "cache_root" not in payload


def test_the_exporting_environment_cannot_change_the_bytes(
    exporter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same sources, different exporting environment, identical output.

    This is the churn itself, expressed so it fails deterministically rather than only when two
    runs straddle a second boundary: ``OPENBLAS_NUM_THREADS`` is read straight out of the export
    process's environment, so flipping it changed the committed file while no measurement moved.
    """
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    cells = {"mldr_de_3k_slice__recursive-t256-o0-gpt2__ollama-bge-m3": _cell("mldr_de_3k_slice")}

    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    first = json.dumps(_build(exporter, tmp_path, cells, ["mldr_de_3k_slice"]), sort_keys=True)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "16")
    second = json.dumps(_build(exporter, tmp_path, cells, ["mldr_de_3k_slice"]), sort_keys=True)

    assert first == second, "the exporting environment leaked into the committed bytes"


def test_a_cell_carries_the_provenance_recorded_when_it_was_measured(
    exporter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stamp a producer wrote at measurement time reaches the committed row unaltered."""
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    stamp = _measured_on("px-semdex-test-embeddings")
    payload = _build(
        exporter,
        tmp_path,
        {"mldr_de_3k_slice__recursive-t256-o0-gpt2__ollama-bge-m3": _cell("mldr_de_3k_slice", stamp)},
        ["mldr_de_3k_slice"],
    )
    assert payload["cells"][0]["measured_on"] == stamp


def test_a_cell_measured_before_provenance_existed_is_reported_as_not_recorded(
    exporter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every committed file today holds such cells. None of them may borrow today's machine."""
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    payload = _build(
        exporter,
        tmp_path,
        {"mldr_de_3k_slice__recursive-t256-o0-gpt2__ollama-bge-m3": _cell("mldr_de_3k_slice")},
        ["mldr_de_3k_slice"],
    )
    assert "measured_on" not in payload["cells"][0]
    assert payload["measured_on"]["recorded"] == 0
    assert payload["measured_on"]["not_recorded"] == 1
    assert payload["measured_on"]["runs"] == []


def test_the_file_reports_every_distinct_run_its_cells_came_from(
    exporter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case a single file-level stamp cannot represent even in principle.

    lancedb needs AVX2 and the Ivy-Bridge cluster nodes have none, so a store-quality file holds
    rows that provably ran on different machines. One stamp has to misdescribe some of them.
    """
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    gpu = _measured_on("px-semdex-test-embeddings", when="2026-08-10T02:12:00+00:00")
    dev = _measured_on("lxc-pydev", when="2026-08-31T03:04:00+00:00")
    payload = _build(
        exporter,
        tmp_path,
        {
            "mldr_de_3k_slice__recursive-t256-o0-gpt2__ollama-bge-m3": _cell("mldr_de_3k_slice", gpu),
            "mldr_en_8k_slice__recursive-t256-o0-gpt2__ollama-bge-m3": _cell("mldr_en_8k_slice", dev),
        },
        ["mldr_de_3k_slice", "mldr_en_8k_slice"],
    )
    summary = payload["measured_on"]
    assert summary["recorded"] == 2
    assert summary["not_recorded"] == 0
    # Chronological, so the runs read as the timeline they are: the GPU box measured first.
    assert [run["host"] for run in summary["runs"]] == ["px-semdex-test-embeddings", "lxc-pydev"]


def test_identical_runs_collapse_to_one_entry(exporter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A sweep measuring 400 cells in one run must not print that run 400 times."""
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    stamp = _measured_on("lxc-pydev")
    payload = _build(
        exporter,
        tmp_path,
        {
            "mldr_de_3k_slice__recursive-t256-o0-gpt2__ollama-bge-m3": _cell("mldr_de_3k_slice", stamp),
            "mldr_en_8k_slice__recursive-t256-o0-gpt2__ollama-bge-m3": _cell("mldr_en_8k_slice", dict(stamp)),
        },
        ["mldr_de_3k_slice", "mldr_en_8k_slice"],
    )
    assert payload["measured_on"]["runs"] == [stamp]
    assert payload["measured_on"]["recorded"] == 2
