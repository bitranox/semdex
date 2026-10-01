"""A cell the audits have judged unusable must not reach the published files.

Two audits already produce the judgement and nothing consumed it: the chunk-dimension audit
names recursive overlap sets whose count moved (a size-guard re-cut them into crumbs), and the
embedder-truncation audit names (chunk set, embedder) pairs where the embedder silently clipped
a share of the text. Four cap512 overlap sets and the whitespace cells on German (92 percent of
chunks clipped, a quarter of the token mass unseen) were published as verdicts. The export now
drops such cells, says so in the file, and keeps the judgement with the numbers.
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
    spec = importlib.util.spec_from_file_location("export_bench_raw_void", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_bench_raw_void"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exporter() -> Any:
    return _load()


def _row(corpus: str, profile: str, embedding: str) -> dict[str, Any]:
    return {"cell": f"{corpus}__{profile}__{embedding}", "corpus": corpus, "profile": profile, "embedding": embedding}


def _raw_dir(tmp_path: Path, *, moved: list[dict[str, Any]], clipped: list[dict[str, Any]]) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "chunk-dimension-audit.json").write_text(
        json.dumps({"chunk_sets": [], "integrity": {"moved_boundaries_under_overlap": moved}})
    )
    (raw / "embedder-truncation-audit.json").write_text(json.dumps({"rows": clipped}))
    return raw


def test_a_cell_on_a_re_cut_chunk_set_is_voided_with_the_reason(exporter: Any, tmp_path: Path) -> None:
    raw = _raw_dir(
        tmp_path,
        moved=[{"corpus": "mldr_en", "profile": "recursive-t512-o10-gpt2", "rows": 78808, "rows_at_overlap_0": 68090}],
        clipped=[],
    )
    voids = exporter.load_voids(raw)
    kept, voided = exporter.void_cells(
        [
            _row("mldr_en", "recursive-t512-o10-gpt2", "fastembed:bge-base"),
            _row("mldr_en", "recursive-t512-o0-gpt2", "fastembed:bge-base"),
        ],
        voids,
    )
    assert [r["profile"] for r in kept] == ["recursive-t512-o0-gpt2"]
    assert voided[0]["cell"] == "mldr_en__recursive-t512-o10-gpt2__fastembed:bge-base"
    assert "re-cut" in voided[0]["reason"] and "68090" in voided[0]["reason"]


def test_a_cell_the_embedder_clipped_is_voided_and_a_lightly_clipped_one_is_kept(exporter: Any, tmp_path: Path) -> None:
    raw = _raw_dir(
        tmp_path,
        moved=[],
        clipped=[
            {
                "chunk_set": "gerdalir__whitespace-t256-o0-gpt2",
                "embedder": "fastembed:bge-base",
                "token_mass_lost_pct": 24.35,
            },
            {
                "chunk_set": "gerdalir__recursive-t256-o256-gpt2",
                "embedder": "fastembed:bge-base",
                "token_mass_lost_pct": 0.0007,
            },
        ],
    )
    voids = exporter.load_voids(raw)
    kept, voided = exporter.void_cells(
        [
            _row("gerdalir", "whitespace-t256-o0-gpt2", "fastembed:bge-base"),
            _row("gerdalir", "recursive-t256-o256-gpt2", "fastembed:bge-base"),
            _row("gerdalir", "whitespace-t256-o0-gpt2", "ollama:bge-m3"),  # not audited for this embedder
        ],
        voids,
    )
    assert [r["cell"] for r in voided] == ["gerdalir__whitespace-t256-o0-gpt2__fastembed:bge-base"]
    assert "24.35" in voided[0]["reason"]
    assert len(kept) == 2


def test_missing_audits_void_nothing_but_are_reported(
    exporter: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    voids = exporter.load_voids(tmp_path / "nowhere")
    kept, voided = exporter.void_cells([_row("c", "recursive-t512-o10-gpt2", "e")], voids)
    assert len(kept) == 1 and voided == []
    assert "chunk-dimension-audit.json" in capsys.readouterr().err


def test_build_export_drops_voided_cells_and_lists_them(
    exporter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    scores = tmp_path / "scores"
    scores.mkdir()
    cells = {
        "mldr_en_8k_slice__recursive-t512-o10-gpt2__fastembed-bge-base": {
            "corpus": "mldr_en_8k_slice",
            "profile": "recursive-t512-o10-gpt2",
            "embedding": "fastembed:bge-base",
            "ndcg@10": 0.6,
        },
        "mldr_en_8k_slice__recursive-t512-o0-gpt2__fastembed-bge-base": {
            "corpus": "mldr_en_8k_slice",
            "profile": "recursive-t512-o0-gpt2",
            "embedding": "fastembed:bge-base",
            "ndcg@10": 0.7,
        },
    }
    (scores / "src.json").write_text(json.dumps(cells))
    raw = _raw_dir(
        tmp_path,
        moved=[
            {
                "corpus": "mldr_en_8k_slice",
                "profile": "recursive-t512-o10-gpt2",
                "rows": 78808,
                "rows_at_overlap_0": 68090,
            }
        ],
        clipped=[],
    )
    spec = {"sources": ["src.json"], "note": "n", "fit_for_chunk_claims": True, "corpora": ["mldr_en_8k_slice"]}
    payload = exporter.build_export("chunk-sweep-mldr.json", spec, voids=exporter.load_voids(raw))
    assert payload["summary"]["cells"] == 1
    assert payload["summary"]["profiles"] == ["recursive-t512-o0-gpt2"]
    assert [v["cell"] for v in payload["voided"]] == ["mldr_en_8k_slice__recursive-t512-o10-gpt2__fastembed-bge-base"]
