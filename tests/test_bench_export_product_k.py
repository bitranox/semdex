"""The product-k export: flat rows per cell and rung, the corpus allowlist, rounding, carried provenance."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "export_bench_raw.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("export_bench_raw_product_k", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exporter() -> Any:
    return _load()


def _rung(k: int, *, product: bool, budgets: list[int]) -> dict[str, Any]:
    return {
        "k": k,
        "budgets": budgets,
        "product": product,
        "ndcg_delivered": 0.512345678,
        "ndcg_delivered_ci_lo": 0.5,
        "ndcg_delivered_ci_hi": 0.52,
        "ndcg_delivered_sd": 0.1,
        "ndcg_documents": 0.6,
        "ndcg_documents_ci_lo": 0.59,
        "ndcg_documents_ci_hi": 0.61,
        "ndcg_documents_sd": 0.1,
        "distinct_docs": 3.25,
        "distinct_docs_ci_lo": 3.1,
        "distinct_docs_ci_hi": 3.4,
        "distinct_docs_sd": 0.5,
        "gap_mean_delta": 0.087654321,
        "gap_ci_lo": 0.08,
        "gap_ci_hi": 0.095,
        "gap_wins": 300,
        "gap_losses": 100,
        "gap_ties": 400,
        "gap_n_shared": 800,
        "gap_resolved": True,
    }


def _cell(corpus: str, profile: str = "recursive-t256-o0-gpt2") -> dict[str, Any]:
    return {
        "corpus": corpus,
        "profile": profile,
        "embedding": "fastembed:bge-base",
        "dim": 768,
        "n_queries": 800,
        "product_k": 5,
        "fetch": 100,
        "rungs": [_rung(5, product=True, budgets=[1280]), _rung(10, product=False, budgets=[2560])],
        "perquery_sha256": "ab" * 32,
        "measured_on": {"host": "bench-box", "measured_utc": "2026-09-30T00:00:00Z"},
    }


def test_the_payload_flattens_rungs_rounds_them_parses_axes_and_carries_the_stamp(exporter: Any) -> None:
    cells = {"mldr_en_8k_slice__recursive-t256-o0-gpt2__fastembed-bge-base": _cell("mldr_en_8k_slice")}
    payload = exporter.product_k_payload(cells)
    assert payload["sources"] == ["product_k_scores.json"]
    assert payload["corpora"] == list(exporter._PRODUCT_K_CORPORA)
    assert payload["product_k"] == 5 and payload["budgets"] == [1280, 2560]
    assert "delivered" in payload["note"] and "dedup" in payload["note"]
    assert [r["k"] for r in payload["rows"]] == [5, 10], "one flat row per rung, in k order"
    row = payload["rows"][0]
    assert row["cell"] == "mldr_en_8k_slice__recursive-t256-o0-gpt2__fastembed-bge-base"
    assert row["axes"]["max_tokens"] == 256 and row["axes"]["strategy"] == "recursive"
    assert row["product"] is True and row["budgets"] == [1280]
    assert row["ndcg_delivered"] == 0.5123
    assert row["gap_mean_delta"] == 0.0877
    assert row["gap_wins"] == 300
    assert row["measured_on"] == {"host": "bench-box", "measured_utc": "2026-09-30T00:00:00Z"}
    assert "rungs" not in row and "fetch" not in row
    assert payload["measured_on"]["recorded"] == 2, "every flat row carries the cell's stamp"
    assert payload["summary"] == {
        "cells": 1,
        "rows": 2,
        "corpora": ["mldr_en_8k_slice"],
        "profiles": ["recursive-t256-o0-gpt2"],
        "embeddings": ["fastembed:bge-base"],
    }


def test_a_foreign_corpus_is_refused_by_name(exporter: Any) -> None:
    cells = {"nfcorpus__recursive-t256-o0-gpt2__fastembed-bge-base": _cell("nfcorpus")}
    with pytest.raises(ValueError, match=r"product-k\.json declares corpora .*nfcorpus"):
        exporter.product_k_payload(cells)


def test_the_writer_skips_a_missing_source_and_writes_the_file_when_present(
    exporter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path / "cache"))
    out = tmp_path / "raw"
    out.mkdir()
    exporter._write_product_k(out)
    assert not (out / "product-k.json").exists()
    assert "product-k.json: source missing" in capsys.readouterr().out
    scores = tmp_path / "cache" / "scores"
    scores.mkdir(parents=True)
    cell_key = "gerdalir_de_12k_slice__recursive-t256-o0-gpt2__fastembed-bge-base"
    (scores / "product_k_scores.json").write_text(json.dumps({cell_key: _cell("gerdalir_de_12k_slice")}))
    exporter._write_product_k(out)
    written = json.loads((out / "product-k.json").read_text())
    assert written["summary"] == {
        "cells": 1,
        "rows": 2,
        "corpora": ["gerdalir_de_12k_slice"],
        "profiles": ["recursive-t256-o0-gpt2"],
        "embeddings": ["fastembed:bge-base"],
    }
    assert (out / "product-k.json").read_text().endswith("\n")
