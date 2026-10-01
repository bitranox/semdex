"""The product-k tables: the k=5 verdict per strategy and embedder, and the budget ladder per cap."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "gen_bench_tables.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("gen_bench_tables_product_k", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator() -> Any:
    return _load()


def _row(
    corpus: str,
    profile: str,
    embedding: str,
    *,
    k: int,
    product: bool,
    budgets: list[int],
    delivered: float,
    documents: float,
) -> dict[str, Any]:
    strategy = profile.split("-", maxsplit=1)[0]
    cap = int(profile.split("-t")[1].split("-", maxsplit=1)[0])
    breakpoint_model = profile.rsplit("-bp", 1)[1] if "-bp" in profile else None
    return {
        "cell": f"{corpus}__{profile}__{embedding.replace(':', '-')}",
        "corpus": corpus,
        "profile": profile,
        "embedding": embedding,
        "axes": {"strategy": strategy, "max_tokens": cap, "overlap_tokens": 0, "breakpoint_model": breakpoint_model},
        "product_k": 5,
        "k": k,
        "budgets": budgets,
        "product": product,
        "ndcg_delivered": delivered,
        "ndcg_documents": documents,
        "distinct_docs": 3.2,
        "gap_mean_delta": documents - delivered,
        "gap_ci_lo": documents - delivered - 0.01,
        "gap_ci_hi": documents - delivered + 0.01,
        "gap_resolved": True,
        "gap_wins": 300,
        "gap_losses": 100,
    }


def _doc() -> dict[str, Any]:
    en, de, bge = "mldr_en_8k_slice", "gerdalir_de_12k_slice", "fastembed:bge-base"
    return {
        "product_k": 5,
        "budgets": [1280, 2560],
        "rows": [
            _row(
                en,
                "recursive-t256-o0-gpt2",
                bge,
                k=5,
                product=True,
                budgets=[1280],
                delivered=0.50,
                documents=0.55,
            ),
            _row(
                en,
                "recursive-t256-o0-gpt2",
                bge,
                k=10,
                product=False,
                budgets=[2560],
                delivered=0.53,
                documents=0.58,
            ),
            _row(
                en,
                "semantic-t256-o0-gpt2-bpbge-m3",
                bge,
                k=5,
                product=True,
                budgets=[1280],
                delivered=0.48,
                documents=0.49,
            ),
            _row(
                en,
                "semantic-t256-o0-gpt2-bpbge-m3",
                bge,
                k=10,
                product=False,
                budgets=[2560],
                delivered=0.50,
                documents=0.52,
            ),
            _row(
                en,
                "recursive-t64-o0-gpt2",
                bge,
                k=5,
                product=True,
                budgets=[],
                delivered=0.40,
                documents=0.45,
            ),
            _row(
                en,
                "recursive-t64-o0-gpt2",
                bge,
                k=20,
                product=False,
                budgets=[1280],
                delivered=0.47,
                documents=0.52,
            ),
            _row(
                en,
                "recursive-t64-o0-gpt2",
                bge,
                k=40,
                product=False,
                budgets=[2560],
                delivered=0.49,
                documents=0.54,
            ),
            _row(
                de,
                "recursive-t256-o0-gpt2",
                bge,
                k=5,
                product=True,
                budgets=[1280],
                delivered=0.30,
                documents=0.31,
            ),
            _row(
                de,
                "recursive-t256-o0-gpt2",
                bge,
                k=10,
                product=False,
                budgets=[2560],
                delivered=0.32,
                documents=0.33,
            ),
        ],
    }


def test_the_verdict_table_has_one_row_per_cap_256_cell_and_names_the_breakpoint(generator: Any) -> None:
    tables: dict[str, Any] = {}
    generator._register_product_k_tables(tables, _doc())
    table = tables["product_k_verdict"]
    assert table["columns"] == [
        "Corpus",
        "Strategy",
        "Embedder",
        "Documents nDCG@5",
        "Delivered nDCG@5",
        "Cost of no dedup",
        "95% CI",
        "Distinct docs in 5",
    ]
    assert [row[0] for row in table["rows"]] == ["gerdalir-de", "mldr-en", "mldr-en"]
    assert table["rows"][1] == [
        "mldr-en",
        "`recursive`",
        "`fastembed:bge-base`",
        "0.5500",
        "0.5000",
        "+0.0500 resolved",
        "[+0.0400, +0.0600]",
        "3.20",
    ]
    assert table["rows"][2][1] == "`semantic` bp bge-m3"
    assert "zero gain" in table["note"]


def test_the_budget_table_has_one_row_per_recursive_cap_and_names_the_missing_rungs(generator: Any) -> None:
    tables: dict[str, Any] = {}
    generator._register_product_k_tables(tables, _doc())
    table = tables["product_k_budget"]
    assert table["columns"] == [
        "Corpus",
        "Embedder",
        "Cap",
        "k at 1280",
        "Delivered nDCG@k, 1280",
        "k at 2560",
        "Delivered nDCG@k, 2560",
    ]
    assert table["rows"] == [
        ["gerdalir-de", "`fastembed:bge-base`", "256", "5", "0.3000", "10", "0.3200"],
        ["mldr-en", "`fastembed:bge-base`", "64", "20", "0.4700", "40", "0.4900"],
        ["mldr-en", "`fastembed:bge-base`", "256", "5", "0.5000", "10", "0.5300"],
    ]
    assert "gerdalir-de: 64, 128, 512" in table["note"] and "mldr-en: 128, 512" in table["note"]


def test_no_rows_registers_no_table(generator: Any) -> None:
    tables: dict[str, Any] = {}
    generator._register_product_k_tables(tables, {"product_k": 5, "budgets": [1280, 2560], "rows": []})
    assert "product_k_verdict" not in tables and "product_k_budget" not in tables
