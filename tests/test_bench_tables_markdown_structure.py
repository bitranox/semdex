"""The markdown-structure table renders from its raw file with one row per strategy and embedder."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "gen_bench_tables.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("gen_bench_tables_mdstruct", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator() -> Any:
    return _load()


def _effect(strategy: str, delta: float, resolved: bool) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "embedding": "fastembed:bge-base",
        "mean_delta": delta,
        "ci_lo": delta - 0.01,
        "ci_hi": delta + 0.01,
        "wins": 300,
        "losses": 200,
        "ties": 300,
        "n_shared": 800,
        "resolved": resolved,
        "marked_rows": 152000,
        "unmarked_rows": 149082,
        "marked_token_p50": 205,
        "unmarked_token_p50": 208,
        "marked_heading_start_share": 0.2712,
    }


def test_the_table_has_one_row_per_pair_and_prints_the_chunk_facts(generator: Any) -> None:
    tables: dict[str, Any] = {}
    doc = {"effects": [_effect("markdown", 0.0123, True), _effect("fast", -0.0004, False)], "missing": []}
    generator._register_markdown_structure_tables(tables, doc)
    table = tables["markdown_structure_effect"]
    assert table["columns"] == [
        "Strategy",
        "Embedder",
        "Marked minus unmarked",
        "95% CI",
        "Wins/losses",
        "Chunks (marked / unmarked)",
        "Median tokens (marked / unmarked)",
        "Chunks starting at a heading",
    ]
    assert table["rows"][0] == [
        "`markdown`",
        "`fastembed:bge-base`",
        "+0.0123 resolved",
        "[+0.0023, +0.0223]",
        "300/200",
        "152,000 / 149,082",
        "205 / 208",
        "27.1%",
    ]
    assert table["rows"][1][2] == "-0.0004 unresolved"
    assert "identical documents, queries and qrels" in table["note"]


def test_missing_pairs_are_named_in_the_note(generator: Any) -> None:
    tables: dict[str, Any] = {}
    doc = {
        "effects": [_effect("markdown", 0.01, True)],
        "missing": [
            {
                "strategy": "semantic",
                "embedding": "ollama:bge-m3",
                "missing": ["x__semantic-t256-o0-gpt2__ollama-bge-m3"],
            }
        ],
    }
    generator._register_markdown_structure_tables(tables, doc)
    assert "semantic / ollama:bge-m3" in tables["markdown_structure_effect"]["note"]


def test_no_effects_registers_no_table(generator: Any) -> None:
    tables: dict[str, Any] = {}
    generator._register_markdown_structure_tables(tables, {"effects": [], "missing": []})
    assert "markdown_structure_effect" not in tables


def test_a_row_whose_dimensions_were_not_audited_renders_n_a_and_keeps_its_verdict(generator: Any) -> None:
    tables: dict[str, Any] = {}
    row = _effect("markdown", 0.0123, True)
    for key in ("marked_rows", "unmarked_rows", "marked_token_p50", "unmarked_token_p50", "marked_heading_start_share"):
        row[key] = None
    generator._register_markdown_structure_tables(tables, {"effects": [row], "missing": []})
    cells = tables["markdown_structure_effect"]["rows"][0]
    assert cells[2] == "+0.0123 resolved"
    assert cells[5:] == ["n/a", "n/a", "n/a"]
