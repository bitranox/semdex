"""The query-length tables and the voided-cells table render from their raw data.

The chunking page's overlap answer turned on QUERY LENGTH, and the cells an audit judged unusable
are withheld from every table. Both facts reach the reader only through generated tables, so each
registrar is checked on a small document with a known shape.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "gen_bench_tables.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("gen_bench_tables_qlen", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator() -> Any:
    return _load()


def _paired(delta: float, resolved: bool) -> dict[str, Any]:
    return {"mean_delta": delta, "ci_lo": delta - 0.01, "ci_hi": delta + 0.01, "resolved": resolved, "n_shared": 10}


def test_the_binned_table_puts_each_bin_in_its_own_column_and_names_the_word_ranges(generator: Any) -> None:
    tables: dict[str, Any] = {}
    doc = {
        "by_query_length": [
            {
                "corpus": "gerdalir",
                "embedding": "fastembed:bge-base",
                "overlap_low": 0,
                "overlap_high": 128,
                "all": _paired(0.0055, True),
                "bins": [
                    {"bin": "Q1", "words_lo": 1, "words_hi": 63, **_paired(-0.0242, True)},
                    {"bin": "Q2", "words_lo": 63, "words_hi": 104, **_paired(-0.0047, False)},
                    {"bin": "<= 38 words", "words_lo": 0, "words_hi": 38, **_paired(-0.0252, True)},
                ],
            }
        ],
        "truncated_queries": [],
    }
    generator._register_query_length_tables(tables, doc)
    table = tables["chunk_overlap_by_query_length"]
    assert table["columns"] == ["Corpus", "Embedder", "Overlap (low to high)", "All queries", "Q1", "Q2", "<= 38 words"]
    assert table["rows"] == [
        [
            "gerdalir",
            "`fastembed:bge-base`",
            "0 to 128",
            "+0.0055 resolved",
            "-0.0242 resolved",
            "-0.0047 unresolved",
            "-0.0252 resolved",
        ]
    ]
    assert "gerdalir: Q1 1-63 words, Q2 63-104 words" in table["note"]
    assert "chunk_overlap_truncated_queries" not in tables, "no causal rows, no causal table"


def test_the_truncated_query_table_shows_full_beside_cut(generator: Any) -> None:
    tables: dict[str, Any] = {}
    doc = {
        "by_query_length": [],
        "truncated_queries": [
            {
                "corpus": "gerdalir",
                "embedding": "ollama:bge-m3",
                "overlap_low": 0,
                "overlap_high": 128,
                "query_words": 20,
                "full_queries": _paired(0.0136, True),
                "truncated_queries": _paired(-0.0021, False),
            }
        ],
    }
    generator._register_query_length_tables(tables, doc)
    table = tables["chunk_overlap_truncated_queries"]
    assert table["columns"][-1] == "First 20 words"
    assert table["rows"] == [["gerdalir", "`ollama:bge-m3`", "0 to 128", "+0.0136 resolved", "-0.0021 unresolved"]]


def test_the_voided_table_lists_every_export_s_dropped_cells(generator: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    docs = {
        "chunk-sweep-mldr.json": {
            "voided": [{"cell": "mldr__recursive-t512-o10-gpt2__fastembed-bge-base", "reason": "re-cut"}]
        },
        "chunk-sweep-gerdalir.json": {
            "voided": [{"cell": "gerdalir__whitespace-t256-o0-gpt2__fastembed-bge-base", "reason": "clipped"}]
        },
    }
    monkeypatch.setattr(generator, "_load", docs.get)
    tables: dict[str, Any] = {}
    generator._register_voided_cells(tables)
    assert [r[1] for r in tables["voided_cells"]["rows"]] == ["re-cut", "clipped"]


def _nothing_voided(_name: str) -> dict[str, list[dict[str, str]]]:
    return {"voided": []}


def test_no_voided_cells_means_no_table(generator: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generator, "_load", _nothing_voided)
    tables: dict[str, Any] = {}
    generator._register_voided_cells(tables)
    assert "voided_cells" not in tables
