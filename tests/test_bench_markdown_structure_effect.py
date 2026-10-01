"""Marked against unmarked: the same queries, the same documents, only the heading markup differs.

The exporter pairs cells within one corpus. The script under test pairs each marked cell with its
unmarked twin across the two corpus ids and runs the shared paired bootstrap. These tests build
tiny per-query arrays with a known shape and require the script to report a shifted pair as
resolved, an identical pair as unresolved with delta zero, and a missing or mismatched twin by name.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa  # pyright: ignore[reportMissingTypeStubs] - no stubs; typed at the facade below
import pyarrow.parquet as pq  # pyright: ignore[reportMissingTypeStubs] - see above
import pytest

# Typed facade over the two unstubbed pyarrow calls this test needs. Declaring the signatures
# under TYPE_CHECKING types every call site below without suppressing anything: the checker reads
# these declarations, the runtime binds the real functions. Drop it when pyarrow ships stubs.
if TYPE_CHECKING:

    def _arrow_table(columns: dict[str, list[str]]) -> Any: ...

    def _write_parquet(table: Any, where: Path) -> None: ...

else:
    _arrow_table = pa.table
    _write_parquet = pq.write_table

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "score_markdown_structure_effect.py"

pytestmark = pytest.mark.os_agnostic

_MARKED = "mldr_en_8k_md_slice"
_UNMARKED = "mldr_en_8k_slice"
_LABEL = "fake:embedder"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("score_markdown_structure_effect", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["score_markdown_structure_effect"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def effect() -> Any:
    return _load()


def _write_cell(perquery: Path, corpus: str, strategy: str, scores: dict[str, float]) -> None:
    perquery.mkdir(parents=True, exist_ok=True)
    qids = sorted(scores)
    with (perquery / f"{corpus}__{strategy}-t256-o0-gpt2__fake-embedder.npz").open("wb") as handle:
        np.savez(handle, qids=np.asarray(qids), ndcg=np.asarray([scores[q] for q in qids]))


def test_a_shifted_pair_resolves_in_favour_of_the_marked_cell(effect: Any, tmp_path: Path) -> None:
    n = 60
    _write_cell(tmp_path, _UNMARKED, "markdown", {f"q{i}": 0.5 for i in range(n)})
    _write_cell(tmp_path, _MARKED, "markdown", {f"q{i}": 0.6 for i in range(n)})
    row = effect.pair_cells(
        tmp_path,
        strategy="markdown",
        label=_LABEL,
        marked=_MARKED,
        unmarked=_UNMARKED,
    )
    assert row["mean_delta"] == pytest.approx(0.1)
    assert row["resolved"] and row["wins"] == n and row["losses"] == 0
    assert row["marked_cell"] == f"{_MARKED}__markdown-t256-o0-gpt2__fake-embedder"
    assert row["unmarked_cell"] == f"{_UNMARKED}__markdown-t256-o0-gpt2__fake-embedder"


def test_an_identical_pair_is_delta_zero_and_unresolved(effect: Any, tmp_path: Path) -> None:
    scores = {f"q{i}": 0.3 + i / 100 for i in range(40)}
    _write_cell(tmp_path, _UNMARKED, "fast", scores)
    _write_cell(tmp_path, _MARKED, "fast", scores)
    row = effect.pair_cells(tmp_path, strategy="fast", label=_LABEL, marked=_MARKED, unmarked=_UNMARKED)
    assert row["mean_delta"] == 0.0 and not row["resolved"] and row["ties"] == 40


def test_a_missing_twin_yields_none_not_a_zero(effect: Any, tmp_path: Path) -> None:
    _write_cell(tmp_path, _MARKED, "semantic", {"q0": 0.5})
    assert (
        effect.pair_cells(
            tmp_path,
            strategy="semantic",
            label=_LABEL,
            marked=_MARKED,
            unmarked=_UNMARKED,
        )
        is None
    )


def test_mismatched_query_ids_are_refused_by_name(effect: Any, tmp_path: Path) -> None:
    _write_cell(tmp_path, _UNMARKED, "recursive", {"q0": 0.5, "q1": 0.5})
    _write_cell(tmp_path, _MARKED, "recursive", {"q0": 0.5, "q2": 0.5})
    with pytest.raises(
        ValueError,
        match=r"recursive-t256-o0-gpt2__fake-embedder.*query ids differ",
    ):
        effect.pair_cells(
            tmp_path,
            strategy="recursive",
            label=_LABEL,
            marked=_MARKED,
            unmarked=_UNMARKED,
        )


def test_heading_start_share_counts_chunks_that_begin_with_the_mark(effect: Any, tmp_path: Path) -> None:
    table = _arrow_table({"text": ["## History\nprose", "plain prose", "## Synthesis\nmore", "x"]})
    _write_parquet(table, tmp_path / "chunks.parquet")
    assert effect.heading_start_share(tmp_path / "chunks.parquet") == pytest.approx(0.5)


def test_judged_wrong_counts_the_x_prefixed_rows(effect: Any, tmp_path: Path) -> None:
    judged = tmp_path / "heading-sample-judged.txt"
    judged.write_text("doc-1\t## History\tprose\nX\tdoc-2\t## Not one\tprose\ndoc-3\t## Synthesis\tprose\n")
    assert effect.judged_wrong(judged) == (1, 3)
    assert effect.judged_wrong(tmp_path / "absent.txt") == (0, 0)


def test_main_writes_effects_missing_pairs_and_carried_provenance(
    effect: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    perquery = cache / "scores" / "perquery"
    n = 30
    _write_cell(perquery, _UNMARKED, "markdown", {f"q{i}": 0.5 for i in range(n)})
    _write_cell(perquery, _MARKED, "markdown", {f"q{i}": 0.55 for i in range(n)})
    _write_cell(perquery, _MARKED, "fast", {f"q{i}": 0.5 for i in range(n)})
    for corpus in (_MARKED, _UNMARKED):
        d = cache / "chunks" / f"{corpus}__markdown-t256-o0-gpt2"
        d.mkdir(parents=True)
        _write_parquet(
            _arrow_table({"text": (["## H\nx", "y"] if corpus == _MARKED else ["H\nx", "y"])}),
            d / "chunks.parquet",
        )
    stamp = {"host": "bench-box", "semdex": "1.2.3"}
    (cache / "scores" / "mldr_md_chunk_scores.json").write_text(
        json.dumps(
            {
                f"{_MARKED}__markdown-t256-o0-gpt2__fake-embedder": {
                    "corpus": _MARKED,
                    "embedding": _LABEL,
                    "measured_on": stamp,
                }
            }
        )
    )
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "chunk_sets": [
                    {
                        "corpus": _MARKED,
                        "profile": "markdown-t256-o0-gpt2",
                        "rows": 12,
                        "token_p50": 200,
                    },
                    {
                        "corpus": _UNMARKED,
                        "profile": "markdown-t256-o0-gpt2",
                        "rows": 10,
                        "token_p50": 210,
                    },
                ]
            }
        )
    )
    slice_dir = tmp_path / _MARKED
    slice_dir.mkdir()
    slice_dir.joinpath("meta.json").write_text(
        json.dumps({"corpus": _MARKED, "marked_lines": 7, "marked_docs": 3, "total": 4})
    )
    slice_dir.joinpath("heading-sample-judged.txt").write_text("a\t## A\tp\nX\tb\t## B\tp\n")
    out = tmp_path / "out.json"
    monkeypatch.setenv("CACHE_ROOT", str(cache))
    monkeypatch.setenv("MDSTRUCT_STRATEGIES", "markdown,fast")
    monkeypatch.setenv("MDSTRUCT_EMBEDDINGS", _LABEL)
    monkeypatch.setenv("MDSTRUCT_SLICE_DIR", str(slice_dir))
    monkeypatch.setenv("MDSTRUCT_AUDIT", str(audit))
    monkeypatch.setenv("OUT", str(out))
    assert effect.main() == 0
    payload = json.loads(out.read_text())
    assert payload["measured_on"]["runs"] == [stamp], "provenance comes from the scorer's stamp, not this machine"
    (row,) = payload["effects"]
    assert row["strategy"] == "markdown" and row["embedding"] == _LABEL
    assert row["mean_delta"] == pytest.approx(0.05) and row["resolved"]
    assert (row["marked_rows"], row["unmarked_rows"]) == (12, 10)
    assert (row["marked_token_p50"], row["unmarked_token_p50"]) == (200, 210)
    assert row["marked_heading_start_share"] == pytest.approx(0.5)
    assert payload["missing"] == [
        {
            "strategy": "fast",
            "embedding": _LABEL,
            "missing": [f"{_UNMARKED}__fast-t256-o0-gpt2__fake-embedder"],
        }
    ]
    assert payload["slice"]["marked_lines"] == 7
    assert (payload["slice"]["sample_judged_wrong"], payload["slice"]["sample_judged"]) == (
        1,
        2,
    )
