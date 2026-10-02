"""A cell that declares a retrieval method pairs on the method axis and on nothing else.

The method-bearing cells (bm25, dense, hybrid and their @20 / +rerank variants) exist to answer
"does adding lexical retrieval or a reranker help". Pairing two of them on the EMBEDDING axis
manufactures rows the embedding is not on the path of - two bm25 cells embed nothing, so their
per-query scores are identical and the row prints as an unresolved zero - and the dense and hybrid
variants restate every dense-only effect under a second label that the effect row does not carry.
Measured before this guard: 52 of the 572 rows on the embedding axis, 20 of them bm25 pairs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "export_bench_raw.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("export_bench_raw_axes", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_bench_raw_axes"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exporter() -> Any:
    return _load()


def _cell(embedding: str, *, method: str | None = None, max_tokens: int = 256) -> dict[str, Any]:
    axes = {"strategy": "recursive", "max_tokens": max_tokens, "overlap_tokens": 0, "breakpoint_model": None}
    cell: dict[str, Any] = {"cell": f"c__p__{embedding}", "corpus": "c", "embedding": embedding, "axes": axes}
    if method is not None:
        cell["method"] = method
        cell["cell"] += f"__{method}"
    return cell


def test_plain_cells_differing_only_in_embedder_pair_on_the_embedding_axis(exporter: Any) -> None:
    assert exporter._one_axis_apart(_cell("a"), _cell("b")) == "embedding"


def test_the_same_embedder_under_two_methods_pairs_on_the_method_axis(exporter: Any) -> None:
    assert exporter._one_axis_apart(_cell("a", method="dense"), _cell("a", method="hybrid")) == "method"


@pytest.mark.parametrize("method", ["bm25", "dense", "hybrid", "bm25@20", "hybrid@20+rerank"])
def test_two_cells_under_the_same_declared_method_pair_on_nothing_else(exporter: Any, method: str) -> None:
    """The embedder is not the question these cells were scored to answer, and for bm25 it is not
    even on the path: both cells hold the same lexical ranking."""
    assert exporter._one_axis_apart(_cell("a", method=method), _cell("b", method=method)) is None
    assert exporter._one_axis_apart(_cell("a", method=method), _cell("a", method=method, max_tokens=512)) is None


def test_a_plain_cell_never_pairs_with_a_method_bearing_one(exporter: Any) -> None:
    """A plain cell IS the dense measurement under another name; pairing it against a declared
    dense cell would compare a cell with itself and report a method effect."""
    assert exporter._one_axis_apart(_cell("a"), _cell("a", method="dense")) is None
    assert exporter._one_axis_apart(_cell("a"), _cell("b", method="hybrid")) is None


# --------------------------------------------------------------------------
# A declared dense cell with no plain twin stands in for it
# --------------------------------------------------------------------------


def _declared(cells: list[dict[str, Any]], exporter: Any) -> list[dict[str, Any]]:
    exporter.mark_dense_stand_ins(cells)
    return cells


def test_a_dense_cell_is_marked_a_stand_in_only_when_its_plain_twin_is_absent(exporter: Any) -> None:
    twin_present = _declared([_cell("a"), _cell("a", method="dense")], exporter)
    assert exporter._STAND_IN not in twin_present[1]
    twin_absent = _declared([_cell("b", method="dense")], exporter)
    assert twin_absent[0][exporter._STAND_IN] is True


def test_a_stand_in_pairs_with_a_plain_cell_of_another_embedder_on_the_embedding_axis(exporter: Any) -> None:
    """The embedder the dense-only sweep never ran would otherwise vanish from the axis: measured,
    the largest embedding effect on the page (0.4927) lived only in such a cell."""
    plain_a, dense_b = _declared([_cell("a"), _cell("b", method="dense")], exporter)
    assert exporter._one_axis_apart(plain_a, dense_b) == "embedding"
    assert exporter._one_axis_apart(dense_b, plain_a) == "embedding"


def test_a_dense_cell_with_a_plain_twin_stays_off_the_embedding_axis(exporter: Any) -> None:
    plain_a, dense_a, plain_b = _declared([_cell("a"), _cell("a", method="dense"), _cell("b")], exporter)
    assert exporter._one_axis_apart(dense_a, plain_b) is None
    assert exporter._one_axis_apart(plain_a, plain_b) == "embedding"


def test_a_stand_in_still_pairs_with_its_own_hybrid_cell_on_the_method_axis(exporter: Any) -> None:
    dense_b, hybrid_b = _declared([_cell("b", method="dense"), _cell("b", method="hybrid")], exporter)
    assert exporter._one_axis_apart(dense_b, hybrid_b) == "method"


# ---------------------------------------------------------------- one measurement under two labels


def _write_per_query(directory: Path, cell: str, scores: list[float], exporter: Any) -> None:
    np = exporter.np
    qids = np.array([f"q{i}" for i in range(len(scores))])
    np.savez(directory / f"{cell}.npz", qids=qids, **{exporter.NPZ_KEYS["ndcg@10"]: np.array(scores)})


def test_two_cells_with_identical_per_query_scores_are_one_measurement_not_a_comparison(
    exporter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """bm25 and bm25@20 are one ranked list read at two shortlist depths, identical at nDCG@10.

    Pairing them reports a comparison that was never a contest: delta zero, no wins, no losses,
    a tie on every query. Such a pair inflates the axis's denominator and its tie count, so it
    must be dropped as a duplicate measurement, decided on the data (every per-query score
    equal) rather than on the level names, so any future same-list-twice pair is caught too.
    """
    monkeypatch.setenv("SEMDEX_SCORE_PERQUERY", str(tmp_path))
    scores = [0.1 * (i % 7) for i in range(24)]
    identical = [_cell("m", method="bm25"), _cell("m", method="bm25@20")]
    for cell in identical:
        _write_per_query(tmp_path, cell["cell"], scores, exporter)
    assert exporter.knob_effects(identical) == []
    # Control: the same pairing with scores that differ on some queries IS a comparison.
    contest = [_cell("m", method="bm25"), _cell("m", method="hybrid")]
    _write_per_query(tmp_path, contest[0]["cell"], scores, exporter)
    _write_per_query(
        tmp_path, contest[1]["cell"], [v + (0.05 if i % 3 else 0.0) for i, v in enumerate(scores)], exporter
    )
    effects = exporter.knob_effects(contest)
    assert [e["axis"] for e in effects] == ["method"]


def _chunk_cell(embedding: str, **axes: Any) -> dict[str, Any]:
    cell = _cell(embedding)
    cell["axes"].update({"tokenizer": "gpt2", "recipe": ""})
    cell["axes"].update(axes)
    return cell


def test_cells_cut_under_different_recipes_are_two_axes_apart_with_any_other_change(exporter: Any) -> None:
    # A markdown-recipe cell differs from a generic one in the recipe AND whatever else changed,
    # so neither the overlap nor the embedder may claim the whole difference.
    generic = _chunk_cell("a")
    markdown = _chunk_cell("a", recipe="markdown", overlap_tokens=51)
    assert exporter._one_axis_apart(generic, markdown) is None
    assert exporter._one_axis_apart(_chunk_cell("a"), _chunk_cell("b", recipe="markdown")) is None


def test_cells_cut_with_different_tokenizers_are_not_an_overlap_or_embedding_effect(exporter: Any) -> None:
    gpt2 = _chunk_cell("a")
    assert exporter._one_axis_apart(gpt2, _chunk_cell("a", tokenizer="cl100k", overlap_tokens=51)) is None
    assert exporter._one_axis_apart(gpt2, _chunk_cell("b", tokenizer="cl100k")) is None


def test_a_recipe_only_difference_is_not_reported_as_a_chunk_knob(exporter: Any) -> None:
    assert exporter._one_axis_apart(_chunk_cell("a"), _chunk_cell("a", recipe="markdown")) is None


def test_a_single_knob_still_pairs_when_recipe_and_tokenizer_match(exporter: Any) -> None:
    assert exporter._one_axis_apart(_chunk_cell("a"), _chunk_cell("a", overlap_tokens=51)) == "overlap_tokens"
    assert exporter._one_axis_apart(_chunk_cell("a"), _chunk_cell("b")) == "embedding"
