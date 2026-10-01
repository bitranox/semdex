"""The overlap effect is read against QUERY LENGTH, on the cells already scored.

Two corpora disagree on whether chunk overlap helps retrieval, and their queries differ by an
order of magnitude in length. The script under test bins the stored per-query deltas by query
length (observational) and reads the same delta at truncated queries (causal). These tests build
tiny per-query arrays with a known shape - overlap helps the LONG queries and hurts the SHORT
ones - and require the bins to say so.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "score_query_length_effect.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("score_query_length_effect", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["score_query_length_effect"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def qlen() -> Any:
    return _load()


_CORPUS = "corpus"
_LABEL = "fake:embedder"


def _queries(n: int) -> dict[str, str]:
    # Query i has i+1 words, so length and id order agree and the bins are predictable.
    return {f"q{i}": " ".join(["w"] * (i + 1)) for i in range(n)}


def _write_cell(perquery: Path, overlap: int, scores: dict[str, float]) -> None:
    perquery.mkdir(parents=True, exist_ok=True)
    qids = sorted(scores)
    with (perquery / f"{_CORPUS}__recursive-t256-o{overlap}-gpt2__fake-embedder.npz").open("wb") as handle:
        np.savez(handle, qids=np.asarray(qids), ndcg=np.asarray([scores[q] for q in qids]))


def _length_dependent_delta(n: int, *, cut: int) -> tuple[dict[str, float], dict[str, float]]:
    """Overlap-0 scores flat at 0.5; overlap-high scores lower for short queries, higher for long."""
    low = {f"q{i}": 0.5 for i in range(n)}
    high = {f"q{i}": 0.5 + (0.1 if i + 1 > cut else -0.1) for i in range(n)}
    return low, high


def test_word_counts_are_whitespace_words(qlen: Any) -> None:
    assert qlen.query_word_counts({"a": "one  two   three", "b": ""}) == {"a": 3, "b": 0}


def test_bin_edges_are_quantiles_from_min_to_max(qlen: Any) -> None:
    assert qlen.bin_edges(range(1, 101), 4) == [1, 25, 50, 75, 100]
    assert qlen.bin_edges([], 4) == []


def test_bins_separate_the_short_queries_that_lose_from_the_long_ones_that_gain(qlen: Any, tmp_path: Path) -> None:
    n = 400
    low, high = _length_dependent_delta(n, cut=200)
    _write_cell(tmp_path, 0, low)
    _write_cell(tmp_path, 128, high)
    row = qlen.by_query_length(
        corpus=_CORPUS, label=_LABEL, pair=(0, 128), queries=_queries(n), perquery_dir=tmp_path, bins=4, short_words=38
    )
    assert row is not None
    by_bin = {b["bin"]: b for b in row["bins"]}
    assert by_bin["Q1"]["mean_delta"] == pytest.approx(-0.1) and by_bin["Q1"]["resolved"]
    assert by_bin["Q4"]["mean_delta"] == pytest.approx(+0.1) and by_bin["Q4"]["resolved"]
    assert by_bin["<= 38 words"]["mean_delta"] == pytest.approx(-0.1)
    assert by_bin["<= 38 words"]["n_shared"] == 38
    assert row["all"]["mean_delta"] == pytest.approx(0.0, abs=1e-9)


def test_every_query_lands_in_exactly_one_quantile_bin(qlen: Any, tmp_path: Path) -> None:
    # Half-open bins with a closed last one: the maximum must not fall off the end, and no
    # boundary query may be counted twice.
    n = 101
    low, high = _length_dependent_delta(n, cut=50)
    _write_cell(tmp_path, 0, low)
    _write_cell(tmp_path, 51, high)
    row = qlen.by_query_length(
        corpus=_CORPUS, label=_LABEL, pair=(0, 51), queries=_queries(n), perquery_dir=tmp_path, bins=4, short_words=10
    )
    assert sum(b["n_shared"] for b in row["bins"] if b["bin"].startswith("Q")) == n


def test_a_missing_cell_yields_no_row_rather_than_a_zero(qlen: Any, tmp_path: Path) -> None:
    _write_cell(tmp_path, 0, {"q0": 0.5})
    assert (
        qlen.by_query_length(
            corpus=_CORPUS,
            label=_LABEL,
            pair=(0, 128),
            queries=_queries(1),
            perquery_dir=tmp_path,
            bins=2,
            short_words=5,
        )
        is None
    )


def test_truncated_queries_report_both_deltas_side_by_side(qlen: Any, tmp_path: Path) -> None:
    n = 50
    low, high = _length_dependent_delta(n, cut=0)  # every query gains at full length
    _write_cell(tmp_path / "full", 0, low)
    _write_cell(tmp_path / "full", 128, high)
    flat = {f"q{i}": 0.5 for i in range(n)}
    _write_cell(tmp_path / "cut", 0, flat)
    _write_cell(tmp_path / "cut", 128, flat)  # and nothing once the queries are cut
    row = qlen.truncated_queries(
        corpus=_CORPUS,
        label=_LABEL,
        pair=(0, 128),
        perquery_dir=tmp_path / "full",
        variant_dir=tmp_path / "cut",
        words=20,
    )
    assert row["full_queries"]["mean_delta"] == pytest.approx(0.1)
    assert row["truncated_queries"]["mean_delta"] == pytest.approx(0.0)
    assert row["query_words"] == 20


def test_main_writes_both_measurements_with_carried_provenance(
    qlen: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    n = 40
    cache = tmp_path / "cache"
    chunks = cache / "chunks" / f"{_CORPUS}__recursive-t256-o0-gpt2"
    chunks.mkdir(parents=True)
    chunks.joinpath("queries.json").write_text(json.dumps(_queries(n)))
    low, high = _length_dependent_delta(n, cut=20)
    _write_cell(cache / "scores" / "perquery", 0, low)
    _write_cell(cache / "scores" / "perquery", 51, high)
    _write_cell(cache / "scores" / "perquery-q20w", 0, low)
    _write_cell(cache / "scores" / "perquery-q20w", 51, low)
    stamp = {"host": "bench-box", "semdex": "1.2.3"}
    (cache / "scores" / "x_scores.json").write_text(
        json.dumps({"cell": {"corpus": _CORPUS, "embedding": _LABEL, "measured_on": stamp}})
    )
    out = tmp_path / "out.json"
    monkeypatch.setenv("CACHE_ROOT", str(cache))
    monkeypatch.setenv("QLEN_CORPORA", _CORPUS)
    monkeypatch.setenv("QLEN_PAIRS", "0:51")
    monkeypatch.setenv("QLEN_BINS", "2")
    monkeypatch.setenv("QLEN_VARIANTS", "20")
    monkeypatch.setenv("OUT", str(out))
    assert qlen.main() == 0
    payload = json.loads(out.read_text())
    assert [r["embedding"] for r in payload["by_query_length"]] == [_LABEL]
    assert payload["truncated_queries"][0]["truncated_queries"]["mean_delta"] == pytest.approx(0.0)
    assert payload["measured_on"]["runs"] == [stamp], "provenance comes from the scorer's stamp, not this machine"
    assert payload["query_shape"] == [
        {
            "corpus": _CORPUS,
            "n_queries": n,
            "median_words": 20.5,
            "p25_words": 10.75,
            "p75_words": 30.25,
            "p90_words": 36.1,
            "max_words": n,
        }
    ]
    flat = payload["by_query_length_bins"]
    assert len(flat) == 3, "two quantile bins plus the short-query bin, one row each"
    assert {r["bin"] for r in flat} == {"Q1", "Q2", "<= 38 words"}
    assert all(r["embedding"] == _LABEL and r["overlap_high"] == 51 for r in flat)
