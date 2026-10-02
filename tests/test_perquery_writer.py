"""Per-query score files: one writer, which refuses to replace a file of a different kind.

Four scorers persist per-query arrays as ``<dir>/<cell>.npz`` and three of them name a cell
``<corpus>__<profile>__<embedder>``. The product-k scorer once shared the dense scorer's directory
too, and one run replaced the dense arrays of every cell it scored with its own; the exporter then
died on the first of them. These tests drive each REAL scorer's writer against a directory that
already holds the other kind of file, so a scorer that stops routing through the guarded writer
fails here rather than in a week-old cache.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

pytestmark = pytest.mark.os_agnostic

_CELL = "mldr_en_8k_slice__recursive-t256-o0-gpt2__fastembed-bge-base"
_DENSE_KEYS = {"qids", "ndcg", "recall", "mrr", "p1"}
_PRODUCT_K_KEYS = {"qids", "ks", "delivered", "documents", "distinct"}


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _metrics() -> dict[str, dict[str, float]]:
    return {q: {"ndcg@10": 0.5, "recall@10": 1.0, "mrr": 0.5, "p@1": 0.0} for q in ("q1", "q2")}


def _write_dense(module: Any, cell: str) -> str:
    return str(module._write_per_query(cell, _metrics()))


def _write_product_k(module: Any, cell: str) -> str:
    rungs = [module.Rung(k=2, budgets=(512,), product=True)]
    per_rung = {2: {q: {"delivered": 0.5, "documents": 1.0, "distinct": 2.0} for q in ("q1", "q2")}}
    return str(module._write_per_query(cell, ["q1", "q2"], rungs, per_rung))


def _plant(path: Path, keys: set[str]) -> None:
    """A per-query file of the given kind, written by plain np.savez the way the old scorers did."""
    path.parent.mkdir(parents=True, exist_ok=True)
    z = np.zeros(2, dtype=np.float32)
    with path.open("wb") as handle:
        if keys == _DENSE_KEYS:
            np.savez(handle, qids=z, ndcg=z, recall=z, mrr=z, p1=z)
        else:
            assert keys == _PRODUCT_K_KEYS
            np.savez(handle, qids=z, ks=z, delivered=z, documents=z, distinct=z)


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    monkeypatch.delenv("SEMDEX_SCORE_PERQUERY", raising=False)
    monkeypatch.delenv("SEMDEX_PRODUCT_K_PERQUERY", raising=False)
    return tmp_path


@pytest.mark.parametrize("scorer", ["score_chunk_sweep", "score_hybrid_sweep", "score_rerank_sweep"])
def test_a_dense_scorer_refuses_to_replace_a_product_k_file(cache: Path, scorer: str) -> None:
    target = cache / "scores" / "perquery" / f"{_CELL}.npz"
    _plant(target, _PRODUCT_K_KEYS)
    before = target.read_bytes()
    with pytest.raises(ValueError, match=r"holds a different kind of per-query file"):
        _write_dense(_load(scorer), _CELL)
    assert target.read_bytes() == before, "the refused write must leave the existing file untouched"


def test_the_product_k_scorer_refuses_to_replace_a_dense_file(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Even when pointed at the dense directory by hand, which is how a shared directory comes back.
    monkeypatch.setenv("SEMDEX_PRODUCT_K_PERQUERY", str(cache / "scores" / "perquery"))
    target = cache / "scores" / "perquery" / f"{_CELL}.npz"
    _plant(target, _DENSE_KEYS)
    before = target.read_bytes()
    with pytest.raises(ValueError, match=r"holds a different kind of per-query file"):
        _write_product_k(_load("score_product_k"), _CELL)
    assert target.read_bytes() == before


def test_the_product_k_scorer_defaults_to_its_own_directory(cache: Path) -> None:
    _write_product_k(_load("score_product_k"), _CELL)
    assert (cache / "scores" / "perquery-product-k" / f"{_CELL}.npz").is_file()
    assert not (cache / "scores" / "perquery" / f"{_CELL}.npz").exists()


def test_rewriting_a_file_of_the_same_kind_is_allowed_and_hashed(cache: Path) -> None:
    module = _load("score_chunk_sweep")
    _write_dense(module, _CELL)
    digest = _write_dense(module, _CELL)
    target = cache / "scores" / "perquery" / f"{_CELL}.npz"
    with np.load(target, allow_pickle=False) as data:
        assert set(data.files) == _DENSE_KEYS
        assert list(data["qids"]) == ["q1", "q2"]
    assert digest == hashlib.sha256(target.read_bytes()).hexdigest()
    assert not list(target.parent.glob("*.tmp")), "an atomic write leaves no temporary behind"
