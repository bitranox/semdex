"""The product-k scorer: delivered against documents nDCG on hand-built and fixture rankings.

A chunk ranking is a list of the DOCUMENT each ranked chunk belongs to, in rank order. The
delivered view is what ``semdex search`` hands back: k slots, a document may fill several. The
documents view is what every other page measures: the list deduplicated to k distinct documents.
"""

from __future__ import annotations

import importlib.util
import json
import math
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
_SCRIPT = _ROOT / "scripts" / "score_product_k.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("score_product_k", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["score_product_k"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def core() -> Any:
    return _load()


def test_a_repeated_document_becomes_the_sentinel_in_the_delivered_view(core: Any) -> None:
    assert core.delivered_view(["a", "a", "b", "a", "c"], 5) == ["a", core.REPEAT, "b", core.REPEAT, "c"]
    assert core.delivered_view(["a", "b", "c"], 2) == ["a", "b"]


def test_one_document_filling_every_slot_earns_only_its_first_slot(core: Any) -> None:
    rels = {"a": 1}
    assert core.ndcg_delivered(["a", "a", "a", "a", "a"], rels, 5) == pytest.approx(1.0)
    assert core.ndcg_documents(["a", "a", "a", "a", "a"], rels, 5) == pytest.approx(1.0)
    two = {"a": 1, "b": 1}
    # Delivered: only slot 1 gains, the ideal has two gains -> 1 / (1 + 1/log2(3)).
    ideal = 1.0 + 1.0 / math.log2(3)
    assert core.ndcg_delivered(["a", "a", "a", "a", "a"], two, 5) == pytest.approx(1.0 / ideal)
    # Documents: dedup leaves ["a"], the same single gain against the same ideal.
    assert core.ndcg_documents(["a", "a", "a", "a", "a"], two, 5) == pytest.approx(1.0 / ideal)


def test_interleaved_documents_score_the_same_both_ways_when_no_slot_is_wasted(core: Any) -> None:
    rels = {"a": 1, "b": 1}
    ranking = ["a", "b", "c", "d", "e"]
    assert core.ndcg_delivered(ranking, rels, 5) == pytest.approx(core.ndcg_documents(ranking, rels, 5))
    assert core.ndcg_delivered(ranking, rels, 5) == pytest.approx(1.0)


def test_a_relevant_document_first_seen_past_k_counts_only_for_documents(core: Any) -> None:
    rels = {"b": 1}
    # Delivered cuts at k=2: ["a", REPEAT] -> no gain. Documents dedups first: ["a", "b"] -> b at rank 2.
    ranking = ["a", "a", "b"]
    assert core.ndcg_delivered(ranking, rels, 2) == pytest.approx(0.0)
    assert core.ndcg_documents(ranking, rels, 2) == pytest.approx(1.0 / math.log2(3))


def test_empty_qrels_score_zero_and_distinct_docs_counts_names(core: Any) -> None:
    assert core.ndcg_delivered(["a", "b"], {}, 2) == 0.0
    assert core.ndcg_documents(["a", "b"], {}, 2) == 0.0
    assert core.distinct_docs(["a", "a", "b", "c", "c"], 5) == 3
    assert core.distinct_docs(["a", "a", "b", "c", "c"], 2) == 1


def test_the_rung_table_follows_the_budgets(core: Any) -> None:
    def rungs(cap: int) -> list[tuple[int, tuple[int, ...], bool]]:
        return [(r.k, r.budgets, r.product) for r in core.rungs_for_cap(cap, product_k=5, budgets=[1280, 2560])]

    assert rungs(64) == [(5, (), True), (20, (1280,), False), (40, (2560,), False)]
    assert rungs(128) == [(5, (), True), (10, (1280,), False), (20, (2560,), False)]
    assert rungs(256) == [(5, (1280,), True), (10, (2560,), False)]
    assert rungs(512) == [(2, (1280,), False), (5, (2560,), True)]
    assert rungs(1024) == [(5, (), True)], "a cap outside the ladder gets the product k only"


_CORPUS = "mldr_en_8k_slice"
_PROFILE = "recursive-t256-o0-gpt2"
_LABEL = "fastembed:bge-base"


def _write_cell(cache: Path, core: Any, *, with_query_cache: bool) -> None:
    """A four-chunk cell over two documents, three queries, and its query-vector cache."""
    cell = f"{_CORPUS}__{_PROFILE}__{core.dirsafe(_LABEL)}"
    vroot = cache / "vectors" / cell
    vroot.mkdir(parents=True)
    # doc-a has three chunks, doc-b one; unit vectors so cosine equals the dot product.
    vectors = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(vroot / "vectors.npy", vectors)
    (vroot / "meta.json").write_text(json.dumps({"count": 4, "dim": 2, "model_id": "BAAI/bge-base-en-v1.5"}))
    chunks = cache / "chunks" / f"{_CORPUS}__{_PROFILE}"
    chunks.mkdir(parents=True)
    _write_parquet(_arrow_table({"source_uri": ["doc-a", "doc-a", "doc-a", "doc-b"]}), chunks / "chunks.parquet")
    queries = {"q1": "about a", "q2": "about b", "q3": "about both"}
    (chunks / "queries.json").write_text(json.dumps(queries))
    qrels = {"q1": {"doc-a": 1}, "q2": {"doc-b": 1}, "q3": {"doc-a": 1, "doc-b": 1}}
    (chunks / "qrels.json").write_text(json.dumps(qrels))
    if with_query_cache:
        path = core.qvec_path(_CORPUS, _LABEL, queries)
        path.parent.mkdir(parents=True, exist_ok=True)
        qvecs = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
        with path.open("wb") as handle:
            np.savez(handle, qids=np.asarray(sorted(queries)), vectors=qvecs)


def test_a_cell_without_a_query_vector_cache_is_refused_by_name(
    core: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    _write_cell(tmp_path, core, with_query_cache=False)
    with pytest.raises(SystemExit, match=r"mldr_en_8k_slice__fastembed-bge-base: no query-vector cache"):
        core.score_cell(_CORPUS, _PROFILE, _LABEL, product_k=2, budgets=[512], fetch=4)


def test_a_fixture_cell_scores_both_views_from_one_top_list(
    core: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("SEMDEX_PRODUCT_K_PERQUERY", str(tmp_path / "perquery"))
    _write_cell(tmp_path, core, with_query_cache=True)
    row = core.score_cell(_CORPUS, _PROFILE, _LABEL, product_k=2, budgets=[512], fetch=4)
    assert row is not None
    assert row["corpus"] == _CORPUS and row["embedding"] == _LABEL and row["n_queries"] == 3
    rungs = {r["k"]: r for r in row["rungs"]}
    assert sorted(rungs) == [2], "cap 256 at budget 512 is k=2, which is also the product k"
    two = rungs[2]
    assert two["product"] is True and two["budgets"] == [512]
    # q2 asks for doc-b: the top-2 chunks are doc-b then doc-a. Both views score 1.0.
    # q1 asks for doc-a: top-2 are two doc-a chunks. Delivered wastes slot 2, documents is fine: both 1.0.
    # q3 wants both: top-2 delivered are doc-a, doc-a (second slot wasted), documents dedups to doc-a, doc-b.
    assert two["ndcg_documents"] > two["ndcg_delivered"]
    assert two["gap_mean_delta"] == pytest.approx(two["ndcg_documents"] - two["ndcg_delivered"])
    assert two["gap_wins"] == 1 and two["gap_losses"] == 0
    assert two["distinct_docs"] == pytest.approx((1 + 2 + 1) / 3)
    npz = tmp_path / "perquery" / f"{_CORPUS}__{_PROFILE}__{core.dirsafe(_LABEL)}.npz"
    with np.load(npz) as data:
        assert list(data["ks"]) == [2]
        assert data["delivered"].shape == (3, 1) and data["documents"].shape == (3, 1)
    assert len(row["perquery_sha256"]) == 64


@pytest.mark.parametrize(
    ("profile", "tokenizer", "recipe", "breakpoint_model"),
    [
        ("recursive-t256-o0-gpt2", "gpt2", "", None),
        ("recursive-t256-o0-gpt2-rmarkdown", "gpt2", "markdown", None),
        ("semantic-t256-o0-gpt2-bpbge-m3", "gpt2", "", "bge-m3"),
        ("semantic-t256-o0-gpt2-rmarkdown-bpbge-m3", "gpt2", "markdown", "bge-m3"),
    ],
)
def test_parse_profile_reads_the_recipe_segment_the_chunker_writes(
    core: Any, profile: str, tokenizer: str, recipe: str, breakpoint_model: str | None
) -> None:
    """The producer writes ``<tokenizer>[-r<recipe>][-bp<model>]``; a recipe cell is not a recipe-"" cell."""
    axes = core.parse_profile(profile)
    assert (axes["tokenizer"], axes["recipe"], axes["breakpoint_model"]) == (tokenizer, recipe, breakpoint_model)
    assert "recipe_unparsed" not in axes


def test_parse_profile_flags_a_recipe_segment_it_does_not_know(core: Any) -> None:
    """An unknown ``-r`` segment may be part of a tokenizer name; it is flagged, never guessed at."""
    axes = core.parse_profile("recursive-t256-o0-gpt2-rfoo")
    assert axes["recipe_unparsed"] == "gpt2-rfoo"


def test_discovery_keeps_only_recipe_free_profiles(core: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A recipe cell measures another rule set, so it must not join the recipe-"" cap-256 profiles."""
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    for profile in ("recursive-t256-o0-gpt2", "recursive-t256-o0-gpt2-rmarkdown", "recursive-t256-o0-gpt2-rfoo"):
        (tmp_path / "vectors" / f"widget__{profile}__fastembed-x").mkdir(parents=True)
    assert core._discover_profiles("widget") == ["recursive-t256-o0-gpt2"]
