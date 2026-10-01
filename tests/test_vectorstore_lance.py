"""Contract tests for the LanceDB embedded vector store.

Skipped when the optional ``lancedb`` dependency is absent (run with
``pytest --extra lance`` / after installing ``semdex[lance]``).
"""

# pyright: basic
# The index tests reach into the adapter (`_db`, `list_indices`, IndexConfig) on purpose -
# whether the ANN index exists AND covers the whole table is the contract under test - and
# lancedb ships no type stubs; strict mode would only flag that intentional white-box access.

from __future__ import annotations

import random
from pathlib import Path

import pytest

pytest.importorskip("lancedb")

from semdex.adapters.vectorstore import LanceVectorStore
from semdex.domain.errors import CollectionModelMismatchError, VectorStoreError
from semdex.domain.models import Chunk, Collection, SourceRef


def _chunk(text: str, path: str) -> Chunk:
    source = SourceRef(uri=str(path), label="curated", content_hash="h", mtime=1.5)
    return Chunk(text=text, source=source, ordinal=0, token_count=len(text.split()))


@pytest.mark.os_agnostic
def test_upsert_persists_across_instances(tmp_path: Path) -> None:
    """A second store instance over the same dir sees the first's writes."""
    coll = Collection(name="c", model_id="m", dim=3)
    first = LanceVectorStore(tmp_path)
    first.ensure_collection(coll)
    first.upsert(collection="c", chunks=[_chunk("body", "/a.md")], vectors=[(1.0, 0.0, 0.0)])

    second = LanceVectorStore(tmp_path)

    assert second.count(collection="c") == 1
    assert second.collections() == [coll]
    hits = second.query(collection="c", vector=(1.0, 0.0, 0.0), k=1)
    assert hits[0].uri == "/a.md"
    assert hits[0].label == "curated"


@pytest.mark.os_agnostic
def test_summary_round_trips_and_persists(tmp_path: Path) -> None:
    """A chunk's per-document summary survives upsert -> query and a reload (None stays None)."""
    with_summary = Chunk(
        text="body",
        source=SourceRef(uri="/a.md", label="curated", content_hash="h", mtime=1.5),
        ordinal=0,
        token_count=1,
        summary="a short doc summary",
    )
    first = LanceVectorStore(tmp_path)
    first.ensure_collection(Collection(name="c", model_id="m", dim=3))
    first.upsert(
        collection="c",
        chunks=[with_summary, _chunk("plain", "/b.md")],
        vectors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
    )

    second = LanceVectorStore(tmp_path)
    a_hit = second.query(collection="c", vector=(1.0, 0.0, 0.0), k=1)[0]
    b_hit = second.query(collection="c", vector=(0.0, 1.0, 0.0), k=1)[0]
    assert a_hit.summary == "a short doc summary"
    assert b_hit.summary is None


@pytest.mark.os_agnostic
def test_cosine_knn_ranks_closest_first(tmp_path: Path) -> None:
    """The store returns the nearest chunk first with a higher cosine score."""
    store = LanceVectorStore(tmp_path)
    store.ensure_collection(Collection(name="c", model_id="m", dim=3))
    store.upsert(
        collection="c",
        chunks=[_chunk("x-axis", "/x.md"), _chunk("y-axis", "/y.md")],
        vectors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
    )

    hits = store.query(collection="c", vector=(0.9, 0.1, 0.0), k=2)

    assert [hit.uri for hit in hits] == ["/x.md", "/y.md"]
    assert hits[0].score > hits[1].score


@pytest.mark.os_agnostic
def test_delete_by_source_persists(tmp_path: Path) -> None:
    """Deleting a source removes its rows durably."""
    first = LanceVectorStore(tmp_path)
    first.ensure_collection(Collection(name="c", model_id="m", dim=3))
    first.upsert(collection="c", chunks=[_chunk("body", "/a.md")], vectors=[(1.0, 0.0, 0.0)])
    first.delete_by_source(collection="c", uri="/a.md")

    assert LanceVectorStore(tmp_path).count(collection="c") == 0


@pytest.mark.os_agnostic
def test_swap_is_blue_green(tmp_path: Path) -> None:
    """swap renames staging over target and adopts the target name."""
    store = LanceVectorStore(tmp_path)
    store.ensure_collection(Collection(name="c__staging", model_id="m", dim=3))
    store.upsert(collection="c__staging", chunks=[_chunk("body", "/a.md")], vectors=[(1.0, 0.0, 0.0)])

    store.swap(staging="c__staging", target="c")

    assert [c.name for c in store.collections()] == ["c"]
    assert store.count(collection="c") == 1


@pytest.mark.os_agnostic
def test_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    """Querying with a wrong-dimension vector is a model mismatch."""
    store = LanceVectorStore(tmp_path)
    store.ensure_collection(Collection(name="c", model_id="m", dim=8))
    with pytest.raises(CollectionModelMismatchError):
        store.query(collection="c", vector=(1.0, 0.0), k=1)


@pytest.mark.os_agnostic
def test_unknown_collection_query_raises(tmp_path: Path) -> None:
    """Querying an unknown collection is a clear VectorStoreError."""
    with pytest.raises(VectorStoreError):
        LanceVectorStore(tmp_path).query(collection="nope", vector=(1.0, 0.0, 0.0), k=1)


@pytest.mark.os_agnostic
def test_empty_store_has_no_collections(tmp_path: Path) -> None:
    """A fresh store dir starts empty."""
    assert LanceVectorStore(tmp_path).collections() == []


def _bulk(store: LanceVectorStore, collection: str, n: int, dim: int = 8) -> None:
    # upsert only appends; compact() is what builds/folds the ANN index (as the index use case does).
    rng = random.Random(7)  # deterministic fixture data, not cryptography  # noqa: S311
    chunks = [_chunk(f"text {i}", f"/doc{i}.md") for i in range(n)]
    vectors = [tuple(rng.random() for _ in range(dim)) for _ in range(n)]
    store.upsert(collection=collection, chunks=chunks, vectors=vectors)
    store.compact(collection=collection)


def _vector_indices(store: LanceVectorStore, collection: str) -> list:
    # Reaches into the adapter deliberately: whether the ANN index exists IS the contract under test.
    table = store._db.open_table(store._require(collection).table)
    return list(table.list_indices())


def _unindexed_rows(store: LanceVectorStore, collection: str) -> int:
    # Rows outside the ANN index that LanceDB would brute-force scan on every query.
    table = store._db.open_table(store._require(collection).table)
    return int(table.list_indices()[0].num_unindexed_rows)


@pytest.mark.os_agnostic
def test_ann_index_builds_when_threshold_crossed(tmp_path: Path) -> None:
    """Crossing index_threshold rows makes upsert build the ANN index, and search still ranks."""
    store = LanceVectorStore(tmp_path, index_threshold=300)
    store.ensure_collection(Collection(name="c", model_id="m", dim=8))
    _bulk(store, "c", 400)

    assert _vector_indices(store, "c"), "expected an ANN index after crossing the threshold"
    hits = store.query(collection="c", vector=tuple([0.5] * 8), k=3)
    assert len(hits) == 3


@pytest.mark.os_agnostic
def test_no_ann_index_below_threshold(tmp_path: Path) -> None:
    """Below the threshold the table stays flat-scan (no index built)."""
    store = LanceVectorStore(tmp_path, index_threshold=300)
    store.ensure_collection(Collection(name="c", model_id="m", dim=8))
    _bulk(store, "c", 50)

    assert _vector_indices(store, "c") == []


@pytest.mark.os_agnostic
def test_existing_index_is_detected_across_instances(tmp_path: Path) -> None:
    """A new store instance reuses the index built earlier and does not rebuild it."""
    first = LanceVectorStore(tmp_path, index_threshold=300)
    first.ensure_collection(Collection(name="c", model_id="m", dim=8))
    _bulk(first, "c", 400)
    assert _vector_indices(first, "c")

    second = LanceVectorStore(tmp_path, index_threshold=300)
    second.upsert(collection="c", chunks=[_chunk("tail", "/tail.md")], vectors=[tuple([0.1] * 8)])

    assert len(_vector_indices(second, "c")) == 1  # reused, not rebuilt (still exactly one index)


@pytest.mark.os_agnostic
def test_bulk_load_leaves_no_unindexed_tail(tmp_path: Path) -> None:
    """The ANN index must COVER every row after a bulk load, not just exist.

    Regression for the tail-scan bug: the index was built on the first batch that
    crossed the threshold, then every later row became an unindexed tail that
    LanceDB brute-force scanned - an index that exists but is not used. upsert must
    fold the tail back in (table.optimize()), so no rows are left unindexed.
    """
    store = LanceVectorStore(tmp_path, index_threshold=300)
    store.ensure_collection(Collection(name="c", model_id="m", dim=8))
    # Many small batches after the index builds: each would otherwise grow the tail.
    for _ in range(8):
        _bulk(store, "c", 100)

    assert _vector_indices(store, "c"), "expected an ANN index after crossing the threshold"
    assert _unindexed_rows(store, "c") == 0, "index must cover all rows, not leave a brute-force-scanned tail"


def test_ann_params_reach_the_query_path(tmp_path: Path) -> None:
    """A lancedb store built with ANN params queries fine (nprobes/refine applied) and, on this
    tiny exact dataset, returns the same top doc as the default - proving the param plumbs
    through without changing correctness."""
    from semdex.domain.ann_tuning import AnnParams

    coll = Collection(name="c", model_id="m", dim=3)
    chunks = [_chunk(text, f"/{i}.md") for i, text in enumerate(("alpha", "beta", "gamma"))]
    vectors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]

    base = LanceVectorStore(tmp_path / "base")
    base.ensure_collection(coll)
    base.upsert(collection="c", chunks=chunks, vectors=vectors)
    base_hit = base.query(collection="c", vector=(1.0, 0.0, 0.0), k=1)[0]
    base.close()

    tuned = LanceVectorStore(tmp_path / "tuned", ann_params=AnnParams(nprobes=5, refine_factor=2))
    tuned.ensure_collection(coll)
    tuned.upsert(collection="c", chunks=chunks, vectors=vectors)
    tuned_hit = tuned.query(collection="c", vector=(1.0, 0.0, 0.0), k=1)[0]
    tuned.close()

    assert base_hit.uri == tuned_hit.uri == "/0.md"
