"""Contract tests for the sqlite-vec embedded vector store.

Skipped when the optional ``sqlite-vec`` dependency is absent; the ``dev`` extra
installs it, so these run in the normal suite.
"""

# pyright: basic
# The query-path test reaches into the adapter (`_db`, `_collection_id`, `_serialize`) on
# purpose - proving the search engages sqlite-vec's KNN path IS the contract under test.

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sqlite_vec")

from semdex.adapters.vectorstore import SqliteVecStore
from semdex.domain.errors import CollectionModelMismatchError, VectorStoreError
from semdex.domain.models import Chunk, Collection, SourceRef


def _chunk(text: str, path: str, *, ordinal: int = 0) -> Chunk:
    source = SourceRef(uri=str(path), label="curated", content_hash="h", mtime=1.5)
    return Chunk(text=text, source=source, ordinal=ordinal, token_count=len(text.split()))


@pytest.mark.os_agnostic
def test_upsert_persists_across_instances(tmp_path: Path) -> None:
    """A second store instance over the same dir sees the first's writes."""
    coll = Collection(name="c", model_id="m", dim=3)
    first = SqliteVecStore(tmp_path)
    first.ensure_collection(coll)
    first.upsert(collection="c", chunks=[_chunk("body", "/a.md")], vectors=[(1.0, 0.0, 0.0)])

    second = SqliteVecStore(tmp_path)

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
    first = SqliteVecStore(tmp_path)
    first.ensure_collection(Collection(name="c", model_id="m", dim=3))
    first.upsert(
        collection="c",
        chunks=[with_summary, _chunk("plain", "/b.md")],
        vectors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
    )

    second = SqliteVecStore(tmp_path)
    a_hit = second.query(collection="c", vector=(1.0, 0.0, 0.0), k=1)[0]
    b_hit = second.query(collection="c", vector=(0.0, 1.0, 0.0), k=1)[0]
    assert a_hit.summary == "a short doc summary"
    assert b_hit.summary is None


@pytest.mark.os_agnostic
def test_cosine_knn_ranks_closest_first(tmp_path: Path) -> None:
    """The store returns the nearest chunk first with a higher cosine score."""
    store = SqliteVecStore(tmp_path)
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
    first = SqliteVecStore(tmp_path)
    first.ensure_collection(Collection(name="c", model_id="m", dim=3))
    first.upsert(collection="c", chunks=[_chunk("body", "/a.md")], vectors=[(1.0, 0.0, 0.0)])
    first.delete_by_source(collection="c", uri="/a.md")

    assert SqliteVecStore(tmp_path).count(collection="c") == 0


@pytest.mark.os_agnostic
def test_swap_is_blue_green(tmp_path: Path) -> None:
    """swap renames staging over target and adopts the target name."""
    store = SqliteVecStore(tmp_path)
    store.ensure_collection(Collection(name="c__staging", model_id="m", dim=3))
    store.upsert(collection="c__staging", chunks=[_chunk("body", "/a.md")], vectors=[(1.0, 0.0, 0.0)])

    store.swap(staging="c__staging", target="c")

    assert [c.name for c in store.collections()] == ["c"]
    assert store.count(collection="c") == 1


@pytest.mark.os_agnostic
def test_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    """Querying with a wrong-dimension vector is a model mismatch."""
    store = SqliteVecStore(tmp_path)
    store.ensure_collection(Collection(name="c", model_id="m", dim=8))
    with pytest.raises(CollectionModelMismatchError):
        store.query(collection="c", vector=(1.0, 0.0), k=1)


@pytest.mark.os_agnostic
def test_unknown_collection_query_raises(tmp_path: Path) -> None:
    """Querying an unknown collection is a clear VectorStoreError."""
    with pytest.raises(VectorStoreError):
        SqliteVecStore(tmp_path).query(collection="nope", vector=(1.0, 0.0, 0.0), k=1)


@pytest.mark.os_agnostic
def test_empty_store_has_no_collections(tmp_path: Path) -> None:
    """A fresh store dir starts empty."""
    assert SqliteVecStore(tmp_path).collections() == []


@pytest.mark.os_agnostic
def test_query_uses_the_vec0_knn_path_not_a_full_scan(tmp_path: Path) -> None:
    """Search must engage sqlite-vec's KNN (MATCH + k=), not a naive `ORDER BY distance` scan.

    sqlite_vec is an EXACT store (no ANN index), so its latency is O(n) by design - but it must
    still use the vec0 KNN top-k path (heap + SIMD), not a degenerate full-table sort. EXPLAIN
    QUERY PLAN shows sqlite-vec's KNN scan as idxNum 3 (the MATCH+k constraint pushed to vec0); a
    query that failed to engage the KNN would show idxNum 1. Regression guard per "assert the
    index/query path is actually used", mirroring test_vectorstore_mariadb.
    """
    from semdex.adapters.vectorstore.sqlitevec import _serialize

    store = SqliteVecStore(tmp_path)
    store.ensure_collection(Collection(name="c", model_id="m", dim=3))
    store.upsert(
        collection="c",
        chunks=[_chunk(f"t{i}", f"/d{i}.md") for i in range(50)],
        vectors=[(float(i), 1.0, 0.0) for i in range(50)],
    )

    coll_id = store._collection_id("c")  # int read back from our own table - no injection surface
    plan = store._db.execute(
        f"EXPLAIN QUERY PLAN SELECT c.text FROM vec_{coll_id} v JOIN chunks c ON c.id = v.rowid "  # noqa: S608
        f"WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance",
        (_serialize((0.5, 1.0, 0.0)), 5),
    ).fetchall()
    plan_text = " ".join(str(row) for row in plan)
    assert "INDEX 0:3" in plan_text, f"sqlite-vec KNN path not engaged; plan was: {plan_text}"
