"""Persistent orjson-file vector store (phase-1 embedded backend)."""

from __future__ import annotations

from pathlib import Path

import pytest

from semdex.adapters.vectorstore import JsonVectorStore
from semdex.domain.errors import CollectionModelMismatchError, VectorStoreError
from semdex.domain.models import Chunk, Collection, SourceRef


def _chunk(text: str, path: str) -> Chunk:
    source = SourceRef(uri=str(path), label="curated", content_hash="h", mtime=1.5)
    return Chunk(text=text, source=source, ordinal=0, token_count=len(text.split()))


@pytest.mark.os_agnostic
def test_close_is_a_noop_and_leaves_the_store_usable(tmp_path: Path) -> None:
    """The json store holds no persistent handle, so close() is a safe no-op."""
    store = JsonVectorStore(tmp_path)
    store.ensure_collection(Collection(name="c", model_id="m", dim=3))
    store.upsert(collection="c", chunks=[_chunk("body", "/a.md")], vectors=[(1.0, 0.0, 0.0)])
    store.close()
    store.close()  # idempotent
    assert store.count(collection="c") == 1  # still usable after close


@pytest.mark.os_agnostic
def test_upsert_persists_across_instances(tmp_path: Path) -> None:
    """A second store instance loads collections and rows written by the first."""
    coll = Collection(name="c", model_id="m", dim=3)
    first = JsonVectorStore(tmp_path)
    first.ensure_collection(coll)
    first.upsert(collection="c", chunks=[_chunk("body", "/a.md")], vectors=[(1.0, 0.0, 0.0)])

    second = JsonVectorStore(tmp_path)

    assert second.count(collection="c") == 1
    assert second.collections() == [coll]
    hits = second.query(collection="c", vector=(1.0, 0.0, 0.0), k=1)
    assert hits[0].uri == "/a.md"
    assert hits[0].label == "curated"


@pytest.mark.os_agnostic
def test_summary_round_trips_and_persists(tmp_path: Path) -> None:
    """A chunk's per-document summary survives upsert -> query and a reload (None stays None)."""
    coll = Collection(name="c", model_id="m", dim=3)
    with_summary = Chunk(
        text="body",
        source=SourceRef(uri="/a.md", label="curated", content_hash="h", mtime=1.5),
        ordinal=0,
        token_count=1,
        summary="a short doc summary",
    )
    first = JsonVectorStore(tmp_path)
    first.ensure_collection(coll)
    first.upsert(
        collection="c", chunks=[with_summary, _chunk("plain", "/b.md")], vectors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    )

    second = JsonVectorStore(tmp_path)
    a_hit = second.query(collection="c", vector=(1.0, 0.0, 0.0), k=1)[0]
    b_hit = second.query(collection="c", vector=(0.0, 1.0, 0.0), k=1)[0]
    assert a_hit.summary == "a short doc summary"
    assert b_hit.summary is None  # a chunk indexed with no summary reads back as None


@pytest.mark.os_agnostic
def test_delete_persists_across_instances(tmp_path: Path) -> None:
    """A delete is durable: a fresh instance sees the row gone."""
    first = JsonVectorStore(tmp_path)
    first.ensure_collection(Collection(name="c", model_id="m", dim=3))
    first.upsert(collection="c", chunks=[_chunk("body", "/a.md")], vectors=[(1.0, 0.0, 0.0)])
    first.delete_by_source(collection="c", uri="/a.md")

    assert JsonVectorStore(tmp_path).count(collection="c") == 0


@pytest.mark.os_agnostic
def test_swap_persists_across_instances(tmp_path: Path) -> None:
    """A blue-green swap is durable and adopts the target name."""
    first = JsonVectorStore(tmp_path)
    first.ensure_collection(Collection(name="c__staging", model_id="m", dim=3))
    first.swap(staging="c__staging", target="c")

    assert [c.name for c in JsonVectorStore(tmp_path).collections()] == ["c"]


@pytest.mark.os_agnostic
def test_inherits_dimension_mismatch_enforcement(tmp_path: Path) -> None:
    """The persistent store keeps the reader slice's mismatch guard."""
    store = JsonVectorStore(tmp_path)
    store.ensure_collection(Collection(name="c", model_id="m", dim=8))
    with pytest.raises(CollectionModelMismatchError):
        store.query(collection="c", vector=(1.0, 0.0), k=1)


@pytest.mark.os_agnostic
def test_empty_store_dir_starts_empty(tmp_path: Path) -> None:
    """A store over a fresh directory has no collections."""
    assert JsonVectorStore(tmp_path).collections() == []


@pytest.mark.os_agnostic
def test_malformed_json_raises_vector_store_error(tmp_path: Path) -> None:
    """A store file that is not valid JSON is a clean VectorStoreError."""
    (tmp_path / "store.json").write_bytes(b"not json {{")
    with pytest.raises(VectorStoreError):
        JsonVectorStore(tmp_path)


@pytest.mark.os_agnostic
def test_schema_violation_raises_vector_store_error(tmp_path: Path) -> None:
    """A store whose JSON violates the persisted schema is rejected, not silently used."""
    (tmp_path / "store.json").write_bytes(b'{"collections": {"c": {"model_id": "m"}}, "rows": {}}')
    with pytest.raises(VectorStoreError):
        JsonVectorStore(tmp_path)
