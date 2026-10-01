"""Composition wiring for the semantic index (testing container)."""

from __future__ import annotations

from pathlib import Path

import pytest

from semdex.composition import IndexServices, build_index_testing
from semdex.domain.models import Collection, SourceRef


@pytest.mark.os_agnostic
def test_build_index_testing_populates_every_field() -> None:
    """build_index_testing wires every required index port to an in-memory adapter.

    ``summarize`` is the only optional port: the summary tier is off by default, so
    it is legitimately ``None`` here (a summarizer is wired only when configured).
    """
    services = build_index_testing()
    assert isinstance(services, IndexServices)
    assert services.summarize is None  # optional tier, off by default
    for field_name in services.__dataclass_fields__:
        if field_name == "summarize":
            continue
        assert getattr(services, field_name) is not None


@pytest.mark.os_agnostic
def test_reader_and_writer_share_one_backing_store() -> None:
    """The reader and writer slices are the same store, so writes are visible."""
    services = build_index_testing()
    coll = Collection(name="c", model_id=services.embedding.model_id, dim=services.embedding.dim)
    services.store_writer.ensure_collection(coll)
    assert services.store_reader.count(collection="c") == 0
    assert services.store_reader.collections() == [coll]


@pytest.mark.os_agnostic
def test_build_index_production_honors_tunable_embedding_dim(tmp_path: Path) -> None:
    """A configured embedding_dim threads through to the placeholder provider."""
    from semdex.composition import build_index_production

    services = build_index_production(tmp_path, embedding_dim=8)
    assert services.embedding.dim == 8


@pytest.mark.os_agnostic
def test_wired_services_index_and_search_end_to_end() -> None:
    """Extract -> chunk -> embed -> upsert -> query composes through the container."""
    documents = {
        Path("/mem/fruit.md"): "banana cherry apple",
        Path("/mem/car.md"): "engine piston valve",
    }
    services = build_index_testing(documents=documents)
    coll = Collection(name="c", model_id=services.embedding.model_id, dim=services.embedding.dim)
    services.store_writer.ensure_collection(coll)

    for path in documents:
        source = SourceRef(uri=str(path), label="curated", content_hash="h", mtime=0.0)
        document = services.extract(source)
        chunks = services.chunk(document, max_tokens=100)
        vectors = services.embedding.embed_passages([chunk.text for chunk in chunks])
        services.store_writer.upsert(collection="c", chunks=chunks, vectors=vectors)

    hits = services.store_reader.query(collection="c", vector=services.embedding.embed_query("apple banana"), k=1)
    assert hits[0].uri == "/mem/fruit.md"
