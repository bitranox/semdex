"""In-memory index adapters: deterministic embeddings + cosine vector store.

These are real, reusable adapters (test doubles and tiny-corpus backends), the
reference implementations the port contract is checked against.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from semdex.adapters.memory.index import (
    InMemoryEmbeddingProvider,
    InMemoryExtractor,
    InMemoryVectorStore,
    InMemoryWatcher,
    chunk_in_memory,
)
from semdex.domain.enums import ChangeKind
from semdex.domain.errors import CollectionModelMismatchError, ExtractionError
from semdex.domain.models import ChangeEvent, Chunk, Collection, ExtractedDocument, SourceRef


def _source(path: str = "file:///mem/a.md") -> SourceRef:
    return SourceRef(uri=str(path), label="curated", content_hash="h", mtime=0.0)


def _chunk(text: str, path: str = "file:///mem/a.md") -> Chunk:
    return Chunk(text=text, source=_source(path), ordinal=0, token_count=len(text.split()))


# --------------------------- embedding provider ---------------------------


@pytest.mark.os_agnostic
def test_embedding_is_deterministic_and_correct_dim() -> None:
    """The same text embeds to the same vector, with the declared dimension."""
    embed = InMemoryEmbeddingProvider(dim=16)
    assert embed.dim == 16
    assert len(embed.embed_query("hello world")) == 16
    assert embed.embed_query("hello world") == embed.embed_query("hello world")


@pytest.mark.os_agnostic
def test_embedding_places_shared_tokens_closer() -> None:
    """Texts sharing tokens embed nearer than disjoint texts (cosine)."""
    embed = InMemoryEmbeddingProvider(dim=64)

    def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    base = embed.embed_query("alpha beta gamma")
    near = embed.embed_query("alpha beta delta")
    far = embed.embed_query("omega psi chi")
    assert cosine(base, near) > cosine(base, far)


# ----------------------------- vector store -----------------------------


@pytest.mark.os_agnostic
def test_store_ranks_matching_chunk_first() -> None:
    """A query ranks the semantically closest chunk highest."""
    embed = InMemoryEmbeddingProvider(dim=64)
    store = InMemoryVectorStore()
    store.ensure_collection(Collection(name="c", model_id=embed.model_id, dim=embed.dim))
    chunks = [_chunk("apple banana cherry", "file:///mem/fruit.md"), _chunk("engine piston valve", "file:///mem/car.md")]
    store.upsert(collection="c", chunks=chunks, vectors=embed.embed_passages([c.text for c in chunks]))

    hits = store.query(collection="c", vector=embed.embed_query("banana cherry apple"), k=2)

    assert hits[0].uri == "file:///mem/fruit.md"
    assert hits[0].score >= hits[1].score


@pytest.mark.os_agnostic
def test_store_hit_carries_chunk_ordinal() -> None:
    """A hit surfaces the matched chunk's ordinal so a caller can locate it in the source."""
    embed = InMemoryEmbeddingProvider(dim=64)
    store = InMemoryVectorStore()
    store.ensure_collection(Collection(name="c", model_id=embed.model_id, dim=embed.dim))
    chunk = Chunk(text="apple banana cherry", source=_source("file:///mem/fruit.md"), ordinal=7, token_count=3)
    store.upsert(collection="c", chunks=[chunk], vectors=embed.embed_passages([chunk.text]))

    hits = store.query(collection="c", vector=embed.embed_query("banana cherry apple"), k=1)

    assert hits[0].ordinal == 7


@pytest.mark.os_agnostic
def test_store_source_hashes_maps_uri_to_content_hash() -> None:
    """source_hashes reports each indexed source uri and its content hash (for reconcile)."""
    embed = InMemoryEmbeddingProvider(dim=8)
    store = InMemoryVectorStore()
    store.ensure_collection(Collection(name="c", model_id=embed.model_id, dim=embed.dim))
    a = Chunk(
        text="alpha",
        source=SourceRef(uri="file:///a.md", label="", content_hash="ha", mtime=0.0),
        ordinal=0,
        token_count=1,
    )
    b = Chunk(
        text="beta",
        source=SourceRef(uri="file:///b.md", label="", content_hash="hb", mtime=0.0),
        ordinal=0,
        token_count=1,
    )
    store.upsert(collection="c", chunks=[a, b], vectors=embed.embed_passages(["alpha", "beta"]))

    assert store.source_hashes(collection="c") == {"file:///a.md": "ha", "file:///b.md": "hb"}


@pytest.mark.os_agnostic
def test_store_rejects_dimension_mismatch() -> None:
    """Querying with a vector of the wrong dimension is a model mismatch."""
    store = InMemoryVectorStore()
    store.ensure_collection(Collection(name="c", model_id="m", dim=8))
    with pytest.raises(CollectionModelMismatchError):
        store.query(collection="c", vector=(1.0, 0.0), k=1)


@pytest.mark.os_agnostic
def test_store_count_delete_and_collections() -> None:
    """Count reflects upserts, delete_by_source removes rows, collections lists."""
    embed = InMemoryEmbeddingProvider(dim=8)
    store = InMemoryVectorStore()
    coll = Collection(name="c", model_id=embed.model_id, dim=embed.dim)
    store.ensure_collection(coll)
    chunk = _chunk("hi", "file:///mem/a.md")
    store.upsert(collection="c", chunks=[chunk], vectors=[embed.embed_query("hi")])

    assert store.count(collection="c") == 1
    assert store.collections() == [coll]

    store.delete_by_source(collection="c", uri="file:///mem/a.md")
    assert store.count(collection="c") == 0


@pytest.mark.os_agnostic
def test_store_swap_adopts_target_identity() -> None:
    """Blue-green swap renames staging over target, adopting the target name."""
    store = InMemoryVectorStore()
    store.ensure_collection(Collection(name="c__staging", model_id="m", dim=8))

    store.swap(staging="c__staging", target="c")

    assert [c.name for c in store.collections()] == ["c"]


# ------------------------------- watcher -------------------------------


@pytest.mark.os_agnostic
def test_watcher_delivers_emitted_events_until_stopped() -> None:
    """The watcher forwards emitted events to the callback and honors stop()."""
    watcher = InMemoryWatcher()
    seen: list[ChangeEvent] = []
    watcher.start(roots=[Path("/mem")], on_change=seen.append)

    watcher.emit(ChangeEvent(path=Path("/mem/a.md"), kind=ChangeKind.MODIFIED))
    watcher.stop()
    watcher.emit(ChangeEvent(path=Path("/mem/b.md"), kind=ChangeKind.CREATED))  # ignored after stop

    assert seen == [ChangeEvent(path=Path("/mem/a.md"), kind=ChangeKind.MODIFIED)]


# ---------------------------- extractor / chunker ----------------------------


@pytest.mark.os_agnostic
def test_extractor_returns_registered_document() -> None:
    """The in-memory extractor returns text registered for the source path."""
    ref = _source("file:///mem/a.md")
    extractor = InMemoryExtractor({Path("/mem/a.md"): "# Title\nbody"})
    assert extractor(ref) == ExtractedDocument(source=ref, text="# Title\nbody")


@pytest.mark.os_agnostic
def test_extractor_raises_for_unknown_path() -> None:
    """An unregistered path is an extraction error, not a silent empty doc."""
    extractor = InMemoryExtractor({})
    with pytest.raises(ExtractionError):
        extractor(_source("file:///mem/missing.md"))


@pytest.mark.os_agnostic
def test_chunker_splits_into_bounded_chunks() -> None:
    """The chunker splits a document into token-bounded chunks preserving order."""
    ref = _source("file:///mem/a.md")
    doc = ExtractedDocument(source=ref, text="one two three four five six")
    chunks = chunk_in_memory(doc, max_tokens=2)

    assert [c.text for c in chunks] == ["one two", "three four", "five six"]
    assert [c.ordinal for c in chunks] == [0, 1, 2]
    assert all(c.token_count <= 2 for c in chunks)
    assert all(c.source is ref for c in chunks)
