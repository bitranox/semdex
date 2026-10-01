"""Conformance tests for the semantic-index application ports.

Each fake is a minimal in-memory adapter; the helper functions are typed to
the port, so pyright (strict, part of the gate) enforces structural
conformance while pytest exercises real behavior. These fakes are the seed of
the contract-test suite every real adapter will later be run against.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from semdex.application.ports import (
    ChunkText,
    EmbeddingProvider,
    Extract,
    VectorStoreReader,
    VectorStoreWriter,
    Watcher,
)
from semdex.domain.enums import ChangeKind
from semdex.domain.models import (
    ChangeEvent,
    Chunk,
    Collection,
    ExtractedDocument,
    Hit,
    SourceRef,
    Vector,
)

# --------------------------- fakes (adapters) ---------------------------


class FakeExtractor:
    """Extract adapter: pretends every file yields one line of markdown."""

    def __call__(self, source: SourceRef) -> ExtractedDocument:
        return ExtractedDocument(source=source, text=f"text of {Path(source.uri).name}")


class FakeChunker:
    """ChunkText adapter: emits the whole document as a single chunk."""

    def __call__(self, document: ExtractedDocument, *, max_tokens: int = 256) -> list[Chunk]:
        token_count = min(len(document.text), max_tokens)
        return [Chunk(text=document.text, source=document.source, ordinal=0, token_count=token_count)]


class FakeEmbedding:
    """EmbeddingProvider adapter: 3-dim toy embeddings with query/passage split."""

    model_id = "fake-embed"
    dim = 3

    def embed_passages(self, texts: Sequence[str]) -> list[Vector]:
        return [(float(len(t)), 0.0, 0.0) for t in texts]

    def embed_query(self, text: str) -> Vector:
        return (float(len(text)), 1.0, 0.0)


class FakeVectorStore:
    """In-memory store implementing BOTH reader and writer port slices."""

    def __init__(self) -> None:
        self._collections: dict[str, Collection] = {}
        self._rows: dict[str, list[tuple[Chunk, Vector]]] = {}
        self.compactions: list[int] = []  # row count at each compact() call, for cadence tests

    def ensure_collection(self, collection: Collection) -> None:
        self._collections[collection.name] = collection
        self._rows.setdefault(collection.name, [])

    def upsert(self, *, collection: str, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        self._rows[collection].extend(zip(chunks, vectors, strict=True))

    def delete_by_source(self, *, collection: str, uri: str) -> None:
        self._rows[collection] = [row for row in self._rows[collection] if row[0].source.uri != uri]

    def swap(self, *, staging: str, target: str) -> None:
        staged = self._collections.pop(staging)
        self._collections[target] = Collection(name=target, model_id=staged.model_id, dim=staged.dim)
        self._rows[target] = self._rows.pop(staging)

    def compact(self, *, collection: str) -> None:
        self.compactions.append(len(self._rows.get(collection, [])))

    def query(self, *, collection: str, vector: Vector, k: int) -> list[Hit]:
        rows = self._rows.get(collection, [])
        return [
            Hit(
                chunk_text=chunk.text,
                score=1.0,
                uri=chunk.source.uri,
                ordinal=chunk.ordinal,
                label=chunk.source.label,
                collection=collection,
            )
            for chunk, _vector in rows[:k]
        ]

    def collections(self) -> list[Collection]:
        return list(self._collections.values())

    def count(self, *, collection: str) -> int:
        return len(self._rows.get(collection, []))

    def source_hashes(self, *, collection: str) -> dict[str, str]:
        return {chunk.source.uri: chunk.source.content_hash for chunk, _ in self._rows.get(collection, [])}


class FakeWatcher:
    """Watcher adapter: records roots, replays one event, tracks stop."""

    def __init__(self) -> None:
        self.roots: tuple[Path, ...] = ()
        self.stopped = False

    def start(self, *, roots: Sequence[Path], on_change: Callable[[ChangeEvent], None]) -> None:
        self.roots = tuple(roots)
        on_change(ChangeEvent(path=roots[0] / "new.md", kind=ChangeKind.CREATED))

    def stop(self) -> None:
        self.stopped = True


# ----------- helpers typed to the PORT (pyright enforces conformance) -----------


def _extract(port: Extract, source: SourceRef) -> ExtractedDocument:
    return port(source)


def _chunk(port: ChunkText, doc: ExtractedDocument) -> list[Chunk]:
    return port(doc, max_tokens=8)


def _count_via_reader(reader: VectorStoreReader, collection: str) -> int:
    return reader.count(collection=collection)


# ------------------------------- tests -------------------------------


@pytest.mark.os_agnostic
def test_extract_port_returns_document() -> None:
    """An Extract adapter converts a source ref to an ExtractedDocument."""
    ref = SourceRef(uri="/mem/a.md", label="curated", content_hash="h", mtime=0.0)
    doc = _extract(FakeExtractor(), ref)
    assert doc == ExtractedDocument(source=ref, text="text of a.md")


@pytest.mark.os_agnostic
def test_chunk_port_returns_chunks_bounded_by_max_tokens() -> None:
    """A ChunkText adapter turns a document into chunks honoring max_tokens."""
    ref = SourceRef(uri="/mem/a.md", label="curated", content_hash="h", mtime=0.0)
    chunks = _chunk(FakeChunker(), ExtractedDocument(source=ref, text="a long body"))
    assert len(chunks) == 1
    assert chunks[0].token_count == 8
    assert chunks[0].source is ref


@pytest.mark.os_agnostic
def test_embedding_provider_exposes_model_and_splits_query_from_passage() -> None:
    """An EmbeddingProvider reports model_id/dim and embeds queries vs passages."""
    embed: EmbeddingProvider = FakeEmbedding()
    assert embed.model_id == "fake-embed"
    assert embed.dim == 3
    assert embed.embed_passages(["ab", "abc"]) == [(2.0, 0.0, 0.0), (3.0, 0.0, 0.0)]
    assert embed.embed_query("xy") == (2.0, 1.0, 0.0)


@pytest.mark.os_agnostic
def test_vector_store_writer_then_reader_round_trip() -> None:
    """Ensure a collection, upsert chunks, then read them back as Hits."""
    store = FakeVectorStore()
    writer: VectorStoreWriter = store
    reader: VectorStoreReader = store
    coll = Collection(name="memory", model_id="fake-embed", dim=3)

    writer.ensure_collection(coll)
    source = SourceRef(uri="/mem/a.md", label="native", content_hash="h", mtime=0.0)
    chunk = Chunk(text="body", source=source, ordinal=0, token_count=1)
    writer.upsert(collection="memory", chunks=[chunk], vectors=[(1.0, 0.0, 0.0)])

    assert reader.count(collection="memory") == 1
    assert reader.collections() == [coll]
    hits = reader.query(collection="memory", vector=(1.0, 0.0, 0.0), k=5)
    assert hits == [
        Hit(
            chunk_text="body",
            score=1.0,
            uri="/mem/a.md",
            ordinal=0,
            label="native",
            collection="memory",
        )
    ]


@pytest.mark.os_agnostic
def test_vector_store_delete_by_source_removes_rows() -> None:
    """delete_by_source drops every chunk from a given source file."""
    store = FakeVectorStore()
    store.ensure_collection(Collection(name="memory", model_id="fake-embed", dim=3))
    source = SourceRef(uri="/mem/a.md", label="curated", content_hash="h", mtime=0.0)
    store.upsert(
        collection="memory",
        chunks=[Chunk(text="body", source=source, ordinal=0, token_count=1)],
        vectors=[(1.0, 0.0, 0.0)],
    )

    store.delete_by_source(collection="memory", uri="/mem/a.md")

    assert store.count(collection="memory") == 0


@pytest.mark.os_agnostic
def test_vector_store_swap_is_blue_green() -> None:
    """swap atomically renames a staging collection over the target."""
    store = FakeVectorStore()
    store.ensure_collection(Collection(name="memory__staging", model_id="fake-embed", dim=3))

    store.swap(staging="memory__staging", target="memory")

    assert [c.name for c in store.collections()] == ["memory"]


@pytest.mark.os_agnostic
def test_reader_slice_accepts_a_full_store() -> None:
    """A consumer needing only the reader slice accepts the combined store."""
    store = FakeVectorStore()
    store.ensure_collection(Collection(name="memory", model_id="fake-embed", dim=3))
    assert _count_via_reader(store, "memory") == 0


@pytest.mark.os_agnostic
def test_watcher_port_starts_emits_and_stops() -> None:
    """A Watcher records roots, delivers change events, and can be stopped."""
    watcher: Watcher = FakeWatcher()
    seen: list[ChangeEvent] = []
    watcher.start(roots=[Path("/mem")], on_change=seen.append)
    watcher.stop()

    assert seen == [ChangeEvent(path=Path("/mem/new.md"), kind=ChangeKind.CREATED)]
    assert isinstance(watcher, FakeWatcher) and watcher.stopped


@pytest.mark.os_agnostic
def test_index_sources_folds_tail_on_record_cadence_and_at_end() -> None:
    """A record-count policy folds the ANN tail every N chunks during a load, then once at the end."""
    from semdex.application.use_cases import index_sources
    from semdex.domain.models import CompactionPolicy

    store = FakeVectorStore()
    sources = [SourceRef(uri=str(f"/f{i}.md"), label="l", content_hash="h", mtime=0.0) for i in range(7)]
    index_sources(
        extract=FakeExtractor(),
        chunk=FakeChunker(),  # 1 chunk per source
        embedding=FakeEmbedding(),
        store=store,
        collection="c",
        sources=sources,
        max_tokens=100,
        compaction=CompactionPolicy(after_records=3, after_seconds=1e9),  # time trigger off
    )
    # 7 chunks, 1 per source: the policy fires as the running count reaches 3 and 6, and the
    # final compact folds the remainder - so compact() sees 3, then 6, then 7 rows.
    assert store.compactions == [3, 6, 7]


@pytest.mark.os_agnostic
def test_index_sources_without_policy_folds_only_once_at_end() -> None:
    """With no compaction policy, upsert is never interrupted; the tail is folded once at the end."""
    from semdex.application.use_cases import index_sources

    store = FakeVectorStore()
    sources = [SourceRef(uri=str(f"/f{i}.md"), label="l", content_hash="h", mtime=0.0) for i in range(5)]
    index_sources(
        extract=FakeExtractor(),
        chunk=FakeChunker(),
        embedding=FakeEmbedding(),
        store=store,
        collection="c",
        sources=sources,
        max_tokens=100,
    )
    assert store.compactions == [5]  # only the final fold
