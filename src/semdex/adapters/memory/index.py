"""In-memory index adapters: deterministic embeddings and a cosine vector store.

Real, dependency-free implementations of the semantic-index ports. They back
the testing composition and serve tiny corpora directly. The production
adapters (LanceDB, sentence-transformers, watchdog, markitdown, chonkie) are
plugged in behind the same ports in phase 2.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from ...domain.errors import CollectionModelMismatchError, ExtractionError, VectorStoreError
from ...domain.models import ChangeEvent, Chunk, Collection, ExtractedDocument, Hit, SourceRef, Vector
from ..discovery.location import from_uri

_DEFAULT_DIM = 256


class InMemoryEmbeddingProvider:
    """Deterministic bag-of-tokens embeddings (stable across processes).

    Tokens are hashed into ``dim`` buckets and the vector is L2-normalized, so
    texts sharing tokens land nearer in cosine space. Uses ``hashlib`` (not the
    salted builtin ``hash``) so a given text always embeds to the same vector.
    """

    def __init__(self, *, model_id: str = "in-memory", dim: int = _DEFAULT_DIM) -> None:
        self._model_id = model_id
        self._dim = dim

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return self._dim

    def embed_passages(self, texts: Sequence[str]) -> list[Vector]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> Vector:
        return self._embed(text)

    def _embed(self, text: str) -> Vector:
        buckets = [0.0] * self._dim
        for token in text.lower().split():
            buckets[self._bucket(token)] += 1.0
        norm = math.sqrt(sum(value * value for value in buckets))
        if norm == 0.0:
            return tuple(buckets)
        return tuple(value / norm for value in buckets)

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self._dim


def _cosine(a: Vector, b: Vector) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore:
    """In-memory vector store implementing the reader and writer port slices.

    Similarity search is exact cosine top-k. A query whose dimension does not
    match the collection is a model mismatch and is rejected rather than
    silently mis-ranked.
    """

    def __init__(self) -> None:
        self._collections: dict[str, Collection] = {}
        self._rows: dict[str, list[tuple[Chunk, Vector]]] = {}

    def close(self) -> None:
        """No-op: an in-memory / json-file store holds no persistent connection."""

    # --- writer slice ---

    def ensure_collection(self, collection: Collection) -> None:
        self._collections[collection.name] = collection
        self._rows.setdefault(collection.name, [])

    def upsert(self, *, collection: str, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        self._require(collection)
        self._rows[collection].extend(zip(chunks, vectors, strict=True))

    def delete_by_source(self, *, collection: str, uri: str) -> None:
        self._require(collection)
        self._rows[collection] = [row for row in self._rows[collection] if row[0].source.uri != uri]

    def swap(self, *, staging: str, target: str) -> None:
        staged = self._require(staging)
        self._collections[target] = Collection(name=target, model_id=staged.model_id, dim=staged.dim)
        self._rows[target] = self._rows.pop(staging)
        del self._collections[staging]

    def compact(self, *, collection: str) -> None:
        """No-op: an in-memory exact scan has no ANN index to maintain."""

    # --- reader slice ---

    def query(self, *, collection: str, vector: Vector, k: int) -> list[Hit]:
        coll = self._require(collection)
        if len(vector) != coll.dim:
            raise CollectionModelMismatchError(
                f"query dim {len(vector)} does not match collection '{collection}' dim {coll.dim}"
            )
        scored = sorted(
            ((_cosine(vector, stored), chunk) for chunk, stored in self._rows[collection]),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return [self._to_hit(chunk, score, collection) for score, chunk in scored[:k]]

    def collections(self) -> list[Collection]:
        return list(self._collections.values())

    def count(self, *, collection: str) -> int:
        return len(self._rows.get(collection, []))

    def source_hashes(self, *, collection: str) -> dict[str, str]:
        self._require(collection)
        return {chunk.source.uri: chunk.source.content_hash for chunk, _ in self._rows[collection]}

    # --- helpers ---

    def _require(self, collection: str) -> Collection:
        try:
            return self._collections[collection]
        except KeyError:
            raise VectorStoreError(f"unknown collection: {collection}") from None

    @staticmethod
    def _to_hit(chunk: Chunk, score: float, collection: str) -> Hit:
        return Hit(
            chunk_text=chunk.text,
            score=score,
            uri=chunk.source.uri,
            ordinal=chunk.ordinal,
            label=chunk.source.label,
            collection=collection,
            summary=chunk.summary,
        )


class InMemoryWatcher:
    """Controllable watcher: a test (or the composition) drives it via emit()."""

    def __init__(self) -> None:
        self.roots: tuple[Path, ...] = ()
        self._on_change: Callable[[ChangeEvent], None] | None = None
        self._stopped = False

    def start(self, *, roots: Sequence[Path], on_change: Callable[[ChangeEvent], None]) -> None:
        self.roots = tuple(roots)
        self._on_change = on_change
        self._stopped = False

    def emit(self, event: ChangeEvent) -> None:
        """Deliver a change event to the callback (no-op once stopped)."""
        if not self._stopped and self._on_change is not None:
            self._on_change(event)

    def stop(self) -> None:
        self._stopped = True


class InMemoryExtractor:
    """Extractor backed by an in-memory ``path -> text`` map (no filesystem)."""

    def __init__(self, documents: Mapping[Path, str]) -> None:
        self._documents = dict(documents)

    def __call__(self, source: SourceRef) -> ExtractedDocument:
        try:
            text = self._documents[from_uri(source.uri)]
        except KeyError:
            raise ExtractionError(f"no in-memory document for {source.uri}") from None
        return ExtractedDocument(source=source, text=text)


def chunk_in_memory(document: ExtractedDocument, *, max_tokens: int = _DEFAULT_DIM) -> list[Chunk]:
    """Split a document into ordered, whitespace-token-bounded chunks.

    The in-memory (testing) chunker; delegates to the same window helper the
    production ``WhitespaceChunker`` fallback uses.
    """
    from ..chunker._base import window_chunks

    return window_chunks(document, max_tokens=max_tokens)


# Static conformance assertions -- pyright verifies each adapter satisfies its port.
if TYPE_CHECKING:
    from ...application.ports import (
        ChunkText,
        EmbeddingProvider,
        Extract,
        VectorStoreReader,
        VectorStoreWriter,
        Watcher,
    )

    _assert_embedding: EmbeddingProvider = InMemoryEmbeddingProvider()
    _assert_store_reader: VectorStoreReader = InMemoryVectorStore()
    _assert_store_writer: VectorStoreWriter = InMemoryVectorStore()
    _assert_watcher: Watcher = InMemoryWatcher()
    _assert_extract: Extract = InMemoryExtractor({})
    _assert_chunk: ChunkText = chunk_in_memory


__all__ = [
    "InMemoryEmbeddingProvider",
    "InMemoryExtractor",
    "InMemoryVectorStore",
    "InMemoryWatcher",
    "chunk_in_memory",
]
