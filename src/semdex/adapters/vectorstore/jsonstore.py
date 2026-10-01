"""Persistent embedded vector store backed by a single orjson file.

The phase-1 default store: it reuses the exact cosine search, dimension-mismatch
guard, delete, and blue-green swap logic of the in-memory store, adding load on
open and an atomic write after each mutation. Right for small corpora (the
bitranox memory case) and cross-platform with no server. LanceDB / sqlite-vec /
pgvector / mariadb plug in behind the same ports for larger corpora in phase 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import orjson
from pydantic import BaseModel, Field, ValidationError

from ...domain.errors import VectorStoreError
from ...domain.models import Chunk, Collection, SourceRef, Vector
from ..memory.index import InMemoryVectorStore

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

_STORE_FILENAME = "store.json"


class _PersistedCollection(BaseModel):
    """On-disk schema for one collection's pinned model and dimension."""

    model_id: str
    dim: int = Field(gt=0)


class _PersistedRow(BaseModel):
    """On-disk schema for one stored chunk plus its vector."""

    text: str
    path: str
    label: str
    content_hash: str
    mtime: float
    ordinal: int
    token_count: int
    vector: list[float]
    # Nullable per-document summary (the opt-in tier); absent/None when tier off.
    summary: str | None = None


class _PersistedStore(BaseModel):
    """On-disk schema for the whole store; validated on load."""

    collections: dict[str, _PersistedCollection] = Field(default_factory=dict)
    rows: dict[str, list[_PersistedRow]] = Field(default_factory=dict)


class JsonVectorStore(InMemoryVectorStore):
    """In-memory vector store logic plus durable orjson-file persistence."""

    def __init__(self, store_dir: Path) -> None:
        super().__init__()
        self._path = store_dir / _STORE_FILENAME
        self._load()

    def ensure_collection(self, collection: Collection) -> None:
        super().ensure_collection(collection)
        self._persist()

    def upsert(self, *, collection: str, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        super().upsert(collection=collection, chunks=chunks, vectors=vectors)
        self._persist()

    def delete_by_source(self, *, collection: str, uri: str) -> None:
        super().delete_by_source(collection=collection, uri=uri)
        self._persist()

    def swap(self, *, staging: str, target: str) -> None:
        super().swap(staging=staging, target=target)
        self._persist()

    # --- persistence ---

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            model = _PersistedStore.model_validate(orjson.loads(self._path.read_bytes()))
        except (orjson.JSONDecodeError, ValidationError) as exc:
            raise VectorStoreError(f"corrupt vector store at {self._path}: {exc}") from exc
        self._collections = {
            name: Collection(name=name, model_id=coll.model_id, dim=coll.dim)
            for name, coll in model.collections.items()
        }
        self._rows = {name: [_from_row(row) for row in rows] for name, rows in model.rows.items()}

    def _persist(self) -> None:
        model = _PersistedStore(
            collections={
                name: _PersistedCollection(model_id=c.model_id, dim=c.dim) for name, c in self._collections.items()
            },
            rows={name: [_to_row(chunk, vector) for chunk, vector in rows] for name, rows in self._rows.items()},
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_bytes(orjson.dumps(model.model_dump()))
        tmp.replace(self._path)  # atomic swap-in; a crash mid-write cannot corrupt the store


def _to_row(chunk: Chunk, vector: Vector) -> _PersistedRow:
    return _PersistedRow(
        text=chunk.text,
        path=chunk.source.uri,
        label=chunk.source.label,
        content_hash=chunk.source.content_hash,
        mtime=chunk.source.mtime,
        ordinal=chunk.ordinal,
        token_count=chunk.token_count,
        vector=list(vector),
        summary=chunk.summary,
    )


def _from_row(row: _PersistedRow) -> tuple[Chunk, Vector]:
    source = SourceRef(uri=row.path, label=row.label, content_hash=row.content_hash, mtime=row.mtime)
    chunk = Chunk(text=row.text, source=source, ordinal=row.ordinal, token_count=row.token_count, summary=row.summary)
    return chunk, tuple(row.vector)


# Static conformance assertions -- JsonVectorStore satisfies both store slices.
if TYPE_CHECKING:
    from pathlib import Path as _Path

    from ...application.ports import VectorStoreReader, VectorStoreWriter

    _assert_reader: VectorStoreReader = JsonVectorStore(_Path())
    _assert_writer: VectorStoreWriter = JsonVectorStore(_Path())


__all__ = [
    "JsonVectorStore",
]
