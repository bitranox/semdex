"""Embedded single-file vector store backed by sqlite + the sqlite-vec extension.

Incremental (no whole-file rewrite), bounded memory, and a native cosine KNN
index via ``vec0`` virtual tables - the embedded upgrade over the JSON store for
larger local corpora. Needs the optional ``semdex[sqlite]`` extra.

Schema: a ``collections`` table (name -> id, model_id, dim), a shared ``chunks``
table, and one ``vec_<id>`` virtual table per collection (each pinned to its own
dimension, so multiple models coexist). The vec table is keyed by the collection
id, not its name, so blue-green ``swap`` is a rename with no table move.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from ...domain.errors import CollectionModelMismatchError, VectorStoreError
from ...domain.models import Collection, Hit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...application.ports import VectorStore
    from ...domain.models import Chunk, Vector

_STORE_FILENAME = "store.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    model_id TEXT NOT NULL,
    dim INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection TEXT NOT NULL,
    path TEXT NOT NULL,
    label TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    mtime REAL NOT NULL,
    ordinal INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    text TEXT NOT NULL,
    summary TEXT
);
CREATE INDEX IF NOT EXISTS chunks_by_source ON chunks (collection, path);
"""


def _connect(path: Path) -> sqlite3.Connection:
    try:
        import sqlite_vec  # type: ignore  # optional dep; see module docstring
    except ImportError as exc:
        raise VectorStoreError("sqlite-vec is not installed; install semdex[sqlite]") from exc
    # check_same_thread=False: the MCP server runs tools in a worker thread while the
    # connection is created on the main thread. sqlite3 (serialized threading mode) makes
    # cross-thread use safe; the strict same-thread guard would otherwise reject it.
    db = sqlite3.connect(path, check_same_thread=False)
    db.enable_load_extension(True)
    sqlite_vec.load(db)  # type: ignore[no-untyped-call]
    db.enable_load_extension(False)
    return db


def _serialize(vector: Vector) -> bytes:
    import sqlite_vec  # type: ignore  # optional dep; see module docstring

    return sqlite_vec.serialize_float32(list(vector))  # type: ignore[no-untyped-call, no-any-return]


class SqliteVecStore:
    """Vector store over sqlite-vec; implements the reader and writer slices."""

    def __init__(self, store_dir: Path) -> None:
        store_dir.mkdir(parents=True, exist_ok=True)
        self._db = _connect(store_dir / _STORE_FILENAME)
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def close(self) -> None:
        """Close the sqlite connection + release its file lock (called on server shutdown)."""
        self._db.close()

    # --- writer slice ---

    def ensure_collection(self, collection: Collection) -> None:
        if self._collection_id(collection.name) is not None:
            return
        cursor = self._db.execute(
            "INSERT INTO collections (name, model_id, dim) VALUES (?, ?, ?)",
            (collection.name, collection.model_id, collection.dim),
        )
        coll_id = int(cursor.lastrowid or 0)
        self._db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_{coll_id} "
            f"USING vec0(embedding float[{collection.dim}] distance_metric=cosine)"
        )
        self._db.commit()

    def upsert(self, *, collection: str, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        coll_id = self._require_id(collection)
        for chunk, vector in zip(chunks, vectors, strict=True):
            cursor = self._db.execute(
                "INSERT INTO chunks "
                "(collection, path, label, content_hash, mtime, ordinal, token_count, text, summary) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    collection,
                    chunk.source.uri,
                    chunk.source.label,
                    chunk.source.content_hash,
                    chunk.source.mtime,
                    chunk.ordinal,
                    chunk.token_count,
                    chunk.text,
                    chunk.summary,
                ),
            )
            self._db.execute(
                f"INSERT INTO vec_{coll_id} (rowid, embedding) VALUES (?, ?)",
                (int(cursor.lastrowid or 0), _serialize(vector)),
            )
        self._db.commit()

    def delete_by_source(self, *, collection: str, uri: str) -> None:
        coll_id = self._require_id(collection)
        ids = [
            int(row[0])
            for row in self._db.execute(
                "SELECT id FROM chunks WHERE collection = ? AND path = ?", (collection, uri)
            ).fetchall()
        ]
        for chunk_id in ids:
            self._db.execute(f"DELETE FROM vec_{coll_id} WHERE rowid = ?", (chunk_id,))
        self._db.execute("DELETE FROM chunks WHERE collection = ? AND path = ?", (collection, uri))
        self._db.commit()

    def swap(self, *, staging: str, target: str) -> None:
        self._require_id(staging)
        target_id = self._collection_id(target)
        if target_id is not None:
            self._db.execute(f"DROP TABLE IF EXISTS vec_{target_id}")
            self._db.execute("DELETE FROM chunks WHERE collection = ?", (target,))
            self._db.execute("DELETE FROM collections WHERE name = ?", (target,))
        self._db.execute("UPDATE collections SET name = ? WHERE name = ?", (target, staging))
        self._db.execute("UPDATE chunks SET collection = ? WHERE collection = ?", (target, staging))
        self._db.commit()

    def compact(self, *, collection: str) -> None:
        """No-op: sqlite_vec does exact KNN (vec0), there is no ANN index to fold."""

    # --- reader slice ---

    def query(self, *, collection: str, vector: Vector, k: int) -> list[Hit]:
        row = self._db.execute("SELECT id, dim FROM collections WHERE name = ?", (collection,)).fetchone()
        if row is None:
            raise VectorStoreError(f"unknown collection: {collection}")
        coll_id, dim = int(row[0]), int(row[1])
        if len(vector) != dim:
            raise CollectionModelMismatchError(
                f"query dim {len(vector)} does not match collection '{collection}' dim {dim}"
            )
        rows = self._db.execute(
            f"SELECT c.text, c.path, c.ordinal, c.label, c.summary, v.distance FROM vec_{coll_id} v "
            f"JOIN chunks c ON c.id = v.rowid WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance",
            (_serialize(vector), k),
        ).fetchall()
        return [
            Hit(
                chunk_text=str(text),
                score=1.0 - float(distance),  # cosine distance -> cosine similarity
                uri=str(path),
                ordinal=int(ordinal),
                label=str(label),
                collection=collection,
                summary=None if summary is None else str(summary),
            )
            for text, path, ordinal, label, summary, distance in rows
        ]

    def collections(self) -> list[Collection]:
        rows = self._db.execute("SELECT name, model_id, dim FROM collections ORDER BY id").fetchall()
        return [Collection(name=str(name), model_id=str(model_id), dim=int(dim)) for name, model_id, dim in rows]

    def count(self, *, collection: str) -> int:
        row = self._db.execute("SELECT COUNT(*) FROM chunks WHERE collection = ?", (collection,)).fetchone()
        return int(row[0])

    def source_hashes(self, *, collection: str) -> dict[str, str]:
        self._require_id(collection)
        rows = self._db.execute(
            "SELECT DISTINCT path, content_hash FROM chunks WHERE collection = ?", (collection,)
        ).fetchall()
        return {str(path): str(content_hash) for path, content_hash in rows}

    # --- helpers ---

    def _collection_id(self, collection: str) -> int | None:
        row = self._db.execute("SELECT id FROM collections WHERE name = ?", (collection,)).fetchone()
        return int(row[0]) if row is not None else None

    def _require_id(self, collection: str) -> int:
        coll_id = self._collection_id(collection)
        if coll_id is None:
            raise VectorStoreError(f"unknown collection: {collection}")
        return coll_id


# Static conformance assertion -- SqliteVecStore satisfies the combined VectorStore port.
if TYPE_CHECKING:
    _assert_store: VectorStore = SqliteVecStore(Path())


__all__ = [
    "SqliteVecStore",
]
