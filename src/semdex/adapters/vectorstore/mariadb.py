"""MariaDB 11.8 native VECTOR server vector store.

A ``collections`` metadata table plus one ``chunks_<id>`` table per collection
(each with a ``VECTOR(dim)`` column and a ``DISTANCE=cosine`` MHNSW index),
keyed by the collection id so blue-green ``swap`` is a rename with no table
move. Cosine KNN via ``VEC_DISTANCE_COSINE`` - the index MUST be built for the
same distance or the optimizer falls back to a full scan.
Needs the optional ``semdex[mariadb]`` extra (PyMySQL, pure-Python) and a
reachable MariaDB 11.7+; the connection URL (``mysql://user:pass@host:port/db``)
comes from ``[vector_store].dsn``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from ...domain.ann_tuning import AnnParams
from ...domain.errors import CollectionModelMismatchError, VectorStoreError
from ...domain.models import Collection, Hit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...domain.models import Chunk, Vector


def _connect(dsn: str) -> Any:
    try:
        import pymysql  # type: ignore  # optional dep; see module docstring
    except ImportError as exc:
        raise VectorStoreError("PyMySQL is not installed; install semdex[mariadb]") from exc
    url = urlparse(dsn)
    return pymysql.connect(  # type: ignore[no-untyped-call]
        host=url.hostname or "127.0.0.1",
        port=url.port or 3306,
        user=url.username or "root",
        password=url.password or "",
        database=(url.path or "").lstrip("/") or None,
        autocommit=True,
    )


def _vector_text(vector: Vector) -> str:
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


class MariaDbVectorStore:
    """Vector store over MariaDB native VECTOR; reader and writer slices."""

    def __init__(self, dsn: str, *, ann_params: AnnParams | None = None, connection: Any | None = None) -> None:
        """Open the store.

        Args:
            dsn: MariaDB connection URL.
            ann_params: resolved ANN knobs; ``None`` leaves every driver default.
        connection: an already-open DB-API connection to use instead of dialling ``dsn``.
            The database is this adapter's one external edge, so it is the seam a test
            substitutes at - injecting here keeps the SQL the adapter emits under test without
            reaching into its internals.
        """
        self._conn = _connect(dsn) if connection is None else connection
        self._execute(
            "CREATE TABLE IF NOT EXISTS collections ("
            " id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255) UNIQUE NOT NULL,"
            " model_id TEXT NOT NULL, dim INT NOT NULL)"
        )
        # Kept, not just consumed: the build-time knob is needed later, when the index is
        # created in ensure_collection.
        self._ann = ann = ann_params or AnnParams()
        if ann.ef_search is not None:
            # Session-level MHNSW recall knob (ann_recall preset / override); fixed per store.
            # int() from a validated Field(gt=0), so the literal is injection-safe.
            self._execute(f"SET SESSION mhnsw_ef_search = {int(ann.ef_search)}")

    @property
    def ann_params(self) -> AnnParams:
        """The ANN knobs this store was actually opened with.

        Introspection, so a caller can ask what a store is configured with rather than assume it
        got what it passed. A benchmark that resolves the right params and then forgets to hand
        them to the factory measures something else entirely and reports it as fact - which is
        what happened to the published lancedb rows.
        """
        return self._ann

    def close(self) -> None:
        """Close the MariaDB connection (called on server shutdown)."""
        self._conn.close()

    # --- writer slice ---

    def ensure_collection(self, collection: Collection) -> None:
        if self._collection_id(collection.name) is not None:
            return
        cursor = self._execute(
            "INSERT INTO collections (name, model_id, dim) VALUES (%s, %s, %s)",
            (collection.name, collection.model_id, collection.dim),
        )
        coll_id = int(cursor.lastrowid)
        self._execute(
            f"CREATE TABLE chunks_{coll_id} ("
            f" id INT AUTO_INCREMENT PRIMARY KEY, path TEXT NOT NULL, label TEXT NOT NULL,"
            f" content_hash TEXT NOT NULL, mtime DOUBLE NOT NULL, ordinal INT NOT NULL,"
            f" token_count INT NOT NULL, chunk_text TEXT NOT NULL, summary TEXT,"
            # DISTANCE=cosine is REQUIRED: the MHNSW index defaults to euclidean, and the
            # optimizer only uses it when the query's distance function matches. Our query is
            # VEC_DISTANCE_COSINE, so without this the index is dead and every search is a full
            # O(n) scan (measured: 21 s/query at 250K rows vs single-digit ms with the index).
            f" embedding VECTOR({collection.dim}) NOT NULL,"
            f" VECTOR INDEX (embedding) DISTANCE=cosine{self._index_options()})"
        )

    def _index_options(self) -> str:
        """MHNSW's build-time neighbour count, or empty for MariaDB's own default.

        MariaDB spells it ``M=`` inside the index definition rather than in a WITH clause, and
        has no ``ef_construction`` equivalent, so only ``m`` is applicable here.
        """
        return f" M={int(self._ann.m)}" if self._ann.m is not None else ""

    def upsert(self, *, collection: str, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        coll_id = self._require_id(collection)
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._execute(
                f"INSERT INTO chunks_{coll_id} "
                f"(path, label, content_hash, mtime, ordinal, token_count, chunk_text, summary, embedding) "
                f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s, VEC_FromText(%s))",
                (
                    chunk.source.uri,
                    chunk.source.label,
                    chunk.source.content_hash,
                    chunk.source.mtime,
                    chunk.ordinal,
                    chunk.token_count,
                    chunk.text,
                    chunk.summary,
                    _vector_text(vector),
                ),
            )

    def delete_by_source(self, *, collection: str, uri: str) -> None:
        coll_id = self._require_id(collection)
        self._execute(f"DELETE FROM chunks_{coll_id} WHERE path = %s", (uri,))

    def swap(self, *, staging: str, target: str) -> None:
        self._require_id(staging)
        target_id = self._collection_id(target)
        if target_id is not None:
            self._execute(f"DROP TABLE IF EXISTS chunks_{target_id}")
            self._execute("DELETE FROM collections WHERE name = %s", (target,))
        self._execute("UPDATE collections SET name = %s WHERE name = %s", (target, staging))

    def compact(self, *, collection: str) -> None:
        """No-op: MariaDB maintains the native VECTOR index incrementally on every insert."""

    # --- reader slice ---

    def query(self, *, collection: str, vector: Vector, k: int) -> list[Hit]:
        cursor = self._execute("SELECT id, dim FROM collections WHERE name = %s", (collection,))
        row = cursor.fetchone()
        if row is None:
            raise VectorStoreError(f"unknown collection: {collection}")
        coll_id, dim = int(row[0]), int(row[1])
        if len(vector) != dim:
            raise CollectionModelMismatchError(
                f"query dim {len(vector)} does not match collection '{collection}' dim {dim}"
            )
        cursor = self._execute(
            f"SELECT chunk_text, path, ordinal, label, summary, "
            f"VEC_DISTANCE_COSINE(embedding, VEC_FromText(%s)) AS dist "
            f"FROM chunks_{coll_id} ORDER BY dist LIMIT %s",
            (_vector_text(vector), k),
        )
        return [
            Hit(
                chunk_text=str(text),
                score=1.0 - float(dist),  # cosine distance -> similarity
                uri=str(path),
                ordinal=int(ordinal),
                label=str(label),
                collection=collection,
                summary=None if summary is None else str(summary),
            )
            for text, path, ordinal, label, summary, dist in cursor.fetchall()
        ]

    def collections(self) -> list[Collection]:
        cursor = self._execute("SELECT name, model_id, dim FROM collections ORDER BY id")
        return [
            Collection(name=str(name), model_id=str(model_id), dim=int(dim))
            for name, model_id, dim in cursor.fetchall()
        ]

    def count(self, *, collection: str) -> int:
        coll_id = self._collection_id(collection)
        if coll_id is None:
            return 0
        cursor = self._execute(f"SELECT COUNT(*) FROM chunks_{coll_id}")
        return int(cursor.fetchone()[0])

    def source_hashes(self, *, collection: str) -> dict[str, str]:
        coll_id = self._require_id(collection)
        cursor = self._execute(f"SELECT DISTINCT path, content_hash FROM chunks_{coll_id}")
        return {str(path): str(content_hash) for path, content_hash in cursor.fetchall()}

    # --- helpers ---

    def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> Any:
        cursor = self._conn.cursor()
        cursor.execute(sql, params)
        return cursor

    def _collection_id(self, collection: str) -> int | None:
        cursor = self._execute("SELECT id FROM collections WHERE name = %s", (collection,))
        row = cursor.fetchone()
        return int(row[0]) if row is not None else None

    def _require_id(self, collection: str) -> int:
        coll_id = self._collection_id(collection)
        if coll_id is None:
            raise VectorStoreError(f"unknown collection: {collection}")
        return coll_id


if TYPE_CHECKING:
    from ...application.ports import VectorStore

    _assert_store: VectorStore = MariaDbVectorStore("")


__all__ = [
    "MariaDbVectorStore",
]
