"""Postgres + pgvector server vector store.

A ``collections`` metadata table plus one ``chunks_<id>`` table per collection
(each with a ``vector(dim)`` column and an HNSW cosine index), keyed by the
collection id so blue-green ``swap`` is a rename with no table move. Needs the
optional ``semdex[pg]`` extra and a reachable Postgres with the ``vector``
extension; the DSN comes from ``[vector_store].dsn``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...domain.ann_tuning import AnnParams
from ...domain.errors import CollectionModelMismatchError, VectorStoreError
from ...domain.models import Collection, Hit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...domain.models import Chunk, Vector

_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS collections (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    model_id TEXT NOT NULL,
    dim INTEGER NOT NULL
);
"""


def _connect(dsn: str) -> Any:
    try:
        import psycopg  # type: ignore  # optional dep; see module docstring
        from pgvector.psycopg import register_vector  # type: ignore
    except ImportError as exc:
        raise VectorStoreError("pgvector/psycopg is not installed; install semdex[pg]") from exc
    connection = psycopg.connect(dsn, autocommit=True)  # type: ignore[no-untyped-call]
    connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(connection)  # type: ignore[no-untyped-call]
    return connection


def _to_vector(vector: Vector) -> Any:
    from pgvector import Vector as PgVector  # type: ignore

    return PgVector(list(vector))  # type: ignore[no-untyped-call]


class PgVectorStore:
    """Vector store over Postgres + pgvector; implements reader and writer slices."""

    def __init__(self, dsn: str, *, ann_params: AnnParams | None = None, connection: Any | None = None) -> None:
        """Open the store.

        Args:
            dsn: libpq connection string.
            ann_params: resolved ANN knobs; ``None`` leaves every driver default.
        connection: an already-open DB-API connection to use instead of dialling ``dsn``.
            The database is this adapter's one external edge, so it is the seam a test
            substitutes at - injecting here keeps the SQL the adapter emits under test without
            reaching into its internals.
        """
        self._conn = _connect(dsn) if connection is None else connection
        self._conn.execute(_SCHEMA)
        # Kept, not just consumed: the build-time knobs are needed later, when the index is
        # created in ensure_collection.
        self._ann = ann = ann_params or AnnParams()
        if ann.ef_search is not None:
            # Session-level HNSW recall knob (ann_recall preset / override); fixed per store, so
            # set once. int() from a validated Field(gt=0), so the literal is injection-safe.
            self._conn.execute(f"SET hnsw.ef_search = {int(ann.ef_search)}")

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
        """Close the Postgres connection (called on server shutdown)."""
        self._conn.close()

    # --- writer slice ---

    def ensure_collection(self, collection: Collection) -> None:
        if self._collection_id(collection.name) is not None:
            return
        row = self._conn.execute(
            "INSERT INTO collections (name, model_id, dim) VALUES (%s, %s, %s) RETURNING id",
            (collection.name, collection.model_id, collection.dim),
        ).fetchone()
        coll_id = int(row[0])
        self._conn.execute(
            f"CREATE TABLE chunks_{coll_id} ("
            f" id SERIAL PRIMARY KEY, path TEXT NOT NULL, label TEXT NOT NULL,"
            f" content_hash TEXT NOT NULL, mtime DOUBLE PRECISION NOT NULL,"
            f" ordinal INTEGER NOT NULL, token_count INTEGER NOT NULL, text TEXT NOT NULL,"
            f" summary TEXT,"
            f" embedding vector({collection.dim}) NOT NULL)"
        )
        self._conn.execute(
            f"CREATE INDEX ON chunks_{coll_id} USING hnsw (embedding vector_cosine_ops){self._index_options()}"
        )

    def _index_options(self) -> str:
        """The HNSW build parameters, as a WITH clause, or empty for pgvector's own defaults.

        Build-time knobs cannot be changed without rebuilding the index, so unlike ``ef_search``
        they belong here rather than on the query. Both are integers coerced by ``int()`` before
        interpolation, so no caller value reaches the DDL as text.
        """
        pairs = (("m", self._ann.m), ("ef_construction", self._ann.ef_construction))
        options = [f"{name} = {int(value)}" for name, value in pairs if value is not None]
        return f" WITH ({', '.join(options)})" if options else ""

    def upsert(self, *, collection: str, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        coll_id = self._require_id(collection)
        with self._conn.cursor() as cursor:
            for chunk, vector in zip(chunks, vectors, strict=True):
                cursor.execute(
                    f"INSERT INTO chunks_{coll_id} "
                    f"(path, label, content_hash, mtime, ordinal, token_count, text, summary, embedding) "
                    f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        chunk.source.uri,
                        chunk.source.label,
                        chunk.source.content_hash,
                        chunk.source.mtime,
                        chunk.ordinal,
                        chunk.token_count,
                        chunk.text,
                        chunk.summary,
                        _to_vector(vector),
                    ),
                )

    def delete_by_source(self, *, collection: str, uri: str) -> None:
        coll_id = self._require_id(collection)
        self._conn.execute(f"DELETE FROM chunks_{coll_id} WHERE path = %s", (uri,))

    def swap(self, *, staging: str, target: str) -> None:
        self._require_id(staging)
        target_id = self._collection_id(target)
        if target_id is not None:
            self._conn.execute(f"DROP TABLE IF EXISTS chunks_{target_id}")
            self._conn.execute("DELETE FROM collections WHERE name = %s", (target,))
        self._conn.execute("UPDATE collections SET name = %s WHERE name = %s", (target, staging))

    def compact(self, *, collection: str) -> None:
        """No-op: Postgres maintains the HNSW index incrementally on every insert."""

    # --- reader slice ---

    def query(self, *, collection: str, vector: Vector, k: int) -> list[Hit]:
        row = self._conn.execute("SELECT id, dim FROM collections WHERE name = %s", (collection,)).fetchone()
        if row is None:
            raise VectorStoreError(f"unknown collection: {collection}")
        coll_id, dim = int(row[0]), int(row[1])
        if len(vector) != dim:
            raise CollectionModelMismatchError(
                f"query dim {len(vector)} does not match collection '{collection}' dim {dim}"
            )
        query_vec = _to_vector(vector)
        rows = self._conn.execute(
            f"SELECT text, path, ordinal, label, summary, embedding <=> %s AS distance FROM chunks_{coll_id} "
            f"ORDER BY embedding <=> %s LIMIT %s",
            (query_vec, query_vec, k),
        ).fetchall()
        return [
            Hit(
                chunk_text=str(text),
                score=1.0 - float(distance),  # cosine distance -> similarity
                uri=str(path),
                ordinal=int(ordinal),
                label=str(label),
                collection=collection,
                summary=None if summary is None else str(summary),
            )
            for text, path, ordinal, label, summary, distance in rows
        ]

    def collections(self) -> list[Collection]:
        rows = self._conn.execute("SELECT name, model_id, dim FROM collections ORDER BY id").fetchall()
        return [Collection(name=str(name), model_id=str(model_id), dim=int(dim)) for name, model_id, dim in rows]

    def count(self, *, collection: str) -> int:
        coll_id = self._collection_id(collection)
        if coll_id is None:
            return 0
        row = self._conn.execute(f"SELECT COUNT(*) FROM chunks_{coll_id}").fetchone()
        return int(row[0])

    def source_hashes(self, *, collection: str) -> dict[str, str]:
        coll_id = self._require_id(collection)
        rows = self._conn.execute(f"SELECT DISTINCT path, content_hash FROM chunks_{coll_id}").fetchall()
        return {str(path): str(content_hash) for path, content_hash in rows}

    # --- helpers ---

    def _collection_id(self, collection: str) -> int | None:
        row = self._conn.execute("SELECT id FROM collections WHERE name = %s", (collection,)).fetchone()
        return int(row[0]) if row is not None else None

    def _require_id(self, collection: str) -> int:
        coll_id = self._collection_id(collection)
        if coll_id is None:
            raise VectorStoreError(f"unknown collection: {collection}")
        return coll_id


if TYPE_CHECKING:
    from ...application.ports import VectorStore

    _assert_store: VectorStore = PgVectorStore("")


__all__ = [
    "PgVectorStore",
]
