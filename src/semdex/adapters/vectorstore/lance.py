"""Embedded columnar ANN vector store backed by LanceDB.

One LanceDB table per collection (each pinned to its own dimension). Collection
metadata (model_id, dim, and a stable table id) lives in a typed orjson sidecar
so a blue-green ``swap`` is a metadata move with no table rename. Needs the
optional ``semdex[lance]`` extra (pulls pyarrow).

``upsert`` only appends rows (cheap, per source). Index maintenance is deferred
to ``compact``, which the index use case calls once after a bulk load (and a
watcher would call periodically) - NOT per upsert, or an ANN store would
re-optimize on every source. ``compact`` builds the IVF-PQ (cosine) index once
the table crosses ``index_threshold`` rows and folds the still-unindexed tail
into it with ``table.optimize()``. Between compactions freshly-added rows live
in a small unindexed tail that LanceDB brute-force scans - cheap while small,
and it keeps new docs immediately searchable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import orjson
from pydantic import BaseModel, Field

from ...domain.ann_tuning import AnnParams
from ...domain.errors import CollectionModelMismatchError, VectorStoreError
from ...domain.models import Collection, Hit

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ...domain.models import Chunk, Vector

_DB_DIRNAME = "lance"
_META_FILENAME = "lance_meta.json"
# Rows before the ANN index is built: below ~100K vectors a flat scan beats the
# index build + probe cost (LanceDB's own guidance and the semdex store
# benchmarks agree), above it ANN is the point of using LanceDB. Overridable
# via [vector_store].lance_index_threshold.
_DEFAULT_INDEX_THRESHOLD = 100_000


class _CollectionMeta(BaseModel):
    """Sidecar record for one collection: its model, dim, and stable table id."""

    model_id: str
    dim: int
    table: str


class _LanceMeta(BaseModel):
    """The whole sidecar: collections by name plus the next table-id counter."""

    collections: dict[str, _CollectionMeta] = Field(default_factory=dict)
    next_id: int = 0


def _connect(uri: str) -> Any:
    try:
        import lancedb  # type: ignore  # optional dep; see module docstring
    except ImportError as exc:
        raise VectorStoreError("lancedb is not installed; install semdex[lance]") from exc
    return lancedb.connect(uri)  # type: ignore[no-untyped-call]


def _schema(dim: int) -> Any:
    import pyarrow as pa  # type: ignore  # comes with lancedb; see module docstring

    p: Any = pa  # untyped optional dep (no stubs): treat as Any so member access is not flagged
    return p.schema(
        [
            p.field("vector", p.list_(p.float32(), dim)),
            p.field("text", p.string()),
            p.field("path", p.string()),
            p.field("label", p.string()),
            p.field("content_hash", p.string()),
            p.field("mtime", p.float64()),
            p.field("ordinal", p.int64()),
            p.field("token_count", p.int64()),
            # Nullable per-document summary (the opt-in tier); None when off.
            p.field("summary", p.string(), nullable=True),
        ]
    )


class LanceVectorStore:
    """Vector store over LanceDB; implements the reader and writer slices."""

    def __init__(
        self, store_dir: Path, *, index_threshold: int = _DEFAULT_INDEX_THRESHOLD, ann_params: AnnParams | None = None
    ) -> None:
        store_dir.mkdir(parents=True, exist_ok=True)
        self._db = _connect(str(store_dir / _DB_DIRNAME))
        self._meta_path = store_dir / _META_FILENAME
        self._index_threshold = index_threshold
        self._ann = ann_params or AnnParams()
        if self._meta_path.exists():
            self._meta = _LanceMeta.model_validate(orjson.loads(self._meta_path.read_bytes()))
        else:
            self._meta = _LanceMeta()

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
        """Release the LanceDB connection (embedded; guarded for version compat)."""
        # LanceDB is an embedded store with no persistent socket; older versions
        # expose no close(), so call it only when present.
        close = getattr(self._db, "close", None)
        if callable(close):
            close()

    # --- writer slice ---

    def ensure_collection(self, collection: Collection) -> None:
        if collection.name in self._meta.collections:
            return
        table_id = f"t{self._meta.next_id}"
        self._meta.next_id += 1
        self._meta.collections[collection.name] = _CollectionMeta(
            model_id=collection.model_id, dim=collection.dim, table=table_id
        )
        self._db.create_table(table_id, schema=_schema(collection.dim))
        self._save_meta()

    def upsert(self, *, collection: str, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None:
        table = self._db.open_table(self._require(collection).table)
        rows = [
            {
                "vector": list(vector),
                "text": chunk.text,
                "path": chunk.source.uri,
                "label": chunk.source.label,
                "content_hash": chunk.source.content_hash,
                "mtime": chunk.source.mtime,
                "ordinal": chunk.ordinal,
                "token_count": chunk.token_count,
                "summary": chunk.summary,
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if rows:
            table.add(rows)

    def delete_by_source(self, *, collection: str, uri: str) -> None:
        table = self._db.open_table(self._require(collection).table)
        escaped = uri.replace("'", "''")
        table.delete(f"path = '{escaped}'")

    def swap(self, *, staging: str, target: str) -> None:
        self._require(staging)
        target_meta = self._meta.collections.get(target)
        if target_meta is not None:
            self._db.drop_table(target_meta.table)
        # the data table id travels with the metadata; no table rename needed.
        self._meta.collections[target] = self._meta.collections.pop(staging)
        self._save_meta()

    def compact(self, *, collection: str) -> None:
        """Build the ANN index once big enough, then fold in the unindexed tail.

        Called once after a bulk load (and periodically by a long-running watcher),
        NOT per ``upsert`` - so a batched load re-optimizes a bounded number of
        times, not on every source. The first run once the table crosses
        ``index_threshold`` rows builds the IVF-PQ index (parameters auto-tuned by
        LanceDB); later runs fold the rows added since - the unindexed tail that
        LanceDB would otherwise brute-force scan on every query - with
        ``table.optimize()``. ``cleanup_older_than`` prunes the superseded fragments
        each optimize leaves behind, so repeated compaction does not bloat the
        table on disk.
        """
        meta = self._meta.collections.get(collection)
        if meta is None:
            return
        table = self._db.open_table(meta.table)
        existing = table.list_indices()
        if not existing:
            if table.count_rows() < self._index_threshold:
                return
            # num_partitions fixes how many IVF cells the vectors are clustered into, and it is
            # what nprobes is later counted against: a probe count only means something relative
            # to the partition count, so tuning one without pinning the other moves two things.
            from lancedb.index import IvfPq  # type: ignore  # optional dep; see module docstring

            partitions = self._ann.num_partitions
            config = (
                IvfPq(distance_type="cosine")
                if partitions is None
                else IvfPq(distance_type="cosine", num_partitions=int(partitions))
            )
            table.create_index("vector", config=config)
            return
        if existing[0].num_unindexed_rows > 0:
            from datetime import timedelta

            table.optimize(cleanup_older_than=timedelta(0))

    # --- reader slice ---

    def query(self, *, collection: str, vector: Vector, k: int) -> list[Hit]:
        meta = self._require(collection)
        if len(vector) != meta.dim:
            raise CollectionModelMismatchError(
                f"query dim {len(vector)} does not match collection '{collection}' dim {meta.dim}"
            )
        table = self._db.open_table(meta.table)
        search = table.search(list(vector)).metric("cosine")
        if self._ann.nprobes is not None:  # query-time recall knob (ann_recall preset / override)
            search = search.nprobes(self._ann.nprobes)
        if self._ann.refine_factor is not None:
            search = search.refine_factor(self._ann.refine_factor)
        rows: list[dict[str, Any]] = search.limit(k).to_list()
        return [
            Hit(
                chunk_text=str(row["text"]),
                score=1.0 - float(row["_distance"]),  # cosine distance -> similarity
                uri=str(row["path"]),
                ordinal=int(row["ordinal"]),
                label=str(row["label"]),
                collection=collection,
                summary=None if row.get("summary") is None else str(row["summary"]),
            )
            for row in rows
        ]

    def collections(self) -> list[Collection]:
        return [
            Collection(name=name, model_id=meta.model_id, dim=meta.dim) for name, meta in self._meta.collections.items()
        ]

    def count(self, *, collection: str) -> int:
        meta = self._meta.collections.get(collection)
        if meta is None:
            return 0
        return int(self._db.open_table(meta.table).count_rows())

    def source_hashes(self, *, collection: str) -> dict[str, str]:
        table = self._db.open_table(self._require(collection).table)
        # Project only the two columns (not the vectors) so the scan stays cheap.
        rows = table.to_arrow().select(["path", "content_hash"]).to_pylist()
        return {str(row["path"]): str(row["content_hash"]) for row in rows}

    # --- helpers ---

    def _require(self, collection: str) -> _CollectionMeta:
        meta = self._meta.collections.get(collection)
        if meta is None:
            raise VectorStoreError(f"unknown collection: {collection}")
        return meta

    def _save_meta(self) -> None:
        self._meta_path.write_bytes(orjson.dumps(self._meta.model_dump()))


if TYPE_CHECKING:
    from pathlib import Path as _Path

    from ...application.ports import VectorStore

    _assert_store: VectorStore = LanceVectorStore(_Path())


__all__ = [
    "LanceVectorStore",
]
