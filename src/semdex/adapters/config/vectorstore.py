"""Vector-store configuration model parsed from the ``[vector_store]`` section.

Selects which persistence backend the index uses. The composition root turns
this into a concrete store adapter behind the ``VectorStore*`` ports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from ...domain.ann_tuning import AnnParams, resolve_ann_params
from ...domain.enums import AnnRecall, Partition, StoreBackend

if TYPE_CHECKING:
    from lib_layered_config import Config


class LancedbParams(BaseModel):
    """Raw lancedb ANN overrides (``[vector_store.lancedb]``); an unset field uses the ann_recall preset."""

    model_config = ConfigDict(frozen=True)
    nprobes: int | None = Field(default=None, gt=0)  # query-time: IVF cells probed per search
    refine_factor: int | None = Field(default=None, gt=0)  # query-time: re-rank multiplier


class PgvectorParams(BaseModel):
    """Raw pgvector ANN overrides (``[vector_store.pgvector]``); an unset field uses the ann_recall preset."""

    model_config = ConfigDict(frozen=True)
    ef_search: int | None = Field(default=None, gt=0)  # query-time: HNSW candidate-list size


class MariadbParams(BaseModel):
    """Raw mariadb ANN overrides (``[vector_store.mariadb]``); an unset field uses the ann_recall preset."""

    model_config = ConfigDict(frozen=True)
    ef_search: int | None = Field(default=None, gt=0)  # query-time: MHNSW candidate-list size


class VectorStoreConfig(BaseModel):
    """Validated, immutable vector-store backend selection.

    Example:
        >>> VectorStoreConfig().backend.value
        'json'
    """

    model_config = ConfigDict(frozen=True)

    backend: StoreBackend = StoreBackend.JSON
    # Connection string for server backends (pgvector / mariadb). Embedded
    # backends (json / sqlite_vec / lancedb) use the store directory instead.
    dsn: str | None = None
    # Rows before the lancedb backend builds its ANN index; below it a flat
    # scan is faster than paying the index build. Ignored by other backends.
    lance_index_threshold: int = 100_000
    # Compaction cadence for an ANN store (lancedb): the index use case folds the
    # freshly-added, still-unindexed rows into the index after this many new
    # records, OR at least this often (seconds) while records keep arriving -
    # whichever comes first. Between folds those rows are brute-force scanned
    # (fine while the tail is small). Ignored by stores that need no compaction.
    # DOWNSIDE: a writer that upserts but never triggers a fold lets the tail grow
    # without bound, silently degrading every query to a full O(n) scan.
    compact_after_records: int = 50_000
    compact_after_seconds: float = 300.0
    # Default partitioning for datasets that do not set their own (decision 11):
    # ``table`` = many collections share one store/database (fewest connections,
    # simplest provisioning, but only coarse DB-level grants and a shared
    # blast-radius); ``database`` = one database (or store file/dir) per dataset
    # (per-dataset grants/backup/isolation, at the cost of a connection per dataset
    # and CREATEDB to provision one). A per-dataset ``partition`` overrides this.
    default_partition: Partition = Partition.TABLE
    # ANN recall/latency preset for the approximate stores (lancedb / pgvector / mariadb),
    # applied query-time (no reindex). ``balanced`` keeps each driver's default (non-breaking);
    # ``fast`` trades recall for speed, ``accurate`` the reverse. Exact stores (json / sqlite_vec)
    # ignore it. A per-dataset ``ann_recall`` overrides this default.
    ann_recall: AnnRecall = AnnRecall.BALANCED
    # Optional raw per-backend overrides; a set field wins over the ann_recall preset.
    lancedb: LancedbParams = Field(default_factory=LancedbParams)
    pgvector: PgvectorParams = Field(default_factory=PgvectorParams)
    mariadb: MariadbParams = Field(default_factory=MariadbParams)

    def resolved_ann_params(self, backend: StoreBackend, recall: AnnRecall | None = None) -> AnnParams:
        """Effective ANN params for ``backend``: the ``ann_recall`` preset (or ``recall`` override
        from a dataset) with any raw per-backend overrides applied. Exact stores get empty params.
        """
        override = AnnParams()
        if backend is StoreBackend.LANCEDB:
            override = AnnParams(nprobes=self.lancedb.nprobes, refine_factor=self.lancedb.refine_factor)
        elif backend is StoreBackend.PGVECTOR:
            override = AnnParams(ef_search=self.pgvector.ef_search)
        elif backend is StoreBackend.MARIADB:
            override = AnnParams(ef_search=self.mariadb.ef_search)
        return resolve_ann_params(backend, recall or self.ann_recall, override)


def get_vector_store_config(config: Config) -> VectorStoreConfig:
    """Parse the ``[vector_store]`` section into a VectorStoreConfig.

    Falls back to the JSON backend when the section is absent.

    Example:
        >>> from lib_layered_config import Config
        >>> get_vector_store_config(Config({}, {})).backend.value
        'json'
    """
    return VectorStoreConfig.model_validate(config.get("vector_store", {}))


__all__ = [
    "LancedbParams",
    "MariadbParams",
    "PgvectorParams",
    "VectorStoreConfig",
    "get_vector_store_config",
]
