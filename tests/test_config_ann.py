"""Config tests for the #44 ANN-tuning + embed-batch knobs.

Covers: ``[vector_store].ann_recall`` + raw per-backend overrides + the resolver method, the
per-dataset ``ann_recall`` override, and ``[embedding].batch``. The non-breaking guarantee is
that the default (``balanced``, no overrides) resolves to empty ANN params for every backend.
"""

from __future__ import annotations

import pytest
from lib_layered_config import Config
from pydantic import ValidationError

from semdex.adapters.config.dataset import DatasetConfig
from semdex.adapters.config.embedding import EmbeddingConfig
from semdex.adapters.config.vectorstore import VectorStoreConfig, get_vector_store_config
from semdex.domain.ann_tuning import AnnParams
from semdex.domain.enums import AnnRecall, StoreBackend

pytestmark = pytest.mark.os_agnostic


def test_defaults_are_balanced_and_leave_every_driver_alone_but_lancedb() -> None:
    """No explicit overrides out of the box, and BALANCED is the default preset.

    It resolves to the driver's own default everywhere except lancedb, where the measured
    frontier showed that default giving up 0.0474 nDCG for nothing.
    """
    cfg = VectorStoreConfig()
    assert cfg.ann_recall is AnnRecall.BALANCED
    assert cfg.lancedb.nprobes is None and cfg.pgvector.ef_search is None
    for backend in StoreBackend:
        if backend is StoreBackend.LANCEDB:
            continue
        assert cfg.resolved_ann_params(backend) == AnnParams()
    assert cfg.resolved_ann_params(StoreBackend.LANCEDB) == AnnParams(nprobes=10, refine_factor=5)


def test_preset_maps_per_backend() -> None:
    cfg = VectorStoreConfig.model_validate({"ann_recall": "fast"})
    assert cfg.resolved_ann_params(StoreBackend.PGVECTOR).ef_search == 20
    assert cfg.resolved_ann_params(StoreBackend.MARIADB).ef_search == 20
    assert cfg.resolved_ann_params(StoreBackend.LANCEDB).nprobes == 10
    assert cfg.resolved_ann_params(StoreBackend.SQLITE_VEC) == AnnParams()  # exact store


def test_raw_override_wins_over_preset() -> None:
    cfg = VectorStoreConfig.model_validate({"ann_recall": "accurate", "pgvector": {"ef_search": 99}})
    assert cfg.resolved_ann_params(StoreBackend.PGVECTOR).ef_search == 99  # override beats 200


def test_dataset_recall_argument_overrides_global() -> None:
    cfg = VectorStoreConfig()  # global balanced
    # a dataset requesting "accurate" searches at that recall despite the global default
    assert cfg.resolved_ann_params(StoreBackend.PGVECTOR, AnnRecall.ACCURATE).ef_search == 200


def test_parse_from_layered_config_nested_tables() -> None:
    cfg = get_vector_store_config(Config({"vector_store": {"ann_recall": "fast", "pgvector": {"ef_search": 50}}}, {}))
    assert cfg.ann_recall is AnnRecall.FAST
    assert cfg.resolved_ann_params(StoreBackend.PGVECTOR).ef_search == 50


def test_embedding_batch_field() -> None:
    assert EmbeddingConfig().batch is None
    assert EmbeddingConfig.model_validate({"batch": 128}).batch == 128
    with pytest.raises(ValidationError):
        EmbeddingConfig.model_validate({"batch": 0})  # gt=0


def test_dataset_ann_recall_override_field() -> None:
    assert DatasetConfig(name="notes").ann_recall is None  # inherits the global default
    assert DatasetConfig.model_validate({"name": "notes", "ann_recall": "fast"}).ann_recall is AnnRecall.FAST


def test_invalid_ann_recall_rejected() -> None:
    with pytest.raises(ValidationError):
        VectorStoreConfig.model_validate({"ann_recall": "turbo"})
