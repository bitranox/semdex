from __future__ import annotations

from pathlib import Path

import pytest

from semdex.adapters.config.dataset import DatasetConfig
from semdex.composition import build_dataset_services
from semdex.domain.enums import EmbeddingBackend, Partition, StoreBackend


@pytest.mark.os_agnostic
def test_builds_embedded_dataset_services(tmp_path: Path) -> None:
    dataset = DatasetConfig(
        name="notes",
        backend=StoreBackend.JSON,
        store_dir=str(tmp_path / "store"),
        collection="notes",
        embedding_provider=EmbeddingBackend.PLACEHOLDER,
        sources=(str(tmp_path / "docs"),),
    )
    services = build_dataset_services(dataset, default_partition=Partition.TABLE)
    assert services.name == "notes"
    assert services.collection == "notes"
    assert services.partition is Partition.TABLE  # inherited (dataset.partition is None)
    assert services.connector.sources() == []  # empty dir -> no sources, no crash
    assert services.model_key == "placeholder||"  # provider|model|endpoint identity
    assert services.writable is False  # source dataset by default
