"""Vector-store configuration model parsed from the [vector_store] section."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from lib_layered_config import Config
from pydantic import ValidationError

from semdex.adapters.config.vectorstore import get_vector_store_config
from semdex.domain.enums import Partition, StoreBackend


@pytest.mark.os_agnostic
def test_default_backend_is_json(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """With no [vector_store] section, the zero-dependency JSON store is default."""
    cfg = get_vector_store_config(config_factory({}))
    assert cfg.backend is StoreBackend.JSON
    assert cfg.dsn is None


@pytest.mark.os_agnostic
def test_reads_backend(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """The backend selector is read from config and coerced to the enum."""
    cfg = get_vector_store_config(config_factory({"vector_store": {"backend": "sqlite_vec"}}))
    assert cfg.backend is StoreBackend.SQLITE_VEC


@pytest.mark.os_agnostic
def test_reads_dsn_for_server_backend(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """A server backend carries its DSN."""
    cfg = get_vector_store_config(
        config_factory({"vector_store": {"backend": "pgvector", "dsn": "postgresql://u@h/db"}})
    )
    assert cfg.backend is StoreBackend.PGVECTOR
    assert cfg.dsn == "postgresql://u@h/db"


@pytest.mark.os_agnostic
def test_lance_index_threshold_default_and_override(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """The lancedb ANN-index threshold defaults to 100K rows and is tunable."""
    assert get_vector_store_config(config_factory({})).lance_index_threshold == 100_000
    cfg = get_vector_store_config(config_factory({"vector_store": {"lance_index_threshold": 20_000}}))
    assert cfg.lance_index_threshold == 20_000


@pytest.mark.os_agnostic
def test_default_partition_defaults_to_table_and_overrides(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """Partitioning defaults to table-per-collection; overridable to database-per-dataset."""
    assert get_vector_store_config(config_factory({})).default_partition is Partition.TABLE
    cfg = get_vector_store_config(config_factory({"vector_store": {"default_partition": "database"}}))
    assert cfg.default_partition is Partition.DATABASE


@pytest.mark.os_agnostic
def test_unknown_partition_is_rejected(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """An unknown partition value is a validation error, not a silent fallback."""
    with pytest.raises(ValidationError):
        get_vector_store_config(config_factory({"vector_store": {"default_partition": "sharded"}}))


@pytest.mark.os_agnostic
def test_unknown_backend_is_rejected(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """An unknown backend name is a validation error, not a silent fallback."""
    with pytest.raises(ValidationError):
        get_vector_store_config(config_factory({"vector_store": {"backend": "nonsense"}}))


@pytest.mark.os_agnostic
def test_model_is_frozen(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """VectorStoreConfig is immutable once parsed."""
    cfg = get_vector_store_config(config_factory({}))
    with pytest.raises(ValidationError):
        cfg.backend = StoreBackend.LANCEDB  # type: ignore[misc]
