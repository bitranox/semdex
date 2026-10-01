"""Dataset configuration: a named store+collection+embedding+sources binding.

Parsed from a ``[[dataset]]`` array; the MCP server loads the list and each tool
call selects a dataset by name. Private vs shared is purely which store/DB the
binding points at - not a code path here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from lib_layered_config import Config
from pydantic import ValidationError

from semdex.adapters.config.dataset import DatasetConfig, get_datasets
from semdex.domain.enums import Partition, StoreBackend
from semdex.domain.errors import ConfigurationError


@pytest.mark.os_agnostic
def test_no_dataset_section_yields_empty(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """With no [[dataset]] entries, the dataset list is empty."""
    assert get_datasets(config_factory({})) == ()


@pytest.mark.os_agnostic
def test_parses_multiple_datasets(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """Each [[dataset]] entry becomes a validated DatasetConfig, in order."""
    cfg = config_factory(
        {
            "dataset": [
                {"name": "notes", "backend": "sqlite_vec", "store_dir": "~/.semdex/notes", "sources": ["~/notes"]},
                {
                    "name": "shared",
                    "backend": "pgvector",
                    "dsn": "postgresql://u@h/semdex_shared",
                    "partition": "database",
                    "read_only": True,
                },
            ]
        }
    )
    datasets = get_datasets(cfg)
    assert [d.name for d in datasets] == ["notes", "shared"]
    assert datasets[0].backend is StoreBackend.SQLITE_VEC
    assert datasets[0].sources == ("~/notes",)
    assert datasets[1].partition is Partition.DATABASE
    assert datasets[1].read_only is True


@pytest.mark.os_agnostic
def test_partition_defaults_to_none_meaning_inherit(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """An unset partition is None, so the dataset inherits [vector_store].default_partition."""
    dataset = get_datasets(config_factory({"dataset": [{"name": "x"}]}))[0]
    assert dataset.partition is None
    assert dataset.read_only is False


@pytest.mark.os_agnostic
def test_server_backend_requires_dsn(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """A server backend (pgvector/mariadb) without a dsn is a validation error."""
    with pytest.raises(ValidationError):
        get_datasets(config_factory({"dataset": [{"name": "x", "backend": "mariadb"}]}))


@pytest.mark.os_agnostic
def test_duplicate_dataset_names_rejected(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """Two datasets with the same name is a configuration error (names select datasets)."""
    with pytest.raises(ConfigurationError):
        get_datasets(config_factory({"dataset": [{"name": "dup"}, {"name": "dup"}]}))


@pytest.mark.os_agnostic
def test_writable_defaults_false() -> None:
    """A dataset is a source-mirrored (non-writable) dataset unless opted in."""
    assert DatasetConfig(name="kb").writable is False


@pytest.mark.os_agnostic
def test_writable_forbids_sources() -> None:
    """A writable knowledge dataset has no connector, so sources are a contradiction."""
    with pytest.raises(ValidationError):
        DatasetConfig(name="kb", writable=True, sources=("/tmp/x",))


@pytest.mark.os_agnostic
def test_writable_forbids_read_only() -> None:
    """writable (agent writes) and read_only (no writes) are mutually exclusive."""
    with pytest.raises(ValidationError):
        DatasetConfig(name="kb", writable=True, read_only=True)


@pytest.mark.os_agnostic
def test_writable_knowledge_dataset_ok() -> None:
    """A writable dataset with no sources validates."""
    cfg = DatasetConfig(name="kb", backend=StoreBackend.JSON, writable=True)
    assert cfg.writable is True and cfg.sources == ()
