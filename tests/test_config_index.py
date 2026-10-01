"""Index configuration model parsed from the [index] config section."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from lib_layered_config import Config
from pydantic import ValidationError

from semdex.adapters.config.index import IndexConfig, get_index_config


@pytest.mark.os_agnostic
def test_defaults_when_no_index_section(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """With no [index] section, all fields fall back to built-in defaults."""
    cfg = get_index_config(config_factory({}))
    assert cfg.store_dir is None
    assert cfg.collection == "default"
    assert cfg.embedding_model is None
    assert cfg.default_label == ""
    assert cfg.default_k == 5


@pytest.mark.os_agnostic
def test_reads_values_from_index_section(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """Values in the [index] section populate the model."""
    cfg = get_index_config(
        config_factory(
            {
                "index": {
                    "store_dir": "/var/semdex",
                    "collection": "memory",
                    "embedding_model": "org/model",
                    "default_label": "curated",
                    "default_k": 3,
                }
            }
        )
    )
    assert cfg.store_dir == "/var/semdex"
    assert cfg.collection == "memory"
    assert cfg.embedding_model == "org/model"
    assert cfg.default_label == "curated"
    assert cfg.default_k == 3


@pytest.mark.os_agnostic
def test_unknown_keys_are_ignored(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """Unknown keys in the section are ignored, not fatal."""
    cfg = get_index_config(config_factory({"index": {"collection": "x", "unknown_key": 1}}))
    assert cfg.collection == "x"


@pytest.mark.os_agnostic
def test_model_is_frozen(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """IndexConfig is immutable once parsed."""
    cfg = get_index_config(config_factory({}))
    with pytest.raises(ValidationError):
        cfg.collection = "changed"  # type: ignore[misc]


@pytest.mark.os_agnostic
def test_invalid_type_is_rejected(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """A wrongly-typed value (non-int default_k) is a validation error."""
    with pytest.raises(ValidationError):
        get_index_config(config_factory({"index": {"default_k": "not-a-number"}}))


@pytest.mark.os_agnostic
def test_direct_construction_has_defaults() -> None:
    """IndexConfig() with no args yields the documented defaults."""
    assert IndexConfig().collection == "default"
    assert IndexConfig().default_k == 5


@pytest.mark.os_agnostic
def test_tunable_defaults(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """The tunable 'assumed' values have their documented built-in defaults."""
    cfg = get_index_config(config_factory({}))
    assert cfg.max_tokens == 256
    assert cfg.max_file_bytes == 5 * 1024 * 1024
    assert cfg.embedding_dim == 256
    assert cfg.snippet_chars == 80
    assert cfg.hash_chunk_bytes == 65536
    assert cfg.extensions == [".md", ".txt"]


@pytest.mark.os_agnostic
def test_tunables_read_from_config(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """Tunable values are configurable via the [index] section."""
    cfg = get_index_config(config_factory({"index": {"max_tokens": 128, "embedding_dim": 64, "extensions": [".rst"]}}))
    assert cfg.max_tokens == 128
    assert cfg.embedding_dim == 64
    assert cfg.extensions == [".rst"]


@pytest.mark.os_agnostic
def test_non_positive_tunable_is_rejected(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """A non-positive size/dimension is a validation error, not a silent bad value."""
    with pytest.raises(ValidationError):
        get_index_config(config_factory({"index": {"max_tokens": 0}}))
