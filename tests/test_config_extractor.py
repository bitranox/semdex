"""Extractor configuration model parsed from the [extractor] section."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from lib_layered_config import Config
from pydantic import ValidationError

from semdex.adapters.config.extractor import get_extractor_config
from semdex.domain.enums import ExtractorBackend


@pytest.mark.os_agnostic
def test_default_backend_is_text(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """With no [extractor] section, the embedded text reader is the default."""
    cfg = get_extractor_config(config_factory({}))
    assert cfg.backend is ExtractorBackend.TEXT
    assert cfg.endpoint is None
    assert cfg.timeout == 120.0
    assert cfg.max_file_bytes == 25 * 1024 * 1024


@pytest.mark.os_agnostic
def test_reads_backend_and_endpoint(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """The backend selector and endpoint are read from config."""
    cfg = get_extractor_config(
        config_factory({"extractor": {"backend": "docling", "endpoint": "http://h:5001", "timeout": 30}})
    )
    assert cfg.backend is ExtractorBackend.DOCLING
    assert cfg.endpoint == "http://h:5001"
    assert cfg.timeout == 30.0


@pytest.mark.os_agnostic
def test_unknown_backend_is_rejected(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """An unknown backend name is a validation error, not a silent fallback."""
    with pytest.raises(ValidationError):
        get_extractor_config(config_factory({"extractor": {"backend": "nonsense"}}))


@pytest.mark.os_agnostic
def test_non_positive_timeout_is_rejected(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """A zero/negative timeout is rejected at the boundary."""
    with pytest.raises(ValidationError):
        get_extractor_config(config_factory({"extractor": {"timeout": 0}}))


@pytest.mark.os_agnostic
def test_model_is_frozen(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """ExtractorConfig is immutable once parsed."""
    cfg = get_extractor_config(config_factory({}))
    with pytest.raises(ValidationError):
        cfg.backend = ExtractorBackend.DOCLING  # type: ignore[misc]
