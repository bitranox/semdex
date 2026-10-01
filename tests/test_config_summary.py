"""Summary-tier configuration: the global [summary] section + the per-dataset knob."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from lib_layered_config import Config
from pydantic import ValidationError

from semdex.adapters.config.dataset import DatasetConfig
from semdex.adapters.config.summary import get_summary_config
from semdex.domain.enums import SummaryBackend

pytestmark = pytest.mark.os_agnostic


def test_default_provider_is_none_tier_off(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """With no [summary] section the tier is off and the sensible defaults hold."""
    cfg = get_summary_config(config_factory({}))
    assert cfg.provider is SummaryBackend.NONE
    assert cfg.model is None
    assert cfg.endpoint is None
    assert cfg.api_key is None
    assert cfg.timeout is None
    assert cfg.max_input_chars == 8000
    assert "summar" in cfg.prompt.lower()


def test_reads_ollama_provider_and_knobs(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """The ollama provider reads its model, endpoint, timeout, clip, and prompt."""
    cfg = get_summary_config(
        config_factory(
            {
                "summary": {
                    "provider": "ollama",
                    "model": "llama3.2",
                    "endpoint": "http://h:11434",
                    "timeout": 120.0,
                    "max_input_chars": 4000,
                    "prompt": "one line only",
                }
            }
        )
    )
    assert cfg.provider is SummaryBackend.OLLAMA
    assert cfg.model == "llama3.2"
    assert cfg.endpoint == "http://h:11434"
    assert cfg.timeout == 120.0
    assert cfg.max_input_chars == 4000
    assert cfg.prompt == "one line only"


def test_reads_openai_provider_endpoint_and_env_api_key(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """The openai provider reads its base-URL endpoint and (env-sourced) api_key."""
    cfg = get_summary_config(
        config_factory({"summary": {"provider": "openai", "endpoint": "http://h:8080/v1", "api_key": "sk-from-env"}})
    )
    assert cfg.provider is SummaryBackend.OPENAI
    assert cfg.endpoint == "http://h:8080/v1"
    assert cfg.api_key == "sk-from-env"


def test_unknown_provider_is_rejected(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """An unknown provider name is a validation error, not a silent fallback."""
    with pytest.raises(ValidationError):
        get_summary_config(config_factory({"summary": {"provider": "nonsense"}}))


def test_non_positive_max_input_chars_is_rejected(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """max_input_chars must be positive (a zero/negative clip makes no sense)."""
    with pytest.raises(ValidationError):
        get_summary_config(config_factory({"summary": {"max_input_chars": 0}}))


def test_summary_config_is_frozen(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """SummaryConfig is immutable once parsed."""
    cfg = get_summary_config(config_factory({}))
    with pytest.raises(ValidationError):
        cfg.provider = SummaryBackend.OLLAMA  # type: ignore[misc]


def test_dataset_summarize_defaults_off_and_parses_overrides() -> None:
    """A dataset opts into the tier with summarize=true and may override model/endpoint."""
    assert DatasetConfig(name="d").summarize is False

    ds = DatasetConfig(
        name="notes",
        sources=("~/notes",),
        summarize=True,
        summary_model="llama3.2",
        summary_endpoint="http://h:11434",
    )
    assert ds.summarize is True
    assert ds.summary_model == "llama3.2"
    assert ds.summary_endpoint == "http://h:11434"
