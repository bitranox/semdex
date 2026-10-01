"""Embedding configuration model parsed from the [embedding] section."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from lib_layered_config import Config
from pydantic import ValidationError

from semdex.adapters.config.embedding import get_embedding_config
from semdex.domain.enums import EmbeddingBackend


@pytest.mark.os_agnostic
def test_default_provider_is_fastembed(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """With no [embedding] section, the light fastembed provider is the default."""
    cfg = get_embedding_config(config_factory({}))
    assert cfg.provider is EmbeddingBackend.FASTEMBED
    assert cfg.model is None
    assert cfg.endpoint is None
    assert cfg.threads is None
    assert cfg.timeout is None
    assert cfg.api_key is None


@pytest.mark.os_agnostic
def test_reads_openai_provider_endpoint_timeout_and_api_key(
    config_factory: Callable[[dict[str, Any]], Config],
) -> None:
    """The openai provider reads its base-URL endpoint, timeout, and (env-sourced) api_key."""
    cfg = get_embedding_config(
        config_factory(
            {
                "embedding": {
                    "provider": "openai",
                    "model": "text-embedding-3-small",
                    "endpoint": "http://px-semdex-test-embeddings:8080/v1",
                    "timeout": 120.0,
                    "api_key": "sk-from-env",
                }
            }
        )
    )
    assert cfg.provider is EmbeddingBackend.OPENAI
    assert cfg.model == "text-embedding-3-small"
    assert cfg.endpoint == "http://px-semdex-test-embeddings:8080/v1"
    assert cfg.timeout == 120.0
    assert cfg.api_key == "sk-from-env"


@pytest.mark.os_agnostic
def test_reads_provider_model_endpoint(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """Provider, model, and endpoint are read from config."""
    cfg = get_embedding_config(
        config_factory({"embedding": {"provider": "ollama", "model": "nomic-embed-text", "endpoint": "http://h:11434"}})
    )
    assert cfg.provider is EmbeddingBackend.OLLAMA
    assert cfg.model == "nomic-embed-text"
    assert cfg.endpoint == "http://h:11434"


@pytest.mark.os_agnostic
def test_unknown_provider_is_rejected(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """An unknown provider name is a validation error, not a silent fallback."""
    with pytest.raises(ValidationError):
        get_embedding_config(config_factory({"embedding": {"provider": "nonsense"}}))


@pytest.mark.os_agnostic
def test_model_is_frozen(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """EmbeddingConfig is immutable once parsed."""
    cfg = get_embedding_config(config_factory({}))
    with pytest.raises(ValidationError):
        cfg.provider = EmbeddingBackend.MODEL2VEC  # type: ignore[misc]
