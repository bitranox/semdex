"""Composition wiring of the [health] tier into a dataset's providers.

Uses an offline placeholder-backed dataset for the end-to-end build (no server),
and drives the http-only wrapping/warm-up helpers with a fake provider. Confirms
the default (health None) path is unchanged and that only server-backed providers
are wrapped and warmed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from semdex.adapters.config.dataset import DatasetConfig
from semdex.adapters.config.health import HealthConfig
from semdex.application.resilience import ResilientEmbedding
from semdex.composition import (
    _resilient_embedding,  # pyright: ignore[reportPrivateUsage]
    _warmup_dataset_providers,  # pyright: ignore[reportPrivateUsage]
    build_dataset_services,
)
from semdex.domain.enums import EmbeddingBackend, Partition

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from semdex.domain.models import Vector

pytestmark = pytest.mark.os_agnostic


class _FakeEmbedding:
    """Minimal EmbeddingProvider recording embed_query calls."""

    def __init__(self) -> None:
        self.query_calls = 0

    @property
    def model_id(self) -> str:
        return "fake-model"

    @property
    def dim(self) -> int:
        return 3

    def embed_passages(self, texts: Sequence[str]) -> list[Vector]:
        return [(0.0, 0.0, 0.0) for _ in texts]

    def embed_query(self, text: str) -> Vector:
        self.query_calls += 1
        return (0.0, 0.0, 0.0)


def _placeholder_dataset(store_dir: str) -> DatasetConfig:
    return DatasetConfig(name="d", store_dir=store_dir, embedding_provider=EmbeddingBackend.PLACEHOLDER)


def test_health_none_leaves_placeholder_dataset_unchanged(tmp_path: Path) -> None:
    """With health None an in-process dataset is built exactly as before (no wrapping)."""
    dataset = _placeholder_dataset(str(tmp_path))
    svc = build_dataset_services(dataset, default_partition=Partition.TABLE, health=None)
    assert not isinstance(svc.embedding, ResilientEmbedding)
    assert svc.summarize is None


def test_restart_command_does_not_wrap_in_process_provider(tmp_path: Path) -> None:
    """A placeholder provider runs in-process, so it is never wrapped even with a restart command."""
    health = HealthConfig(restart_command="true", warmup=True)
    dataset = _placeholder_dataset(str(tmp_path))
    svc = build_dataset_services(dataset, default_partition=Partition.TABLE, health=health)
    assert not isinstance(svc.embedding, ResilientEmbedding)


def test_resilient_embedding_wraps_ollama_when_restart_set() -> None:
    """An ollama dataset with a restart command is wrapped in the self-heal decorator."""
    dataset = DatasetConfig(name="d", embedding_provider=EmbeddingBackend.OLLAMA, embedding_endpoint="http://h:11434")
    fake = _FakeEmbedding()
    wrapped = _resilient_embedding(fake, dataset, HealthConfig(restart_command="true"))
    assert isinstance(wrapped, ResilientEmbedding)
    assert wrapped.model_id == "fake-model"  # delegates to the inner provider


def test_resilient_embedding_returns_same_without_restart_command() -> None:
    """No restart command means no wrapping - the provider is returned untouched."""
    dataset = DatasetConfig(name="d", embedding_provider=EmbeddingBackend.OLLAMA)
    fake = _FakeEmbedding()
    assert _resilient_embedding(fake, dataset, HealthConfig()) is fake


def test_warmup_fires_for_http_provider_and_skips_in_process() -> None:
    """Warm-up calls the http provider once and is a no-op for an in-process one."""
    ollama = DatasetConfig(name="d", embedding_provider=EmbeddingBackend.OLLAMA)
    fake = _FakeEmbedding()
    _warmup_dataset_providers(ollama, fake, None, HealthConfig(warmup=True))
    assert fake.query_calls == 1

    placeholder = DatasetConfig(name="d", embedding_provider=EmbeddingBackend.PLACEHOLDER)
    other = _FakeEmbedding()
    _warmup_dataset_providers(placeholder, other, None, HealthConfig(warmup=True))
    assert other.query_calls == 0


def test_warmup_skipped_when_disabled() -> None:
    """warmup=false suppresses the warm-up call even for an http provider."""
    ollama = DatasetConfig(name="d", embedding_provider=EmbeddingBackend.OLLAMA)
    fake = _FakeEmbedding()
    _warmup_dataset_providers(ollama, fake, None, HealthConfig(warmup=False))
    assert fake.query_calls == 0
