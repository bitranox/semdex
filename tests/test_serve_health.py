"""serve wiring of the active health-check loop: endpoint-pair derivation.

Drives the pure helpers (no thread started, no network): which (probe, hook) pairs
serve builds from the [embedding]/[summary] config, that a shared host is probed
once, that in-process embedding providers are skipped, and that the loop stays off
when check_enabled is false.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from lib_layered_config import Config

from semdex.adapters.cli.commands.serve import (
    _health_check_pairs,  # pyright: ignore[reportPrivateUsage]
    _start_health_loop,  # pyright: ignore[reportPrivateUsage]
)
from semdex.adapters.config.health import HealthConfig

pytestmark = pytest.mark.os_agnostic


def test_shared_host_is_probed_once(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """Embedding and summary on the same ollama host collapse to one probe pair."""
    config = config_factory(
        {
            "embedding": {"provider": "ollama", "endpoint": "http://h:11434"},
            "summary": {"provider": "ollama", "endpoint": "http://h:11434"},
        }
    )
    pairs = _health_check_pairs(config, HealthConfig(restart_command="true"))
    assert len(pairs) == 1
    assert pairs[0][1] is not None  # the shared restart hook is attached


def test_distinct_endpoints_yield_two_pairs(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """An ollama embedding host and an openai summary host give two distinct probes."""
    config = config_factory(
        {
            "embedding": {"provider": "ollama", "endpoint": "http://e:11434"},
            "summary": {"provider": "openai", "endpoint": "http://s:8080/v1"},
        }
    )
    pairs = _health_check_pairs(config, HealthConfig(restart_command="true"))
    assert len(pairs) == 2


def test_in_process_embedding_and_no_summary_yield_no_pairs(
    config_factory: Callable[[dict[str, Any]], Config],
) -> None:
    """A fastembed embedding exposes no endpoint and the summary tier is off: no probes."""
    config = config_factory({"embedding": {"provider": "fastembed"}})
    assert _health_check_pairs(config, HealthConfig(restart_command="true")) == []


def test_pairs_have_no_hook_without_restart_command(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """With no restart_command the loop can detect but not restart: hook is None."""
    config = config_factory({"embedding": {"provider": "ollama", "endpoint": "http://h:11434"}})
    pairs = _health_check_pairs(config, HealthConfig())
    assert len(pairs) == 1
    assert pairs[0][1] is None


def test_loop_off_when_check_disabled(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """check_enabled false means no loop is started (returns None)."""
    config = config_factory({"embedding": {"provider": "ollama", "endpoint": "http://h:11434"}})
    assert _start_health_loop(config, HealthConfig(check_enabled=False)) is None


def test_loop_none_when_enabled_but_no_endpoints(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """check_enabled with only in-process providers has nothing to probe: no loop."""
    config = config_factory({"embedding": {"provider": "fastembed"}})
    loop = _start_health_loop(config, HealthConfig(check_enabled=True, restart_command="true"))
    assert loop is None
