"""Health-tier configuration: the [health] section defaults and a full override."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from lib_layered_config import Config
from pydantic import ValidationError

from semdex.adapters.config.health import get_health_config

pytestmark = pytest.mark.os_agnostic


def test_defaults_when_section_absent(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """With no [health] section the safe defaults hold: warm-up on, restart/loop off."""
    cfg = get_health_config(config_factory({}))
    assert cfg.warmup is True
    assert cfg.keep_alive is None
    assert cfg.restart_command is None
    assert cfg.restart_timeout == 120.0
    assert cfg.check_enabled is False
    assert cfg.check_interval == 30.0
    assert cfg.recover_timeout == 120.0


def test_full_override(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """Every knob is read from the section when present."""
    cfg = get_health_config(
        config_factory(
            {
                "health": {
                    "warmup": False,
                    "keep_alive": "30m",
                    "restart_command": "systemctl restart ollama",
                    "restart_timeout": 45.0,
                    "check_enabled": True,
                    "check_interval": 10.0,
                    "recover_timeout": 90.0,
                }
            }
        )
    )
    assert cfg.warmup is False
    assert cfg.keep_alive == "30m"
    assert cfg.restart_command == "systemctl restart ollama"
    assert cfg.restart_timeout == 45.0
    assert cfg.check_enabled is True
    assert cfg.check_interval == 10.0
    assert cfg.recover_timeout == 90.0


def test_non_positive_intervals_rejected(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    """A non-positive restart_timeout / check_interval is a validation error."""
    with pytest.raises(ValidationError):
        get_health_config(config_factory({"health": {"restart_timeout": 0}}))
    with pytest.raises(ValidationError):
        get_health_config(config_factory({"health": {"check_interval": 0}}))
