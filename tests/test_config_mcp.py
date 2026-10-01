from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from lib_layered_config import Config
from pydantic import ValidationError

from semdex.adapters.config.mcp import get_mcp_config
from semdex.domain.enums import McpAuthMode, McpTransport


@pytest.mark.os_agnostic
def test_defaults_are_stdio_and_no_auth(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    cfg = get_mcp_config(config_factory({}))
    assert cfg.transport is McpTransport.STDIO
    assert cfg.auth is McpAuthMode.NONE
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8080


@pytest.mark.os_agnostic
def test_reads_http_bearer(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    cfg = get_mcp_config(config_factory({"mcp": {"transport": "http", "auth": "bearer", "port": 9000}}))
    assert cfg.transport is McpTransport.HTTP
    assert cfg.auth is McpAuthMode.BEARER
    assert cfg.port == 9000


@pytest.mark.os_agnostic
def test_unknown_auth_rejected(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    with pytest.raises(ValidationError):
        get_mcp_config(config_factory({"mcp": {"auth": "basic"}}))


@pytest.mark.os_agnostic
def test_rrf_k_default_and_override(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    assert get_mcp_config(config_factory({})).rrf_k == 60
    assert get_mcp_config(config_factory({"mcp": {"rrf_k": 100}})).rrf_k == 100


@pytest.mark.os_agnostic
def test_mask_error_details_defaults_true(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    assert get_mcp_config(config_factory({})).mask_error_details is True


@pytest.mark.os_agnostic
def test_mask_error_details_parses_from_toml(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    cfg = get_mcp_config(config_factory({"mcp": {"mask_error_details": False}}))
    assert cfg.mask_error_details is False


@pytest.mark.os_agnostic
def test_oauth_revocation_url_defaults_none(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    assert get_mcp_config(config_factory({})).oauth_revocation_url is None


@pytest.mark.os_agnostic
def test_oauth_revocation_url_parses_from_toml(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    cfg = get_mcp_config(config_factory({"mcp": {"oauth_revocation_url": "https://idp.example.com/revoke"}}))
    assert cfg.oauth_revocation_url == "https://idp.example.com/revoke"
