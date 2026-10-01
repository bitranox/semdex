from __future__ import annotations

import pytest

from semdex.adapters.config.mcp import McpConfig
from semdex.adapters.mcp.auth import build_mcp_auth
from semdex.domain.enums import McpAuthMode
from semdex.domain.errors import ConfigurationError


@pytest.mark.os_agnostic
def test_none_yields_no_auth() -> None:
    assert build_mcp_auth(McpConfig(auth=McpAuthMode.NONE), env={}) is None


@pytest.mark.os_agnostic
def test_bearer_builds_static_verifier_from_env() -> None:
    from fastmcp.server.auth import StaticTokenVerifier

    cfg = McpConfig(auth=McpAuthMode.BEARER, bearer_token_env="TOKENS")
    auth = build_mcp_auth(cfg, env={"TOKENS": "tok-a,tok-b"})
    assert isinstance(auth, StaticTokenVerifier)


@pytest.mark.os_agnostic
def test_bearer_without_tokens_is_configuration_error() -> None:
    cfg = McpConfig(auth=McpAuthMode.BEARER, bearer_token_env="TOKENS")
    with pytest.raises(ConfigurationError):
        build_mcp_auth(cfg, env={})


@pytest.mark.os_agnostic
def test_oauth_requires_jwks_and_issuer() -> None:
    cfg = McpConfig(auth=McpAuthMode.OAUTH)  # no jwks/issuer
    with pytest.raises(ConfigurationError):
        build_mcp_auth(cfg, env={})


@pytest.mark.os_agnostic
def test_oauth_verify_only_builds_jwt_verifier() -> None:
    from fastmcp.server.auth.providers.jwt import JWTVerifier

    cfg = McpConfig(
        auth=McpAuthMode.OAUTH,
        oauth_jwks_uri="https://idp.example.com/jwks.json",
        oauth_issuer="https://idp.example.com/",
    )
    auth = build_mcp_auth(cfg, env={})
    assert isinstance(auth, JWTVerifier)


@pytest.mark.os_agnostic
def test_oauth_with_upstream_fields_builds_oauth_proxy() -> None:
    from fastmcp.server.auth import OAuthProxy

    cfg = McpConfig(
        auth=McpAuthMode.OAUTH,
        oauth_jwks_uri="https://idp.example.com/jwks.json",
        oauth_issuer="https://idp.example.com/",
        oauth_authorize_url="https://idp.example.com/authorize",
        oauth_token_url="https://idp.example.com/token",
        oauth_client_id="semdex-mcp",
        oauth_client_secret_env="SECRET",
        oauth_base_url="https://semdex.example.com",
    )
    auth = build_mcp_auth(cfg, env={"SECRET": "shh"})
    assert isinstance(auth, OAuthProxy)


@pytest.mark.os_agnostic
def test_oauth_proxy_without_base_url_is_configuration_error() -> None:
    """FastMCP's ``OAuthProxy.get_routes(mcp_path=...)`` raises a bare
    ``AssertionError`` when built with ``base_url=None`` (it does NOT fall back to
    per-request derivation, despite what an earlier comment here claimed), and the
    redirect/consent URLs it builds meanwhile render as the literal string
    "None/...". ADR 0002 requires a misconfigured mode to fail fast with a clear
    ``ConfigurationError`` instead, so proxy mode (the interactive-flow fields set)
    without ``oauth_base_url`` must be refused at build time, naming the key."""
    cfg = McpConfig(
        auth=McpAuthMode.OAUTH,
        oauth_jwks_uri="https://idp.example.com/jwks.json",
        oauth_issuer="https://idp.example.com/",
        oauth_authorize_url="https://idp.example.com/authorize",
        oauth_token_url="https://idp.example.com/token",
        oauth_client_id="semdex-mcp",
        oauth_client_secret_env="SECRET",
        # oauth_base_url omitted on purpose.
    )
    with pytest.raises(ConfigurationError, match="oauth_base_url"):
        build_mcp_auth(cfg, env={"SECRET": "shh"})


@pytest.mark.os_agnostic
def test_oauth_revocation_url_reaches_the_proxy() -> None:
    from fastmcp.server.auth import OAuthProxy

    cfg = McpConfig(
        auth=McpAuthMode.OAUTH,
        oauth_jwks_uri="https://idp.example.com/jwks.json",
        oauth_issuer="https://idp.example.com/",
        oauth_authorize_url="https://idp.example.com/authorize",
        oauth_token_url="https://idp.example.com/token",
        oauth_client_id="semdex-mcp",
        oauth_client_secret_env="SECRET",
        oauth_base_url="https://semdex.example.com",
        oauth_revocation_url="https://idp.example.com/revoke",
    )
    auth = build_mcp_auth(cfg, env={"SECRET": "shh"})
    assert isinstance(auth, OAuthProxy)
    # No public accessor for the constructor kwarg; read the private attribute directly.
    assert getattr(auth, "_upstream_revocation_endpoint") == "https://idp.example.com/revoke"  # noqa: B009
