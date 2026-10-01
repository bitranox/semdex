"""Build a FastMCP auth provider from the ``[mcp]`` config.

Maps ``McpAuthMode`` to a concrete FastMCP verifier / proxy, loading every secret
(bearer tokens, the OAuth client secret) from an injected ``env`` mapping - never
from a tracked file. ``env`` is injected so tests can supply tokens without
touching the real process environment.

``fastmcp`` is the optional ``mcp-server`` extra, so it is imported lazily inside
the builder; a missing install surfaces as a ``ConfigurationError`` pointing at the
extra rather than an ImportError.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...domain.enums import McpAuthMode
from ...domain.errors import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastmcp.server.auth import AuthProvider, OAuthProxy, StaticTokenVerifier
    from fastmcp.server.auth.providers.jwt import JWTVerifier

    from ..config.mcp import McpConfig


def build_mcp_auth(config: McpConfig, *, env: Mapping[str, str]) -> AuthProvider | None:
    """Return a FastMCP auth provider for ``config``, or ``None`` for no auth.

    Args:
        config: The parsed ``[mcp]`` settings.
        env: Environment mapping the secrets load from (e.g. ``os.environ``).

    Returns:
        A FastMCP auth provider (``StaticTokenVerifier`` / ``JWTVerifier`` /
        ``OAuthProxy``) for the ``bearer`` / ``oauth`` modes, or ``None`` for
        ``none``.

    Raises:
        ConfigurationError: When the selected mode is misconfigured (no bearer
            tokens in the env, missing OAuth jwks/issuer) or ``fastmcp`` is not
            installed.
    """
    if config.auth is McpAuthMode.NONE:
        return None
    if config.auth is McpAuthMode.BEARER:
        return _build_bearer(config, env=env)
    return _build_oauth(config, env=env)


def _build_bearer(config: McpConfig, *, env: Mapping[str, str]) -> StaticTokenVerifier:
    """Build a static-token verifier from the comma-separated tokens in the env."""
    raw = env.get(config.bearer_token_env, "")
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    if not tokens:
        raise ConfigurationError(
            f"bearer auth requires token(s) in ${config.bearer_token_env} (comma-separated); none found."
        )
    static_token_verifier = _import_static_token_verifier()
    return static_token_verifier(
        tokens={t: {"client_id": "semdex", "scopes": ["read", "write"]} for t in tokens},
        required_scopes=["read"],
    )


def _build_oauth(config: McpConfig, *, env: Mapping[str, str]) -> JWTVerifier | OAuthProxy:
    """Build a JWT verifier (verify-only) or an OAuth proxy wrapping it."""
    if not config.oauth_jwks_uri or not config.oauth_issuer:
        raise ConfigurationError("oauth auth requires [mcp].oauth_jwks_uri and [mcp].oauth_issuer.")
    jwt_verifier_cls = _import_jwt_verifier()
    verifier = jwt_verifier_cls(
        jwks_uri=config.oauth_jwks_uri,
        issuer=config.oauth_issuer,
        audience=config.oauth_audience,
    )
    # Verify-only unless the interactive-proxy endpoints are all configured.
    if not (config.oauth_authorize_url and config.oauth_token_url and config.oauth_client_id):
        return verifier
    client_secret = env.get(config.oauth_client_secret_env, "")
    if not client_secret:
        raise ConfigurationError(
            f"oauth proxy requires the client secret in ${config.oauth_client_secret_env}; none found."
        )
    if not config.oauth_base_url:
        raise ConfigurationError(
            "oauth proxy requires [mcp].oauth_base_url (this server's own public URL); "
            "without it fastmcp's OAuthProxy fails at request time with a bare AssertionError "
            "and builds redirect/consent URLs as the literal string 'None/...'."
        )
    oauth_proxy_cls = _import_oauth_proxy()
    return oauth_proxy_cls(
        upstream_authorization_endpoint=config.oauth_authorize_url,
        upstream_token_endpoint=config.oauth_token_url,
        upstream_client_id=config.oauth_client_id,
        upstream_client_secret=client_secret,
        upstream_revocation_endpoint=config.oauth_revocation_url,
        token_verifier=verifier,
        base_url=config.oauth_base_url,
    )


def _import_static_token_verifier() -> type[StaticTokenVerifier]:
    try:
        from fastmcp.server.auth import StaticTokenVerifier
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ConfigurationError("MCP server needs fastmcp; install semdex[mcp-server].") from exc
    return StaticTokenVerifier


def _import_jwt_verifier() -> type[JWTVerifier]:
    try:
        from fastmcp.server.auth.providers.jwt import JWTVerifier
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ConfigurationError("MCP server needs fastmcp; install semdex[mcp-server].") from exc
    return JWTVerifier


def _import_oauth_proxy() -> type[OAuthProxy]:
    try:
        from fastmcp.server.auth import OAuthProxy
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ConfigurationError("MCP server needs fastmcp; install semdex[mcp-server].") from exc
    return OAuthProxy


__all__ = ["build_mcp_auth"]
