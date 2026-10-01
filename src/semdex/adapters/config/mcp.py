"""MCP server configuration parsed from the ``[mcp]`` section.

Selects the transport (stdio / Streamable HTTP) and how the HTTP endpoint is
authenticated. Secrets (bearer tokens, the OAuth client secret) are NEVER stored
here - only the names of the environment variables that hold them; the auth
builder reads the real values at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from ...domain.enums import McpAuthMode, McpTransport

if TYPE_CHECKING:
    from lib_layered_config import Config


class McpConfig(BaseModel):
    """Validated, immutable MCP server settings.

    Example:
        >>> McpConfig().transport.value
        'stdio'
    """

    model_config = ConfigDict(frozen=True)

    transport: McpTransport = McpTransport.STDIO
    host: str = "127.0.0.1"  # http only; default localhost (safe)
    port: int = Field(default=8080, gt=0, le=65535)
    auth: McpAuthMode = McpAuthMode.NONE
    # Reciprocal Rank Fusion constant for the fan-out search_datasets tool - see
    # 17-mcp.toml. Never affects single-dataset search.
    rrf_k: int = Field(default=60, gt=0)
    # bearer: token(s) supplied at runtime via env, never inline here. This names
    # the env var holding a comma-separated token list (S105: the default is an env
    # var NAME, not a secret - the secret only ever lives in that env var).
    bearer_token_env: str = "SEMDEX_MCP_BEARER_TOKENS"  # noqa: S105
    # oauth: JWT verification + optional OAuth proxy. Secrets via env.
    oauth_jwks_uri: str | None = None
    oauth_issuer: str | None = None
    oauth_audience: str | None = None
    oauth_authorize_url: str | None = None
    oauth_token_url: str | None = None
    oauth_client_id: str | None = None
    oauth_client_secret_env: str = "SEMDEX_MCP_OAUTH_CLIENT_SECRET"  # noqa: S105  (env var NAME, not the secret)
    oauth_base_url: str | None = None  # this server's public URL (OAuth proxy)
    oauth_revocation_url: str | None = None  # upstream token-revocation endpoint (OAuth proxy only)
    # Mask an unexpected tool exception's message/traceback from MCP clients - see
    # 17-mcp.toml. A deliberately raised ToolError always reaches the client verbatim.
    mask_error_details: bool = True


def get_mcp_config(config: Config) -> McpConfig:
    """Parse the ``[mcp]`` section into an McpConfig (stdio + no auth by default).

    Example:
        >>> from lib_layered_config import Config
        >>> get_mcp_config(Config({}, {})).auth.value
        'none'
    """
    return McpConfig.model_validate(config.get("mcp", {}))


__all__ = ["McpConfig", "get_mcp_config"]
