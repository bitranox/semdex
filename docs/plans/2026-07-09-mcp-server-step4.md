# MCP server (serve command) - Step 4 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use bitranox:process-agents-subagent-driven-development (recommended) or bitranox:process-plan-executor to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Also invoke bitranox:process-test-driven-development (each task is RED -> GREEN) and bitranox:coding-python-clean-architecture.

**Goal:** Ship a `semdex serve` command exposing an MCP server (tools: `list_datasets`, `search`, `reindex`) over stdio (local) and Streamable HTTP (network), with configurable auth (none | bearer | oauth), wiring the Dataset config from step 3 into per-dataset services.

**Architecture:** The MCP server is an ADAPTER/delivery surface (a sibling of the CLI), a thin layer of tool functions that call the existing use cases (`search`, `reconcile`) and composition. Domain/application stay framework-free; the server is source-type-blind (it passes opaque `uri`s through). FastMCP picks the transport at launch; `store.close()` runs in the FastMCP lifespan.

**Tech Stack:** `fastmcp` v3 (PrefectHQ; built on the official `mcp` SDK already vendored here for the markitdown client), pydantic config models, rich-click CLI, uv, pytest. FastMCP is TDD-able in-memory via `fastmcp.Client(mcp_server)`.

## Handover context (read first - you have zero prior context)

- **Worktree:** `<semdex-checkout>/.claude/worktrees/mcp-source-identity`, branch `worktree-mcp-source-identity` (based on `origin/main`). Work here. If the branch is not pushed, `git -C <worktree> log --oneline -8` shows commits `5bccda5`..`bdccaea` = steps 1-3.
- **The design** this implements: `docs/systemdesign/mcp-multi-dataset-design.md` (committed on branch `bench-persistent-store-knobs`; if absent in this worktree, `git show bench-persistent-store-knobs:docs/systemdesign/mcp-multi-dataset-design.md`). Key settled decisions restated inline below so this plan is self-contained.
- **What already exists (steps 1-3, done):**
  - Sources are keyed by an opaque `uri` (`file://...`): `SourceRef.uri: str`, `Hit.uri: str`, `Chunk.ordinal`. Stores are source-type-blind. Helper: `semdex.adapters.discovery.location.to_uri`/`from_uri`.
  - `application/use_cases/reconciling.py`: `reconcile(*, connector, extract, chunk, embedding, store, collection, max_tokens=256) -> ReconcileReport` (prune vanished, re-index changed, skip unchanged). This is `reindex`'s engine.
  - `application/use_cases/searching.py`: `search(*, embedding, store, collection, query, k) -> list[Hit]`.
  - `application/ports.py`: `SourceConnector` (`sources() -> Sequence[SourceRef]`), `VectorStoreReader.source_hashes(collection) -> dict[str,str]`, `VectorStore` combined port.
  - `adapters/discovery/filesystem.py`: `FilesystemConnector(roots, *, label, extensions, hash_chunk_size)`.
  - `adapters/config/dataset.py`: `DatasetConfig` + `get_datasets(config) -> tuple[DatasetConfig, ...]`. Fields: `name, backend, store_dir, dsn, collection, embedding_provider, embedding_model, embedding_endpoint, sources, partition, read_only`.
  - `adapters/config/vectorstore.py`: `VectorStoreConfig.default_partition: Partition` (enum `Partition.TABLE|DATABASE` in `domain/enums.py`).
  - `composition/__init__.py`: `build_vector_store(backend, store_dir, *, dsn=None, lance_index_threshold=None)`, `build_embedding(provider, *, model=None, endpoint=None, ...)`, `build_index_production(store_dir)`. Read this file before Task 3.
- **Gates (run from the worktree; `make test` also hits codecov, so use these directly):**
  - `uv run pytest -m "not local_only and not integration" -q -p no:cacheprovider`
  - `uv run ruff check src/ tests/` and `uv run ruff format src/ tests/`
  - `uv run pyright src/ tests/`
  - `uv run lint-imports`
  - The `VIRTUAL_ENV ... does not match` warning from uv is harmless.

## Global Constraints

- **Python 3.10+**; type-annotate everything; pyright strict must pass (0 errors).
- **Clean architecture (import-linter):** domain pure; the MCP server lives in `adapters/mcp/` and may import application + domain + composition-built services, never the reverse. Run `lint-imports` after each task.
- **TDD:** every task is RED (watch it fail) -> GREEN (minimal) -> commit. FastMCP tools are tested in-memory with `fastmcp.Client(mcp_server)`.
- **Secrets never in tracked files:** the bearer token and any OAuth client secret load at runtime from env/keyfile, never inline in committed TOML or code. (Iron rule.)
- **Document every TOML setting** comprehensively (default, effect, when to set, consequences, per-scenario recommendation) - the new `17-mcp.toml` must match the style of `15-vectorstore.toml`.
- **ASCII only** in all files (a tell-sweep hook rejects em/en-dashes, curly quotes, ellipsis). Use `-`, straight quotes, `...`.
- **No Claude/AI attribution** in commits. Commit after each task.
- **Config as the only source of tunables** - the serve command reads `[mcp]` + `[[dataset]]`, no hardcoded host/port/paths.

---

### Task 0: Add the fastmcp dependency + confirm its API

**Files:**
- Modify: `pyproject.toml` (add `fastmcp` to a new `mcp`/`serve` optional-dependency group AND to `[dev]` so CI type-checks it, per this project's ci-devextra convention).

- [ ] **Step 1: Confirm the FastMCP v3 API before writing code.** Use context7 (`resolve-library-id "FastMCP"` -> `/prefecthq/fastmcp`, then `query-docs`) for: (a) `FastMCP`, `@mcp.tool`, `mcp.run(transport=...)`; (b) `run_http_async` host/port/`allowed_hosts`/`host_origin_protection`; (c) auth: `fastmcp.server.auth.StaticTokenVerifier`, `providers.jwt.JWTVerifier`, `OAuthProxy`; (d) in-memory testing via `fastmcp.Client(server)`. (Captured 2026-07-09: `FastMCP(name, auth=...)`, `mcp.run(transport="stdio")` / `mcp.run(transport="http", host=, port=)`, `StaticTokenVerifier(tokens={token: {"client_id":.., "scopes":[..]}}, required_scopes=[..])`, `OAuthProxy(upstream_authorization_endpoint=, upstream_token_endpoint=, upstream_client_id=, upstream_client_secret=, token_verifier=JWTVerifier(jwks_uri=, issuer=, audience=), base_url=)`.) Fix any drift below to match the installed version.
- [ ] **Step 2: Add the dependency.** In `pyproject.toml` under `[project.optional-dependencies]` add `mcp-server = ["fastmcp>=3.2"]` (keep the existing `markitdown`/`mcp` client dep untouched), and add `fastmcp>=3.2` to the `dev` extra list. Then `uv sync --extra dev --extra mcp-server`.
- [ ] **Step 3: Verify import.** Run: `uv run python -c "import fastmcp; from fastmcp.server.auth import StaticTokenVerifier; print(fastmcp.__version__)"` Expected: prints a 3.x version, no error.
- [ ] **Step 4: Commit.** `git add pyproject.toml uv.lock && git commit -m "build: add fastmcp for the MCP server (mcp-server extra + dev)"`

---

### Task 1: `[mcp]` config model + get_mcp_config + 17-mcp.toml docs

**Files:**
- Create: `src/semdex/adapters/config/mcp.py`
- Create: `src/semdex/adapters/config/defaultconfig.d/17-mcp.toml`
- Create: `tests/test_config_mcp.py`

**Interfaces:**
- Produces: `McpAuthMode` (enum `none|bearer|oauth`), `McpTransport` (enum `stdio|http`), `McpConfig` (pydantic frozen model), `get_mcp_config(config) -> McpConfig`.

- [ ] **Step 1: Write the failing tests.**

```python
# tests/test_config_mcp.py
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
```

- [ ] **Step 2: Run to verify it fails.** `uv run pytest tests/test_config_mcp.py -q` Expected: ImportError (McpAuthMode / config.mcp missing).
- [ ] **Step 3: Add the enums** to `src/semdex/domain/enums.py` (after `Partition`):

```python
class McpTransport(str, Enum):
    """MCP server transport. ``STDIO`` is local (the client spawns the process);
    ``HTTP`` is Streamable HTTP for the network (SSE is deprecated)."""

    STDIO = "stdio"
    HTTP = "http"


class McpAuthMode(str, Enum):
    """How the HTTP MCP endpoint is authenticated. ``NONE`` = unauthenticated
    (bind localhost / front with a reverse proxy). ``BEARER`` = static bearer
    token(s). ``OAUTH`` = OAuth 2.1 (JWT verification / OAuth proxy). Ignored by
    the stdio transport, which is bounded by the process it runs in."""

    NONE = "none"
    BEARER = "bearer"
    OAUTH = "oauth"
```

- [ ] **Step 4: Add the config model** `src/semdex/adapters/config/mcp.py`:

```python
"""MCP server configuration parsed from the ``[mcp]`` section."""

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
    # bearer: token(s) supplied at runtime via env, never inline here. This names
    # the env var holding a comma-separated token list.
    bearer_token_env: str = "SEMDEX_MCP_BEARER_TOKENS"
    # oauth: JWT verification + optional OAuth proxy. Secrets via env.
    oauth_jwks_uri: str | None = None
    oauth_issuer: str | None = None
    oauth_audience: str | None = None
    oauth_authorize_url: str | None = None
    oauth_token_url: str | None = None
    oauth_client_id: str | None = None
    oauth_client_secret_env: str = "SEMDEX_MCP_OAUTH_CLIENT_SECRET"
    oauth_base_url: str | None = None  # this server's public URL (OAuth proxy)


def get_mcp_config(config: Config) -> McpConfig:
    """Parse the ``[mcp]`` section into an McpConfig (stdio + no auth by default).

    Example:
        >>> from lib_layered_config import Config
        >>> get_mcp_config(Config({}, {})).auth.value
        'none'
    """
    return McpConfig.model_validate(config.get("mcp", {}))


__all__ = ["McpConfig", "get_mcp_config"]
```

- [ ] **Step 5: Run tests.** `uv run pytest tests/test_config_mcp.py -q` Expected: PASS (3 tests).
- [ ] **Step 6: Write `17-mcp.toml`** in the `15-vectorstore.toml` documentation style: a `[mcp]` block with every key documented (default, effect, when to change, consequences, recommendation). Cover: transport (stdio local vs http network; SSE gone), host/port (default 127.0.0.1 = localhost-only; set 0.0.0.0 only behind a trusted proxy), auth (none = localhost/proxy-fronted; bearer = SEMDEX_MCP_BEARER_TOKENS env holds the token(s); oauth = jwks/issuer/audience + optional proxy creds, secret via SEMDEX_MCP_OAUTH_CLIENT_SECRET). State plainly: NEVER put a token/secret inline here; load from env. Keep it ASCII.
- [ ] **Step 7: Gates + commit.** Run pyright + ruff + `uv run python -c "from semdex.adapters.config.loader import get_config; from semdex.adapters.config.mcp import get_mcp_config; print(get_mcp_config(get_config()).transport)"` (expect `McpTransport.STDIO`). `git add -A && git commit -m "feat(config): [mcp] server config (transport/host/port/auth) + 17-mcp.toml docs"`

---

### Task 2: Auth builder (none | bearer | oauth)

**Files:**
- Create: `src/semdex/adapters/mcp/__init__.py` (empty package marker)
- Create: `src/semdex/adapters/mcp/auth.py`
- Create: `tests/test_mcp_auth.py`

**Interfaces:**
- Consumes: `McpConfig`, `McpAuthMode` (Task 1).
- Produces: `build_mcp_auth(config: McpConfig, *, env: Mapping[str, str]) -> object | None` - returns a FastMCP auth provider (or None for `NONE`). `env` is injected (default `os.environ`) so tests pass tokens without touching the real environment.

- [ ] **Step 1: Write failing tests.**

```python
# tests/test_mcp_auth.py
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
```

- [ ] **Step 2: Run to verify it fails.** `uv run pytest tests/test_mcp_auth.py -q` Expected: ImportError (mcp.auth missing).
- [ ] **Step 3: Implement** `src/semdex/adapters/mcp/auth.py`. Map the mode to a FastMCP provider; load secrets from `env`; raise `ConfigurationError` (from `domain.errors`) on missing config. Bearer -> `StaticTokenVerifier(tokens={t: {"client_id": "semdex", "scopes": ["read", "write"]} for t in tokens})`. OAuth: if `oauth_authorize_url`/`oauth_token_url`/`oauth_client_id` set -> `OAuthProxy(upstream_authorization_endpoint=..., upstream_token_endpoint=..., upstream_client_id=..., upstream_client_secret=env[oauth_client_secret_env], token_verifier=JWTVerifier(jwks_uri, issuer, audience), base_url=oauth_base_url)`; else -> the verifier alone `JWTVerifier(jwks_uri, issuer, audience)` (accept tokens from an external IdP without proxying). Require `oauth_jwks_uri` and `oauth_issuer` for OAUTH. Keep imports of fastmcp INSIDE the function (optional dep - `mcp-server` extra), mapping ImportError to `ConfigurationError("install semdex[mcp-server]")`.
- [ ] **Step 4: Run tests.** `uv run pytest tests/test_mcp_auth.py -q` Expected: PASS (4 tests).
- [ ] **Step 5: Gates + commit.** pyright/ruff/lint-imports. `git add -A && git commit -m "feat(mcp): configurable auth builder (none/bearer/oauth)"`

---

### Task 3: Per-dataset services composition

**Files:**
- Modify: `src/semdex/composition/__init__.py` (add `DatasetServices` + `build_dataset_services`)
- Create: `tests/test_dataset_services.py`

**Interfaces:**
- Consumes: `DatasetConfig`, `Partition` (step 3), `build_vector_store`, `build_embedding`, `FilesystemConnector`, `TextExtractor`, `build_chunker`.
- Produces: `DatasetServices` (frozen dataclass: `name: str`, `store: VectorStore`, `embedding: EmbeddingProvider`, `extract: Extract`, `chunk: ChunkText`, `connector: SourceConnector`, `collection: str`, `partition: Partition`, `read_only: bool`), and `build_dataset_services(dataset: DatasetConfig, *, default_partition: Partition) -> DatasetServices`.

- [ ] **Step 1: Read `composition/__init__.py`** to copy the exact `build_vector_store` / `build_embedding` / extractor / chunker construction the CLI already uses (do not invent new wiring; reuse `build_index_production`'s internals where possible).
- [ ] **Step 2: Write failing test** (embedded backend, offline - use `placeholder` embedding + `whitespace` chunker to stay offline, matching the repo's e2e convention):

```python
# tests/test_dataset_services.py
from __future__ import annotations
from pathlib import Path
import pytest
from semdex.adapters.config.dataset import DatasetConfig
from semdex.composition import build_dataset_services
from semdex.domain.enums import EmbeddingBackend, Partition, StoreBackend


@pytest.mark.os_agnostic
def test_builds_embedded_dataset_services(tmp_path: Path) -> None:
    dataset = DatasetConfig(
        name="notes",
        backend=StoreBackend.JSON,
        store_dir=str(tmp_path / "store"),
        collection="notes",
        embedding_provider=EmbeddingBackend.PLACEHOLDER,
        sources=(str(tmp_path / "docs"),),
    )
    services = build_dataset_services(dataset, default_partition=Partition.TABLE)
    assert services.name == "notes"
    assert services.collection == "notes"
    assert services.partition is Partition.TABLE  # inherited (dataset.partition is None)
    assert services.connector.sources() == []  # empty dir -> no sources, no crash
```

- [ ] **Step 3: Run to verify it fails.** `uv run pytest tests/test_dataset_services.py -q` Expected: ImportError (build_dataset_services missing).
- [ ] **Step 4: Implement** `DatasetServices` + `build_dataset_services` in `composition/__init__.py`: resolve `partition = dataset.partition or default_partition`; `store = build_vector_store(dataset.backend, Path(expanduser(dataset.store_dir or default)), dsn=dataset.dsn)`; `embedding = build_embedding(dataset.embedding_provider, model=dataset.embedding_model, endpoint=dataset.embedding_endpoint)`; `connector = FilesystemConnector([Path(expanduser(s)) for s in dataset.sources])`; `extract = TextExtractor()`; `chunk = build_chunker(...)` matching the CLI default. Expand `~` in paths.
- [ ] **Step 5: Run tests.** Expected: PASS.
- [ ] **Step 6: Gates + commit.** `git add -A && git commit -m "feat(composition): build_dataset_services (per-dataset store/embedding/connector)"`

---

### Task 4: MCP tools + server assembly

**Files:**
- Create: `src/semdex/adapters/mcp/server.py`
- Create: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: `DatasetServices` (Task 3), `build_mcp_auth` (Task 2), `search`, `reconcile`.
- Produces: `build_mcp_server(datasets: Sequence[DatasetServices], *, auth: object | None = None, name: str = "semdex") -> FastMCP`. Tools: `list_datasets() -> list[dict]`; `search(dataset: str, query: str, k: int = 5) -> list[dict]`; `reindex(dataset: str) -> dict` (rejects a `read_only` dataset).

- [ ] **Step 1: Write failing tests** using FastMCP's in-memory client (no network):

```python
# tests/test_mcp_server.py
from __future__ import annotations
from pathlib import Path
import pytest
from fastmcp import Client
from semdex.adapters.config.dataset import DatasetConfig
from semdex.adapters.mcp.server import build_mcp_server
from semdex.composition import build_dataset_services
from semdex.domain.enums import EmbeddingBackend, Partition, StoreBackend


def _services(tmp_path: Path, *, read_only: bool = False):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("banana cherry apple", encoding="utf-8")
    dataset = DatasetConfig(
        name="notes",
        backend=StoreBackend.JSON,
        store_dir=str(tmp_path / "store"),
        collection="notes",
        embedding_provider=EmbeddingBackend.PLACEHOLDER,
        sources=(str(tmp_path / "docs"),),
        read_only=read_only,
    )
    return build_dataset_services(dataset, default_partition=Partition.TABLE)


@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_list_datasets_and_reindex_then_search(tmp_path: Path) -> None:
    server = build_mcp_server([_services(tmp_path)])
    async with Client(server) as client:
        listed = await client.call_tool("list_datasets", {})
        assert any(d["name"] == "notes" for d in listed.data)
        await client.call_tool("reindex", {"dataset": "notes"})
        hits = await client.call_tool("search", {"dataset": "notes", "query": "banana apple", "k": 3})
        assert hits.data and hits.data[0]["uri"].endswith("a.md")


@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_reindex_rejects_read_only_dataset(tmp_path: Path) -> None:
    server = build_mcp_server([_services(tmp_path, read_only=True)])
    async with Client(server) as client:
        with pytest.raises(Exception):  # FastMCP surfaces a tool error
            await client.call_tool("reindex", {"dataset": "notes"})
```

(Confirm the `Client` result attribute - `.data` vs `.structured_content` - against the installed FastMCP via context7 in Task 0; adjust the assertions to match. `pytest-asyncio` is needed - add it to the `dev` extra if absent and mark with `asyncio_mode=auto` or the decorator.)

- [ ] **Step 2: Run to verify it fails.** Expected: ImportError (mcp.server missing).
- [ ] **Step 3: Implement** `build_mcp_server`: create `FastMCP(name, auth=auth)`, index the datasets by name in a dict, register three `@mcp.tool` functions. `search` -> `search(embedding=svc.embedding, store=svc.store, collection=svc.collection, query=query, k=k)` then map each `Hit` to `{"uri":.., "ordinal":.., "score":.., "text":.., "label":.., "collection":..}`. `reindex` -> reject if `svc.read_only` (raise a ValueError/ToolError), else `reconcile(connector=svc.connector, extract=svc.extract, chunk=svc.chunk, embedding=svc.embedding, store=svc.store, collection=svc.collection)` and return the report as a dict. `list_datasets` -> name/backend/collection/read_only/partition per dataset (never expose the dsn/secret). Unknown dataset name -> a clear ToolError. Keep the `fastmcp` import at module top guarded (optional dep).
- [ ] **Step 4: Run tests.** Expected: PASS (2 tests).
- [ ] **Step 5: Gates + commit.** `git add -A && git commit -m "feat(mcp): MCP server with list_datasets/search/reindex tools"`

---

### Task 5: `serve` CLI command + lifespan close()

**Files:**
- Create: `src/semdex/adapters/cli/commands/serve.py`
- Modify: `src/semdex/adapters/cli/root.py` (register the command)
- Create: `tests/test_cli_serve.py`

**Interfaces:**
- Consumes: `get_mcp_config`, `get_datasets`, `build_dataset_services`, `build_mcp_auth`, `build_mcp_server`.
- Produces: a `serve` command: `semdex serve [--transport stdio|http] [--host H] [--port P]` (flags override `[mcp]` config).

- [ ] **Step 1: Write a failing test** that does NOT actually block on a server: monkeypatch `build_mcp_server` to return a fake whose `.run(...)` records its args, and assert the command resolves config -> datasets -> `run(transport=..., host=..., port=...)`. Use the repo's `cli_runner` + `inject_config` fixtures (see `tests/conftest.py`).

```python
# tests/test_cli_serve.py (sketch - follow existing test_cli_index_search.py patterns)
# - inject a Config with one [[dataset]] + [mcp].transport="http"
# - monkeypatch semdex.adapters.cli.commands.serve.build_mcp_server -> returns Recorder()
# - invoke ["serve", "--port", "9001"]; assert Recorder.run called with transport="http", port=9001
```

- [ ] **Step 2: Run to verify it fails.** Expected: no `serve` command / import error.
- [ ] **Step 3: Implement** `serve.py`: read `get_config()`, `get_mcp_config`, `get_vector_store_config().default_partition`, `get_datasets`; build `DatasetServices` per dataset; `auth = build_mcp_auth(mcp_cfg, env=os.environ)`; `server = build_mcp_server(services, auth=auth)`; then for http `server.run(transport="http", host=host, port=port)` else `server.run(transport="stdio")`. Register a FastMCP lifespan/atexit that calls `store.close()` on each dataset's store IF it has one (guard with `getattr(store, "close", None)`; the store `close()` is still deferred - add a no-op default in the memory/json stores if needed, or guard). Wire into `root.py` next to `index`/`search`.
- [ ] **Step 4: Run tests.** Expected: PASS.
- [ ] **Step 5: Manual smoke (both transports).** stdio: `uv run python -c "from fastmcp import Client; import asyncio; ..."` against an in-process build, OR run `SEMDEX___MCP__TRANSPORT=stdio semdex serve` and connect a client. http: `SEMDEX___MCP__TRANSPORT=http semdex serve --port 9001 &` then a `fastmcp.Client("http://127.0.0.1:9001/mcp")` lists tools; kill by pid (never `pkill -f "semdex serve"` - self-match). Document the exact commands you used in the commit body.
- [ ] **Step 6: Gates + commit.** `git add -A && git commit -m "feat(cli): serve command running the MCP server over stdio/http"`

---

### Task 6: Docs, package metadata, final gate

**Files:**
- Modify: `README.md` / `docs/COMPONENT_SETUP.md` (a short "serve as an MCP server" section: install `semdex[mcp-server]`, the `[mcp]` knobs, stdio vs http, the three auth modes, example client config).
- Modify: `CLAUDE.md` "Code Quality" (record the accepted stances: MCP server is an adapter delivery surface; auth secrets via env; fastmcp is the server, `mcp` stays the markitdown client).
- Modify: `CHANGELOG.md`.

- [ ] **Step 1:** Write the docs (ASCII; no live secrets, generic examples per the docs-describe-current-code rule).
- [ ] **Step 2: Full gate.** Run all four gate commands; all green. Then `uv run pytest -m "not local_only and not integration" -q` full count should be > the step-3 count of 638.
- [ ] **Step 3: Commit + optionally open a PR** for the whole branch (steps 1-4).

---

## Self-review checklist (done while writing; re-verify before executing)

- Spec coverage: transports (stdio+http) = Task 5; auth none/bearer/oauth = Tasks 1-2; datasets wired = Tasks 3-4; reconcile gets its caller = Task 4 `reindex`; partition resolved = Task 3; TOML docs = Task 1; secrets-from-env = Tasks 1-2.
- Type consistency: `DatasetServices` fields (Task 3) are consumed verbatim in Task 4; `build_mcp_auth(...) -> object|None` (Task 2) feeds `build_mcp_server(auth=...)` (Task 4) and `serve` (Task 5).
- Deferred / not in this plan (carry as follow-ons): title/breadcrumb metadata on Hit (decision 10 baseline), no-re-embed `relocate` for moves, RRF fan-out (step 5 / #25), watchdog + IMAP connector (step 6 / #26). The `search` tool is single-dataset here; multi-dataset fan-out is step 5.
- Store `close()` is still a stub (Task 5 guards for it); implement it properly for the SQL/lance stores when the long-lived server proves it necessary (CLAUDE.md deferral).

## References

- Design: `docs/systemdesign/mcp-multi-dataset-design.md` (branch bench-persistent-store-knobs).
- FastMCP: context7 `/prefecthq/fastmcp` (transports, auth, in-memory Client).
- Tasks tracker: #24 (this), #25 (fan-out), #26 (watchers/IMAP), #20 (summary metadata).
