# ADR 0002: MCP Server as an Adapter Delivery Surface

**Status:** Accepted

## Context

`semdex` needs to expose its search and reindex use cases to MCP clients
(Claude Desktop, IDE agents, other LLM tooling) in addition to the CLI. The
questions were where the server code belongs in the clean architecture, how
it authenticates network clients without leaking secrets, and how it relates
to the `mcp` client dependency already vendored for the markitdown extractor.

## Decision

- The MCP server is an **adapter / delivery surface**, a sibling of the CLI.
  It lives in `src/semdex/adapters/mcp/` and may import the application use
  cases, the domain, and composition-built services, never the reverse. The
  tool functions are thin: they call the existing `search` and `reconcile`
  use cases and map results to plain dicts. Domain and application stay
  framework-free, and the server is source-type-blind (it passes the opaque
  `uri` through unchanged).
- **Auth secrets load from the environment at runtime, never from tracked
  config.** The `[mcp]` config stores only the *names* of the environment
  variables that hold the bearer token(s) and the OAuth client secret; the
  auth builder reads the real values from an injected `env` mapping. A
  misconfigured mode (bearer with no tokens, OAuth with no jwks/issuer) fails
  fast with a clear `ConfigurationError`.
- **`fastmcp` is the server; the pre-existing `mcp` dependency stays the
  markitdown client.** They are separate concerns: `fastmcp` (the
  `mcp-server` optional extra) builds and runs the server; `mcp` (pulled by
  the `markitdown` extractor extra) is a converter client. Neither replaces
  the other.

## Consequences

- Installing the server is opt-in (`semdex[mcp-server]`); the core CLI does
  not carry `fastmcp`. The server modules import `fastmcp` lazily so importing
  them without the extra fails only when the server is actually built, with a
  message pointing at the extra.
- The import-linter layer contracts continue to hold: `adapters/mcp` depends
  downward only. The server is unit-testable in-memory via
  `fastmcp.Client(server)` with no network.
- Secrets never appear in committed TOML, code, or logs; rotating a token is
  an environment change, not a code change.
- `store.close()` is called when the server stops but remains optional on the
  store adapters (guarded with `getattr`); implementing it properly for the
  long-lived SQL / lance stores is a documented follow-on.
- Fan-out search (`search_datasets`) fuses the per-dataset rankings by Reciprocal
  Rank Fusion (rank only), never by raw cross-model similarity score, because
  different embedding models' scores are on different scales; the query is
  embedded once per distinct model. This keeps a shared corpus indexed once with
  department-specific datasets layered on top, searched together - access control
  stays at the database (per-dataset DSN grants), not in the server.
- semdex serves two dataset KINDS. A SOURCE dataset is connector-mirrored: read-only
  over its source and reconcile-driven (filesystem now, an external IMAP connector
  later). A KNOWLEDGE dataset (`writable`) has no connector; the client writes its
  content directly via the `remember` / `forget` tools. They never mix - a knowledge
  dataset has no connector, so `reconcile` cannot prune it and no source content can
  land in it, and `reindex` refuses it while source datasets refuse the write tools.
  Write authorization is a database GRANT on the dataset's DSN, not server logic; the
  `writable` flag is only the advisory signal. This keeps the "never mutates the
  sources" invariant intact (a knowledge dataset has no external source to mutate)
  while letting an agent deposit and recall its own knowledge.
