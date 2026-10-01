# Implementation plan: fixes from the fastmcp 3-to-4 audit of the MCP adapter ([35])

Source: the read-only audit report of 2026-09-28 (`.bitranox/sdd/audit-35-fastmcp4-report.md`,
gitignored; its findings are restated here in full so the plan stands alone). The adapter under
`src/semdex/adapters/mcp/` needs no change to run on fastmcp 4.0.10, but the audit found one
Critical defect one directory away, three Important gaps that would let the next fastmcp release
change behaviour with every test green, and three Minor items.

## Global Constraints

- Work on `main`, commit after every task, push after the gate; never a PR (repo rule).
- ASCII punctuation only in files (no em-dash, no arrow character, no curly quotes).
- pyright strict: DEFINE the missing types (a `TYPE_CHECKING` import, a typed facade), never a
  suppression; ruff clean; imports at module top unless the lazy optional-dependency import is the
  stated reason.
- Any new knob is a config key under `[mcp]` in `src/semdex/adapters/config/mcp.py` (`McpConfig`)
  and `src/semdex/adapters/config/defaultconfig.d/17-mcp.toml`, documented in the TOML in the
  file's existing style (default, effect, when to change, consequences, recommendation).
- Tests exercise the real fastmcp `Client` in-memory and the real `mcp.types` models; no stub
  standing in for a type the real package provides.
- `make test` is the gate, read from its RC line and result envelope, never the tail; the [22]
  sweep runs on its own frozen venv, so the gate is allowed.
- Never `pkill -f`; the sweep runs on this box.

## Ground truth (2026-09-28)

- fastmcp 4.0.10, mcp 2.2.0 installed; pyproject pins `fastmcp>=4.0.10`, `mcp>=2.2.0`.
- `mcp.types.CallToolResult` fields: `content`, `is_error`, `meta`, `result_type`,
  `structured_content`. `getattr(result, "isError", ...)` resolves ONLY after `import fastmcp`
  (its camelCase compatibility bridge), and then with a `FastMCPDeprecationWarning`.
- `src/semdex/adapters/extractor/markitdown.py:105-114` `_text_of(result: object, url)` guards
  on `getattr(result, "isError", False)` and otherwise returns the first text block.
  `tests/test_extractor_markitdown.py:86-88` defines a `_Result` dataclass with an `isError`
  field, so the test answers the attribute the code asks for and never meets the real name.
- `src/semdex/adapters/mcp/server.py:44` `build_mcp_server(datasets, *, name, auth: object | None,
  rrf_k)`; line 68 `mcp = fastmcp_cls(name, auth=auth)`; tools annotated `-> list[dict[str, Any]]`
  / `-> dict[str, Any]` and returning `.model_dump(mode="json")` of the `schemas.py` models, so
  fastmcp publishes `{"additionalProperties": true, "type": "object"}` as every output schema.
  `ToolError` is raised at lines 74, 124, 126, 213, 237 for caller-facing messages.
  `mask_error_details` is left at fastmcp's default `False`: an unexpected exception's text
  reaches the client verbatim. Comment at 84-86 says "these three closures" above six.
- `src/semdex/adapters/mcp/auth.py:69-91` builds `JWTVerifier` / `OAuthProxy` via untyped
  `_import_*() -> type` helpers; `tests/test_mcp_auth.py` (4 tests) never constructs either.
  `OAuthProxy` is built without `upstream_revocation_endpoint`.
- `tests/test_mcp_server.py` (8 tests) calls tools and reads `result.data[i]["uri"]`; nothing calls
  `list_tools()`. `tests/test_e2e_mcp_knowledge_matrix.py` (21 tests, local_only) indexes rows
  the same way.
- `src/semdex/adapters/cli/commands/serve.py:116` is the one production caller of
  `build_mcp_server`.

### Task 1: The markitdown extractor reads the real error flag, pinned by the real type

**Files:**
- Modify: `src/semdex/adapters/extractor/markitdown.py` (`_text_of`)
- Modify: `tests/test_extractor_markitdown.py` (replace the `_Result` / `_Block` stubs)

**Interfaces:** `_text_of(result: CallToolResult, url: str) -> str` with the type imported under
`TYPE_CHECKING` from `mcp.types` (the runtime import stays lazy: the markitdown extra is optional).

**Out of scope:** the MCP call itself (`_call_convert_tool`), the Docker integration tests.

**STOP conditions:** `mcp` is not importable in the project venv; a test outside this file fails.

- [ ] Step 1: Write the failing test. In `tests/test_extractor_markitdown.py`, build results with
  the real `mcp.types.CallToolResult` and `mcp.types.TextContent`; add
  `test_an_error_result_raises_instead_of_returning_the_error_text` constructing
  `CallToolResult(content=[TextContent(type="text", text="conversion failed")], is_error=True)`
  and asserting `_text_of` raises `ExtractionError` whose message names the URL. Run it without
  `import fastmcp` anywhere in the test module: it must FAIL on the current code by RETURNING
  "conversion failed" (record the assertion output). Rewrite the two existing stub-based tests
  onto the real types; delete `_Result` and `_Block`.
- [ ] Step 2: Fix `_text_of`: `if result.is_error:` with `result: CallToolResult`; keep the text
  scan over `result.content` reading `block.text` only for `TextContent` instances
  (`isinstance`), drop the `getattr` reads. Update the docstring.
- [ ] Step 3: Run the module's tests, `ruff check`, `ruff format --check`, `pyright` on both files.
  Also run `env -u VIRTUAL_ENV .venv/bin/python -m pytest tests/test_extractor_markitdown.py -q
  -p no:cacheprovider -W error::DeprecationWarning`: must pass with zero warnings.
- [ ] Step 4: Commit (`fix(extractor): markitdown reads is_error, an error result is never
  indexed as text`), message via `-F`.

### Task 2: The server masks unexpected errors, and the tests pin the tool surface and the auth providers

**Files:**
- Modify: `src/semdex/adapters/mcp/server.py` (constructor call, the stale comment)
- Modify: `src/semdex/adapters/config/mcp.py`, `src/semdex/adapters/config/defaultconfig.d/17-mcp.toml`
  (new keys `mask_error_details`, `oauth_revocation_url`)
- Modify: `src/semdex/adapters/mcp/auth.py` (forward `upstream_revocation_endpoint`)
- Modify: `src/semdex/adapters/cli/commands/serve.py` (pass the new knob through)
- Test: `tests/test_mcp_server.py`, `tests/test_mcp_auth.py`, the config test module that covers
  `McpConfig` (find it with `grep -rl McpConfig tests/`)

**Interfaces:** `build_mcp_server(..., mask_error_details: bool = True)`; `McpConfig.mask_error_details:
bool = True`; `McpConfig.oauth_revocation_url: str | None = None`; `build_auth_provider` forwards it
as `upstream_revocation_endpoint` when set.

**Out of scope:** return-type changes (Task 3); any new tool.

**STOP conditions:** fastmcp 4.0.10's `FastMCP.__init__` lacks `mask_error_details` or
`OAuthProxy.__init__` lacks `upstream_revocation_endpoint` (check with `inspect.signature` first).

- [ ] Step 1: Failing tests first. (a) In `tests/test_mcp_server.py`: a tool that raises
  `RuntimeError("SECRET-INTERNAL-DETAIL")` registered on a server built by `build_mcp_server` (add
  the tool through the returned `FastMCP` object's `.tool` decorator in the test) surfaces to the
  `Client` as a `ToolError` whose text does NOT contain the original message; and a second test
  that a deliberate `ToolError` message (an unknown dataset name) still reaches the client
  verbatim. (b) `test_the_tool_surface_is_exactly_the_six_tools`: `await client.list_tools()`
  yields exactly `{"list_datasets", "search", "reindex", "search_datasets", "remember", "forget"}`
  and, per tool, the `required` parameter list you read from `tool.inputSchema` today (write the
  list into the test from a one-off print, then assert it). (c) In `tests/test_mcp_auth.py`: the
  OAUTH branch with a fabricated `jwks_uri` and issuer builds a `JWTVerifier`; with the upstream
  fields set too it builds an `OAuthProxy`; and with `oauth_revocation_url` set the proxy's
  `upstream_revocation_endpoint` equals it (read the attribute name off the instance). (d) The
  config test: `McpConfig().mask_error_details is True`, `oauth_revocation_url is None`, and both
  parse from a TOML section.
- [ ] Step 2: Implement: config fields plus TOML docs; `build_mcp_server` passes
  `mask_error_details=mask_error_details` to the constructor; `serve.py` passes
  `mcp_cfg.mask_error_details`; auth forwards the revocation endpoint; fix the "three closures"
  comment to "these closures".
- [ ] Step 3: Run the three test modules, ruff, pyright on every touched file.
- [ ] Step 4: Commit (`feat(mcp): mask unexpected tool errors by default; pin the tool surface and
  the auth providers in tests`), `-F`.

### Task 3: The tools publish their Pydantic output schemas, and the fastmcp types are defined at the boundary

**Files:**
- Modify: `src/semdex/adapters/mcp/server.py` (return annotations and returns), `schemas.py`
  (`Field(description=...)` on the fields carrying a rule, above all `rrf_score`)
- Modify: `src/semdex/adapters/mcp/auth.py` and `server.py` typing (`TYPE_CHECKING` imports of
  `FastMCP`, `AuthProvider`, `JWTVerifier`, `OAuthProxy`, `StaticTokenVerifier`; `_import_*()
  -> type[X]`; `auth: AuthProvider | None`)
- Test: `tests/test_mcp_server.py` (assertions move from `row["uri"]` to the parsed shape),
  `tests/test_e2e_mcp_knowledge_matrix.py` (local_only, same move; run it only if its markers let
  it run offline, else adapt by reading and say so in the report)

**Interfaces:** tools annotated `-> list[SearchHit]`, `-> list[DatasetInfo]`, `-> ReindexResult`
and so on, returning model instances; fastmcp derives `outputSchema` and serialises.

**Out of scope:** changing any tool's fields or semantics.

**STOP conditions:** a `Client.call_tool(...).data` shape that the tests cannot read without
monkeypatching; pyright cannot type a `TYPE_CHECKING` import because a fastmcp symbol lacks a
stub (then a typed Protocol facade per the house rule, and say so).

- [ ] Step 1: Failing test first: extend the Task 2 surface test so each tool's `outputSchema`
  is non-empty and names the model's fields (e.g. `search` output items require `uri`); it fails
  today with `additionalProperties: true`.
- [ ] Step 2: Change the annotations and returns; add the field descriptions; type the boundary.
- [ ] Step 3: Adapt the assertions in the two test files to the parsed shape (`.data` is now
  model-like; `structured_content` keeps the JSON); run them, ruff, pyright.
- [ ] Step 4: Full gate `make test` (RC line and envelope), commit (`refactor(mcp): tools return
  their Pydantic models so fastmcp publishes the output schema; boundary types defined`), `-F`,
  push, then close the [35] line in `OPEN-WORK.md` (`| closed: 2026-09-28, ...`) in a follow-up
  docs commit and push.
