# Changelog

All notable changes to this project will be documented in this file following
the [Keep a Changelog](https://keepachangelog.com/) format.


## [Unreleased]

## [1.5.8] 2026-10-29 00:39:20

### Fixed

- The `sqlite_vec` store refuses a Python whose `sqlite3` was built without SQLite extension loading
  (the python.org macOS installer is one) with a `VectorStoreError` naming that reason. It raised
  `AttributeError: 'sqlite3.Connection' object has no attribute 'enable_load_extension'` instead. The
  sqlite-vec tests now ask one shared helper whether sqlite-vec is importable AND loadable here, and
  skip with that reason where it is not.
- `logdemo` refuses a `LOG_*` variable lib_log_rich refuses (`LOG_CONSOLE_LEVEL=bogus`) with exit 78
  and lib_log_rich's own line, as `config` does. The demo reads the variables itself, so its
  `ValueError` reached `main()` and exited with the OS errno (22 on POSIX, 87 on Windows).
- The test suite no longer reads the app and host configuration layers of the machine it runs on.
  An autouse fixture points lib_layered_config's three app-root variables (`LIB_LAYERED_CONFIG_ETC`,
  `_MAC_APP_ROOT`, `_PROGRAMDATA`) at an empty directory under `tmp_path`, so a system-wide deploy
  (for example a `/etc/xdg/<slug>/config.d/99-*.toml`) can no longer fail tests locally while CI stays
  green. The subprocess tests in `test_logging_dotenv_isolation.py` dropped every
  `LIB_LAYERED_CONFIG*` variable and so read the real system layers; they now keep the redirect.
- `logdemo` works with lib_log_rich 7.0, which takes a `ConsoleTheme` member rather than a string
  (it failed with `AttributeError: 'str' object has no attribute 'value'`). `--theme` offers
  lib_log_rich's own theme list, so an unknown name is a usage error (exit 2) instead of the
  library's `ValueError` reaching `main()`, whose exit code was the OS errno (22 on POSIX, 87 on
  Windows).
- The test suite no longer reads the developer's own user configuration layer. `user_layer_in_tmp_path`
  is now autouse and points the user layer (`XDG_CONFIG_HOME`, `LIB_LAYERED_CONFIG_MAC_HOME_ROOT`,
  `_APPDATA`, `_LOCALAPPDATA`) at an empty directory under `tmp_path`; `HOME` is no longer redirected,
  because git and uv read it.
- The doctests under `src/` no longer run against the machine's configuration layers: a fixture in
  `tests/conftest.py` never reaches them. The two layer redirects now live in a rootdir `conftest.py`,
  their variable lists in `tests/layer_roots.py`; pyproject sets `testpaths` to include `src` and
  `tests` (which keeps the root `conftest.py` out of doctest collection), adds `tests` to `pythonpath`
  and pyright's `extraPaths`, and gives pyright an execution environment rooted at `tests/`. A test
  checks that every doctest collected from `src/` gets both redirects.

## [1.5.7] 2026-10-24 15:27:00

### Changed
- Raise the btx_lib_mail floor to 4.6.0.

### Security
- pip-audit ignores 37 vulnerabilities in pillow 12.3.0 (fixed in pillow 13.0.0). fastembed 0.9.0,
  the newest release, requires `pillow<13.0`, so the lock cannot take the fix; semdex uses only
  fastembed's `TextEmbedding` and never its image path. The ignores go when a fastembed release
  allows pillow 13.

## [1.5.6] 2026-10-15 12:49:43

### Added
- `uv.lock` is committed, and `[tool.uv] exclude-newer` sets a cooldown that keeps releases younger
  than it out of the lock. CI installs exactly the locked tree on push and pull request; the nightly
  scheduled run still resolves afresh to catch upstream breakage early.

### Changed
- The install section explains why installing with uv protects against supply-chain attacks and why
  a plain `pip install` from PyPI does not.

## [1.5.5] 2026-10-12 20:12:11

### Fixed
- **A digit-only `.env` value no longer stops the program.** lib_layered_config 7.1 reads an
  unquoted `.env` or environment value that spells a number losslessly as an int, so
  `INDEX__COLLECTION=2026`, `EMBEDDING__API_KEY=123456` or `MCP__HOST=127` reached a text setting
  as a number and failed with `Input should be a valid string`. Every configuration section model
  (index, embedding, summary, extractor, chunker, health, mcp, vector store, sanitize, watch,
  dataset), the email model and the logging model now turn such a number back into the text the
  operator wrote. Numeric settings still read as numbers and out-of-range values are still refused.

### Changed
- **Synced with bitranox_template_py_cli 2.4.0** (a three-way merge from the template of
  2026-07-24, the version this app last matched before the 2.0 fixes were ported by hand). The
  hand-ported fixes are replaced by the template's own versions; what only this app has stays: the
  `index`, `search` and `serve` commands, the indexing, search, MCP, watch and health services and
  their configuration files, and the `--env-file` option. semdex keeps having no `hello` command.
- `EmailConfig` is a subclass of btx_lib_mail's `ConfMail` (`btx_lib_mail>=4.0.0`). The Python
  field names follow ConfMail (`smtphosts`, `smtp_use_starttls`, `smtp_timeout`); the keys in the
  configuration files stay the same.
- An unknown key in `[email]` or `[email.attachments]` is refused (exit 78), with one `Error:` line
  per problem that names the file key (`email.timeout`, not `email`). A refused SMTP host no longer
  repeats the host in the message, since a host entry can carry credentials.
- New `[email]` keys: `starttls_verify`, `local_hostname`, `delivery_deadline`,
  `recipient_max_count`, and `attachments.max_count`.
- A delivery error keeps the library's message; it is no longer replaced when it contains a word
  such as "password".
- `config --profile` checks the logging configuration with `lib_log_rich.validate_config` and
  refuses what `config` without a profile refuses.
- A `--set` value appears in the `config` output with the source `cli`.
- `send-notification` no longer offers `--raise-on-missing-attachments`; it takes no attachments.
  `send-email` keeps it.
- Console output, `index`, `search` and `serve` included, goes through one adapter that replaces a
  character the console codepage cannot show instead of crashing on it (a cp1252 Windows console).
- **A refused attachment exits 77, not 1 with a traceback (exit code change).** An attachment
  btx_lib_mail's security checks refuse (a blocked extension such as `.exe`, a file under a
  blocked directory, a symlink, a file over the size limit) made `send-email` exit 1 with
  `Error: Unexpected error - Attachment security violation (...)` and log the refusal with a
  traceback, as if the program had crashed. Since btx_lib_mail 4.0.0 every blocked extension is
  refused on every platform, so this is an ordinary outcome on Linux and macOS too. It now exits
  77 (`ExitCode.ATTACHMENT_REFUSED`, sysexits `EX_NOPERM`: not permitted by policy) with one
  line, `Error: Attachment refused by security policy - <the library's reason>`, logs no
  traceback, and delivers nothing. A script that checks for 1 after a refused attachment must
  check for 77. With `email.attachments.raise_on_security_violation = false` the attachment is
  still skipped with a warning and the message sent, as before; a violation that names no attachment
  left to drop is refused in warn mode too, and now exits 77 where it exited 1.

### Removed
- `adapters/config/permissions.py` (`PermissionDefaults`, `get_permission_defaults`,
  `get_modes_for_target`, `parse_mode`): lib_layered_config reads and validates the permission
  section now; a caller that wants the settings uses its `deploy_permissions_from_config`.

### Fixed
- **BREAKING (behaviour): a `.env` reaches logging and nothing else.** The logging setup called
  lib_log_rich's `enable_dotenv()`, which copied every line of the nearest `.env` into the process
  environment, so a later configuration load (`config --profile`, the deploy's permission read)
  took an app-prefixed `.env` line for the environment layer: a prefixed
  `SEMDEX___LIB_LAYERED_CONFIG__DEFAULT_PERMISSIONS__USER_FILE=400` in the working directory
  refused `config-deploy` even under `--env-file`, and a prefixed `SEMDEX___EMAIL__...` line
  overrode `config --profile`. Logging now copies only the `LOG_*` lines, never over a variable
  that is already set, from the `--env-file` when one is given, otherwise from the nearest `.env`
  up to the project root, without changing directory and passing over a directory it cannot
  read. A `.env` that is not UTF-8 no longer stops logging from starting. Other `.env` lines no
  longer reach the environment: `SEMDEX_MCP_BEARER_TOKENS`, `SEMDEX_MCP_OAUTH_CLIENT_SECRET`,
  `FASTEMBED_CACHE_PATH` and `DEVELOPMENT_MODE`, which semdex reads from the process environment,
  must be exported (`serve` refuses with exit 78 naming a missing MCP secret, it never starts
  unauthenticated).
- **An invalid `[lib_log_rich]` value or `LOG_*` variable no longer disables every command (exit
  code change).** A value lib_log_rich refuses in the section (`rate_limit = "100:60"`,
  `queue_maxsize = 0`, an unknown `console_level`) exited 22 with pydantic's multi-line report
  from every command, `config-deploy --force` included, and one in a `LOG_*` variable, exported
  or in the `.env` logging reads (`LOG_CONSOLE_LEVEL=bogus`), exited 1. Both are now a
  configuration failure like a broken file: `config`, `index`, `search`, `serve`, `send-email`
  and `send-notification` refuse with exit 78, and the other commands run (exit 0). A problem the
  section's type check finds is one line per problem naming the key, never the value
  (`lib_log_rich.rate_limit: Input should be a valid tuple`); a value only lib_log_rich refuses
  carries lib_log_rich's own message (`Error: lib_log_rich: Unknown log level: 'bogus'`), which
  may name neither the setting nor where it was set. Logging falls back to its defaults: with the
  valid `LOG_*` variables still in force when the section was refused, and with every `LOG_*`
  variable hidden for that start (and put back afterwards) only when one of them is the refused
  setting. `InvalidLoggingConfigError` (a `ConfigurationError`) is what the logging setup raises
  for it, after logging has started.
- **The documented `.env` and environment syntax for logging tables and lists works.**
  `.env.example`, `defaultconfig.d/90-logging.toml` and the README still showed comma-separated
  lists (`SEMDEX___EMAIL__SMTP_HOSTS="a:587,b:587"`) and `LEVEL=style` / `field=regex` pairs, but a
  comma-separated value arrives as ONE string in both layers: one bogus SMTP host, or a logging
  table that lib_log_rich refuses. They now show a JSON array or object (unquoted in `.env`,
  shell-quoted in the environment) or one key per entry
  (`LIB_LOG_RICH__SCRUB_PATTERNS__API_KEY=.+`), and say that every `.env` value is text that the
  setting's own validation reads.
- **BREAKING (exit code): an invalid `[email]` section exits 78, no longer 22.** `send-email` and
  `send-notification` let the `ValidationError` escape to `main()`'s catch-all, which exited 22
  with pydantic's multi-line report and documentation URL. They now exit 78 (EX_CONFIG) with one
  line per problem, `Error: Invalid configuration: email.<key>: <reason>` (an `[email.attachments]`
  setting is named by its nested key, and the refused value, which can be the SMTP password, is
  never shown). An invalid option value (`--timeout -5`) still exits 22, now in the same one-line
  form (`Error: Invalid option value: ...`). A script that checks for 22 after a refused
  configuration must check for 78. `describe_validation_error` in `adapters/email/config.py` is
  the exported renderer.
- **`config-deploy` leaves every permission decision to lib_layered_config (exit code change).**
  The command read `[lib_layered_config.default_permissions]` from its own configuration load,
  `.env` included, and used only `enabled` from it: a `.env` in the working directory (or
  `--env-file`) decided whether modes were set, and a `--set` of a mode never reached the deploy.
  It now hands its options and any `--set` of the section to lib_layered_config's `deploy_config`,
  which reads the section itself, from the bundled defaults, the configuration files the deploy
  does not overwrite and the environment, never from `.env`, so a `.env` can neither change a
  deployed mode nor block a deploy, and `config-deploy --force` replaces a deployed file that
  holds a bad value without further options. A setting the library refuses (a malformed, unsafe
  or bare-integer mode such as TOML `user_file = 400`, which is decimal 400, i.e. `0o620`; a
  non-boolean `enabled`; an unknown key; a section that is not a table) exited 1 with
  "Failed to deploy configuration: ..."; it now exits 78 before anything is written, with one
  `Error:` line per problem naming the key and where it was set (`(source: override)` for a
  `--set`), then, for a configured value, a hint that both `--dir-mode` and `--file-mode` deploy
  anyway.
- **`--dir-mode`/`--file-mode` refuse an unsafe or malformed mode as a usage error (exit code
  change).** They were parsed with a bare `int(value, 8)`, so `-1`, `777` or a group-writable mode
  reached the deploy and failed there as "Failed to deploy configuration" (exit 1). They are now
  parsed by lib_layered_config's `DeployMode`, the rule the library applies to configured modes,
  and a refused mode exits 2 naming the offending bits before anything is written;
  `--no-permissions` together with a mode option is a usage error too. `config-deploy` logs
  "Deployed configuration" after the deploy instead of "Deploying configuration" before it, and
  says "(permissions not set)" only for an explicit `--no-permissions`.
- **A broken configuration file no longer disables every command (exit code change).** The root
  group loaded the configuration before any subcommand option was parsed and let a load error
  escape, so a malformed `config.toml`, an unreadable one or an `--env-file` that is not UTF-8 made
  every command, `--help` and `config-deploy` (the command that replaces the file) exit 1 with
  empty stdout. The root now records the failure (`adapters/cli/config_load.py`): `config`,
  `index`, `search`, `serve`, `send-email` and `send-notification` refuse with exit 78 (was 1) and
  one line naming the file, while `info`, `config-generate-examples` and help run (exit 0, was 1),
  and `config-deploy` runs too: it reads nothing from that configuration, so
  `config-deploy --force` replaces the broken file. `--traceback` prints the
  loader's chained traceback before the line. Any other exception from the loader is a bug and
  propagates as one. `config --profile X` reloads with the root's `--env-file` instead of
  searching for another `.env`, and a profile file that does not load exits 78 there too.
- **Command-line mistakes are usage errors, checked before the configuration loads (exit code
  change).** A malformed `--set` is refused for every command whether or not the file loads. Two
  `--set` values that give one key a value and put a key under it (`--set a.b=1 --set a.b.c=2`)
  escaped as a `TypeError` in one order and silently dropped the earlier value in the other; both
  orders now exit 2 naming the pair. An invalid `--profile` name such as `../x` now exits 2 for
  every command: the root's `--profile` and `config --profile` exited 22, and
  `config-deploy --profile` exited 1 ("Failed to deploy configuration"). A script that tells a bad
  profile name apart by exit code 22 has to test for 2.
- **No more `SystemExit: N` on stderr.** The email, config, `index`, `search` and `serve` commands
  raised a bare `SystemExit`, which `main()`'s catch-all branch printed as `SystemExit: 78` after the
  real error message. They now exit through click's context (`ctx.exit`), and `main()` returns the
  exit code rich_click's `main()` hands back instead of discarding it and returning 0. `typed_click`
  gains a typed `get_current_context` wrapper for the helpers that have no `ctx` parameter. The exit
  codes themselves are unchanged.
- **A failed send is reported once.** click's `Exit` subclasses `RuntimeError`, so the send-result
  exit raised inside the delivery `try` was caught again by the `DeliveryError`/`RuntimeError`
  branch, adding "SMTP delivery failed" / "Failed to send email" to the correct "sending failed".
  The send result is handled in the `try`'s `else`, and `config-deploy` re-raises an `Exit` before
  its catch-all.
- **`build_testing()` can run a command.** The in-memory logging initializer was a no-op while every
  command binds job context onto the lib_log_rich runtime, so any command under the testing
  composition raised `RuntimeError('lib_log_rich.init() must be called before using the logging
  API')`. It now starts a quiet runtime (no journald, event log, Graylog or queue; console at ERROR;
  no `.env` loading), and the conftest fixtures that build services use it too, so a `logger.error`
  line now precedes a command's `Error:` line on CliRunner's stderr. That runtime starts with every
  `LOG_*` variable hidden and puts each back afterwards: a `LOG_CONSOLE_LEVEL=bogus` in the shell
  running the tests made every command under `build_testing()` fail with `ValueError: Unknown log
  level: 'bogus'`, and a valid one changed the quiet runtime. Production logging is unaffected.
- **Tests no longer pass or fail by order or by machine.** The autouse fixture that shut the
  lib_log_rich runtime down after each test now also restores the root logger's handlers, level and
  propagate flag, which production `init_logging` changes and `runtime.shutdown()` does not undo. A
  second autouse fixture resets rich-click's import-time colour and width globals, so CI (where
  GitHub sets `GITHUB_ACTIONS`) and a 79-column Windows runner render the same plain 120-column
  output as a developer machine.
- **The embedder-truncation audit runs on model2vec 0.10.** model2vec 0.10 dropped
  `StaticModel.tokenize`'s `max_length` and moved both of `encode`'s cuts (the character
  pre-cut and the token truncation) into its batching step, so
  `scripts/audit_embedder_truncation.py` failed type-checking and would have refused every
  model2vec model at run time. It now tokenizes through that batching step and reads the cap
  from the model's own `max_length`; a test on a real in-memory model pins the character cut.
  The `model2vec` floor is now 0.10.0.
- **`email.smtp_hosts` and `email.recipients` set to nothing mean not configured.** A bare YAML
  key or an environment `null` was refused as "Input should be a valid list" for both settings.
  Both now read `None` as an empty list.
- **A symlinked document keeps its own name.** Source URIs followed symlinks, so a document
  linked to an extension-less file (a HuggingFace cache, git-annex, nix) was uploaded to the
  HTTP extractors under the target's name as `application/octet-stream` - Xberg answered 500
  `UnsupportedFormatError` - and search hits showed the target path. A source URI is now the
  absolute, lexically normalised path it was discovered under. Two links to one file are now
  two sources, and a dataset whose configured root runs through a symlinked directory
  reindexes once under the new URIs.

### Security
- An email attachment allow or block list given in a form it cannot be read in is now refused instead of silently ignored. A comma-separated value (`.pdf,.txt` in an environment variable or `.env`) arrives as one string, and the validators turned any value that was not a list into "not configured", so btx_lib_mail's default lists applied in place of the configured ones. A list or tuple is read as its items (an empty one still means not configured), a set is kept as given, an empty value means not configured, and anything else fails validation naming the setting and the accepted form (a JSON array such as `[".pdf", ".txt"]`). `smtp_hosts` and `recipients` no longer turn a tuple or another non-list value into an empty list. The shipped `50-mail.toml` and `.env.example` documented the comma form for all six list settings; they now show the JSON-array and per-index forms.

### Changed
- **Requires python-dotenv** (already installed through lib_log_rich): the logging setup reads the
  `LOG_*` lines of a `.env` itself. **`InitLogging` takes `dotenv_path`**: the port is
  `init_logging(config, *, dotenv_path=None)`, and a custom logging double needs the keyword.
- **`click` and `rich` are declared dependencies, and `pyarrow` is part of the `lance` extra.**
  The package imports `click` (`adapters/cli/main.py`, `commands/config.py`) and `rich`
  (`adapters/config/display.py`) directly but only had them through rich-click, and the lance
  adapter imports `pyarrow`, which only came through lancedb. A new test fails when a module the
  package imports at run time is in no dependency list or user extra, or when an import that runs
  on module import is not a core dependency.
- **Requires `btx_lib_mail>=4.0.0`.** What a user of `send-email` and `send-notification` can
  see: the default blocked attachment extensions are the POSIX and the Windows lists on every
  platform, so a Linux or macOS sender now refuses `.exe`, `.bat`, `.ps1`, `.dll`, `.lnk` and the
  rest of the Windows list (and a Windows sender `.sh`, `.py`, ...), with exit 1 like any other
  refused attachment; one send accepts at most 1000 recipients and 100 attachments, more are
  refused before any delivery (exit 22); a sender or recipient longer than RFC 5321 allows, and a
  subject with a control character or over 4096 characters, are refused. A delivery failure is
  still reported as `DeliveryError` (exit 69). The shipped `50-mail.toml` now lists the two
  default extension lists as the library defines them (it named a macOS list and Office macro
  extensions the library never blocked). See the btx_lib_mail 4.0.0 changelog for the full list.
- **The deploy port (`DeployConfiguration`) takes `set_permissions: bool | None` and
  `permission_overrides`.** None, the new default, lets lib_layered_config follow the configured
  `enabled`; a custom deploy double needs both parameters.
- **BREAKING (behaviour, not API): `ann_recall = "balanced"` no longer means "lancedb's driver default".** It now resolves to `nprobes=10, refine_factor=5` for lancedb only; pgvector and mariadb are untouched. A recall/latency sweep (`scripts/score_ann_frontier.py`) measured lancedb's own default keeping just 0.567 of the exact top-10 and giving up 0.0474 nDCG@10, which was the entire "approximation costs up to 5.6 percent" figure this project published. The new setting measures 0.986 recall and nDCG 0.8412 - 0.4 percent under an exact scan - for about 9 percent more search latency. Set `ann_recall = "fast"` to get the previous behaviour back. Only affects lancedb deployments that resolve ANN params through config.
- Every MCP tool schema now carries one-line, plain-text descriptions. Pydantic had copied each output model's developer docstring, and the `StoreBackend` / `Partition` enums' multi-paragraph docstrings, into the published `outputSchema` (RST markup and deployment advice included). Each model and enum field now states its own wire description; the enum fields still publish their allowed values. Validation is unchanged.

### Added
- `nest_overrides` in `adapters/config/overrides.py` (exported): the `--set` nesting that refuses
  a conflicting pair, shared by the root's command-line check and `apply_overrides`.
- `semdex serve`: an MCP server (a delivery-surface adapter, sibling of the CLI) exposing the search and reindex use cases over stdio (local) and Streamable HTTP (network). Tools: `list_datasets`, `search`, `reindex` (`reindex` refused for a `read_only` dataset). Install with the new `semdex[mcp-server]` extra (`fastmcp`).
- `[mcp]` config section (transport / host / port / auth) with a fully documented `defaultconfig.d/17-mcp.toml`. Auth modes: `none`, `bearer`, `oauth`; every secret loads from an environment variable named in config, never inline. OAuth proxy mode (the interactive-flow fields set) requires `[mcp].oauth_base_url`; without it the server now refuses to start with a clear `ConfigurationError` instead of failing later inside fastmcp. `[mcp].oauth_revocation_url` forwards an upstream token-revocation endpoint to the proxy when set.
- An unexpected exception raised inside an MCP tool (a bug, not a deliberately raised error) is masked from clients by default: `[mcp].mask_error_details` (default `true`) reaches FastMCP's own masking, so only a message we chose to send - never internal text or a traceback - ever crosses the wire. Set it `false` to see the original message while developing.
- Every MCP tool now returns its typed Pydantic model (`DatasetInfo`, `SearchHit`, `FusedSearchHit`, `ReindexResult`, `RememberResult`, `ForgetResult`) instead of an untyped dict, so FastMCP publishes a real `outputSchema` per tool rather than an open `{"type": "object"}` bag.
- `build_dataset_services` in composition: wires one dataset's store / embedding / connector / extractor / chunker, reusing the CLI index factories, from a `[[dataset]]` binding.
- ADR 0002 recording the MCP-server-as-adapter, secrets-from-env, and fastmcp-vs-mcp-client stances; `docs/COMPONENT_SETUP.md` gains an "MCP server (serve)" section.
- Fan-out search: `search_datasets(query, k, datasets)` MCP tool searches one, several, or all datasets (omit or pass `["all"]`) and fuses the per-dataset rankings by Reciprocal Rank Fusion (`[mcp].rrf_k`, default 60), embedding the query once per distinct model and never comparing raw cross-model similarity scores. Adds the `search_across_datasets` use case, `reciprocal_rank_fusion`, `FusedHit`, and `DatasetServices.model_key`.
- Writable knowledge datasets: a `[[dataset]]` with `writable = true` (no `sources`, not `read_only`) is a database-backed store an MCP client writes to via the new `remember(dataset, text, title, entry_id)` and `forget(dataset, entry_id)` tools; content is keyed by `semdex://<dataset>/<id>`, `reindex` refuses writable datasets, and `list_datasets` reports `writable`. Source-mirrored datasets stay read-only over their sources. Adds the `remember` / `forget` use cases (reuse `index_sources`).

## [1.5.4] 2026-06-14

### Changed
- Added a `typed_click.py` facade wrapping rich-click's `option` / `version_option` decorators behind explicit, fully-known signatures, keeping the CLI strict-clean under pyright 1.1.410 (`reportUnknownMemberType`) without disabling the rule (ignore isolated to the facade).
- Bumped internal dependency floors: `lib_cli_exit_tools>=2.3.2`, `lib_log_rich>=6.3.5`, `lib_layered_config>=5.5.2`, `btx_lib_mail>=1.3.2`.

## [1.5.3] - 2026-03-30

### Changed
- Bumped Codecov GitHub Action to V6
- Updated CVE exclusion list: removed stale entries, added inline documentation for remaining exclusions
- pip-audit set to warning-only to reduce CI noise

### Fixed
- Email transport: minor improvements to SMTP handling

## [1.5.2] - 2026-03-05

### Fixed
- Re-release as 1.5.2: PyPI rejected 1.5.1 due to filename reuse after prior upload deletion

### Changed
- `get_permission_defaults()` now returns a `PermissionDefaults` Pydantic model instead of a raw dict, with typed field access and `dir_mode_for()`/`file_mode_for()` methods
- `EmailSpy` now stores captured calls as `CapturedEmail` and `CapturedNotification` frozen dataclasses instead of `list[dict[str, Any]]`

### Added
- Tests for `--env-file` CLI option (argument passing, validation, end-to-end integration)

## [1.5.1] - 2026-03-05 [YANKED]

### Changed
- `get_permission_defaults()` now returns a `PermissionDefaults` Pydantic model instead of a raw dict, with typed field access and `dir_mode_for()`/`file_mode_for()` methods
- `EmailSpy` now stores captured calls as `CapturedEmail` and `CapturedNotification` frozen dataclasses instead of `list[dict[str, Any]]`

### Added
- Tests for `--env-file` CLI option (argument passing, validation, end-to-end integration)

## [1.5.0] - 2026-03-02

### Added
- `--env-file PATH` global CLI option to load an explicit `.env` file instead of searching upward from the working directory
- `dotenv_path` parameter added to `GetConfig` protocol, config loader, and in-memory adapter
- `__all__` to `__init__conf__.py` listing all public symbols
- `@pytest.mark.integration` marker on email integration tests

### Fixed
- Subprocess tests (`test_module_entry_subprocess_help`, `test_module_entry_subprocess_version`) now pass without editable install by setting `PYTHONPATH` to `src/`
- pip-audit CVE-2025-8869 ignore (pip is environment-level, not a project dependency)

### Changed
- CI workflow: dynamic Python version matrix extracted from `pyproject.toml` classifiers
- CI workflow: ruff cache restricted to GitHub-hosted runners
- CI workflow: bandit reads configuration from `pyproject.toml` (`-c pyproject.toml`)
- CI workflow: Codecov upload uses dynamically resolved latest Python version
- Release workflow: `actions/upload-artifact` upgraded to v7
- Added `[tool.hatch.metadata]` and `[tool.bashate]` sections to `pyproject.toml`
- Documentation updates: CONFIG.md, README.md, DEVELOPMENT.md for `--env-file` option

## [1.4.1] - 2026-02-13

### Fixed
- `reset_git_history.sh`: fixed shellcheck SC1083 warning by quoting `HEAD^{tree}` and normalized indentation to 4 spaces (shfmt)

### Changed
- Updated Makefile from v2.2.1 to v2.3.3
- Bumped dependency minimums: `lib_cli_exit_tools` >=2.3.0, `lib_log_rich` >=6.3.3, `lib_layered_config` >=5.4.1
- Added CVE ignore entries for CVE-2026-26007 and CVE-2026-25990
- Updated CI/CD workflows and added Bash 4+ requirement for macOS

## [1.4.0] - 2026-02-13

### Changed
- **Build automation**: replaced `scripts/` directory with `bmk`-based Makefile (`uvx bmk@latest`); all build, test, bump, push, and release tasks now delegated to `bmk`
- Makefile updated to v2.2.1 with alias targets, trailing argument forwarding, and new commands (config, email, info, logdemo)

## [1.3.1] - 2026-02-13

### Fixed
- `tests/test_metadata_sync.py`: replaced `importlib.metadata` lookups with direct `pyproject.toml` reads - tests no longer fail when the package is not installed in the test environment (uvx)
- Makefile `dev` target now correctly installs dev extras (`uv pip install -e ".[dev]"`)

### Changed
- CLAUDE.md: updated project structure trees to reflect actual codebase (added `entry.py`, `domain/errors.py`, `adapters/memory/`, `adapters/config/permissions.py`, `adapters/email/validation.py`; removed deleted `traceback.py`)
- CLAUDE.md: rewrote Make targets table to match new `bmk`-based Makefile (added aliases, new targets; removed obsolete `menu`, `test-slow`)
- CLAUDE.md: corrected versioning documentation - runtime metadata is served from `__init__conf__.py` constants, not `importlib.metadata`
- CLAUDE.md: replaced stale `scripts/` instrumentation section with `bmk` delegation note
- CLAUDE.md: updated `make test-slow` references to `make testintegration`

## [1.3.0] - 2026-02-01

### Added
- **File permission options for `config-deploy`**: `--permissions/--no-permissions`, `--dir-mode`, `--file-mode`
- **Configurable permission defaults** in `[lib_layered_config.default_permissions]` (app/host: 755/644, user: 700/600)
- **Octal string support** in config files (`"0o755"`, `"755"`, or decimal `493`)

### Changed
- `deploy_configuration()` accepts `set_permissions`, `dir_mode`, `file_mode` parameters
- CONFIG.md: comprehensive CLI options reference, `sudo -u` deployment examples

## [1.2.1] - 2026-02-01

### Changed
- **Profile validation** now delegates to `lib_layered_config.validate_profile_name()` with comprehensive security checks:
  - Maximum length enforcement (64 characters)
  - Empty string rejection
  - Windows reserved name rejection (CON, PRN, AUX, NUL, COM1-9, LPT1-9)
  - Leading character validation (must start with alphanumeric)
  - Path traversal prevention (/, \, ..)
- `validate_profile()` now accepts optional `max_length` parameter for customization

### Added
- `40-layered-config.toml` in `defaultconfig.d/` documenting lib_layered_config integration settings
- Profile validation tests for length limits, empty strings, Windows reserved names, and leading character rules
- Profile name requirements documentation in CONFIG.md and README.md

### Removed
- Custom `_PROFILE_PATTERN` regex - replaced by lib_layered_config's built-in validation

## [1.2.0] - 2026-01-30

### Added
- **Attachment security settings** for email configuration (`[email.attachments]` section in `50-mail.toml`)
  - `allowed_extensions` / `blocked_extensions` - whitelist/blacklist file extensions
  - `allowed_directories` / `blocked_directories` - whitelist/blacklist attachment source directories
  - `max_size_bytes` - maximum attachment file size (default 25 MiB, 0 to disable)
  - `allow_symlinks` - whether symbolic links are permitted (default false)
  - `raise_on_security_violation` - raise or skip on violations (default true)
- New `EmailConfig` fields for attachment security with Pydantic validators
- `load_email_config_from_dict()` now flattens nested `[email.attachments]` section

### Changed
- Bumped `btx_lib_mail` dependency from `>=1.2.1` to `>=1.3.0` for attachment security features

## [1.1.2] - 2026-01-28

### Fixed
- Coverage SQLite "database is locked" errors on Python 3.14 free-threaded builds and network mounts (SMB/NFS)
- Removed bogus `COVERAGE_NO_SQL=1` environment variable from `scripts/test.py` (not a real coverage.py setting)
- CI workflow now sets `COVERAGE_FILE` to `runner.temp` so coverage always writes to local disk
- **Import-linter was a silent no-op** in `make test` / `make push` - `python -m importlinter.cli lint` silently exits 0 without checking; replaced with `lint-imports` (the working console entry point)
- CI/local parameter mismatches: ruff now targets `.` (not hardcoded `src tests notebooks`), pytest uses `python -m pytest` with `--cov=src/$PACKAGE_MODULE`, `--cov-fail-under=90`, and `-vv` matching local runs
- `scripts/test.py` bandit source path now reads `src-path` from `[tool.scripts.test]` instead of hardcoding `Path("src")`
- `scripts/test.py` module-level `_default_env` now rebuilt with configured `src_path` before running checks
- `run_slow_tests()` now reads pytest verbosity from `[tool.scripts.test].pytest-verbosity` instead of hardcoding `"-vv"`

### Changed
- **pyproject.toml as single source of truth**: CI workflow extracts all tool configuration (src-path, pytest-verbosity, coverage-report-file, fail_under, bandit skips) from `pyproject.toml` via metadata step - workflow is portable across projects without editing
- `scripts/test.py` removed module-level `PACKAGE_SRC` constant; bandit source path computed from `config.src_path` inside the functions that need it
- `make push` now accepts an unquoted message as trailing words (e.g. `make push fix typo in readme`); commit message format is `<version> - <message>`, defaulting to `<version> - chores` when no message is given
- Removed interactive commit-message prompt from `push.py` - message is either provided via CLI args / `COMMIT_MESSAGE` env var, or defaults to `"chores"`

### Added
- `pytest_configure` hook in `tests/conftest.py` that redirects coverage data to `tempfile.gettempdir()` and purges stale SQLite journal files before each run

## [1.1.1] - 2026-01-28

### Fixed
- CLAUDE.md: replaced stale package name `bitranox_template_cli_app_config_log_mail` with `semdex` throughout
- Brittle SMTP mock assertions in `test_cli.py` now use structured `call_args` attributes instead of `str()` coercion
- Stale docstring in `__init__conf__.py` claiming "adapters/platform layer" - corrected to "Package-level metadata module"
- Weak OR assertion in `test_cli.py` for SMTP host display - replaced with two independent assertions
- Removed stale `# type: ignore[reportUnknownVariableType]` from `sender.py` (`btx_lib_mail.ConfMail` now has proper type annotations)
- Late function-body imports in `adapters/cli/commands/config.py` moved to module-level for consistency

### Removed
- Dead code: unused `_format_value()` and `_format_source()` wrappers in `adapters/config/display.py`

### Added
- `__all__` to `__init__conf__.py` listing all public symbols
- `tests/test_enums.py` with parametrized tests for `OutputFormat` and `DeployTarget`
- Expanded `tests/test_behaviors.py` with return type, constant value, and constant-usage checks
- Python 3.14 classifier in `pyproject.toml`
- Codecov upload step in CI workflow (gated to `ubuntu-latest` + `3.13`)
- Edge-case tests for `parse_override`: bare `=value`, bare `=`, and CLI `--set ""` empty string
- Duplication-tracking comments for CI metadata extraction scripts

### Changed
- `tests/test_display.py` rewritten to test `_format_raw_value` and `_format_source_line` directly (replacing dead wrapper tests)

## [1.1.0] - 2026-01-27

### Changed
- Replaced `MockConfig` in-memory adapter with real `Config` objects in all tests (`config_factory` / `inject_config` fixtures)
- Replaced `MagicMock` Config objects in CLI email tests with real `Config` instances
- Unified test names to BDD-style `test_when_<condition>_<behavior>` pattern in `test_cli.py`
- Email integration tests now load configuration via `lib_layered_config` instead of dedicated `TEST_SMTP_SERVER` / `TEST_EMAIL_ADDRESS` environment variables

### Added
- Cache effectiveness tests for `get_config()` and `get_default_config_path()` LRU caches (`tests/test_cache_effectiveness.py`)
- Callable Protocol definitions in `application/ports.py` for all adapter functions, with static conformance assertions and `tests/test_ports.py`
- `ExitCode` IntEnum (`adapters/cli/exit_codes.py`) with POSIX-conventional exit codes for all CLI error paths
- `logdemo` and `config-generate-examples` CLI commands
- `--set SECTION.KEY=VALUE` repeatable CLI option for runtime configuration overrides (`adapters.config.overrides` module)
- Unit tests for config overrides and display module (sensitive key matching, redaction, nested rendering)

### Removed
- Dead code: `raise_intentional_failure()`, `noop_main()`, `cli_main()`, duplicate `cli_session` orchestration, catch-log-reraise in `send_email()`
- Replaced dead `ConfigPort`/`EmailPort` protocol classes with callable Protocol definitions

### Fixed
- POSIX-conventional exit codes across all CLI error paths (replacing hardcoded `SystemExit(1)`)
- Sensitive value redaction: word-boundary matching to avoid false positives, nested dict/list redaction, TOML sub-section rendering
- Email validation: reject bogus addresses (`@`, `user@`, `@domain`); IPv6 SMTP host support; credential construction
- Profile name validation against path traversal
- Security: list-based subprocess calls in scripts, sensitive env-var redaction in test output, stale CVE exclusion cleanup
- Documentation: wrong project name references, truncated CLI command names, stale import paths, wrong layer descriptions
- CI: `actions/download-artifact` version mismatch, stale `codecov.yml` ignore patterns
- Unified `__main__.py` and `adapters/cli/main.py` error handling via delegation

### Changed
- Precompile all regex patterns in `scripts/` as module-level constants for consistent compilation
- **LIBRARIES**: Replace custom redaction/validation with `lib_layered_config` redaction API and `btx_lib_mail` validators; bump both libraries
- **LIBRARIES**: Replace stdlib `json` with `orjson`; replace `urllib` with `httpx` in scripts
- **ARCHITECTURE**: Purified domain layer - `emit_greeting()` renamed to `build_greeting()` (returns `str`, no I/O); decoupled `display.py` from Click
- **DATA ARCHITECTURE**: Consolidated `EmailConfig` into single Pydantic `BaseModel` (eliminated dataclass conversion chain)

## [1.0.0] - 2026-01-15

### Added
- Slow integration test infrastructure (`make test-slow`, `@pytest.mark.slow` marker)
- `pydantic>=2.0.0` dependency for boundary validation
- `CLIContext` dataclass replacing untyped `ctx.obj` dict
- Pydantic models: `EmailSectionModel`, `LoggingConfigModel`
- `application/ports.py` with Protocol definitions; `composition/__init__.py` wiring layer

### Changed
- **BREAKING**: Full Clean Architecture refactoring into explicit layer directories (`domain/`, `application/`, `adapters/`, `composition/`)
- CLI restructured from monolithic `cli.py` into focused `cli/` package with single-responsibility modules
- Type hints modernized to Python 3.10+ style
- Removed backward compatibility re-exports; tests import from canonical module paths
- `import-linter` contracts enforce layer dependency direction
- `make test` excludes slow tests by default

## [0.2.5] - 2026-01-01

### Changed
- Bumped `lib_log_rich` to >=6.1.0 and `lib_layered_config` to >=5.2.0

## [0.2.4] - 2025-12-27

### Fixed
- Intermittent test failures on Windows when parsing JSON config output (switched to `result.stdout`)

## [0.2.3] - 2025-12-15

### Changed
- Lowered minimum Python version from 3.13 to 3.10; expanded CI matrix accordingly

## [0.2.2] - 2025-12-15

### Added
- Global `--profile` option for profile-specific configuration across all commands

### Changed
- **BREAKING**: Configuration loaded once in root CLI command and stored in Click context for subcommands
- Subcommand `--profile` options act as overrides that reload config when specified

## [0.2.0] - 2025-12-07

### Added
- `--profile` option for `config` and `config-deploy` commands
- `OutputFormat` and `DeployTarget` enums for type-safe CLI options
- LRU caching for `get_config()` (maxsize=4) and `get_default_config_path()`

### Fixed
- UTF-8 encoding issues in subprocess calls across different locales

## [0.1.0] - 2025-12-07

### Added
- Email sending via `btx-lib-mail` integration: `send-email` and `send-notification` CLI commands
- Email configuration support with `EmailConfig` dataclass and validation
- Real SMTP integration tests using `.env` configuration

## [0.0.1] - 2025-11-11
- Bootstrap
