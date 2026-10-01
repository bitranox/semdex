# Design: MCP server - multi-user, multi-dataset, source lifecycle

**Status:** Approved design (implementation not yet started)

## Context

semdex today is a short-lived CLI. Its package metadata already calls it a "file-watching semantic-search
MCP server" (`pyproject.toml`), but no serve/watch/MCP-server surface exists yet: the only console script
is `semdex` (`semdex.entry:main`), the production watcher is the `InMemoryWatcher` stub
(`composition/__init__.py`), and there is no `Dataset`/user/tenant concept anywhere - the only corpus
unit is `Collection(name, model_id, dim)`, selected by a bare name string.

This design prepares the MCP-server phase. It answers three questions:

1. multi-user / multi-dataset - how to structure it,
2. what happens to indexed content when files (later: emails) are renamed / moved / deleted,
3. can a search result point back to the file (later: message) location.

It is a design/architecture doc, not yet a file-by-file implementation plan.

## Decisions

1. **Per-user MCP process.** Each user runs their own semdex MCP server. Isolation is the process/OS
   boundary, so semdex builds NO in-app auth/ACL/tenant layer.
2. **Storage location is orthogonal to private/shared.** A dataset can be embedded (local
   json/sqlite/lance) OR on a DB server (pgvector/mariadb), and independently private or shared.
3. **The database enforces auth for DB-backed datasets.** Each user's server connects with that user's
   own DB credentials. Private DB dataset = a collection in a per-user database (`semdex_<user>`) only
   that role can reach; shared DB dataset = a collection in a shared database the DBA grants to a group.
   Read-only vs read-write IS the grant (`SELECT` vs `SELECT`+write). Per-user *database* (not schema):
   MariaDB has no separate schema concept, so per-database is the only choice uniform across both
   backends, and both adapters already take a full DSN naming the database - zero adapter change.
4. **Secrets: keyfile/env recommended, inline possible.** The recommended pattern is a DSN with the
   password injected at runtime from the user's keyfile/env. A password inline in a user-local
   (untracked) config toml is also permitted, just discouraged. The hard line: never a secret in a
   TRACKED/committed file (repo defaultconfig, VCS-tracked user config, commit, or script) - that is
   history and needs rotation.
5. **One universal source-identity contract, pluggable connectors.** Every indexed item is modeled as
   three facets; what fills them is connector-specific:

| Facet              | Filesystem connector | IMAP connector (later)   | Used for                     |
|--------------------|----------------------|--------------------------|------------------------------|
| Content identity   | sha256 content_hash  | `Message-ID` header      | dedup + move detection       |
| Location / address | `file://<path>`      | `imap://<mailbox>/<uid>` | what a Hit points to         |
| Version token      | mtime / hash         | flags (body immutable)   | decide "changed -> re-index" |

   Build the universal contract + filesystem connector now; IMAP is later just a second connector, with
   NO re-keying.
6. **Lifecycle = reconcile sweep** on `(id, version)`, connector-agnostic: id gone from source -> delete
   (prune chunks); same content identity at a new location -> move (re-point, no re-embed); same
   location, new version -> re-index. Live layers (watchdog for files, IMAP IDLE for email) are optional
   and land later ON TOP of the sweep, which is also their startup safety net.
7. **Search selects one-or-many datasets.** Default single-target for clean provenance; fan-out when
   asked. Fan-out embeds the query once per distinct model in the selected set, queries each store with
   its own model, and fuses results by RANK (Reciprocal Rank Fusion), never by raw cross-model cosine
   score (different models' scores are on different scales).
8. **Provenance points into the source.** A `Hit` carries the location URI plus the chunk's `ordinal`
   (already stored, currently dropped) and a char offset, so it points to path+position (or the message),
   not just the container.
9. **The server/store/search/reconcile are source-type-blind.** They handle ONLY the opaque triple
   (source-id, location URI, version). Nothing branches on file-vs-email; a Hit returns the location URI
   verbatim and the server never interprets it. ALL type-specific logic (enumerate, fetch, extract to
   text, what id/location/version are) is confined to the connector; adding email = one new connector,
   zero change to server/store/search/reconcile. Interpreting/opening a returned URI (`file://` in an
   editor, `imap://` in a mail client) is the CLIENT's job, outside semdex.
10. **Hit carries a short description for selection (the standard RAG metadata pattern).** Baseline,
    built now: the connector attaches per-source metadata to each Hit - a `title` (filename / first
    heading for files; `Subject`+`From`/`Date` for email) and a structural `breadcrumb` (heading path
    for docs). This is the near-universal convention (LangChain / LlamaIndex `Document.metadata` carried
    on every retrieved chunk; the metadata payload in every vector DB), deterministic and free, and it
    generalizes the existing free-form `label`. Premium tier, opt-in later and OFF by default (lean
    ethos, costs index-time LLM calls): an index-time `summary`/context line per chunk-or-document -
    Anthropic Contextual Retrieval (context prepended before embedding) or a LlamaIndex-style
    per-document summary index. The connector (type-aware) produces all of these; the server stays blind
    per decision 9 and just passes them through.
11. **Store partitioning is a CONFIGURABLE per-dataset knob; the hybrid below is only the DEFAULT.** A
    per-dataset `partition` setting (backed by a global `[vector_store].default_partition`) chooses how a
    dataset's collections are separated. The user can set ANY dataset either way - including `database`
    for local/private datasets if they prefer separate local databases/files over tables:
    - `table` - many collections share one store/database (a `chunks_<id>` table per collection; for
      embedded, many collections in one store file).
    - `database` - one database per dataset (DB backends) or one store file/dir per dataset (embedded).

    It is a deployment/config choice, NOT an adapter branch: the pg/mariadb adapters create their
    `collections` + `chunks_<id>` tables in whatever database the DSN names (pg does `CREATE EXTENSION
    vector` per-connect); embedded stores already isolate by `store_dir`.

    **Default policy (overridable per dataset):** private -> `table` in the user's own database (one
    connection, self-serve `CREATE TABLE`, isolated by the user boundary); shared -> `database` per
    dataset (independent per-group `GRANT`, backup/`DROP`/snapshot, contained blast-radius, no shared
    `collections` list leak).

    **The upsides/downsides MUST be documented in the config toml** (`15-vectorstore.toml`) per the
    standing rules (magic-numbers-configurable + document-every-toml-setting): `table` = fewest
    connections + simplest provisioning (`CREATE TABLE`), but only coarse DB-level grants, a shared
    `collections` list, and a shared blast-radius; `database` = per-dataset grants/backup/isolation, but
    a connection per dataset and `CREATEDB` (or admin provisioning) to create one.

## The Dataset concept (new config layer, not new domain complexity)

A **Dataset** is a named config binding:

- `name` - what an MCP tool selects.
- `store` - `embedded` (backend + local `store_dir`) OR `db` (backend + `dsn`; password from keyfile/env
  recommended, inline in untracked config allowed but discouraged).
- `collection` + `embedding` - collection name and its model (model must match the collection's
  `model_id`/`dim`).
- `sources` - the connector + roots this dataset indexes (e.g. filesystem roots; later IMAP mailbox).
- `partition` (optional) - `table` | `database`; how this dataset's collections are separated (decision
  11). Defaults from `[vector_store].default_partition` (private -> `table`, shared -> `database`).
- `read_only` (optional) - advisory UX hint so the server does not advertise an index tool for it; the DB
  grant is the real enforcer.

The MCP server loads a LIST of these. Private vs shared is purely which store/DB the binding points at
and what the DB granted - not a semdex code path. Every store backend already CAN hold many collections
at once (json/memory by name; sqlite/pg/mariadb one `chunks_<id>` table per collection; lance one table
per collection), so the store layer supports either partitioning. Per decision 11 the default deliberately
uses that many-collections-per-store capability for a user's private datasets (tables in the user's own
database), and gives each shared dataset its own database instead, so DB-native grants and backups apply
per shared dataset. The partitioning is chosen per Dataset by config - no adapter branch.

## Concrete code impact (for the later implementation plan)

- **Domain** (`domain/models.py`): generalize `SourceRef.path` to a `uri`/source-id (file path is the
  file case), keep `content_hash` as the version/identity token; add per-source `title` + `breadcrumb`
  (+ optional `summary`) for hit selection; add an offset field to `Chunk`; add
  `location`/`ordinal`/offset + `title`/`breadcrumb`/`summary` to `Hit` (today `Hit` flattens to
  `source_path`+`label`+`collection` and drops `ordinal`). Add a `MOVED` change kind or model moves as a
  content-id match.
- **Ports** (`application/ports.py`): re-key `delete_by_source(collection, path)` -> by source-id; add a
  `SourceConnector` port (`list() -> (id, location, version)*`, `fetch(id) -> content`) and a reconcile
  use case.
- **Stores** (all five adapters): key `chunks_<id>` rows by the source-id/URI instead of `path` (touches
  jsonstore, sqlitevec, pgvector, mariadb, lance) so move/delete work uniformly; the `delete_by_source`
  primitive already exists on all five, only its key generalizes.
- **Discovery -> filesystem connector**: wrap `discover_sources` (`adapters/discovery/filesystem.py`) as
  the first `SourceConnector` implementation (it already streams sha256 + mtime).
- **Reconciler**: new use case that diffs a connector's `(id, version)` list against the store's recorded
  set and issues prune/move/re-index. Reuse `store.collections()`/`count()` and the stored (currently
  unused) `content_hash`.
- **Config layer** (`adapters/config/` + `defaultconfig.d/`): a Dataset config model (name, store,
  collection, embedding, sources, `partition`, `read_only`) and the MCP dataset list; add
  `[vector_store].default_partition`. Document the `partition` and `default_partition` upsides/downsides
  fully in `15-vectorstore.toml` (decision 11) per the standing "document every toml setting" rule
  (default, effect, when to set, consequences, per-scenario recommendation).
- **New surface**: a `serve` command + MCP server (FastMCP; the `mcp` dep is present but only used as a
  client for the markitdown extractor today) exposing tools: `list_datasets`, `search(dataset|datasets,
  query, k)`, and an admin `reindex(dataset)` (offered only for writable datasets). Add store `close()`
  for the long-lived process (deferred until this lands).
- **Later, no re-keying**: IMAP connector, watchdog + IMAP IDLE live watchers, RRF fan-out.

## Build order (YAGNI - ship the file path first)

1. Universal `SourceRef`/source-id + Hit provenance (offset/ordinal) + re-key stores by source-id.
2. `SourceConnector` port + filesystem connector + reconcile use case (prune/move/re-index).
3. Dataset config layer (embedded + DB bindings; DSN password from keyfile; `partition` knob + toml docs).
4. `serve` + MCP server with `list_datasets` + single-target `search` + `reindex`; store `close()`.
5. Multi-dataset fan-out search with RRF fusion.
6. Live watchers (watchdog) and the IMAP connector + IDLE - later, drop-in.

## Verification

- Reconcile: index a temp corpus; rename a file -> chunks re-point to the new URI with NO re-embed
  (assert embedding not recomputed) and no stale old-URI rows; delete a file -> its chunks are pruned;
  edit in place -> re-indexed. The same test is reused later against the IMAP connector (Message-ID move).
- Provenance: a search hit resolves to location URI + ordinal/offset, not just the container.
- Multi-user/DB auth: a per-user DSN can only reach its own database; a read-only grant makes a write
  tool fail at the DB (exercised against a real pgvector/mariadb instance).
- Fan-out: RRF over two datasets with different embedding models returns a sensibly fused ranking; assert
  raw cross-model scores are NOT compared directly.
- `make test` green; import-linter contracts still hold (the new connector port lives in application, its
  implementations in adapters).
