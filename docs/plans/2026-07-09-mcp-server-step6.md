# MCP server - Step 6: writable knowledge datasets (remember / forget)

> **For Claude:** REQUIRED SUB-SKILL: Use bitranox:process-plan-executor (or bitranox:process-agents-subagent-driven-development) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. Also invoke bitranox:process-test-driven-development (each task RED -> GREEN) and bitranox:coding-python-clean-architecture.

**Goal:** Let an MCP client WRITE knowledge into designated writable, database-backed datasets (and delete it), while the existing source-mirrored datasets (filesystem, later IMAP) stay strictly read-only over their sources. The agent selects which allowed dataset to write into; the database GRANT on that dataset's DSN is the real enforcer.

**Architecture:** A dataset is one of two kinds, kept separate. A SOURCE dataset mirrors an external source through a connector (read-only over the source; `reconcile` owns its lifecycle). A KNOWLEDGE dataset (`writable = true`) has NO connector; the client writes its content directly, so `reconcile` must never run on it (it would prune everything). The write path is an application use case (`remember`) that reuses `index_sources` - for already-extracted text, extraction is the identity - plus `forget` (drop a uri's chunks). The MCP server (adapter) adds two tools gated to writable datasets and makes `reindex` refuse them. No new store or embedding code: a knowledge dataset is a normal collection written under synthetic `semdex://<dataset>/<id>` uris.

**Tech Stack:** pydantic config, fastmcp v3 (in-memory `fastmcp.Client` for tests), pytest, uv. No new third-party dependency.

## Handover context (read first - you may have zero prior context)

- **Worktree:** `<semdex-checkout>/.claude/worktrees/mcp-source-identity`, branch `worktree-mcp-source-identity`. Work here.
- **Why this shape** (settled with the user): semdex stays a read-only INDEX of external sources for filesystem/IMAP datasets; a separate class of DB-backed "knowledge" datasets is writable by the agent. The two cannot collide because a knowledge dataset has no connector, so the filewatcher's files / IMAP mail can never land in it. IMAP and the live file watcher are OUT of this step (IMAP is a future external connector; the watcher is a possible later step).
- **What steps 1-5 shipped that this builds on:**
  - `application/use_cases/indexing.py`: `index_sources(*, extract, chunk, embedding, store, collection, sources, max_tokens=256, compaction=None, clock=...) -> IndexReport(sources_indexed, chunks_indexed)`. Per source it does `store.ensure_collection(...)`, `store.delete_by_source(collection, uri=source.uri)` (drop-first = idempotent replace), chunk, `embed_passages`, `upsert`, and a final `store.compact`.
  - `application/ports.py`: `Extract` = callable `(SourceRef) -> ExtractedDocument`; `ChunkText` = callable `(ExtractedDocument, *, max_tokens=...) -> list[Chunk]`; `VectorStoreWriter.ensure_collection/upsert/delete_by_source(*, collection, uri)/swap/compact`; `VectorStore` combines reader+writer.
  - `domain/models.py`: `SourceRef(uri, label, content_hash, mtime)`, `ExtractedDocument(source, text)`, `Chunk(text, source, ordinal, token_count)`, `Hit(chunk_text, score, uri, ordinal, label, collection)`.
  - `adapters/config/dataset.py`: `DatasetConfig(name, backend, store_dir, dsn, collection, embedding_provider, embedding_model, embedding_endpoint, sources, partition, read_only)` with a `@model_validator(mode="after") _validate` (raises on empty name / server-backend-without-dsn). `get_datasets(config)`.
  - `composition/__init__.py`: `DatasetServices(name, backend, model_key, store, embedding, extract, chunk, connector, collection, partition, read_only)` + `build_dataset_services(dataset, *, default_partition)` (builds a `FilesystemConnector` from `dataset.sources`, an embedding, a store, a chunker; computes `model_key`).
  - `adapters/mcp/server.py`: `build_mcp_server(datasets, *, auth=None, name="semdex", rrf_k=60)`; tools `list_datasets` / `search` / `search_datasets` / `reindex`. Tools are `@mcp.tool` closures each carrying `# pyright: ignore[reportUnusedFunction]`. Errors raise `fastmcp.exceptions.ToolError` via the `_tool_error` helper. `_dataset(name)` returns the `DatasetServices` or raises a clear ToolError.
- **Layer rule (import-linter, `lint-imports`):** `adapters` may import `application` + `domain`, NEVER `composition`. `adapters/mcp/server.py` imports application use cases at runtime (allowed) and `DatasetServices` only under `TYPE_CHECKING`. Keep it that way. After each task run `lint-imports` and confirm `Contracts: 2 kept, 0 broken` (grep for `Contracts:`; do NOT trust `tail -2`).
- **Gates (run from the worktree):**
  - `uv run pytest -m "not local_only and not integration" -q -p no:cacheprovider`
  - `uv run ruff check src/ tests/` and `uv run ruff format src/ tests/`
  - `uv run pyright src/ tests/`
  - `uv run lint-imports`
  - The `VIRTUAL_ENV ... does not match` warning from uv is harmless.

## Global Constraints

- **Python 3.10+**; type-annotate everything; pyright strict must pass (0 errors).
- **Clean architecture:** the `remember`/`forget` use cases live in `application/use_cases/`; the tools live in `adapters/mcp`. Run `lint-imports` after each task, confirm `2 kept, 0 broken`.
- **TDD:** every task is RED (watch it fail) -> GREEN (minimal) -> commit.
- **The two dataset kinds never mix:** a `writable` dataset must have no `sources` (validator-enforced) and `reindex` must refuse it; a source dataset must refuse `remember`/`forget`. A knowledge dataset's content is keyed by `semdex://<dataset>/<id>`.
- **ruff `A002`:** do NOT name a function argument `id` (shadows the builtin). Use `entry_id`.
- **Config as the only source of tunables**; **document every new TOML key** comprehensively (default, effect, when to set, consequences, recommendation) in `16-datasets.toml`, matching the existing style.
- **ASCII only** in all files (a tell-sweep hook rejects em/en-dashes, curly quotes, ellipsis).
- **No Claude/AI attribution** in commits. Commit after each task. `uv.lock` is gitignored here - never `git add uv.lock`.

---

### Task 1: `writable` dataset flag (config + validation + docs + services)

**Files:**
- Modify: `src/semdex/adapters/config/dataset.py` (add `writable`; validate)
- Modify: `src/semdex/adapters/config/defaultconfig.d/16-datasets.toml` (document `writable`)
- Modify: `src/semdex/composition/__init__.py` (add `DatasetServices.writable`; thread it)
- Modify: `tests/test_config_dataset.py` (validation tests) - if that file does not exist, create it
- Modify: `tests/test_dataset_services.py` (assert `writable` default)

**Interfaces:**
- Produces: `DatasetConfig.writable: bool` (default `False`); `DatasetServices.writable: bool`.

- [ ] **Step 1: Write failing tests.** First check the existing dataset-config test file name: `ls tests | grep dataset`. Add to the config test module (create `tests/test_config_dataset.py` if none exists):

```python
# tests/test_config_dataset.py  (add these; keep existing tests if the file exists)
from __future__ import annotations

import pytest
from pydantic import ValidationError

from semdex.adapters.config.dataset import DatasetConfig
from semdex.domain.enums import StoreBackend


@pytest.mark.os_agnostic
def test_writable_defaults_false() -> None:
    assert DatasetConfig(name="kb").writable is False


@pytest.mark.os_agnostic
def test_writable_forbids_sources() -> None:
    with pytest.raises(ValidationError):
        DatasetConfig(name="kb", writable=True, sources=("/tmp/x",))


@pytest.mark.os_agnostic
def test_writable_forbids_read_only() -> None:
    with pytest.raises(ValidationError):
        DatasetConfig(name="kb", writable=True, read_only=True)


@pytest.mark.os_agnostic
def test_writable_knowledge_dataset_ok() -> None:
    cfg = DatasetConfig(name="kb", backend=StoreBackend.JSON, writable=True)
    assert cfg.writable is True and cfg.sources == ()
```

  And in `tests/test_dataset_services.py`, inside `test_builds_embedded_dataset_services`, add:

```python
    assert services.writable is False  # source dataset by default
```

- [ ] **Step 2: Run to verify it fails.** `uv run pytest tests/test_config_dataset.py tests/test_dataset_services.py -q -p no:cacheprovider` Expected: FAIL (no `writable`).

- [ ] **Step 3: Add the field + validation** in `src/semdex/adapters/config/dataset.py`. Add the field after `read_only`:

```python
    # A writable KNOWLEDGE dataset: the client writes its content directly via the
    # remember/forget tools; it has no connector and is never reconciled. Mutually
    # exclusive with sources and read_only. The DB GRANT is the real enforcer.
    writable: bool = False
```

  and extend `_validate` (before `return self`):

```python
        if self.writable and self.sources:
            raise ValueError(f"dataset '{self.name}': a writable dataset has no sources (the client writes its content)")
        if self.writable and self.read_only:
            raise ValueError(f"dataset '{self.name}': writable and read_only are mutually exclusive")
```

- [ ] **Step 4: Document it** in `src/semdex/adapters/config/defaultconfig.d/16-datasets.toml`, in the same style as the other keys (default / effect / when to set / consequences / recommendation). Cover: `writable = true` makes the dataset a KNOWLEDGE store the MCP client writes to via `remember`/`forget`; it must have no `sources` and cannot be `read_only`; it is never reconciled/reindexed; intended on a database backend whose DSN grant scopes who may write; content is keyed by `semdex://<dataset>/<id>`. Default `false` (a normal source-mirrored dataset). ASCII only. If the file uses an `[[dataset]]` example block, show `writable = true` commented in an example knowledge dataset.

- [ ] **Step 5: Thread it into services** in `src/semdex/composition/__init__.py`: add `writable: bool` to the `DatasetServices` dataclass (after `read_only`), and add `writable=dataset.writable,` to the `DatasetServices(...)` constructor call in `build_dataset_services`.

- [ ] **Step 6: Run tests.** `uv run pytest tests/test_config_dataset.py tests/test_dataset_services.py -q -p no:cacheprovider` Expected: PASS.

- [ ] **Step 7: Gates + commit.** ruff/pyright/lint-imports (`2 kept, 0 broken`). `git add -A && git commit -m "feat(config): writable knowledge-dataset flag (+ validation, docs, services)"`

---

### Task 2: remember / forget write use cases

**Files:**
- Create: `src/semdex/application/use_cases/writing.py`
- Modify: `src/semdex/application/use_cases/__init__.py` (export `forget`, `remember`)
- Create: `tests/test_writing.py`

**Interfaces:**
- Consumes: `index_sources` (Task reuse), `SourceRef`, `ExtractedDocument`, ports `ChunkText`/`EmbeddingProvider`/`VectorStoreWriter`.
- Produces: `remember(*, chunk, embedding, store, collection, uri, text, label="", max_tokens=256) -> IndexReport`; `forget(*, store, collection, uri) -> None`.

- [ ] **Step 1: Write failing tests.**

```python
# tests/test_writing.py
from __future__ import annotations

from pathlib import Path

import pytest

from semdex.adapters.config.dataset import DatasetConfig
from semdex.application.use_cases.searching import search
from semdex.application.use_cases.writing import forget, remember
from semdex.composition import build_dataset_services
from semdex.domain.enums import EmbeddingBackend, Partition, StoreBackend


def _knowledge(tmp_path: Path):
    dataset = DatasetConfig(
        name="kb",
        backend=StoreBackend.JSON,
        store_dir=str(tmp_path / "kb"),
        collection="kb",
        embedding_provider=EmbeddingBackend.PLACEHOLDER,
        writable=True,
    )
    return build_dataset_services(dataset, default_partition=Partition.TABLE)


@pytest.mark.os_agnostic
def test_remember_then_search_then_forget(tmp_path: Path) -> None:
    svc = _knowledge(tmp_path)
    report = remember(
        chunk=svc.chunk,
        embedding=svc.embedding,
        store=svc.store,
        collection=svc.collection,
        uri="semdex://kb/n1",
        text="banana cherry apple",
        label="fruit",
    )
    assert report.chunks_indexed >= 1
    hits = search(embedding=svc.embedding, store=svc.store, collection=svc.collection, query="banana apple", k=5)
    assert hits and hits[0].uri == "semdex://kb/n1" and hits[0].label == "fruit"
    forget(store=svc.store, collection=svc.collection, uri="semdex://kb/n1")
    assert search(embedding=svc.embedding, store=svc.store, collection=svc.collection, query="banana apple", k=5) == []


@pytest.mark.os_agnostic
def test_remember_same_uri_replaces(tmp_path: Path) -> None:
    svc = _knowledge(tmp_path)
    remember(
        chunk=svc.chunk,
        embedding=svc.embedding,
        store=svc.store,
        collection=svc.collection,
        uri="semdex://kb/n1",
        text="alpha",
        label="",
    )
    remember(
        chunk=svc.chunk,
        embedding=svc.embedding,
        store=svc.store,
        collection=svc.collection,
        uri="semdex://kb/n1",
        text="omega",
        label="",
    )
    # drop-first replace: only the second write's chunk(s) remain under that uri
    assert svc.store.count(collection="kb") == 1
```

- [ ] **Step 2: Run to verify it fails.** `uv run pytest tests/test_writing.py -q -p no:cacheprovider` Expected: ImportError (no `writing`).

- [ ] **Step 3: Implement** `src/semdex/application/use_cases/writing.py`:

```python
"""Write use cases: deposit (remember) and delete (forget) client-written knowledge.

A writable KNOWLEDGE dataset has no connector; its content is written directly by
the MCP client, not mirrored from a source. ``remember`` reuses ``index_sources``
(for already-provided text, extraction is the identity), so it inherits the
drop-first idempotency and the ANN compaction. ``forget`` drops a uri's chunks.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from ...domain.models import ExtractedDocument, SourceRef
from .indexing import index_sources

if TYPE_CHECKING:
    from ...domain.models import Chunk
    from ..ports import ChunkText, EmbeddingProvider, VectorStoreWriter
    from .indexing import IndexReport

_DEFAULT_MAX_TOKENS = 256


def remember(
    *,
    chunk: ChunkText,
    embedding: EmbeddingProvider,
    store: VectorStoreWriter,
    collection: str,
    uri: str,
    text: str,
    label: str = "",
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> IndexReport:
    """Write ``text`` into a knowledge ``collection`` under ``uri``.

    Replaces any prior chunks stored under the same ``uri`` (drop-first), then
    chunks and embeds ``text`` with the dataset's own chunker and model. Returns
    the index report (chunk count).
    """
    ref = SourceRef(uri=uri, label=label, content_hash=_sha256(text), mtime=0.0)

    def _extract(source: SourceRef) -> ExtractedDocument:
        return ExtractedDocument(source=source, text=text)

    return index_sources(
        extract=_extract,
        chunk=chunk,
        embedding=embedding,
        store=store,
        collection=collection,
        sources=[ref],
        max_tokens=max_tokens,
    )


def forget(*, store: VectorStoreWriter, collection: str, uri: str) -> None:
    """Delete every chunk stored under ``uri`` (idempotent) and fold the index."""
    store.delete_by_source(collection=collection, uri=uri)
    store.compact(collection=collection)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = ["forget", "remember"]
```

  Note: `Chunk` is imported under TYPE_CHECKING only if referenced; if pyright reports it unused, drop it. `index_sources` accepts `store: VectorStoreWriter`, so passing the writer slice is correct.

- [ ] **Step 4: Export** from `src/semdex/application/use_cases/__init__.py`: add `from .writing import forget, remember` and add `"forget"`, `"remember"` to `__all__` (and mention `.writing` in the module docstring's Contents list).

- [ ] **Step 5: Run tests.** `uv run pytest tests/test_writing.py -q -p no:cacheprovider` Expected: PASS (2 tests).

- [ ] **Step 6: Gates + commit.** ruff/pyright/lint-imports. `git add -A && git commit -m "feat(write): remember/forget knowledge use cases (reuse index_sources)"`

---

### Task 3: MCP remember / forget tools + reindex refuses writable

**Files:**
- Modify: `src/semdex/adapters/mcp/server.py` (import writing use cases; add `remember`/`forget` tools; `reindex` refuses writable; `list_datasets` exposes `writable`)
- Modify: `tests/test_mcp_server.py` (write/forget + rejection tests)

**Interfaces:**
- Consumes: `remember`/`forget` (Task 2), `DatasetServices.writable` (Task 1).
- Produces: MCP tools `remember(dataset, text, title="", entry_id=None) -> dict` and `forget(dataset, entry_id) -> dict`; `list_datasets` rows gain `"writable"`.

- [ ] **Step 1: Write failing tests** in `tests/test_mcp_server.py` (append):

```python
def _writable_services(tmp_path: Path) -> DatasetServices:
    dataset = DatasetConfig(
        name="kb",
        backend=StoreBackend.JSON,
        store_dir=str(tmp_path / "kb"),
        collection="kb",
        embedding_provider=EmbeddingBackend.PLACEHOLDER,
        writable=True,
    )
    return build_dataset_services(dataset, default_partition=Partition.TABLE)


@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_remember_then_search_then_forget_via_mcp(tmp_path: Path) -> None:
    server = build_mcp_server([_writable_services(tmp_path)])
    async with Client(server) as client:
        listed = await client.call_tool("list_datasets", {})
        assert any(d["name"] == "kb" and d["writable"] is True for d in listed.data)
        written = await client.call_tool("remember", {"dataset": "kb", "text": "banana cherry apple", "title": "fruit"})
        entry_id = written.data["id"]
        assert written.data["uri"] == f"semdex://kb/{entry_id}"
        hits = await client.call_tool("search", {"dataset": "kb", "query": "banana apple", "k": 5})
        assert hits.data and hits.data[0]["uri"] == f"semdex://kb/{entry_id}"
        await client.call_tool("forget", {"dataset": "kb", "entry_id": entry_id})
        gone = await client.call_tool("search", {"dataset": "kb", "query": "banana apple", "k": 5})
        assert gone.data == []


@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_remember_rejects_source_dataset(tmp_path: Path) -> None:
    server = build_mcp_server([_services(tmp_path)])  # a normal source dataset
    async with Client(server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("remember", {"dataset": "notes", "text": "x"})


@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_reindex_rejects_writable_dataset(tmp_path: Path) -> None:
    server = build_mcp_server([_writable_services(tmp_path)])
    async with Client(server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("reindex", {"dataset": "kb"})
```

- [ ] **Step 2: Run to verify it fails.** `uv run pytest tests/test_mcp_server.py -q -p no:cacheprovider` Expected: FAIL (no `remember`/`forget` tools; `list_datasets` lacks `writable`).

- [ ] **Step 3: Implement** in `src/semdex/adapters/mcp/server.py`:
  - Add `import uuid` at the top (stdlib).
  - Add the writing imports (adapters -> application allowed), aliased to avoid clashing with the tool names:

```python
from ...application.use_cases.writing import forget as _forget_knowledge, remember as _remember_knowledge
```

  - In `list_datasets`, add `"writable": svc.writable,` to each row dict.
  - In `reindex`, after the `read_only` check, add:

```python
        if svc.writable:
            raise _tool_error(f"dataset '{dataset}' is a writable knowledge dataset; use remember/forget, not reindex")
```

  - Add the two tools (next to the others, each with the pyright-ignore on the def line):

```python
@mcp.tool
def remember(  # pyright: ignore[reportUnusedFunction]
    dataset: str, text: str, title: str = "", entry_id: str | None = None
) -> dict[str, Any]:
    """Write knowledge into a WRITABLE dataset and return its uri.

    Use this to save a note/fact the client wants remembered. Only datasets
    with ``writable: true`` (see ``list_datasets``) accept writes; a source
    dataset (filesystem / mail) returns an error.

    - ``text`` is the content; it is chunked and embedded like any document.
    - ``title`` is an optional label shown on search hits.
    - ``entry_id`` is an optional stable id: pass the same id again to UPDATE
      that entry (its old content is replaced); omit it to create a new entry
      with a generated id. The returned ``uri`` / ``id`` are what ``forget``
      takes.
    """
    svc = _dataset(dataset)
    if not svc.writable:
        raise _tool_error(f"dataset '{dataset}' is not writable; remember is only for knowledge datasets")
    resolved_id = entry_id or uuid.uuid4().hex
    uri = f"semdex://{svc.name}/{resolved_id}"
    report = _remember_knowledge(
        chunk=svc.chunk,
        embedding=svc.embedding,
        store=svc.store,
        collection=svc.collection,
        uri=uri,
        text=text,
        label=title,
    )
    return {"uri": uri, "id": resolved_id, "chunks": report.chunks_indexed}


@mcp.tool
def forget(dataset: str, entry_id: str) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
    """Delete a knowledge entry (by its ``entry_id``) from a WRITABLE dataset.

    Idempotent: forgetting an unknown id is a no-op. Only writable datasets
    accept this; a source dataset returns an error.
    """
    svc = _dataset(dataset)
    if not svc.writable:
        raise _tool_error(f"dataset '{dataset}' is not writable; forget is only for knowledge datasets")
    uri = f"semdex://{svc.name}/{entry_id}"
    _forget_knowledge(store=svc.store, collection=svc.collection, uri=uri)
    return {"uri": uri, "forgotten": True}
```

- [ ] **Step 4: Run tests.** `uv run pytest tests/test_mcp_server.py -q -p no:cacheprovider` Expected: PASS (existing + 3 new).

- [ ] **Step 5: Gates + commit.** ruff/pyright; `lint-imports` MUST print `Contracts: 2 kept, 0 broken` (grep for it). `git add -A && git commit -m "feat(mcp): remember/forget tools for writable knowledge datasets"`

---

### Task 4: Docs + final gate

**Files:**
- Modify: `docs/COMPONENT_SETUP.md` (writable-datasets subsection under the MCP server section)
- Modify: `docs/adr/0002-mcp-server-adapter.md` (one bullet on the two dataset kinds)
- Modify: `CHANGELOG.md` ([Unreleased])

- [ ] **Step 1: Update `docs/COMPONENT_SETUP.md`.** In the "MCP server (serve)" section, add a "Writable knowledge datasets" subsection: a dataset with `writable = true` (no `sources`, not `read_only`) is a database-backed store the client writes to via `remember(dataset, text, title, entry_id)` and `forget(dataset, entry_id)`; content is keyed by `semdex://<dataset>/<id>`; `reindex` refuses it and source datasets refuse `remember`/`forget`; who may write is enforced by the DB grant on that dataset's DSN. Note the split: filesystem/mail datasets mirror external sources read-only; knowledge datasets are agent-written. ASCII, generic examples.

- [ ] **Step 2: Update ADR 0002.** Add one bullet under Decision or Consequences: semdex serves two dataset kinds - SOURCE (connector-mirrored, read-only over the source, reconcile-driven) and KNOWLEDGE (`writable`, no connector, client-written via remember/forget); they never mix (a knowledge dataset has no connector, so reconcile cannot prune it and source content cannot land in it); write authorization is a database GRANT, not server logic.

- [ ] **Step 3: Update `CHANGELOG.md`** `[Unreleased] / ### Added`:

```markdown
- Writable knowledge datasets: a `[[dataset]]` with `writable = true` (no `sources`, not `read_only`) is a database-backed store an MCP client writes to via the new `remember(dataset, text, title, entry_id)` and `forget(dataset, entry_id)` tools; content is keyed by `semdex://<dataset>/<id>`, `reindex` refuses writable datasets, and `list_datasets` reports `writable`. Source-mirrored datasets stay read-only over their sources. New `remember`/`forget` use cases (reuse `index_sources`).
```

  Keep ASCII (the commit-tell-sweep hook rejects em-dashes; if it flags pre-existing dashes elsewhere in the file, normalize only what blocks the commit).

- [ ] **Step 4: Full gate.** All four gate commands green; `lint-imports` = `2 kept, 0 broken`. `uv run pytest -m "not local_only and not integration" -q -p no:cacheprovider` full count > the step-5 count of 665.

- [ ] **Step 5: Optional smoke.** In-process: build a writable dataset, `build_mcp_server`, connect a `fastmcp.Client`, `remember` a note, `search` it back, `forget` it, confirm it is gone; confirm `reindex` on it errors. (The in-memory Client tests already cover this.)

- [ ] **Step 6: Commit.** `git add -A && git commit -m "docs: writable knowledge datasets (remember/forget) in COMPONENT_SETUP, ADR 0002, CHANGELOG"`

---

## Self-review checklist (verify before executing)

- Spec coverage: `writable` flag + validation + docs = Task 1; write/delete engine = Task 2; MCP tools + reindex-refusal + list_datasets flag = Task 3; docs = Task 4. "Select which allowed database to write" = the `dataset` arg on `remember`/`forget` + the `writable` gate + (deployment) per-dataset DSN grants.
- Separation invariant: a writable dataset has no `sources` (validator), so `build_dataset_services` builds an empty `FilesystemConnector` that is never used because `reindex` refuses writable datasets. Source datasets refuse `remember`/`forget`. Verified by `test_reindex_rejects_writable_dataset` and `test_remember_rejects_source_dataset`.
- Type consistency: `remember(...) -> IndexReport` (Task 2) is consumed as `report.chunks_indexed` in Task 3; `DatasetServices.writable` (Task 1) is read by all three tools (Task 3); the `entry_id` argument name (not `id`) avoids ruff A002.
- Reuse: `remember` delegates to `index_sources` (drop-first replace + compaction inherited); `forget` uses `delete_by_source` + `compact`. No new store/embedding code.
- Deferred / not in this plan: the live file watcher (a possible later step; the ports `Watcher`/`WatchMode`/`ChangeEvent` already exist but nothing wires them into `serve`); the IMAP connector (a future EXTERNAL package implementing the public `SourceConnector` port, deliberately not in core); a CLI `remember`/`forget` (MCP-only here); a delete-count return on `forget` (idempotent no-op is enough).

## References

- Design: `docs/systemdesign/mcp-multi-dataset-design.md` (branch bench-persistent-store-knobs) - decisions 5/6/9 (source-type-blind, connector-confined, reconcile lifecycle).
- Tasks tracker: #26 (this - redefined from "watchers/IMAP" to "writable knowledge datasets"; watcher + IMAP deferred).
