# MCP server - Step 5: multi-dataset fan-out search with RRF fusion

> **For Claude:** REQUIRED SUB-SKILL: Use bitranox:process-plan-executor (or bitranox:process-agents-subagent-driven-development) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. Also invoke bitranox:process-test-driven-development (each task RED -> GREEN) and bitranox:coding-python-clean-architecture.

**Goal:** Add a fan-out `search_datasets` MCP tool that searches one, several, or all datasets at once and fuses the per-dataset ranked lists into a single ranking by Reciprocal Rank Fusion (RRF), so a client (an accounting agent) can query a shared corpus plus department-specific datasets together and get one honest ranking.

**Architecture:** A new pure fusion function + a fan-out use case live in the framework-free application layer; the MCP server (adapter) adds one tool that maps its already-held `DatasetServices` to application search targets and calls the use case. RRF fuses by RANK, never by raw cross-model cosine score (different embedding models' scores are on different scales). Query is embedded ONCE per distinct `(provider, model, endpoint)`. No in-app access control - the deployment scopes datasets per server and the database GRANT / per-dataset DSN is the real enforcer.

**Tech Stack:** pydantic config, fastmcp v3 (in-memory `fastmcp.Client` for tests), pytest (+ pytest-asyncio, already wired), uv.

## Handover context (read first - you may have zero prior context)

- **Worktree:** `<semdex-checkout>/.claude/worktrees/mcp-source-identity`, branch `worktree-mcp-source-identity`. Work here.
- **Design decision this implements** (from `docs/systemdesign/mcp-multi-dataset-design.md`, on branch `bench-persistent-store-knobs`; `git show bench-persistent-store-knobs:docs/systemdesign/mcp-multi-dataset-design.md`): decision 7 - "Search selects one-or-many datasets. Default single-target for clean provenance; fan-out when asked. Fan-out embeds the query once per distinct model in the selected set, queries each store with its own model, and fuses results by RANK (Reciprocal Rank Fusion), never by raw cross-model cosine score." Acceptance: "RRF over two datasets with different embedding models returns a sensibly fused ranking; assert raw cross-model scores are NOT compared directly."
- **What step 4 already shipped (done):**
  - `application/use_cases/searching.py`: `search(*, embedding, store, collection, query, k=5) -> list[Hit]`.
  - `domain/models.py`: `Hit(chunk_text, score, uri, ordinal, label, collection)`; `Vector: TypeAlias = tuple[float, ...]`.
  - `application/ports.py`: `EmbeddingProvider` (`.model_id`, `.dim`, `embed_query(text) -> Vector`, `embed_passages`), `VectorStoreReader.query(*, collection, vector, k) -> list[Hit]`.
  - `composition/__init__.py`: `DatasetServices(name, backend, store, embedding, extract, chunk, connector, collection, partition, read_only)` + `build_dataset_services(dataset, *, default_partition)`; injected via `Bootstrap.dataset_services_factory`.
  - `adapters/mcp/server.py`: `build_mcp_server(datasets, *, auth=None, name="semdex") -> FastMCP` with tools `list_datasets` / `search` / `reindex`. Tools are `@mcp.tool` closures; each carries `# pyright: ignore[reportUnusedFunction]`. Tool errors raise `fastmcp.exceptions.ToolError` (helper `_tool_error`).
  - `adapters/config/mcp.py`: `McpConfig` (frozen pydantic) + `get_mcp_config`.
  - `adapters/cli/commands/serve.py`: builds `DatasetServices` per dataset via the injected factory and calls `build_mcp_server`.
- **Layer rule (import-linter, enforced by `lint-imports`):** `adapters` may import `application` + `domain`, NEVER `composition`. `adapters/mcp/server.py` receives `DatasetServices` as an argument and reads its fields; it imports `DatasetServices` only under `TYPE_CHECKING` (excluded from the contract). It MAY import application types (`DatasetSearchTarget`, `search_across_datasets`) at runtime. Do NOT add a runtime `from ...composition import` anywhere in adapters.
- **Gates (run from the worktree):**
  - `uv run pytest -m "not local_only and not integration" -q -p no:cacheprovider`
  - `uv run ruff check src/ tests/` and `uv run ruff format src/ tests/`
  - `uv run pyright src/ tests/`
  - `uv run lint-imports` (MUST read the last line; verify `2 kept, 0 broken` - do not trust `tail -2`, grep for `Contracts:`)
  - The `VIRTUAL_ENV ... does not match` warning from uv is harmless.

## Global Constraints

- **Python 3.10+**; type-annotate everything; pyright strict must pass (0 errors).
- **Clean architecture:** the fusion function + fan-out use case + `DatasetSearchTarget` live in `application`; the tool lives in `adapters/mcp`. Run `lint-imports` after each task and confirm `2 kept, 0 broken`.
- **TDD:** every task is RED (watch it fail) -> GREEN (minimal) -> commit. Fan-out tools are tested in-memory with `fastmcp.Client(server)`.
- **RRF fuses by RANK only.** Never compare or sum raw `Hit.score` (cosine) across datasets - they are on different model scales. The fused ranking uses `1/(k+rank)` contributions only.
- **Config as the only source of tunables:** the RRF constant is `[mcp].rrf_k` (default 60), never hardcoded at the call site. Document it in `17-mcp.toml` in the same comprehensive style as the other keys.
- **ASCII only** in all files (a tell-sweep hook rejects em/en-dashes, curly quotes, ellipsis). Use `-`, straight quotes, `...`.
- **No Claude/AI attribution** in commits. Commit after each task. `uv.lock` is gitignored here - never `git add uv.lock`.

---

### Task 1: FusedHit model + reciprocal_rank_fusion (pure)

**Files:**
- Modify: `src/semdex/domain/models.py` (add `FusedHit`)
- Modify: `src/semdex/application/use_cases/searching.py` (add `reciprocal_rank_fusion`)
- Create: `tests/test_fusion.py`

**Interfaces:**
- Consumes: `Hit` (domain).
- Produces: `FusedHit(hit: Hit, dataset: str, rrf_score: float)`; `reciprocal_rank_fusion(ranked_lists: Sequence[tuple[str, Sequence[Hit]]], *, k: int = 60, limit: int) -> list[FusedHit]`.

- [ ] **Step 1: Write the failing test.**

```python
# tests/test_fusion.py
from __future__ import annotations

import pytest

from semdex.application.use_cases.searching import reciprocal_rank_fusion
from semdex.domain.models import Hit


def _hit(uri: str, ordinal: int = 0, score: float = 0.0) -> Hit:
    return Hit(chunk_text=uri, score=score, uri=uri, ordinal=ordinal, label="", collection="c")


@pytest.mark.os_agnostic
def test_disjoint_lists_interleave_by_rank() -> None:
    # Two datasets, disjoint uris. Rank-1 of each ties (same 1/(k+1)); the fused
    # order is deterministic by (uri, ordinal) on ties.
    left = ("acc", [_hit("a1"), _hit("a2")])
    right = ("shared", [_hit("s1"), _hit("s2")])
    fused = reciprocal_rank_fusion([left, right], k=60, limit=10)
    assert [f.hit.uri for f in fused] == ["a1", "s1", "a2", "s2"]
    assert fused[0].dataset == "acc"
    # rank-1 hits outrank rank-2 hits
    assert fused[0].rrf_score == fused[1].rrf_score > fused[2].rrf_score


@pytest.mark.os_agnostic
def test_same_item_across_lists_sums_contributions() -> None:
    # Same (uri, ordinal) surfaced by both datasets: contributions sum, so it wins.
    a = ("acc", [_hit("dup"), _hit("a2")])
    b = ("shared", [_hit("s1"), _hit("dup")])
    fused = reciprocal_rank_fusion([a, b], k=60, limit=10)
    assert fused[0].hit.uri == "dup"
    assert fused[0].rrf_score == pytest.approx(1.0 / 61 + 1.0 / 62)


@pytest.mark.os_agnostic
def test_limit_truncates() -> None:
    lst = ("d", [_hit("a"), _hit("b"), _hit("c")])
    assert len(reciprocal_rank_fusion([lst], k=60, limit=2)) == 2


@pytest.mark.os_agnostic
def test_never_uses_raw_score() -> None:
    # A rank-2 hit with a huge cosine must NOT outrank a rank-1 hit (rank only).
    lst = ("d", [_hit("first", score=0.01), _hit("second", score=99.0)])
    fused = reciprocal_rank_fusion([lst], k=60, limit=10)
    assert [f.hit.uri for f in fused] == ["first", "second"]
```

- [ ] **Step 2: Run to verify it fails.** `uv run pytest tests/test_fusion.py -q -p no:cacheprovider` Expected: ImportError (`FusedHit` / `reciprocal_rank_fusion` missing).

- [ ] **Step 3: Add `FusedHit`** to `src/semdex/domain/models.py` (after the `Hit` class; add `"FusedHit"` to `__all__`):

```python
@dataclass(frozen=True, slots=True)
class FusedHit:
    """A search hit after cross-dataset Reciprocal Rank Fusion.

    Wraps the original :class:`Hit` (whose ``score`` stays the per-dataset cosine,
    NOT comparable across datasets) with the dataset that produced it and the
    fused ``rrf_score`` that actually ranks it.
    """

    hit: Hit
    dataset: str
    rrf_score: float
```

- [ ] **Step 4: Add `reciprocal_rank_fusion`** to `src/semdex/application/use_cases/searching.py`. Add imports at the top: change the `TYPE_CHECKING` block to also import `Vector`, and import the concrete `FusedHit` for the return value. Add `from collections.abc import Sequence` under `TYPE_CHECKING`. Add the function and `_DEFAULT_RRF_K = 60` module constant:

```python
# at module level, near _DEFAULT_K
_DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[tuple[str, Sequence[Hit]]],
    *,
    k: int = _DEFAULT_RRF_K,
    limit: int,
) -> list[FusedHit]:
    """Fuse per-dataset ranked hit lists into one ranking by Reciprocal Rank Fusion.

    Each hit contributes ``1 / (k + rank)`` (rank 1-based) to its identity; the
    fused ranking uses those summed contributions ONLY, never the raw similarity
    ``Hit.score`` (which is not comparable across different embedding models).
    Hits are identified across lists by ``(uri, ordinal)``; a hit surfaced by
    several datasets sums their contributions. The ``dataset`` on a result is the
    dataset whose single contribution to it was largest. Ties in the fused score
    break deterministically by ``(uri, ordinal)``. Returns the top ``limit``.

    Example:
        >>> from semdex.domain.models import Hit
        >>> h = Hit(chunk_text="x", score=0.0, uri="u", ordinal=0, label="", collection="c")
        >>> reciprocal_rank_fusion([("d", [h])], limit=1)[0].dataset
        'd'
    """
    scores: dict[tuple[str, int], float] = {}
    best: dict[tuple[str, int], tuple[float, str, Hit]] = {}
    for dataset, hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            key = (hit.uri, hit.ordinal)
            contribution = 1.0 / (k + rank)
            scores[key] = scores.get(key, 0.0) + contribution
            current = best.get(key)
            if current is None or contribution > current[0]:
                best[key] = (contribution, dataset, hit)
    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    fused = [FusedHit(hit=best[key][2], dataset=best[key][1], rrf_score=scores[key]) for key in ordered]
    return fused[:limit]
```

  Import note: at the top of `searching.py`, add `from ...domain.models import FusedHit` at RUNTIME (not just TYPE_CHECKING - it is constructed), and keep `Hit` / `Vector` / `EmbeddingProvider` / `VectorStoreReader` / `Sequence` under `TYPE_CHECKING`. Add `"reciprocal_rank_fusion"` and (Task 3) the fan-out names to `__all__`.

- [ ] **Step 5: Run tests.** `uv run pytest tests/test_fusion.py src/semdex/application/use_cases/searching.py src/semdex/domain/models.py -q -p no:cacheprovider` Expected: PASS (4 tests + doctests).

- [ ] **Step 6: Gates + commit.** ruff format+check, pyright, lint-imports (`2 kept, 0 broken`). `git add -A && git commit -m "feat(search): FusedHit + reciprocal_rank_fusion (rank-only cross-dataset fusion)"`

---

### Task 2: fan-out use case (search_across_datasets)

**Files:**
- Modify: `src/semdex/application/use_cases/searching.py` (add `DatasetSearchTarget` + `search_across_datasets`)
- Modify: `src/semdex/application/use_cases/__init__.py` (export the new names)
- Create: `tests/test_search_across_datasets.py`

**Interfaces:**
- Consumes: `EmbeddingProvider`, `VectorStoreReader` (ports), `reciprocal_rank_fusion` (Task 1).
- Produces: `DatasetSearchTarget(name: str, model_key: str, embedding: EmbeddingProvider, store: VectorStoreReader, collection: str)`; `search_across_datasets(*, targets: Sequence[DatasetSearchTarget], query: str, k: int = 5, rrf_k: int = 60) -> list[FusedHit]`.

- [ ] **Step 1: Write the failing test.** Uses tiny fakes so the test is offline and asserts the embed-once-per-model behaviour.

```python
# tests/test_search_across_datasets.py
from __future__ import annotations

from collections.abc import Sequence

import pytest

from semdex.application.use_cases.searching import DatasetSearchTarget, search_across_datasets
from semdex.domain.models import Hit, Vector


class _Embed:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.calls = 0

    @property
    def model_id(self) -> str:
        return self.tag

    @property
    def dim(self) -> int:
        return 1

    def embed_passages(self, texts: Sequence[str]) -> list[Vector]:
        return [(0.0,) for _ in texts]

    def embed_query(self, text: str) -> Vector:
        self.calls += 1
        return (float(len(self.tag)),)


class _Store:
    def __init__(self, hits: list[Hit]) -> None:
        self._hits = hits
        self.seen_vector: Vector | None = None

    def query(self, *, collection: str, vector: Vector, k: int) -> list[Hit]:
        self.seen_vector = vector
        return self._hits[:k]


def _hit(uri: str) -> Hit:
    return Hit(chunk_text=uri, score=0.5, uri=uri, ordinal=0, label="", collection="c")


@pytest.mark.os_agnostic
def test_embeds_once_per_distinct_model_and_fuses() -> None:
    shared_embed = _Embed("m1")  # two datasets share this exact provider object
    acc_store = _Store([_hit("acc-1")])
    shared_store = _Store([_hit("shared-1")])
    targets = [
        DatasetSearchTarget(name="acc", model_key="m1", embedding=shared_embed, store=acc_store, collection="a"),
        DatasetSearchTarget(name="shared", model_key="m1", embedding=shared_embed, store=shared_store, collection="s"),
    ]
    fused = search_across_datasets(targets=targets, query="q", k=5, rrf_k=60)
    # one embed for the shared model_key, reused for both stores
    assert shared_embed.calls == 1
    assert acc_store.seen_vector == shared_store.seen_vector
    assert {f.hit.uri for f in fused} == {"acc-1", "shared-1"}
    assert {f.dataset for f in fused} == {"acc", "shared"}


@pytest.mark.os_agnostic
def test_distinct_models_embed_separately() -> None:
    e1, e2 = _Embed("aa"), _Embed("bbbb")
    t = [
        DatasetSearchTarget(name="d1", model_key="k1", embedding=e1, store=_Store([_hit("x")]), collection="c"),
        DatasetSearchTarget(name="d2", model_key="k2", embedding=e2, store=_Store([_hit("y")]), collection="c"),
    ]
    search_across_datasets(targets=t, query="q", k=5, rrf_k=60)
    assert e1.calls == 1 and e2.calls == 1
```

- [ ] **Step 2: Run to verify it fails.** `uv run pytest tests/test_search_across_datasets.py -q -p no:cacheprovider` Expected: ImportError (`DatasetSearchTarget` / `search_across_datasets` missing).

- [ ] **Step 3: Implement** in `src/semdex/application/use_cases/searching.py`. Add the dataclass (needs a runtime `from dataclasses import dataclass`) and the use case:

```python
@dataclass(frozen=True, slots=True)
class DatasetSearchTarget:
    """One dataset to search in a fan-out.

    ``model_key`` is a stable identity for the embedding model (provider + model +
    endpoint) so the query is embedded ONCE per distinct model across the selected
    datasets; ``embedding`` and ``store`` are the read ports, ``collection`` the
    dataset's collection.
    """

    name: str
    model_key: str
    embedding: EmbeddingProvider
    store: VectorStoreReader
    collection: str


def search_across_datasets(
    *,
    targets: Sequence[DatasetSearchTarget],
    query: str,
    k: int = _DEFAULT_K,
    rrf_k: int = _DEFAULT_RRF_K,
) -> list[FusedHit]:
    """Fan ``query`` out across datasets and fuse the results by RRF.

    Embeds the query ONCE per distinct ``model_key`` (datasets sharing a model
    reuse the vector), queries each dataset's store for its top-``k`` hits, then
    fuses every per-dataset list into one top-``k`` ranking via
    :func:`reciprocal_rank_fusion`. Similarity scores are never compared across
    datasets.
    """
    vectors: dict[str, Vector] = {}
    for target in targets:
        if target.model_key not in vectors:
            vectors[target.model_key] = target.embedding.embed_query(query)
    ranked: list[tuple[str, Sequence[Hit]]] = [
        (target.name, target.store.query(collection=target.collection, vector=vectors[target.model_key], k=k))
        for target in targets
    ]
    return reciprocal_rank_fusion(ranked, k=rrf_k, limit=k)
```

  Move `EmbeddingProvider`, `VectorStoreReader`, `Hit`, `Vector`, `Sequence` imports so the names resolve: `dataclass` and `FusedHit` at runtime; the ports / `Hit` / `Vector` / `Sequence` may stay under `TYPE_CHECKING` (they appear only in annotations). Update `__all__` to include `"DatasetSearchTarget"`, `"reciprocal_rank_fusion"`, `"search_across_datasets"`.

- [ ] **Step 4: Export** from `src/semdex/application/use_cases/__init__.py`: add `DatasetSearchTarget, reciprocal_rank_fusion, search_across_datasets` to the `from .searching import ...` line and to `__all__`.

- [ ] **Step 5: Run tests.** `uv run pytest tests/test_search_across_datasets.py -q -p no:cacheprovider` Expected: PASS (2 tests).

- [ ] **Step 6: Gates + commit.** ruff/pyright/lint-imports. `git add -A && git commit -m "feat(search): search_across_datasets fan-out use case (embed once per model)"`

---

### Task 3: [mcp].rrf_k config knob

**Files:**
- Modify: `src/semdex/adapters/config/mcp.py` (add `rrf_k`)
- Modify: `src/semdex/adapters/config/defaultconfig.d/17-mcp.toml` (document it)
- Modify: `tests/test_config_mcp.py` (assert the default + a set value)

**Interfaces:**
- Produces: `McpConfig.rrf_k: int` (default 60, > 0).

- [ ] **Step 1: Add the failing test** to `tests/test_config_mcp.py`:

```python
@pytest.mark.os_agnostic
def test_rrf_k_default_and_override(config_factory: Callable[[dict[str, Any]], Config]) -> None:
    assert get_mcp_config(config_factory({})).rrf_k == 60
    assert get_mcp_config(config_factory({"mcp": {"rrf_k": 100}})).rrf_k == 100
```

- [ ] **Step 2: Run to verify it fails.** `uv run pytest tests/test_config_mcp.py -q -p no:cacheprovider` Expected: FAIL (AttributeError: no `rrf_k`).

- [ ] **Step 3: Add the field** to `McpConfig` in `src/semdex/adapters/config/mcp.py`, after `auth`:

```python
    # Reciprocal Rank Fusion constant for fan-out (search_datasets) - see 17-mcp.toml.
    rrf_k: int = Field(default=60, gt=0)
```

- [ ] **Step 4: Document it** in `src/semdex/adapters/config/defaultconfig.d/17-mcp.toml`. Add a `rrf_k` block in the same style as the other keys (default / effect / when to change / consequences / recommendation). Cover: it is the RRF constant used to fuse the per-dataset ranked lists in the `search_datasets` fan-out tool; a hit contributes `1/(rrf_k + rank)`; larger `rrf_k` flattens the weight of top ranks (results from many datasets mix more evenly), smaller sharpens the lead of each list's top hits; 60 is the standard literature default; it never affects single-dataset `search`. Keep it ASCII. Place the key `rrf_k = 60` under `[mcp]`.

- [ ] **Step 5: Run tests.** `uv run pytest tests/test_config_mcp.py -q -p no:cacheprovider` Expected: PASS.

- [ ] **Step 6: Gates + commit.** `git add -A && git commit -m "feat(config): [mcp].rrf_k fan-out fusion constant + 17-mcp.toml docs"`

---

### Task 4: DatasetServices.model_key (composition)

**Files:**
- Modify: `src/semdex/composition/__init__.py` (add `model_key` to `DatasetServices`; compute in `build_dataset_services`)
- Modify: `tests/test_dataset_services.py` (assert `model_key`)

**Interfaces:**
- Produces: `DatasetServices.model_key: str` = `"<provider.value>|<model or ''>|<endpoint or ''>"`.

- [ ] **Step 1: Add the failing assertion** to the existing test in `tests/test_dataset_services.py` (inside `test_builds_embedded_dataset_services`, after the existing asserts):

```python
    assert services.model_key == "placeholder||"
```

- [ ] **Step 2: Run to verify it fails.** `uv run pytest tests/test_dataset_services.py -q -p no:cacheprovider` Expected: FAIL (no `model_key`).

- [ ] **Step 3: Add the field** to the `DatasetServices` dataclass in `src/semdex/composition/__init__.py` (after `name`):

```python
    model_key: str
```

- [ ] **Step 4: Compute it** in `build_dataset_services`, and pass it in the constructor. Add before the `return`:

```python
    model_key = f"{dataset.embedding_provider.value}|{dataset.embedding_model or ''}|{dataset.embedding_endpoint or ''}"
```

  and add `model_key=model_key,` to the `DatasetServices(...)` call (right after `name=dataset.name,`).

- [ ] **Step 5: Run tests.** `uv run pytest tests/test_dataset_services.py -q -p no:cacheprovider` Expected: PASS.

- [ ] **Step 6: Gates + commit.** lint-imports must stay `2 kept, 0 broken`. `git add -A && git commit -m "feat(composition): DatasetServices.model_key (embedding identity for fan-out)"`

---

### Task 5: MCP search_datasets tool + serve wiring

**Files:**
- Modify: `src/semdex/adapters/mcp/server.py` (add the `search_datasets` tool; `build_mcp_server(..., rrf_k=60)`)
- Modify: `src/semdex/adapters/cli/commands/serve.py` (pass `rrf_k=mcp_cfg.rrf_k`)
- Modify: `tests/test_mcp_server.py` (fan-out tool tests)

**Interfaces:**
- Consumes: `DatasetServices` (name/model_key/embedding/store/collection), `DatasetSearchTarget` + `search_across_datasets` (application), `McpConfig.rrf_k`.
- Produces: `build_mcp_server(datasets, *, auth=None, name="semdex", rrf_k=60)`; MCP tool `search_datasets(query: str, k: int = 5, datasets: list[str] | None = None) -> list[dict]`.

- [ ] **Step 1: Write failing tests** in `tests/test_mcp_server.py`. Reuse the existing `_services` helper; add a second dataset builder and fan-out tests. Add at the end of the file:

```python
def _services_named(tmp_path: Path, name: str, word: str):
    docs = tmp_path / name
    docs.mkdir()
    (docs / "d.md").write_text(word, encoding="utf-8")
    dataset = DatasetConfig(
        name=name,
        backend=StoreBackend.JSON,
        store_dir=str(tmp_path / f"store-{name}"),
        collection=name,
        embedding_provider=EmbeddingBackend.PLACEHOLDER,
        sources=(str(docs),),
    )
    return build_dataset_services(dataset, default_partition=Partition.TABLE)


@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_search_datasets_fans_out_over_all(tmp_path: Path) -> None:
    shared = _services_named(tmp_path, "shared", "alpha bravo")
    acc = _services_named(tmp_path, "accounting", "charlie delta")
    server = build_mcp_server([shared, acc])
    async with Client(server) as client:
        await client.call_tool("reindex", {"dataset": "shared"})
        await client.call_tool("reindex", {"dataset": "accounting"})
        # omit datasets -> all; each fused hit is tagged with its dataset
        res = await client.call_tool("search_datasets", {"query": "alpha charlie", "k": 5})
        datasets_seen = {row["dataset"] for row in res.data}
        assert datasets_seen == {"shared", "accounting"}
        assert all("rrf_score" in row and "uri" in row for row in res.data)


@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_search_datasets_subset_and_all_token(tmp_path: Path) -> None:
    shared = _services_named(tmp_path, "shared", "alpha bravo")
    acc = _services_named(tmp_path, "accounting", "charlie delta")
    server = build_mcp_server([shared, acc])
    async with Client(server) as client:
        await client.call_tool("reindex", {"dataset": "shared"})
        await client.call_tool("reindex", {"dataset": "accounting"})
        only = await client.call_tool("search_datasets", {"query": "alpha", "datasets": ["shared"]})
        assert {row["dataset"] for row in only.data} == {"shared"}
        allrows = await client.call_tool("search_datasets", {"query": "alpha", "datasets": ["all"]})
        assert {row["dataset"] for row in allrows.data} == {"shared", "accounting"}


@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_search_datasets_unknown_name_errors(tmp_path: Path) -> None:
    server = build_mcp_server([_services_named(tmp_path, "shared", "alpha")])
    async with Client(server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("search_datasets", {"query": "x", "datasets": ["nope"]})
```

- [ ] **Step 2: Run to verify it fails.** `uv run pytest tests/test_mcp_server.py -q -p no:cacheprovider` Expected: FAIL (no `search_datasets` tool / unknown tool).

- [ ] **Step 3: Implement.** In `src/semdex/adapters/mcp/server.py`:
  - Add runtime imports (adapters -> application is allowed): `from ...application.use_cases.searching import DatasetSearchTarget, search_across_datasets`.
  - Change the signature: `def build_mcp_server(datasets, *, auth: object | None = None, name: str = "semdex", rrf_k: int = 60) -> FastMCP:`.
  - Add a selection helper inside `build_mcp_server` (next to `_dataset`), then the tool. `_ALL_TOKEN = "all"`.

```python
def _select(names: list[str] | None) -> list[DatasetServices]:
    if not names or any(n.strip().lower() == "all" for n in names):
        return list(datasets)
    return [_dataset(name) for name in names]  # _dataset raises a clear ToolError on unknown


@mcp.tool
def search_datasets(  # pyright: ignore[reportUnusedFunction]
    query: str, k: int = 5, datasets: list[str] | None = None
) -> list[dict[str, Any]]:
    """Search several datasets at once and return ONE fused ranking.

    Use this to answer a question that may span more than one dataset (for
    example a shared corpus plus a department's own documents).

    Choosing datasets:
    - First call ``list_datasets`` to get the valid dataset names.
    - To search EVERYTHING this server serves: omit ``datasets`` or pass
      ["all"].
    - To search a SUBSET: pass their exact names, e.g. ["shared", "accounting"].
    - An unknown name returns an error that lists the valid names; retry with
      one of those.

    Ranking: results from different datasets are fused by RANK (Reciprocal Rank
    Fusion). Order results by the returned ``rrf_score``, NOT the per-dataset
    ``score`` - raw similarity scores are not comparable across datasets. Each
    row is tagged with its source ``dataset``. ``k`` is how many fused results
    to return.
    """
    targets = [
        DatasetSearchTarget(
            name=svc.name, model_key=svc.model_key, embedding=svc.embedding, store=svc.store, collection=svc.collection
        )
        for svc in _select(datasets)
    ]
    fused = search_across_datasets(targets=targets, query=query, k=k, rrf_k=rrf_k)
    return [
        {
            "uri": fh.hit.uri,
            "ordinal": fh.hit.ordinal,
            "score": fh.hit.score,
            "rrf_score": fh.rrf_score,
            "dataset": fh.dataset,
            "text": fh.hit.chunk_text,
            "label": fh.hit.label,
            "collection": fh.hit.collection,
        }
        for fh in fused
    ]
```

- [ ] **Step 4: Wire rrf_k** in `src/semdex/adapters/cli/commands/serve.py`: change the `build_mcp_server(services, auth=auth)` call to `build_mcp_server(services, auth=auth, rrf_k=mcp_cfg.rrf_k)`.

- [ ] **Step 5: Run tests.** `uv run pytest tests/test_mcp_server.py tests/test_cli_serve.py -q -p no:cacheprovider` Expected: PASS (existing + 3 new).

- [ ] **Step 6: Gates + commit.** ruff/pyright; `lint-imports` MUST print `Contracts: 2 kept, 0 broken` (grep for it, do not trust `tail -2`). `git add -A && git commit -m "feat(mcp): search_datasets fan-out tool (RRF over named datasets or all)"`

---

### Task 6: Docs + final gate

**Files:**
- Modify: `docs/COMPONENT_SETUP.md` (add fan-out to the MCP server section)
- Modify: `docs/adr/0002-mcp-server-adapter.md` (one line: fan-out fuses by rank, never cross-model score)
- Modify: `CHANGELOG.md` ([Unreleased])

- [ ] **Step 1: Update `docs/COMPONENT_SETUP.md`.** In the "MCP server (serve)" section, add a short subsection after the tools list: `search_datasets(query, k, datasets=None)` searches several datasets and fuses by RRF; omit `datasets` or pass `["all"]` for every dataset, else a subset of `list_datasets` names; use it for a shared corpus + department-specific datasets on one server (accounting = shared + accounting), fused into one ranking; `[mcp].rrf_k` tunes the fusion. ASCII only, generic examples (no live data).

- [ ] **Step 2: Update ADR 0002.** Add one bullet under Decision or Consequences: fan-out (`search_datasets`) fuses per-dataset rankings by Reciprocal Rank Fusion (rank only), never by raw cross-model similarity score, because different embedding models' scores are on different scales; the query is embedded once per distinct model.

- [ ] **Step 3: Update `CHANGELOG.md`** `[Unreleased] / ### Added`:

```markdown
- Fan-out search: `search_datasets(query, k, datasets)` MCP tool searches one, several, or all datasets and fuses the per-dataset rankings by Reciprocal Rank Fusion (`[mcp].rrf_k`, default 60), embedding the query once per distinct model and never comparing raw cross-model similarity scores. New `search_across_datasets` use case + `reciprocal_rank_fusion` + `FusedHit`.
```

  Keep ASCII (the commit-tell-sweep hook rejects em-dashes; if the tell-sweep flags pre-existing dashes elsewhere in the file, leave them - only your added lines must be clean, but if the hook blocks the commit, normalize the flagged lines to ASCII as a separate courtesy).

- [ ] **Step 4: Full gate.** All four gate commands green; `lint-imports` = `2 kept, 0 broken`. `uv run pytest -m "not local_only and not integration" -q -p no:cacheprovider` full count > the step-4 count of 654.

- [ ] **Step 5: Optional smoke.** Build two datasets in-process, `build_mcp_server`, connect a `fastmcp.Client`, reindex both, call `search_datasets` with `datasets` omitted and with a subset; confirm each row carries `dataset` + `rrf_score`. (The in-memory Client tests already cover this; a live-socket smoke like step 4 is optional.)

- [ ] **Step 6: Commit.** `git add -A && git commit -m "docs: fan-out search (search_datasets) in COMPONENT_SETUP, ADR 0002, CHANGELOG"`

---

## Self-review checklist (verify before executing)

- Spec coverage: RRF fusion = Task 1; embed-once-per-model + fan-out use case = Task 2; rrf_k config = Task 3; model identity on services = Task 4; the `search_datasets` tool (named subset / "all" / omit) + serve wiring = Task 5; docs = Task 6. Acceptance "raw cross-model scores NOT compared" = `test_never_uses_raw_score` (Task 1) + rank-only fusion.
- Type consistency: `FusedHit(hit, dataset, rrf_score)` (Task 1) is consumed verbatim in Task 5's dict mapping; `DatasetSearchTarget(name, model_key, embedding, store, collection)` (Task 2) is built from `DatasetServices` fields (name/model_key from Task 4, embedding/store/collection from step 4) in Task 5; `search_across_datasets(*, targets, query, k, rrf_k)` (Task 2) is called by the tool (Task 5); `rrf_k` flows McpConfig (Task 3) -> serve (Task 5) -> build_mcp_server (Task 5) -> use case.
- Layer contract: the tool imports `DatasetSearchTarget` / `search_across_datasets` from `application` at runtime (allowed) and reads `DatasetServices` fields (imported only under `TYPE_CHECKING`). NO adapter imports `composition` at runtime. Confirm `lint-imports` = 2 kept, 0 broken after Tasks 4 and 5.
- Deferred / not in this plan (follow-ons): a CLI fan-out command (the fan-out is MCP-only here; the CLI `search` stays single-dataset); per-token dataset access scoping (deployment scopes datasets per server + DB grants enforce); title/breadcrumb metadata on Hit (decision 10); live watchers + IMAP connector (step 6 / #26).

## References

- Design: `docs/systemdesign/mcp-multi-dataset-design.md` (branch bench-persistent-store-knobs), decision 7 + Verification "Fan-out".
- RRF: Cormack et al. 2009, "Reciprocal Rank Fusion outperforms Condorcet"; the standard `k=60`.
- Tasks tracker: #25 (this), #26 (watchers/IMAP).
