# Product-k Scoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use bitranox:process-agents-subagent-driven-development (recommended) or bitranox:process-plan-executor to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score the chunk list `semdex search` delivers (k chunks, no dedup) beside the deduplicated document list every page measures, at the product's k and at two fixed retrieved-token budgets, and publish it on the chunking page.

**Architecture:** One model-free scorer streams a top-100 chunk list per query from the cached query vectors and the cell's `vectors.npy`, derives delivered and documents nDCG@k plus distinct-document counts per rung, and writes its own score file. The exporter gains a dedicated writer for that file (its row schema differs from the chunk-sweep exports), the table generator two registrars, the chart script one collector and renderer, and the chunking page one subsection with claims. The design is `docs/plans/2026-09-27-product-k-design.md`.

**Tech Stack:** Python 3.10+ (scripts run under the project `.venv`), numpy, pyarrow, pytest; the `scripts/` env-config convention (no argparse); pyright strict over `scripts/` and `tests/`.

## Global Constraints

- Work on `main`, commit after every task, push after the gate; never a PR (repo rule).
- Every new figure in prose gets a `[[claim]]` in `tests/benchmarks/claims/03-chunking.toml`; a quote sits on ONE physical line.
- ASCII punctuation only in files (no em-dash, no arrow character in source).
- Provenance: every cell row is written as `results[cell] = stamped(row)` (the stamp test detects that exact shape) and carried through the exporter unchanged, never collected on the copying machine.
- No hardcoded magic numbers without a knob: `SEMDEX_PRODUCT_K` (default `5`, the shipped `default_k`), `SEMDEX_PRODUCT_BUDGETS` (default `1280,2560`), `SEMDEX_PRODUCT_FETCH` (default `100`), `SEMDEX_PRODUCT_CORPORA` (default `gerdalir_de_12k_slice,mldr_de_3k_slice,mldr_en_8k_slice`), `SEMDEX_PRODUCT_EMBEDDINGS` (default the six labels below), `SEMDEX_PRODUCT_PROFILES` (default: every cap-256 zero-overlap profile present for the corpus plus `recursive-t64-o0-gpt2`, `recursive-t128-o0-gpt2`, `recursive-t512-o0-gpt2`), `SEMDEX_SCORE_OUT` (default `<cache>/scores/product_k_scores.json`), `SEMDEX_SCORE_PERQUERY`, `SEMDEX_PRODUCT_FORCE`.
- Budget ladder caps are `64, 128, 256, 512`; a rung's k is `budget // cap`; a cap outside the ladder gets the product k only.
- Embedder labels: `model2vec:potion-retrieval-32M`, `model2vec:potion-base-8M`, `fastembed:bge-base`, `ollama:bge-m3`, `ollama:qwen3-embedding-4b`, `ollama:qwen3-embedding-8b`. Cell id `<corpus>__<profile>__<dirsafe label>` where dirsafe replaces `:` with `-`.
- The scorer is model-free: a cell whose query-vector cache is absent is refused by name with a non-zero exit, never embedded.
- Never `pkill -f`; judge a background job by its RC line. The scoring run (Task 6) starts only after the [22] sweep's `ITEM22-DONE` line with every RC=0.
- The tests type pyarrow through the `TYPE_CHECKING` facade shown in Task 2, never `# pyright: basic`.

---

## Ground truth read before planning (2026-09-27)

- `scripts/score_chunk_sweep.py` exports `dirsafe`, `metrics`, `query_vectors` in `__all__`; `_qvec_path(corpus, label, queries) -> Path` and `_load_uris(parquet) -> list[str]` are private and are exported in Task 2. `_query_vectors` EMBEDS on a cache miss, which is why the new scorer reads the npz itself.
- `scripts/_score_kernel.topk_stream(vectors_path, queries, *, fetch, block_bytes) -> (index, scores)`; `scripts/_score_stats.bootstrap_ci(values) -> {mean, ci_lo, ci_hi, sd, n, half_width}`; `paired_ci(left, right) -> {mean_delta, ci_lo, ci_hi, wins, ties, losses, n_shared, resolved}`; `scripts/_provenance.stamped(row) -> dict`, `MEASURED_ON`; `scripts/preembed_vectors._ndcg(ranked, rels, k)`, `_cache_root()`, `_read_json(path)`; `scripts/audit_chunk_dimensions.parse_profile(profile) -> {strategy, max_tokens, overlap_tokens, tokenizer, recipe, breakpoint_model}`.
- `tests/test_bench_scorers_stamp_provenance.py` treats a script as a cell producer when its source matches `/ "scores" / "<name>.json"` and then requires a `stamped(...)` call inside a subscript assignment; the new scorer's default output path expression satisfies the first and Task 2's write site the second.
- `scripts/export_bench_raw.py`: `export_row` is built for the chunk-sweep metric set (`_METRICS`), so the product-k file gets its own writer beside `_write_knob_effects`; `_reject_foreign_corpora(name, allowed, rows)` reads `row["corpus"]` and `row["cell"]`; `measured_on_summary(rows)`; `_round`; `_cache_root()`; `unscored_cells()` unions the sources of `_EXPORTS` only, so a writer outside `_EXPORTS` changes no unscored count.
- `scripts/gen_bench_tables.py`: `_table(title, columns, rows, note="")`, `_verdict_cell(paired)`, `_SIMPLE_SOURCES` tuple of `(raw name, registrar)`, `_load(name)`. The generator and its tests walk every `.md` under `docs/`, so this plan names markers by id and never spells the marker lines out.
- `scripts/gen_bench_charts.py`: `_read(path)`, `_RAW`, `_style()`, `_save(fig, name)`, `_c`, `_BLUE`, `_AQUA`, `_ORANGE`, `_VIOLET`, `_GREEN`, `_RED`, `_muted()`, `_ink()`, `collect_chart_data()` dict literal, `_render_all(data)`. `tests/test_bench_charts_current.py` seeds a fixture raw dir with `shutil.copytree(gen._RAW, raw)` because several collectors read raw files with no missing-file guard.
- The chunking page's settings table (`## What to set`) and the subsection `### What document-level scoring counts but does not deliver` are in `docs/benchmarks/03-chunking.md`; the selection page has no k row, so the write-up touches the chunking page only (a correction to the design's section D).
- `/embeddings/vectors` holds, at zero overlap: MLDR English `recursive` at caps 64, 128, 256, 512 and eleven cap-256 profiles (`fast`, `late`, `markdown`, `recursive`, `semantic` plus five breakpoint variants, `whitespace`); GerDaLIR `recursive` at 64, 128, 256; MLDR German at 256 and 512. Query-vector caches exist per (corpus, embedder) from the chunk-sweep scoring.

---

### Task 1: The metric core of `scripts/score_product_k.py`

**Files:**
- Create: `scripts/score_product_k.py` (module docstring, constants, the four pure functions)
- Modify: `scripts/score_chunk_sweep.py:456-457` (export two helpers the new scorer imports)
- Test: `tests/test_score_product_k.py`

**Interfaces:**
- Consumes: `preembed_vectors._ndcg(ranked: list[str], rels: dict[str, int], k: int) -> float`; `score_chunk_sweep._qvec_path(corpus, label, queries) -> Path` and `_load_uris(parquet) -> list[str]`, exported here as `qvec_path` and `load_uris`.
- Produces: `delivered_view(chunk_docs: list[str], k: int) -> list[str]`, `ndcg_delivered(chunk_docs, rels, k) -> float`, `ndcg_documents(chunk_docs, rels, k) -> float`, `distinct_docs(chunk_docs, k) -> int`, `rungs_for_cap(cap: int, *, product_k: int, budgets: list[int]) -> list[Rung]` with `Rung` a frozen dataclass `(k: int, budgets: tuple[int, ...], product: bool)`; constants `LADDER_CAPS = (64, 128, 256, 512)`, `DEFAULT_PRODUCT_K = 5`, `DEFAULT_BUDGETS = (1280, 2560)`, `DEFAULT_FETCH = 100`, `REPEAT = ""` (the sentinel a repeated document becomes in the delivered view).

**Out of scope** - do NOT touch, though they look related:
- `scripts/score_chunk_sweep.py` - Task 2 exports two of its helpers; nothing else in it changes.
- `scripts/score_span_integrity.py` - measures answer spans on other corpora; the design keeps it as is.

**STOP conditions** - stop and report rather than improvise, if:
- `preembed_vectors._ndcg` does not have the signature above;
- `score_chunk_sweep.py` lines 456-457 are not `query_vectors = _query_vectors` and `__all__ = ["dirsafe", "metrics", "query_vectors"]`, or `_qvec_path` / `_load_uris` are missing there;
- a step's verification fails twice after one reasonable fix attempt;
- the change turns out to need a file that is not listed in **Files:**.

- [ ] **Step 1: Write the failing test**

The header carries the pyarrow facade Task 2's fixture needs, so Task 2 only appends tests.

```python
"""The product-k scorer: delivered against documents nDCG on hand-built and fixture rankings.

A chunk ranking is a list of the DOCUMENT each ranked chunk belongs to, in rank order. The
delivered view is what ``semdex search`` hands back: k slots, a document may fill several. The
documents view is what every other page measures: the list deduplicated to k distinct documents.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa  # pyright: ignore[reportMissingTypeStubs] - no stubs; typed at the facade below
import pyarrow.parquet as pq  # pyright: ignore[reportMissingTypeStubs] - see above
import pytest

# Typed facade over the two unstubbed pyarrow calls this test needs. Declaring the signatures
# under TYPE_CHECKING types every call site below without suppressing anything: the checker reads
# these declarations, the runtime binds the real functions. Drop it when pyarrow ships stubs.
if TYPE_CHECKING:

    def _arrow_table(columns: dict[str, list[str]]) -> Any: ...

    def _write_parquet(table: Any, where: Path) -> None: ...

else:
    _arrow_table = pa.table
    _write_parquet = pq.write_table

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "score_product_k.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("score_product_k", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["score_product_k"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def core() -> Any:
    return _load()


def test_a_repeated_document_becomes_the_sentinel_in_the_delivered_view(core: Any) -> None:
    assert core.delivered_view(["a", "a", "b", "a", "c"], 5) == ["a", core.REPEAT, "b", core.REPEAT, "c"]
    assert core.delivered_view(["a", "b", "c"], 2) == ["a", "b"]


def test_one_document_filling_every_slot_earns_only_its_first_slot(core: Any) -> None:
    rels = {"a": 1}
    assert core.ndcg_delivered(["a", "a", "a", "a", "a"], rels, 5) == pytest.approx(1.0)
    assert core.ndcg_documents(["a", "a", "a", "a", "a"], rels, 5) == pytest.approx(1.0)
    two = {"a": 1, "b": 1}
    # Delivered: only slot 1 gains, the ideal has two gains -> 1 / (1 + 1/log2(3)).
    ideal = 1.0 + 1.0 / math.log2(3)
    assert core.ndcg_delivered(["a", "a", "a", "a", "a"], two, 5) == pytest.approx(1.0 / ideal)
    # Documents: dedup leaves ["a"], the same single gain against the same ideal.
    assert core.ndcg_documents(["a", "a", "a", "a", "a"], two, 5) == pytest.approx(1.0 / ideal)


def test_interleaved_documents_score_the_same_both_ways_when_no_slot_is_wasted(core: Any) -> None:
    rels = {"a": 1, "b": 1}
    ranking = ["a", "b", "c", "d", "e"]
    assert core.ndcg_delivered(ranking, rels, 5) == pytest.approx(core.ndcg_documents(ranking, rels, 5))
    assert core.ndcg_delivered(ranking, rels, 5) == pytest.approx(1.0)


def test_a_relevant_document_first_seen_past_k_counts_only_for_documents(core: Any) -> None:
    rels = {"b": 1}
    # Delivered cuts at k=2: ["a", REPEAT] -> no gain. Documents dedups first: ["a", "b"] -> b at rank 2.
    ranking = ["a", "a", "b"]
    assert core.ndcg_delivered(ranking, rels, 2) == pytest.approx(0.0)
    assert core.ndcg_documents(ranking, rels, 2) == pytest.approx(1.0 / math.log2(3))


def test_empty_qrels_score_zero_and_distinct_docs_counts_names(core: Any) -> None:
    assert core.ndcg_delivered(["a", "b"], {}, 2) == 0.0
    assert core.ndcg_documents(["a", "b"], {}, 2) == 0.0
    assert core.distinct_docs(["a", "a", "b", "c", "c"], 5) == 3
    assert core.distinct_docs(["a", "a", "b", "c", "c"], 2) == 1


def test_the_rung_table_follows_the_budgets(core: Any) -> None:
    def rungs(cap: int) -> list[tuple[int, tuple[int, ...], bool]]:
        return [(r.k, r.budgets, r.product) for r in core.rungs_for_cap(cap, product_k=5, budgets=[1280, 2560])]

    assert rungs(64) == [(5, (), True), (20, (1280,), False), (40, (2560,), False)]
    assert rungs(128) == [(5, (), True), (10, (1280,), False), (20, (2560,), False)]
    assert rungs(256) == [(5, (1280,), True), (10, (2560,), False)]
    assert rungs(512) == [(2, (1280,), False), (5, (2560,), True)]
    assert rungs(1024) == [(5, (), True)], "a cap outside the ladder gets the product k only"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `env -u VIRTUAL_ENV .venv/bin/python -m pytest tests/test_score_product_k.py -q -p no:cacheprovider`
Expected: every test errors with `FileNotFoundError` or `AttributeError` because `scripts/score_product_k.py` does not exist. (ruff will flag `json`, `np`, and the facade as unused until Task 2; that is expected and is cleared there, so Step 5 below checks ruff on the script only.)

- [ ] **Step 3: Export the two helpers from `score_chunk_sweep.py`**

Replace lines 456-457 with:

```python
query_vectors = _query_vectors
qvec_path = _qvec_path
load_uris = _load_uris
__all__ = ["dirsafe", "load_uris", "metrics", "query_vectors", "qvec_path"]
```

- [ ] **Step 3b: Write the module with the metric core**

Create `scripts/score_product_k.py`:

```python
#!/usr/bin/env python3
"""Score the chunk list ``semdex search`` delivers, not the deduplicated document list.

Every chunking verdict in docs/benchmarks/03-chunking.md is nDCG@10 over DOCUMENTS: the chunk-sweep
scorer over-fetches chunks, deduplicates to ten distinct documents, and scores that. The product
returns ``default_k`` chunks with no deduplication, so five slots can fill with one document's
neighbouring chunks where the document metric sees one hit. This scorer measures both views from
the SAME top list per query, at the product's k and at fixed retrieved-token budgets, so a chunk
size can be compared at equal tokens read.

Model-free by design: it reads the query-vector cache the chunk-sweep scorer wrote and REFUSES a
cell whose cache is absent, so it never loads an embedder and can share a box with a sweep.

Env (all optional):
  SEMDEX_PRODUCT_CORPORA     comma list (default gerdalir_de_12k_slice,mldr_de_3k_slice,mldr_en_8k_slice)
  SEMDEX_PRODUCT_PROFILES    comma list of profile tags (default: every cap-256 zero-overlap profile
                             present for the corpus, plus recursive-t64/t128/t512-o0-gpt2)
  SEMDEX_PRODUCT_EMBEDDINGS  comma list of embedder labels (default: the six sweep embedders)
  SEMDEX_PRODUCT_K           the product's k (default 5, the shipped [index].default_k)
  SEMDEX_PRODUCT_BUDGETS     comma list of retrieved-token budgets (default 1280,2560)
  SEMDEX_PRODUCT_FETCH       chunks streamed per query (default 100; raised to the largest rung k)
  SEMDEX_PRODUCT_FORCE       1 to re-score cells already in the results file
  SEMDEX_SCORE_OUT           results json (default <cache>/scores/product_k_scores.json)
  SEMDEX_SCORE_PERQUERY      per-query npz dir (default <cache>/scores/perquery)
  SEMDEX_SCORE_QVECS         query-vector cache dir (default <cache>/scores/qvecs)
  SEMDEX_SCORE_BLOCK_BYTES   streaming block size for topk_stream
  CACHE_ROOT                 the vector cache root (default /embeddings)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import the sibling driver helpers

from _provenance import stamped
from _score_kernel import topk_stream
from _score_stats import bootstrap_ci, paired_ci
from audit_chunk_dimensions import parse_profile
from preembed_vectors import _cache_root, _ndcg, _read_json
from score_chunk_sweep import dirsafe, load_uris, qvec_path

LADDER_CAPS = (64, 128, 256, 512)
DEFAULT_PRODUCT_K = 5
DEFAULT_BUDGETS = (1280, 2560)
DEFAULT_FETCH = 100
DEFAULT_CORPORA = ("gerdalir_de_12k_slice", "mldr_de_3k_slice", "mldr_en_8k_slice")
DEFAULT_EMBEDDINGS = (
    "model2vec:potion-retrieval-32M",
    "model2vec:potion-base-8M",
    "fastembed:bge-base",
    "ollama:bge-m3",
    "ollama:qwen3-embedding-4b",
    "ollama:qwen3-embedding-8b",
)
LADDER_PROFILES = ("recursive-t64-o0-gpt2", "recursive-t128-o0-gpt2", "recursive-t512-o0-gpt2")
# A repeated document's later slot in the delivered view. Never a document id, so it earns no gain.
REPEAT = ""
_DEFAULT_BLOCK_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class Rung:
    """One k at which a cell is scored, and why: a budget rung, the product k, or both."""

    k: int
    budgets: tuple[int, ...]
    product: bool


def delivered_view(chunk_docs: list[str], k: int) -> list[str]:
    """The first k chunks as the product hands them back: a repeated document becomes REPEAT."""
    seen: set[str] = set()
    view: list[str] = []
    for doc in chunk_docs[:k]:
        view.append(REPEAT if doc in seen else doc)
        seen.add(doc)
    return view


def _documents_view(chunk_docs: list[str], k: int) -> list[str]:
    """The ranking deduplicated to k distinct documents, in order (the chunk-sweep scorer's unit)."""
    seen: list[str] = []
    for doc in chunk_docs:
        if doc not in seen:
            seen.append(doc)
            if len(seen) >= k:
                break
    return seen


def ndcg_delivered(chunk_docs: list[str], rels: dict[str, int], k: int) -> float:
    """nDCG@k over the delivered view: a slot spent on a document already delivered gains nothing."""
    return _ndcg(delivered_view(chunk_docs, k), rels, k)


def ndcg_documents(chunk_docs: list[str], rels: dict[str, int], k: int) -> float:
    """nDCG@k over k distinct documents, as every other chunking table scores."""
    return _ndcg(_documents_view(chunk_docs, k), rels, k)


def distinct_docs(chunk_docs: list[str], k: int) -> int:
    """How many distinct documents the first k chunks name."""
    return len(set(chunk_docs[:k]))


def rungs_for_cap(cap: int, *, product_k: int, budgets: list[int]) -> list[Rung]:
    """The ks a cell of this cap is scored at, sorted by k: the product k plus each budget's k."""
    by_k: dict[int, list[int]] = {product_k: []}
    if cap in LADDER_CAPS:
        for budget in budgets:
            k = budget // cap
            if k >= 1:
                by_k.setdefault(k, []).append(budget)
    return [Rung(k=k, budgets=tuple(sorted(by_k[k])), product=(k == product_k)) for k in sorted(by_k)]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `env -u VIRTUAL_ENV .venv/bin/python -m pytest tests/test_score_product_k.py tests/test_bench_score_out_guard.py tests/test_bench_score_query_variant.py -q -p no:cacheprovider`
Expected: `6 passed` from the new module and the sibling's guard tests unchanged.

- [ ] **Step 5: Gates**

Run: `env -u VIRTUAL_ENV .venv/bin/ruff check scripts/score_product_k.py scripts/score_chunk_sweep.py && env -u VIRTUAL_ENV .venv/bin/ruff format --check scripts/score_product_k.py scripts/score_chunk_sweep.py tests/test_score_product_k.py && env -u VIRTUAL_ENV .venv/bin/pyright --pythonpath .venv/bin/python scripts/score_product_k.py scripts/score_chunk_sweep.py tests/test_score_product_k.py`
Expected: `All checks passed!`, `3 files already formatted`, `0 errors`. The test file's unused imports (`json`, `np`, the facade) are the only ruff findings allowed at this task and are consumed by Task 2; if ruff's F401 fires on them, keep them and note it in the report rather than deleting the facade.

- [ ] **Step 6: Commit**

Subject: `bench(product-k): [27] metric core for the delivered chunk list`. Message file, pathspec `scripts/score_product_k.py scripts/score_chunk_sweep.py tests/test_score_product_k.py`, no trailers.

---

### Task 2: The cell scorer and main of `scripts/score_product_k.py`

**Files:**
- Modify: `scripts/score_product_k.py` (append the cell scorer, the summary, the per-query writer, discovery and `main`)
- Test: `tests/test_score_product_k.py` (append a fixture writer and two tests)

**Interfaces:**
- Consumes: Task 1's core and its exports `qvec_path`, `load_uris`; `topk_stream`; `bootstrap_ci`, `paired_ci`; `stamped`; `parse_profile`.
- Produces: `cached_query_vectors(corpus, label, queries) -> tuple[list[str], np.ndarray]` (refuses on a miss); `score_cell(corpus, profile, label, *, product_k, budgets, fetch) -> dict[str, Any] | None`; the results file shape: `{cell: {corpus, profile, embedding, dim, n_queries, product_k, fetch, rungs: [...], perquery_sha256, measured_on}}` where each rung is `{k, budgets, product, ndcg_delivered, ndcg_delivered_ci_lo, ndcg_delivered_ci_hi, ndcg_delivered_sd, ndcg_documents (+ the same three), distinct_docs (+ the same three), gap_mean_delta, gap_ci_lo, gap_ci_hi, gap_wins, gap_losses, gap_ties, gap_n_shared, gap_resolved}`; the per-query npz `{qids, ks, delivered (n_queries x n_rungs), documents, distinct}`.

**Out of scope** - do NOT touch, though they look related:
- `scripts/score_chunk_sweep.py` - Task 1 exported what this task needs; its guards and defaults stay.
- `scripts/export_bench_raw.py` - Task 3.

**STOP conditions** - stop and report rather than improvise, if:
- `scripts/score_product_k.py` does not import `qvec_path` and `load_uris` from `score_chunk_sweep` (Task 1 not landed);
- a step's verification fails twice after one reasonable fix attempt.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_score_product_k.py`; the imports and the pyarrow facade are already in the module header from Task 1)

```python
_CORPUS = "mldr_en_8k_slice"
_PROFILE = "recursive-t256-o0-gpt2"
_LABEL = "fastembed:bge-base"


def _write_cell(cache: Path, core: Any, *, with_query_cache: bool) -> None:
    """A four-chunk cell over two documents, three queries, and its query-vector cache."""
    cell = f"{_CORPUS}__{_PROFILE}__{core.dirsafe(_LABEL)}"
    vroot = cache / "vectors" / cell
    vroot.mkdir(parents=True)
    # doc-a has three chunks, doc-b one; unit vectors so cosine equals the dot product.
    vectors = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(vroot / "vectors.npy", vectors)
    (vroot / "meta.json").write_text(json.dumps({"count": 4, "dim": 2, "model_id": "BAAI/bge-base-en-v1.5"}))
    chunks = cache / "chunks" / f"{_CORPUS}__{_PROFILE}"
    chunks.mkdir(parents=True)
    _write_parquet(_arrow_table({"source_uri": ["doc-a", "doc-a", "doc-a", "doc-b"]}), chunks / "chunks.parquet")
    queries = {"q1": "about a", "q2": "about b", "q3": "about both"}
    (chunks / "queries.json").write_text(json.dumps(queries))
    (chunks / "qrels.json").write_text(
        json.dumps({"q1": {"doc-a": 1}, "q2": {"doc-b": 1}, "q3": {"doc-a": 1, "doc-b": 1}})
    )
    if with_query_cache:
        path = core.qvec_path(_CORPUS, _LABEL, queries)
        path.parent.mkdir(parents=True, exist_ok=True)
        qvecs = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
        with path.open("wb") as handle:
            np.savez(handle, qids=np.asarray(sorted(queries)), vectors=qvecs)


def test_a_cell_without_a_query_vector_cache_is_refused_by_name(
    core: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    _write_cell(tmp_path, core, with_query_cache=False)
    with pytest.raises(SystemExit, match=r"mldr_en_8k_slice__fastembed-bge-base: no query-vector cache"):
        core.score_cell(_CORPUS, _PROFILE, _LABEL, product_k=2, budgets=[512], fetch=4)


def test_a_fixture_cell_scores_both_views_from_one_top_list(
    core: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("SEMDEX_SCORE_PERQUERY", str(tmp_path / "perquery"))
    _write_cell(tmp_path, core, with_query_cache=True)
    row = core.score_cell(_CORPUS, _PROFILE, _LABEL, product_k=2, budgets=[512], fetch=4)
    assert row is not None
    assert row["corpus"] == _CORPUS and row["embedding"] == _LABEL and row["n_queries"] == 3
    rungs = {r["k"]: r for r in row["rungs"]}
    assert sorted(rungs) == [2], "cap 256 at budget 512 is k=2, which is also the product k"
    two = rungs[2]
    assert two["product"] is True and two["budgets"] == [512]
    # q2 asks for doc-b: the top-2 chunks are doc-b then doc-a. Both views score 1.0.
    # q1 asks for doc-a: top-2 are two doc-a chunks. Delivered wastes slot 2, documents is fine: both 1.0.
    # q3 wants both: top-2 delivered are doc-a, doc-a (second slot wasted), documents dedups to doc-a, doc-b.
    assert two["ndcg_documents"] > two["ndcg_delivered"]
    assert two["gap_mean_delta"] == pytest.approx(two["ndcg_documents"] - two["ndcg_delivered"])
    assert two["gap_wins"] == 1 and two["gap_losses"] == 0
    assert two["distinct_docs"] == pytest.approx((1 + 2 + 1) / 3)
    npz = tmp_path / "perquery" / f"{_CORPUS}__{_PROFILE}__{core.dirsafe(_LABEL)}.npz"
    with np.load(npz) as data:
        assert list(data["ks"]) == [2]
        assert data["delivered"].shape == (3, 1) and data["documents"].shape == (3, 1)
    assert len(row["perquery_sha256"]) == 64
```

- [ ] **Step 2: Run them to verify they fail**

Run: `env -u VIRTUAL_ENV .venv/bin/python -m pytest tests/test_score_product_k.py -q -p no:cacheprovider`
Expected: `2 failed, 6 passed`, both with `AttributeError: module 'score_product_k' has no attribute 'score_cell'`.

- [ ] **Step 3: Append the cell scorer, summary, per-query writer, discovery and main**

Append to `scripts/score_product_k.py`:

```python
def cached_query_vectors(corpus: str, label: str, queries: dict[str, str]) -> tuple[list[str], np.ndarray]:
    """The query vectors the chunk-sweep scorer cached for this (corpus, embedder); refused if absent.

    Reading the npz directly rather than through ``score_chunk_sweep.query_vectors`` is the point:
    that seam EMBEDS on a miss, which would load a model on a box that may be running a sweep.
    """
    path = qvec_path(corpus, label, queries)
    qids = sorted(queries)
    if not path.exists():
        raise SystemExit(
            f"{corpus}__{dirsafe(label)}: no query-vector cache at {path}; score this corpus with "
            "scripts/score_chunk_sweep.py first (this scorer never embeds)"
        )
    with np.load(path) as data:
        cached = [str(q) for q in data["qids"]]
        if cached != qids:
            raise SystemExit(f"{corpus}__{dirsafe(label)}: query-vector cache at {path} holds a different query set")
        return qids, np.asarray(data["vectors"], dtype=np.float32)


def _per_query_at(chunk_docs: list[str], rels: dict[str, int], k: int) -> dict[str, float]:
    return {
        "delivered": ndcg_delivered(chunk_docs, rels, k),
        "documents": ndcg_documents(chunk_docs, rels, k),
        "distinct": float(distinct_docs(chunk_docs, k)),
    }


def _summarize_rung(rung: Rung, per_query: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Means with intervals for the three views, and the paired documents-minus-delivered gap."""
    out: dict[str, Any] = {"k": rung.k, "budgets": list(rung.budgets), "product": rung.product}
    for view, key in (("delivered", "ndcg_delivered"), ("documents", "ndcg_documents"), ("distinct", "distinct_docs")):
        stats = bootstrap_ci([scores[view] for scores in per_query.values()])
        out[key] = stats["mean"]
        out[f"{key}_ci_lo"] = stats["ci_lo"]
        out[f"{key}_ci_hi"] = stats["ci_hi"]
        out[f"{key}_sd"] = stats["sd"]
    gap = paired_ci(
        {qid: scores["documents"] for qid, scores in per_query.items()},
        {qid: scores["delivered"] for qid, scores in per_query.items()},
    )
    for key in ("mean_delta", "ci_lo", "ci_hi", "wins", "losses", "ties", "n_shared", "resolved"):
        out[f"gap_{key}"] = gap[key]
    return out


def _write_per_query(
    cell: str, qids: list[str], rungs: list[Rung], per_rung: dict[int, dict[str, dict[str, float]]]
) -> str:
    """Per-query arrays beside the cache, one column per rung, referenced from the row by hash."""
    root = Path(os.environ.get("SEMDEX_SCORE_PERQUERY", str(_cache_root() / "scores" / "perquery")))
    root.mkdir(parents=True, exist_ok=True)
    ks = [rung.k for rung in rungs]

    def matrix(view: str) -> np.ndarray:
        return np.asarray([[per_rung[k][qid][view] for k in ks] for qid in qids], dtype=np.float32)

    path = root / f"{cell}.npz"
    with path.open("wb") as handle:
        np.savez(
            handle,
            qids=np.asarray(qids),
            ks=np.asarray(ks, dtype=np.int32),
            delivered=matrix("delivered"),
            documents=matrix("documents"),
            distinct=matrix("distinct"),
        )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_cell(
    corpus: str, profile: str, label: str, *, product_k: int, budgets: list[int], fetch: int
) -> dict[str, Any] | None:
    """Both views at every rung of one cell, from ONE streamed top list per query."""
    cell = f"{corpus}__{profile}__{dirsafe(label)}"
    vroot = _cache_root() / "vectors" / cell
    meta = _read_json(vroot / "meta.json")
    if not meta or "count" not in meta:
        return None  # not embedded yet
    chunks_dir = _cache_root() / "chunks" / f"{corpus}__{profile}"
    queries = _read_json(chunks_dir / "queries.json") or {}
    qrels = _read_json(chunks_dir / "qrels.json") or {}
    qids, query_matrix = cached_query_vectors(corpus, label, queries)
    uris = load_uris(chunks_dir / "chunks.parquet")
    rows = int(np.load(vroot / "vectors.npy", mmap_mode="r").shape[0])  # header read, no data
    if len(uris) != rows:
        raise SystemExit(f"{cell}: uri/vector row mismatch ({len(uris)} vs {rows})")
    axes = parse_profile(profile)
    if "max_tokens" not in axes:
        raise SystemExit(f"{cell}: {axes.get('parse_error', 'profile has no max_tokens axis')}")
    rungs = rungs_for_cap(int(axes["max_tokens"]), product_k=product_k, budgets=budgets)
    fetch = max(fetch, *(rung.k for rung in rungs))
    block_bytes = int(os.environ.get("SEMDEX_SCORE_BLOCK_BYTES", _DEFAULT_BLOCK_BYTES))
    index, _ = topk_stream(vroot / "vectors.npy", query_matrix, fetch=fetch, block_bytes=block_bytes)
    ranked = {qid: [uris[int(r)] for r in index[i]] for i, qid in enumerate(qids)}
    per_rung = {rung.k: {qid: _per_query_at(ranked[qid], qrels.get(qid, {}), rung.k) for qid in qids} for rung in rungs}
    row: dict[str, Any] = {
        "corpus": corpus,
        "profile": profile,
        "embedding": label,
        "dim": int(meta["dim"]),
        "n_queries": len(qids),
        "product_k": product_k,
        "fetch": fetch,
        "rungs": [_summarize_rung(rung, per_rung[rung.k]) for rung in rungs],
    }
    row["perquery_sha256"] = _write_per_query(cell, qids, rungs, per_rung)
    return row


def _discover_profiles(corpus: str) -> list[str]:
    """Every cap-256 zero-overlap profile present for the corpus, plus the recursive ladder rungs."""
    found: set[str] = set()
    for entry in (_cache_root() / "vectors").glob(f"{corpus}__*"):
        parts = entry.name.split("__")
        if len(parts) != 3:
            continue
        axes = parse_profile(parts[1])
        if axes.get("max_tokens") == 256 and axes.get("overlap_tokens") == 0:
            found.add(parts[1])
        if parts[1] in LADDER_PROFILES:
            found.add(parts[1])
    return sorted(found)


def _env_list(name: str, default: tuple[str, ...]) -> list[str]:
    raw = os.environ.get(name)
    return [item for item in raw.split(",") if item] if raw else list(default)


def main() -> None:
    cache = _cache_root()
    corpora = _env_list("SEMDEX_PRODUCT_CORPORA", DEFAULT_CORPORA)
    embeddings = _env_list("SEMDEX_PRODUCT_EMBEDDINGS", DEFAULT_EMBEDDINGS)
    profiles_env = os.environ.get("SEMDEX_PRODUCT_PROFILES")
    product_k = int(os.environ.get("SEMDEX_PRODUCT_K", str(DEFAULT_PRODUCT_K)))
    budgets = [int(b) for b in _env_list("SEMDEX_PRODUCT_BUDGETS", tuple(str(b) for b in DEFAULT_BUDGETS))]
    fetch = int(os.environ.get("SEMDEX_PRODUCT_FETCH", str(DEFAULT_FETCH)))
    force = os.environ.get("SEMDEX_PRODUCT_FORCE") == "1"
    out = Path(os.environ.get("SEMDEX_SCORE_OUT", str(cache / "scores" / "product_k_scores.json")))
    out.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = _read_json(out) or {}
    for corpus in corpora:
        profiles = profiles_env.split(",") if profiles_env else _discover_profiles(corpus)
        for label in embeddings:
            for profile in profiles:
                cell = f"{corpus}__{profile}__{dirsafe(label)}"
                if cell in results and not force:
                    print(f"[product-k] skip (cached) {cell}", flush=True)
                    continue
                row = score_cell(corpus, profile, label, product_k=product_k, budgets=budgets, fetch=fetch)
                if row is None:
                    print(f"[product-k] not-embedded {cell}", flush=True)
                    continue
                results[cell] = stamped(row)
                product = next(r for r in row["rungs"] if r["product"])
                print(
                    f"[product-k] {cell}: documents@{product_k}={product['ndcg_documents']:.4f} "
                    f"delivered@{product_k}={product['ndcg_delivered']:.4f} "
                    f"gap={product['gap_mean_delta']:+.4f} "
                    f"[{product['gap_ci_lo']:+.4f}, {product['gap_ci_hi']:+.4f}] "
                    f"distinct={product['distinct_docs']:.2f} (n={row['n_queries']})",
                    flush=True,
                )
                out.write_text(json.dumps(results, indent=2, sort_keys=True))  # persist incrementally
    print(f"[product-k] {len(results)} cells in {out}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `env -u VIRTUAL_ENV .venv/bin/python -m pytest tests/test_score_product_k.py -q -p no:cacheprovider`
Expected: `8 passed`.

- [ ] **Step 5: The stamp detector sees the new scorer, and the sibling's tests still pass**

Run: `env -u VIRTUAL_ENV .venv/bin/python -m pytest tests/test_bench_scorers_stamp_provenance.py tests/test_bench_score_out_guard.py tests/test_bench_score_query_variant.py -q -p no:cacheprovider`
Expected: all pass. `test_the_detector_finds_the_known_cell_producers` lists the known producers with `>=`-style membership, not equality; if it asserts an exact set, ADD `"score_product_k.py"` to that expected set in the same commit and say so in the report.

- [ ] **Step 6: Gates**

Run: `env -u VIRTUAL_ENV .venv/bin/ruff check scripts/score_product_k.py tests/test_score_product_k.py && env -u VIRTUAL_ENV .venv/bin/ruff format --check scripts/score_product_k.py tests/test_score_product_k.py && env -u VIRTUAL_ENV .venv/bin/pyright --pythonpath .venv/bin/python scripts/score_product_k.py tests/test_score_product_k.py`
Expected: `All checks passed!`, `2 files already formatted`, `0 errors`.

- [ ] **Step 7: Commit**

Subject: `bench(product-k): [27] score both views at every rung from one top list`. Pathspec: `scripts/score_product_k.py tests/test_score_product_k.py` (plus `tests/test_bench_scorers_stamp_provenance.py` only if Step 5 required it).

---

### Task 3: The exporter writes `product-k.json` as flat rows

**Files:**
- Modify: `scripts/export_bench_raw.py` (constants after `_EXPORTS`; `product_k_rows`, `product_k_payload`, `_write_product_k` after `_write_knob_effects`; one call in `main`)
- Test: `tests/test_bench_export_product_k.py`

**Interfaces:**
- Consumes: Task 2's results-file shape (one entry per cell with a nested `rungs` list); `_reject_foreign_corpora`, `measured_on_summary`, `_round`, `_cache_root`, `MEASURED_ON`, `parse_profile`.
- Produces: `product_k_rows(cell: str, row: dict[str, Any]) -> list[dict[str, Any]]`, ONE flat row per (cell, rung): `{cell, corpus, profile, embedding, dim, n_queries, axes, product_k, k, budgets, product, ndcg_delivered (+_ci_lo/_ci_hi/_sd), ndcg_documents (+3), distinct_docs (+3), gap_mean_delta, gap_ci_lo, gap_ci_hi, gap_wins, gap_losses, gap_ties, gap_n_shared, gap_resolved, perquery_sha256?, measured_on?}`; `product_k_payload(cells) -> {measured_on, note, sources, corpora, product_k, budgets, summary: {cells, rows, corpora, profiles, embeddings}, rows}`; constants `_PRODUCT_K_FILE = "product-k.json"`, `_PRODUCT_K_SOURCE = "product_k_scores.json"`, `_PRODUCT_K_CORPORA = ("gerdalir_de_12k_slice", "mldr_de_3k_slice", "mldr_en_8k_slice")`. Flat because the claim checker's dotted getter walks dicts only: a figure inside a nested list could never carry a claim.

**Out of scope** - do NOT touch, though they look related:
- `_EXPORTS` and `export_row` - the product-k rows have a different metric set; adding the file there would print null chunk-sweep metrics for every cell.
- `unscored_cells` - it unions `_EXPORTS` sources on purpose; the product-k writer scores a subset of cells and must not shrink the census.

**STOP conditions** - stop and report rather than improvise, if:
- `_write_knob_effects` or `main` in `export_bench_raw.py` differ from the ground-truth section (the `_write_knob_effects(out_dir)` call followed by `report_unscored`);
- a step's verification fails twice after one reasonable fix attempt.

- [ ] **Step 1: Write the failing test**

```python
"""The product-k export: flat rows per cell and rung, the corpus allowlist, rounding, carried provenance."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "export_bench_raw.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("export_bench_raw_product_k", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exporter() -> Any:
    return _load()


def _rung(k: int, *, product: bool, budgets: list[int]) -> dict[str, Any]:
    return {
        "k": k,
        "budgets": budgets,
        "product": product,
        "ndcg_delivered": 0.512345678,
        "ndcg_delivered_ci_lo": 0.5,
        "ndcg_delivered_ci_hi": 0.52,
        "ndcg_delivered_sd": 0.1,
        "ndcg_documents": 0.6,
        "ndcg_documents_ci_lo": 0.59,
        "ndcg_documents_ci_hi": 0.61,
        "ndcg_documents_sd": 0.1,
        "distinct_docs": 3.25,
        "distinct_docs_ci_lo": 3.1,
        "distinct_docs_ci_hi": 3.4,
        "distinct_docs_sd": 0.5,
        "gap_mean_delta": 0.087654321,
        "gap_ci_lo": 0.08,
        "gap_ci_hi": 0.095,
        "gap_wins": 300,
        "gap_losses": 100,
        "gap_ties": 400,
        "gap_n_shared": 800,
        "gap_resolved": True,
    }


def _cell(corpus: str, profile: str = "recursive-t256-o0-gpt2") -> dict[str, Any]:
    return {
        "corpus": corpus,
        "profile": profile,
        "embedding": "fastembed:bge-base",
        "dim": 768,
        "n_queries": 800,
        "product_k": 5,
        "fetch": 100,
        "rungs": [_rung(5, product=True, budgets=[1280]), _rung(10, product=False, budgets=[2560])],
        "perquery_sha256": "ab" * 32,
        "measured_on": {"host": "bench-box", "measured_utc": "2026-09-30T00:00:00Z"},
    }


def test_the_payload_flattens_rungs_rounds_them_parses_axes_and_carries_the_stamp(exporter: Any) -> None:
    cells = {"mldr_en_8k_slice__recursive-t256-o0-gpt2__fastembed-bge-base": _cell("mldr_en_8k_slice")}
    payload = exporter.product_k_payload(cells)
    assert payload["sources"] == ["product_k_scores.json"]
    assert payload["corpora"] == list(exporter._PRODUCT_K_CORPORA)
    assert payload["product_k"] == 5 and payload["budgets"] == [1280, 2560]
    assert "delivered" in payload["note"] and "dedup" in payload["note"]
    assert [r["k"] for r in payload["rows"]] == [5, 10], "one flat row per rung, in k order"
    row = payload["rows"][0]
    assert row["cell"] == "mldr_en_8k_slice__recursive-t256-o0-gpt2__fastembed-bge-base"
    assert row["axes"]["max_tokens"] == 256 and row["axes"]["strategy"] == "recursive"
    assert row["product"] is True and row["budgets"] == [1280]
    assert row["ndcg_delivered"] == 0.5123
    assert row["gap_mean_delta"] == 0.0877
    assert row["gap_wins"] == 300
    assert row["measured_on"] == {"host": "bench-box", "measured_utc": "2026-09-30T00:00:00Z"}
    assert "rungs" not in row and "fetch" not in row
    assert payload["measured_on"]["recorded"] == 2, "every flat row carries the cell's stamp"
    assert payload["summary"] == {
        "cells": 1,
        "rows": 2,
        "corpora": ["mldr_en_8k_slice"],
        "profiles": ["recursive-t256-o0-gpt2"],
        "embeddings": ["fastembed:bge-base"],
    }


def test_a_foreign_corpus_is_refused_by_name(exporter: Any) -> None:
    cells = {"nfcorpus__recursive-t256-o0-gpt2__fastembed-bge-base": _cell("nfcorpus")}
    with pytest.raises(ValueError, match=r"product-k\.json declares corpora .* nfcorpus"):
        exporter.product_k_payload(cells)


def test_the_writer_skips_a_missing_source_and_writes_the_file_when_present(
    exporter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path / "cache"))
    out = tmp_path / "raw"
    out.mkdir()
    exporter._write_product_k(out)
    assert not (out / "product-k.json").exists()
    assert "product-k.json: source missing" in capsys.readouterr().out
    scores = tmp_path / "cache" / "scores"
    scores.mkdir(parents=True)
    (scores / "product_k_scores.json").write_text(
        json.dumps(
            {"gerdalir_de_12k_slice__recursive-t256-o0-gpt2__fastembed-bge-base": _cell("gerdalir_de_12k_slice")}
        )
    )
    exporter._write_product_k(out)
    written = json.loads((out / "product-k.json").read_text())
    assert written["summary"] == {
        "cells": 1,
        "rows": 2,
        "corpora": ["gerdalir_de_12k_slice"],
        "profiles": ["recursive-t256-o0-gpt2"],
        "embeddings": ["fastembed:bge-base"],
    }
    assert (out / "product-k.json").read_text().endswith("\n")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `env -u VIRTUAL_ENV .venv/bin/python -m pytest tests/test_bench_export_product_k.py -q -p no:cacheprovider`
Expected: `3 failed` with `AttributeError: ... has no attribute 'product_k_payload'` / `'_write_product_k'`.

- [ ] **Step 3: Add the constants, the row and payload builders, the writer, and the call**

Insert directly after the closing `}` of `_EXPORTS` in `scripts/export_bench_raw.py`:

```python
# The product-k measurement scores the DELIVERED chunk list beside the deduplicated document list
# (scripts/score_product_k.py). Its rows carry per-rung views, not the chunk-sweep metric set, so
# it is written by its own writer below rather than through export_row, and FLAT - one row per
# cell and rung - because the claim checker's dotted getter walks dicts only, so a figure inside a
# nested list could never carry a claim. It is deliberately NOT in _EXPORTS: unscored_cells unions
# those sources, and this file scores a subset of cells by design.
_PRODUCT_K_FILE = "product-k.json"
_PRODUCT_K_SOURCE = "product_k_scores.json"
_PRODUCT_K_CORPORA = ("gerdalir_de_12k_slice", "mldr_de_3k_slice", "mldr_en_8k_slice")
_PRODUCT_K_NOTE = (
    "The product's own unit: semdex search returns default_k chunks with NO dedup, while every "
    "other chunking table scores ten deduplicated documents. One row per cell and rung: nDCG@k "
    "over the delivered chunk list (a repeated document earns zero gain), nDCG@k over k distinct "
    "documents from the SAME top list, the mean distinct documents in the k slots, and the paired "
    "documents-minus-delivered gap. Budget rungs compare chunk sizes at a fixed number of tokens "
    "read (k = budget // cap); the product rung is the shipped default_k."
)
```

Insert directly after `_write_knob_effects`:

```python
_PRODUCT_K_CELL_FIELDS = ("corpus", "profile", "embedding", "dim", "n_queries", "product_k")


def product_k_rows(cell: str, row: dict[str, Any]) -> list[dict[str, Any]]:
    """One committed row per rung of the cell: the cell's fields, the rounded rung, the stamp verbatim."""
    base: dict[str, Any] = {"cell": cell}
    for field in _PRODUCT_K_CELL_FIELDS:
        base[field] = row.get(field)
    base["axes"] = parse_profile(str(row.get("profile", "")))
    if row.get("perquery_sha256"):
        base["perquery_sha256"] = row["perquery_sha256"]
    if isinstance(row.get(MEASURED_ON), dict):
        base[MEASURED_ON] = row[MEASURED_ON]
    rungs = sorted(row.get("rungs", []), key=lambda r: int(r["k"]))
    return [{**base, **{key: _round(value) for key, value in rung.items()}} for rung in rungs]


def product_k_payload(cells: dict[str, Any]) -> dict[str, Any]:
    """The committed product-k file from the scorer's results, guarded by the corpus allowlist."""
    rows = [flat for cell in sorted(cells) for flat in product_k_rows(cell, cells[cell])]
    _reject_foreign_corpora(_PRODUCT_K_FILE, list(_PRODUCT_K_CORPORA), rows)
    budgets = sorted({int(b) for row in rows for b in row.get("budgets", [])})
    product_ks = sorted({int(row["product_k"]) for row in rows if row.get("product_k") is not None})
    return {
        MEASURED_ON: measured_on_summary(rows),
        "note": _PRODUCT_K_NOTE,
        "sources": [_PRODUCT_K_SOURCE],
        "corpora": list(_PRODUCT_K_CORPORA),
        "product_k": product_ks[0] if len(product_ks) == 1 else product_ks,
        "budgets": budgets,
        "summary": {
            "cells": len({str(r["cell"]) for r in rows}),
            "rows": len(rows),
            "corpora": sorted({str(r["corpus"]) for r in rows}),
            "profiles": sorted({str(r["profile"]) for r in rows}),
            "embeddings": sorted({str(r["embedding"]) for r in rows}),
        },
        "rows": rows,
    }


def _write_product_k(out_dir: Path) -> None:
    """Write product-k.json from the scorer's results file, or say plainly that there is none."""
    source = _cache_root() / "scores" / _PRODUCT_K_SOURCE
    if not source.exists():
        print(f"[export] {_PRODUCT_K_FILE}: source missing, not written: {source}", flush=True)
        return
    payload = product_k_payload(json.loads(source.read_text()))
    (out_dir / _PRODUCT_K_FILE).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    summary = payload["summary"]
    print(
        f"[export] {_PRODUCT_K_FILE}: {summary['cells']} cells, {summary['rows']} rows, "
        f"corpora={len(summary['corpora'])} profiles={len(summary['profiles'])} "
        f"embedders={len(summary['embeddings'])}",
        flush=True,
    )
```

In `main`, replace the line `_write_knob_effects(out_dir)` with:

```python
    _write_knob_effects(out_dir)
    try:
        _write_product_k(out_dir)
    except ValueError as exc:
        # Same stance as a rejected export: report, keep going, fail the run at the end.
        print(f"[export] {_PRODUCT_K_FILE}: REJECTED, not written: {exc}", flush=True)
        rejected.append(_PRODUCT_K_FILE)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `env -u VIRTUAL_ENV .venv/bin/python -m pytest tests/test_bench_export_product_k.py tests/test_bench_export_corpus_guard.py tests/test_bench_export_unscored_guard.py -q -p no:cacheprovider`
Expected: all pass (the unscored guard is untouched because the writer is outside `_EXPORTS`).

- [ ] **Step 5: Gates**

Run: `env -u VIRTUAL_ENV .venv/bin/ruff check scripts/export_bench_raw.py tests/test_bench_export_product_k.py && env -u VIRTUAL_ENV .venv/bin/ruff format --check scripts/export_bench_raw.py tests/test_bench_export_product_k.py && env -u VIRTUAL_ENV .venv/bin/pyright --pythonpath .venv/bin/python scripts/export_bench_raw.py tests/test_bench_export_product_k.py`
Expected: clean, `0 errors`.

- [ ] **Step 6: Commit**

Subject: `bench(export): [27] the product-k file gets its own writer, flat rows and a corpus allowlist`.

---

### Task 4: The two table registrars

**Files:**
- Modify: `scripts/gen_bench_tables.py` (add `_register_product_k_tables` and its helpers after `_register_markdown_structure_tables`; register `("product-k.json", _register_product_k_tables)` in `_SIMPLE_SOURCES` after the markdown-structure entry)
- Test: `tests/test_bench_tables_product_k.py`

**Interfaces:**
- Consumes: `_table(title, columns, rows, note="")`, `_verdict_cell(paired)` where `paired` needs keys `mean_delta` and `resolved`; the flat `product-k.json` shape from Task 3.
- Produces: table ids `product_k_verdict` with columns `["Corpus", "Strategy", "Embedder", "Documents nDCG@5", "Delivered nDCG@5", "Cost of no dedup", "95% CI", "Distinct docs in 5"]` (the 5 from the file's `product_k`) and `product_k_budget` with columns `["Corpus", "Embedder", "Cap", "k at 1280", "Delivered nDCG@k, 1280", "k at 2560", "Delivered nDCG@k, 2560"]` (the budgets from the file's `budgets`).

**Out of scope** - do NOT touch, though they look related:
- `docs/benchmarks/03-chunking.md` - the marker pairs are placed in Task 6 after the run, so no doc references the ids yet and the marker tests stay green.
- `_register_markdown_structure_tables` - a sibling; leave it.

**STOP conditions** - stop and report rather than improvise, if:
- `_table` or `_verdict_cell` signatures differ from `def _table(title: str, columns: list[str], rows: list[list[str]], note: str = "") -> dict[str, Any]` and `def _verdict_cell(paired: dict[str, Any]) -> str`;
- a step's verification fails twice after one reasonable fix attempt.

- [ ] **Step 1: Write the failing test**

```python
"""The product-k tables: the k=5 verdict per strategy and embedder, and the budget ladder per cap."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "gen_bench_tables.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("gen_bench_tables_product_k", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator() -> Any:
    return _load()


def _row(
    corpus: str,
    profile: str,
    embedding: str,
    *,
    k: int,
    product: bool,
    budgets: list[int],
    delivered: float,
    documents: float,
) -> dict[str, Any]:
    strategy = profile.split("-")[0]
    cap = int(profile.split("-t")[1].split("-")[0])
    breakpoint_model = profile.rsplit("-bp", 1)[1] if "-bp" in profile else None
    return {
        "cell": f"{corpus}__{profile}__{embedding.replace(':', '-')}",
        "corpus": corpus,
        "profile": profile,
        "embedding": embedding,
        "axes": {"strategy": strategy, "max_tokens": cap, "overlap_tokens": 0, "breakpoint_model": breakpoint_model},
        "product_k": 5,
        "k": k,
        "budgets": budgets,
        "product": product,
        "ndcg_delivered": delivered,
        "ndcg_documents": documents,
        "distinct_docs": 3.2,
        "gap_mean_delta": documents - delivered,
        "gap_ci_lo": documents - delivered - 0.01,
        "gap_ci_hi": documents - delivered + 0.01,
        "gap_resolved": True,
        "gap_wins": 300,
        "gap_losses": 100,
    }


def _doc() -> dict[str, Any]:
    en, de, bge = "mldr_en_8k_slice", "gerdalir_de_12k_slice", "fastembed:bge-base"
    return {
        "product_k": 5,
        "budgets": [1280, 2560],
        "rows": [
            _row(en, "recursive-t256-o0-gpt2", bge, k=5, product=True, budgets=[1280], delivered=0.50, documents=0.55),
            _row(
                en, "recursive-t256-o0-gpt2", bge, k=10, product=False, budgets=[2560], delivered=0.53, documents=0.58
            ),
            _row(
                en,
                "semantic-t256-o0-gpt2-bpbge-m3",
                bge,
                k=5,
                product=True,
                budgets=[1280],
                delivered=0.48,
                documents=0.49,
            ),
            _row(
                en,
                "semantic-t256-o0-gpt2-bpbge-m3",
                bge,
                k=10,
                product=False,
                budgets=[2560],
                delivered=0.50,
                documents=0.52,
            ),
            _row(en, "recursive-t64-o0-gpt2", bge, k=5, product=True, budgets=[], delivered=0.40, documents=0.45),
            _row(en, "recursive-t64-o0-gpt2", bge, k=20, product=False, budgets=[1280], delivered=0.47, documents=0.52),
            _row(en, "recursive-t64-o0-gpt2", bge, k=40, product=False, budgets=[2560], delivered=0.49, documents=0.54),
            _row(de, "recursive-t256-o0-gpt2", bge, k=5, product=True, budgets=[1280], delivered=0.30, documents=0.31),
            _row(
                de, "recursive-t256-o0-gpt2", bge, k=10, product=False, budgets=[2560], delivered=0.32, documents=0.33
            ),
        ],
    }


def test_the_verdict_table_has_one_row_per_cap_256_cell_and_names_the_breakpoint(generator: Any) -> None:
    tables: dict[str, Any] = {}
    generator._register_product_k_tables(tables, _doc())
    table = tables["product_k_verdict"]
    assert table["columns"] == [
        "Corpus",
        "Strategy",
        "Embedder",
        "Documents nDCG@5",
        "Delivered nDCG@5",
        "Cost of no dedup",
        "95% CI",
        "Distinct docs in 5",
    ]
    assert [row[0] for row in table["rows"]] == ["gerdalir-de", "mldr-en", "mldr-en"]
    assert table["rows"][1] == [
        "mldr-en",
        "`recursive`",
        "`fastembed:bge-base`",
        "0.5500",
        "0.5000",
        "+0.0500 resolved",
        "[+0.0400, +0.0600]",
        "3.20",
    ]
    assert table["rows"][2][1] == "`semantic` bp bge-m3"
    assert "zero gain" in table["note"]


def test_the_budget_table_has_one_row_per_recursive_cap_and_names_the_missing_rungs(generator: Any) -> None:
    tables: dict[str, Any] = {}
    generator._register_product_k_tables(tables, _doc())
    table = tables["product_k_budget"]
    assert table["columns"] == [
        "Corpus",
        "Embedder",
        "Cap",
        "k at 1280",
        "Delivered nDCG@k, 1280",
        "k at 2560",
        "Delivered nDCG@k, 2560",
    ]
    assert table["rows"] == [
        ["gerdalir-de", "`fastembed:bge-base`", "256", "5", "0.3000", "10", "0.3200"],
        ["mldr-en", "`fastembed:bge-base`", "64", "20", "0.4700", "40", "0.4900"],
        ["mldr-en", "`fastembed:bge-base`", "256", "5", "0.5000", "10", "0.5300"],
    ]
    assert "gerdalir-de: 64, 128, 512" in table["note"] and "mldr-en: 128, 512" in table["note"]


def test_no_rows_registers_no_table(generator: Any) -> None:
    tables: dict[str, Any] = {}
    generator._register_product_k_tables(tables, {"product_k": 5, "budgets": [1280, 2560], "rows": []})
    assert "product_k_verdict" not in tables and "product_k_budget" not in tables
```

- [ ] **Step 2: Run it to verify it fails**

Run: `env -u VIRTUAL_ENV .venv/bin/python -m pytest tests/test_bench_tables_product_k.py -q -p no:cacheprovider`
Expected: `3 failed`, `AttributeError: ... has no attribute '_register_product_k_tables'`.

- [ ] **Step 3: Add the registrar and register the source**

Insert after `_register_markdown_structure_tables` in `scripts/gen_bench_tables.py`:

```python
_PRODUCT_K_LADDER_CAPS = (64, 128, 256, 512)


def _product_k_corpus(corpus: str) -> str:
    """Body and language, the chart script's convention, so a table row and a chart label agree."""
    parts = corpus.split("_")
    return "-".join(parts[:2]) if len(parts) > 1 else corpus


def _product_k_strategy(axes: dict[str, Any]) -> str:
    strategy = f"`{axes.get('strategy')}`"
    breakpoint_model = axes.get("breakpoint_model")
    return f"{strategy} bp {breakpoint_model}" if breakpoint_model else strategy


def _register_product_k_verdict(tables: dict[str, Any], doc: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    k = int(doc["product_k"])
    out: list[list[str]] = []
    for row in rows:
        axes = row["axes"]
        if not row.get("product") or axes.get("max_tokens") != 256 or axes.get("overlap_tokens") != 0:
            continue
        out.append(
            [
                _product_k_corpus(row["corpus"]),
                _product_k_strategy(axes),
                f"`{row['embedding']}`",
                f"{row['ndcg_documents']:.4f}",
                f"{row['ndcg_delivered']:.4f}",
                _verdict_cell({"mean_delta": row["gap_mean_delta"], "resolved": row["gap_resolved"]}),
                f"[{row['gap_ci_lo']:+.4f}, {row['gap_ci_hi']:+.4f}]",
                f"{row['distinct_docs']:.2f}",
            ]
        )
    if not out:
        return
    out.sort(key=lambda r: (r[0], r[1], r[2]))
    tables["product_k_verdict"] = _table(
        f"The product's {k} slots: delivered chunks against deduplicated documents, paired per query",
        [
            "Corpus",
            "Strategy",
            "Embedder",
            f"Documents nDCG@{k}",
            f"Delivered nDCG@{k}",
            "Cost of no dedup",
            "95% CI",
            f"Distinct docs in {k}",
        ],
        out,
        f"Both views come from the same top list per query at cap 256, zero overlap. `Documents` "
        f"deduplicates to {k} distinct documents, as every other table here does; `Delivered` scores "
        f"the {k} chunks `semdex search` returns, where a repeated document earns zero gain. The cost "
        "is documents minus delivered; positive means the product's no-dedup loses that much nDCG. "
        "`Distinct docs` is the mean number of different documents in the delivered slots.",
    )


def _register_product_k_budget(tables: dict[str, Any], doc: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    budgets = [int(b) for b in doc["budgets"]]
    ladder = [r for r in rows if r["axes"].get("strategy") == "recursive" and r["axes"].get("overlap_tokens") == 0]
    ladder = [r for r in ladder if int(r["axes"]["max_tokens"]) in _PRODUCT_K_LADDER_CAPS]
    if not ladder:
        return
    by_cell: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in ladder:
        key = (_product_k_corpus(row["corpus"]), str(row["embedding"]), int(row["axes"]["max_tokens"]))
        by_cell.setdefault(key, []).append(row)
    out: list[list[str]] = []
    present: dict[tuple[str, str], set[int]] = {}
    for (label, embedding, cap), rungs in sorted(by_cell.items()):
        present.setdefault((label, embedding), set()).add(cap)
        line = [label, f"`{embedding}`", str(cap)]
        for budget in budgets:
            rung = next((r for r in rungs if budget in r.get("budgets", [])), None)
            line.extend([str(rung["k"]), f"{rung['ndcg_delivered']:.4f}"] if rung else ["n/a", "n/a"])
        out.append(line)
    missing = sorted(
        {
            f"{label}: {', '.join(str(c) for c in _PRODUCT_K_LADDER_CAPS if c not in caps)}"
            for (label, _), caps in present.items()
            if any(c not in caps for c in _PRODUCT_K_LADDER_CAPS)
        }
    )
    columns = ["Corpus", "Embedder", "Cap"]
    for budget in budgets:
        columns.extend([f"k at {budget}", f"Delivered nDCG@k, {budget}"])
    tables["product_k_budget"] = _table(
        "Chunk size at a fixed number of tokens read: delivered nDCG@k with k = budget / cap",
        columns,
        out,
        "`recursive`, zero overlap. A consumer reading k chunks of cap tokens reads k times cap "
        "tokens whatever the cap, so each column holds the cap's k at that budget and the delivered "
        "score there. " + (f"Not measured (no cell at that cap): {'; '.join(missing)}." if missing else ""),
    )


def _register_product_k_tables(tables: dict[str, Any], doc: dict[str, Any]) -> None:
    """The product's delivered chunk list against deduplicated documents, and chunk size at a fixed budget."""
    rows = doc.get("rows") or []
    if not rows:
        return
    _register_product_k_verdict(tables, doc, rows)
    _register_product_k_budget(tables, doc, rows)
```

Add to `_SIMPLE_SOURCES` after the `markdown-structure-effect.json` entry:

```python
(("product-k.json", _register_product_k_tables),)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `env -u VIRTUAL_ENV .venv/bin/python -m pytest tests/test_bench_tables_product_k.py tests/test_bench_tables_current.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Gates**

Run: `env -u VIRTUAL_ENV .venv/bin/ruff check scripts/gen_bench_tables.py tests/test_bench_tables_product_k.py && env -u VIRTUAL_ENV .venv/bin/ruff format --check scripts/gen_bench_tables.py tests/test_bench_tables_product_k.py && env -u VIRTUAL_ENV .venv/bin/pyright --pythonpath .venv/bin/python scripts/gen_bench_tables.py tests/test_bench_tables_product_k.py`
Expected: clean, `0 errors`.

- [ ] **Step 6: Commit**

Subject: `bench(chunking): [27] register the product-k verdict and budget tables`.

---

### Task 5: The budget-ladder chart

**Files:**
- Modify: `scripts/gen_bench_charts.py` (add `_product_k_budget()` collector after `_markdown_structure`, a `"product_k_budget"` key in `collect_chart_data`, `render_product_k_budget(d)` after `render_markdown_structure`, and its call in `_render_all`)
- Modify: `docs/benchmarks/img/charts.manifest.json` (regenerated by `scripts/gen_bench_charts.py`; the manifest test requires it)
- Test: extend `tests/test_bench_charts_current.py` with one test

**Interfaces:**
- Consumes: `_read(path)`, `_RAW`, `_style()`, `_c`, `_BLUE`, `_AQUA`, `_ORANGE`, `_VIOLET`, `_GREEN`, `_RED`, `_muted()`, `_ink()`, `_save(fig, name)`; the flat `product-k.json` shape.
- Produces: chart name `chunk_product_k_budget`; collector shape `{"caps": [int], "budgets": [int], "series": {str(budget): {embedding: [float | None per cap]}}}`.

**Out of scope** - do NOT touch, though they look related:
- `render_markdown_structure` / `_markdown_structure` - siblings from item [22].
- PNG files under `docs/benchmarks/img/` - none changes while the raw file is absent; if the run leaves any PNG modified, report it and do not commit it.

**STOP conditions** - stop and report rather than improvise, if:
- `_save` or `_style` do not exist with those names;
- a step's verification fails twice after one reasonable fix attempt.

- [ ] **Step 1: Write the failing test** (append to `tests/test_bench_charts_current.py`)

```python
@pytest.mark.os_agnostic
def test_the_product_k_budget_chart_has_one_series_per_embedder_and_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delivered nDCG@k against cap, one line per embedder, one panel per budget."""
    gen = _load_generator()
    raw = tmp_path / "raw"
    # Seed the fixture dir from the real raw tree so every collector has all its input files;
    # several collectors have no missing-file guard and crash on a fixture-only dir.
    shutil.copytree(gen._RAW, raw)

    def row(cap: int, k: int, budget: int, delivered: float) -> dict[str, Any]:
        return {
            "corpus": "mldr_en_8k_slice",
            "profile": f"recursive-t{cap}-o0-gpt2",
            "embedding": "fastembed:bge-base",
            "axes": {"strategy": "recursive", "max_tokens": cap, "overlap_tokens": 0, "breakpoint_model": None},
            "k": k,
            "budgets": [budget],
            "ndcg_delivered": delivered,
        }

    raw.joinpath("product-k.json").write_text(
        json.dumps(
            {
                "product_k": 5,
                "budgets": [1280, 2560],
                "rows": [
                    row(64, 20, 1280, 0.41),
                    row(64, 40, 2560, 0.44),
                    row(256, 5, 1280, 0.50),
                    row(256, 10, 2560, 0.53),
                ],
            }
        )
    )
    monkeypatch.setattr(gen, "_RAW", raw)
    d = gen._product_k_budget()
    assert d["caps"] == [64, 128, 256, 512] and d["budgets"] == [1280, 2560]
    assert d["series"]["1280"]["fastembed:bge-base"] == [0.41, None, 0.50, None]
    assert d["series"]["2560"]["fastembed:bge-base"] == [0.44, None, 0.53, None]
    collected = gen.collect_chart_data()["product_k_budget"]
    assert collected["series"]["1280"]["fastembed:bge-base"][0] == 0.41, "the collector is wired into the manifest"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `env -u VIRTUAL_ENV .venv/bin/python -m pytest tests/test_bench_charts_current.py -q -p no:cacheprovider -k product_k`
Expected: `1 failed`, `AttributeError: module 'gen_bench_charts' has no attribute '_product_k_budget'`.

- [ ] **Step 3: Add collector, renderer and wiring**

Insert after `_markdown_structure` in `scripts/gen_bench_charts.py`:

```python
_PRODUCT_K_CAPS = [64, 128, 256, 512]
_PRODUCT_K_CORPUS = "mldr_en_8k_slice"


def _product_k_budget() -> dict[str, Any]:
    """Delivered nDCG@k against cap at fixed token budgets, MLDR English, recursive, zero overlap."""
    doc = _read(_RAW / "product-k.json")
    if not doc:
        return {"caps": [], "budgets": [], "series": {}}
    budgets = [int(b) for b in doc["budgets"]]
    series: dict[str, dict[str, list[float | None]]] = {str(b): {} for b in budgets}
    for row in doc["rows"]:
        axes = row["axes"]
        if row["corpus"] != _PRODUCT_K_CORPUS or axes.get("strategy") != "recursive" or axes.get("overlap_tokens") != 0:
            continue
        cap = int(axes["max_tokens"])
        if cap not in _PRODUCT_K_CAPS:
            continue
        for budget in budgets:
            if budget not in row.get("budgets", []):
                continue
            line = series[str(budget)].setdefault(str(row["embedding"]), [None] * len(_PRODUCT_K_CAPS))
            line[_PRODUCT_K_CAPS.index(cap)] = float(row["ndcg_delivered"])
    return {"caps": list(_PRODUCT_K_CAPS), "budgets": budgets, "series": series}
```

Add to the `collect_chart_data` dict literal after `"markdown_structure": _markdown_structure(),`:

```python
        "product_k_budget": _product_k_budget(),
```

Insert after `render_markdown_structure`:

```python
def render_product_k_budget(d: dict[str, Any]) -> None:
    """One panel per budget: delivered nDCG@k against cap, a line per embedder.

    Read left to right as "the same number of tokens handed to the reader, cut finer or coarser":
    at a fixed budget a smaller cap means more, shorter chunks. The x axis is log2 so the four
    caps sit evenly.
    """
    if not d["series"]:
        return
    plt = _style()
    palette = [_BLUE, _AQUA, _ORANGE, _VIOLET, _GREEN, _RED]
    fig, axes = plt.subplots(1, len(d["budgets"]), figsize=(4.0 * len(d["budgets"]), 3.6), sharey=True)
    panels = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, budget in zip(panels, d["budgets"], strict=True):
        lines = d["series"][str(budget)]
        for colour, (embedding, values) in zip(palette, sorted(lines.items()), strict=False):
            xs = [cap for cap, v in zip(d["caps"], values, strict=True) if v is not None]
            ys = [v for v in values if v is not None]
            ax.plot(xs, ys, "o-", color=_c(colour), lw=1.6, ms=4.5, label=embedding.split(":")[1])
        ax.set_xscale("log", base=2)
        ax.set_xticks(d["caps"])
        ax.set_xticklabels([str(c) for c in d["caps"]], fontsize=8)
        ax.set_xlabel("cap (tokens per chunk)", fontsize=8, color=_muted())
        ax.set_title(f"{budget} tokens read (k = {budget} / cap)", fontsize=9, color=_ink())
        ax.tick_params(labelsize=8)
    panels[0].set_ylabel("delivered nDCG@k", fontsize=8, color=_muted())
    panels[0].legend(fontsize=7, frameon=False, loc="lower right", title="embedder", title_fontsize=7)
    fig.suptitle("Chunk size at a fixed number of tokens read, MLDR English", fontsize=11, color=_ink(), y=1.0)
    _save(fig, "chunk_product_k_budget")
    plt.close(fig)
```

Add to `_render_all` after `render_markdown_structure(data["markdown_structure"])`:

```python
    render_product_k_budget(data["product_k_budget"])
```

- [ ] **Step 4: Run the charts tests, then regenerate the manifest**

Run: `env -u VIRTUAL_ENV .venv/bin/python -m pytest tests/test_bench_charts_current.py -q -p no:cacheprovider`
Expected: the new test passes and `test_committed_charts_match_current_raw_data` FAILS (stale manifest: the new key is absent). Then run `env -u VIRTUAL_ENV .venv/bin/python scripts/gen_bench_charts.py`, expected `wrote 17 charts x 2 themes + manifest` (the new chart draws nothing while the raw file is absent). Run the test module again: all pass. Check `git status --porcelain`: only the two scripts, the test file and `docs/benchmarks/img/charts.manifest.json` may be modified; a modified PNG is reported, not committed.

- [ ] **Step 5: Gates**

Run: `env -u VIRTUAL_ENV .venv/bin/ruff check scripts/gen_bench_charts.py tests/test_bench_charts_current.py && env -u VIRTUAL_ENV .venv/bin/ruff format --check scripts/gen_bench_charts.py tests/test_bench_charts_current.py && env -u VIRTUAL_ENV .venv/bin/pyright --pythonpath .venv/bin/python scripts/gen_bench_charts.py tests/test_bench_charts_current.py`
Expected: clean, `0 errors`.

- [ ] **Step 6: Commit**

Subject: `bench(chunking): [27] budget-ladder chart for the delivered chunk list`. Pathspec: the two scripts, the test file, `docs/benchmarks/img/charts.manifest.json`.

---

### Task 6: After `ITEM22-DONE`: run the scorer, export, tables, charts

Operational task, no new code. Runs only once `~/semdex-sweep-run/logs/item22-markdown-structure-20260927-1711.log` holds `ITEM22-DONE` with every `RC_` line at 0, and after the [22] plan's own Task 8 has run its export (the two share the export command; run it once after both scorers).

**Files:**
- Modify: `tests/benchmarks/raw/product-k.json` (new, written by the export), `docs/benchmarks/img/charts.manifest.json`, `docs/benchmarks/img/chunk_product_k_budget.png`, `docs/benchmarks/img/chunk_product_k_budget-dark.png`, `docs/benchmarks/03-chunking.md` (two marker pairs and one `<picture>` block).

**STOP conditions** - stop and report rather than improvise, if:
- the scorer refuses a cell for a missing query-vector cache: that (corpus, embedder) was never scored by the chunk-sweep scorer; report the name, do not embed;
- the export reports `product-k.json: REJECTED`: a foreign corpus reached the score file; report, do not widen the allowlist;
- the scorer's `[product-k]` lines show a documents nDCG@5 far from the chunk-sweep nDCG@10 for the same cell (more than 0.10 apart): the query cache or the uris are out of step with the vectors; stop.

- [ ] **Step 1: Score, detached, with RC lines**

Write `~/semdex-sweep-run/item27-product-k.sh` (chmod +x) with this body and run it with `setsid nohup ./item27-product-k.sh > /dev/null 2>&1 &` from that directory; judge it by the log:

```bash
#!/usr/bin/env bash
set -u
cd ~/semdex-sweep-run || exit 1
L="logs/item27-product-k-$(date +%Y%m%d-%H%M).log"
{
    echo "start $(date -Is)"
    nice -n 19 ionice -c3 .venv-sweep/bin/python <semdex-checkout>/scripts/score_product_k.py
    echo "RC_SCORE=$?"
    echo "ITEM27-DONE $(date -Is)"
} > "$L" 2>&1
```

Expected: `RC_SCORE=0` and `[product-k] N cells in /embeddings/scores/product_k_scores.json` with N = the cap-256 zero-overlap profiles times six embedders on the three corpora plus the recursive ladder cells (about 100; count them, do not assume). The frozen `.venv-sweep` interpreter carries the sibling scripts' deps; the scorer imports its siblings from the repo's `scripts/` directory beside it, as written above.

- [ ] **Step 2: Export**

Run from the repo: `env -u VIRTUAL_ENV .venv/bin/python scripts/export_bench_raw.py`
Expected: `[export] product-k.json: <N> cells, <R> rows, corpora=3 profiles=<P> embedders=6`, no `REJECTED`, exit 0.

- [ ] **Step 3: Place the markers and the picture, regenerate**

In `docs/benchmarks/03-chunking.md`, directly after the END marker of the `span_blind_spot` block (inside `### What document-level scoring counts but does not deliver`), insert a new subsection heading `### The product's five slots` followed by an empty line, an empty BEGIN/END marker pair for the id `product_k_verdict` in exactly the form of the `chunk_knob_strategy` pair (copy those two lines and swap the id), an empty line, a `<picture>` block for `img/chunk_product_k_budget.png` and its `-dark` twin in the form of the `chunk_knob_strategy` picture block with the alt text `Delivered nDCG@k against chunk cap at 1280 and 2560 tokens read, MLDR English, one line per embedder`, an empty line, and an empty marker pair for `product_k_budget`. This plan does not spell the marker lines out on purpose: the generator and its tests scan every `.md` under `docs/`, plans included.

Then: `env -u VIRTUAL_ENV .venv/bin/python scripts/gen_bench_tables.py && env -u VIRTUAL_ENV .venv/bin/python scripts/gen_bench_charts.py`
Expected: the `[tables] docs/benchmarks/03-chunking.md: <B> blocks` line with B two higher than before this task (count and report the figure), and `wrote 17 charts x 2 themes + manifest`.

- [ ] **Step 4: Gate and commit**

Run: `env -u VIRTUAL_ENV .venv/bin/python -m pytest tests/test_bench_tables_current.py tests/test_bench_charts_current.py tests/test_bench_export_product_k.py -q -p no:cacheprovider`
Expected: all pass. Commit the raw file, the manifest, the two PNGs and the page. Subject: `bench(chunking): [27] product-k results: tables, chart and raw data`.

---

### Task 7: The write-up and the claims

**Files:**
- Modify: `docs/benchmarks/03-chunking.md` (prose under `### The product's five slots`; the `A reader consumes the retrieved chunk` row and, if the verdict warrants, the `Any long documents, to start from` row of `## What to set`; `## Not measured here`)
- Modify: `tests/benchmarks/claims/03-chunking.toml` (a `[rowsets.product_k]` with `source = "product-k.json"` and `path = "rows"`, plus one `[[claim]]` per quoted figure)
- Modify: `docs/benchmarks/08-gaps.md` (close or narrow the gap that names the product's k and the missing dedup, if one exists; grep `dedup` and `k=5` there first)

**STOP conditions** - stop and report rather than improvise, if:
- `scripts/check_bench_claims.py` reports a quote missing after the prose is written: the quote wrapped across lines; re-lay the paragraph, never change the claim to match a broken quote;
- the verdict table shows the gap resolved in opposite directions across corpora and the prose would have to pick one: report the split and write it as a split.

- [ ] **Step 1: Write the subsection**

Under `### The product's five slots`, before the verdict marker pair, three paragraphs: (1) what the product returns and what every table above measured, in the design's words; (2) the verdict read off `product_k_verdict`: the range of the cost across strategies and embedders per corpus, how many pairs resolve, the mean distinct documents in five slots, and which strategy wastes the most slots (expect the fine-cutting strategies); (3) the budget reading off `product_k_budget` and the chart: at 1280 and at 2560 tokens read, which cap delivers most, per corpus, and whether the two budgets agree. Every number quoted comes from the generated tables, printed exactly as the table prints it, and each sits on one physical line.

- [ ] **Step 2: Update the settings row**

Change the `A reader consumes the retrieved chunk` row of `## What to set` to cite the measured cost of no dedup and the budget winner beside the delivered-answer rate it already quotes; keep every existing figure in the row. If the cost is material (0.005 nDCG or more, resolved, on two of the three corpora), add one sentence to the `Any long documents, to start from` row saying so, otherwise say in the subsection that it is not.

- [ ] **Step 3: Claims**

Add to `tests/benchmarks/claims/03-chunking.toml`:

```toml
[rowsets.product_k]
source = "product-k.json"
path = "rows"
```

Then one `[[claim]]` per figure quoted in Steps 1 and 2, in the file's existing form. A single-cell figure selects its flat row by cell id and k:

```toml
[[claim]]
id = "product-k-recursive-bge-base-mldr-en-cost"
quote = "<the sentence fragment as printed, on one line>"
published = "<the figure exactly as printed>"
rows = "product_k"
reduce = "value"
field = "gap_mean_delta"
where = { cell = "mldr_en_8k_slice__recursive-t256-o0-gpt2__fastembed-bge-base", k = 5 }
why = "the paired documents-minus-delivered gap at the product's k for that cell"
```

A count of resolved pairs uses `reduce = "count"` with `where = { product = true, "axes.max_tokens" = 256, "axes.overlap_tokens" = 0, gap_resolved = true }` (the checker's dotted keys reach into `axes`); a mean distinct-documents figure uses `reduce = "mean"` with `field = "distinct_docs"` over the same filter.

Run: `env -u VIRTUAL_ENV .venv/bin/python scripts/check_bench_claims.py`
Expected: every claim passes, including the new ones; the run prints the total.

- [ ] **Step 4: Gate, commit, push**

Run the full gate through the jig: `python3 <plugin>/skills/compuse-toolbox/scripts/gate.py --gate "make test"` with `env -u VIRTUAL_ENV`, read `GATE_RC` from its output. Commit with subject `docs(benchmarks): [27] the product's five slots: delivered against deduplicated, and size at a fixed budget`, then push. CI stays at steps=0 while the account billing block ([15]) stands; the local gate is the validation.
