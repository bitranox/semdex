# Markdown Structure Effect Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use bitranox:process-agents-subagent-driven-development (recommended) or bitranox:process-plan-executor to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure what marking MLDR English's bare-line headings as markdown buys each chunk strategy, on identical documents and queries, and publish the paired result on the chunking page.

**Architecture:** A twin slice `mldr_en_8k_md_slice` is rendered from `mldr_en_8k_slice` by a pure heading rule; the existing sweep, audit and scorer run unchanged over it (its name follows the `mldr_*_slice` family); a new paired script compares each marked cell with its unmarked twin per query; the export, tables, charts and claim gate learn the new corpus and one new raw file. The design is `docs/plans/2026-09-27-markdown-structure-design.md`.

**Tech Stack:** Python 3.10+ (scripts run under the project `.venv`, the sweep under the frozen `.venv-sweep`), numpy, pyarrow, orjson, pytest; the `scripts/` env-config convention (no argparse).

## Global Constraints

- Work on `main`, commit after every task, push at the end; never a PR (repo rule).
- Every new figure in prose gets a `[[claim]]` in `tests/benchmarks/claims/<page>.toml`; a quote sits on ONE physical line.
- ASCII punctuation only in files (no em-dash, no arrow character in source: write the arrow as a `\u2192` escape).
- Provenance is carried from the scorer's stamps on cells, never collected on the machine that copies.
- Never `pkill -f`; judge a background job by its RC line; the sweep is `setsid`-detached and runs from `~/semdex-sweep-run` with the frozen interpreter.
- No hardcoded magic numbers without a config knob: the heading rule's parameters are module constants with env overrides and are recorded in `meta.json`.
- The marked slice name is `mldr_en_8k_md_slice`; the unmarked twin is `mldr_en_8k_slice`; profiles are `markdown-t256-o0-gpt2`, `recursive-t256-o0-gpt2`, `fast-t256-o0-gpt2`, `semantic-t256-o0-gpt2`; embedder labels `model2vec:potion-retrieval-32M`, `model2vec:potion-base-8M`, `fastembed:bge-base`, `ollama:bge-m3`, `ollama:qwen3-embedding-4b`, `ollama:qwen3-embedding-8b`.
- Cell id: `<corpus>__<profile>__<dirsafe label>` where dirsafe replaces `:` and `/` with `-`.

---

## Ground truth read before planning (2026-09-27)

- `/embeddings/vectors` holds the unmarked `mldr_en_8k_slice` cells at `t256-o0` for `markdown`, `recursive`, `fast` with all six embedders, and `semantic` with five: `semantic-t256-o0-gpt2__ollama-bge-m3` is MISSING. The GPU chain therefore runs over BOTH corpora so preembed fills that one cell (it skips complete cells).
- `score_chunk_sweep._reject_corpus_mixing` refuses to add a new corpus to `mldr_chunk_scores.json`, so the marked corpus scores into `/embeddings/scores/mldr_md_chunk_scores.json`, added as a second source of the `chunk-sweep-mldr.json` export (Task 7).
- `export_bench_raw` fails on a vector cell of a declared corpus that has no score, so all 24 marked cells must be scored before the export runs.
- Chunker code since the twins were cut (2026-08-12): one commit, `49f2702` (`late` recount only); none of the four strategies here changed.
- `~/semdex-sweep-run/scripts/*.py` is byte-identical to `scripts/` for the four sweep scripts; `.venv-sweep` is the frozen interpreter.
- ollama on `px-semdex-test-embeddings:11434` is idle and serves `bge-m3`, `qwen3-embedding:4b`, `qwen3-embedding:8b`.

---

### Task 1: The slice builder, `scripts/build_mldr_markdown_slice.py`

**Files:**
- Create: `scripts/build_mldr_markdown_slice.py`
- Test: `tests/test_build_mldr_markdown_slice.py`

**Interfaces:**
- Produces: `mark_headings(text: str) -> tuple[str, int]` (marked text, count of marked lines); `is_heading(line: str, following: str | None) -> bool`; `strip_marks(text: str) -> str`; `build(source: Path, out: Path, *, sample_size: int, seed: int) -> dict[str, Any]` (returns the meta written); `main() -> None`.
- Env: `SEMDEX_MDSLICE_SOURCE` (default `/corpora/mldr-slices/mldr_en_8k_slice`), `SEMDEX_MDSLICE_OUT` (default `/corpora/mldr-slices/mldr_en_8k_md_slice`), `SEMDEX_MDSLICE_SAMPLE` (default 200), `SEMDEX_MDSLICE_SEED` (default 20260927).

**Out of scope** - do NOT touch:
- `scripts/build_mldr_slices.py` - the source slice is frozen; this script READS it.
- `scripts/preembed_vectors.py` - `_SLICE_FAMILIES` already resolves `mldr_*_slice` under `/corpora/mldr-slices`.

**STOP conditions:**
- `/corpora/mldr-slices/mldr_en_8k_slice/corpus.jsonl` is absent or its documents carry no `\n` (the rule is line-based; Task 2 checks the marked count).
- the strip-back identity test fails after one fix.

- [ ] **Step 1: Write the failing tests**

```python
"""The markdown twin of the MLDR English slice is the source text with its headings marked.

The rule is a heuristic; these tests pin what it must and must not do on synthetic text, and that
stripping the marks gives the source back byte for byte, so the paired comparison isolates markup.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "build_mldr_markdown_slice.py"

pytestmark = pytest.mark.os_agnostic

_PROSE = (
    "The compound was first synthesised in the laboratory in the early part of the twentieth century and "
    "later produced at industrial scale by several manufacturers across three continents."
)
assert len(_PROSE) >= 120 and _PROSE.endswith(".")


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("build_mldr_markdown_slice", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_mldr_markdown_slice"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder() -> Any:
    return _load()


def test_a_short_line_before_a_prose_paragraph_is_a_heading(builder: Any) -> None:
    marked, count = builder.mark_headings(f"History\n{_PROSE}\n")
    assert marked == f"## History\n{_PROSE}\n"
    assert count == 1


def test_a_heading_followed_by_a_blank_line_then_prose_is_still_a_heading(builder: Any) -> None:
    marked, count = builder.mark_headings(f"Early life\n\n{_PROSE}\n")
    assert marked.startswith("## Early life\n\n")
    assert count == 1


@pytest.mark.parametrize(
    "line",
    [
        "See also",  # followed by short lines in the fixture below, never prose
        "Population: 1234",  # a run of three digits
        "x = y + z",  # formula characters
        "Ends with a full stop.",
        "A rather long line that has many more than eight spaces in it and so is a sentence not a heading",
        "",
    ],
)
def test_lines_that_are_not_headings_are_left_alone(builder: Any, line: str) -> None:
    following = "Short item\nAnother\n" if line == "See also" else _PROSE
    text = f"{line}\n{following}\n"
    marked, count = builder.mark_headings(text)
    assert marked == text
    assert count == 0


def test_a_list_item_before_prose_is_not_a_heading(builder: Any) -> None:
    text = f"- an item\n{_PROSE}\n"
    assert builder.mark_headings(text) == (text, 0)


def test_stripping_the_marks_gives_the_source_back_byte_for_byte(builder: Any) -> None:
    source = f"Robert Bilott investigation\n{_PROSE}\nSynthesis\n{_PROSE}\nSee also\nShort\n"
    marked, count = builder.mark_headings(source)
    assert count == 2
    assert builder.strip_marks(marked) == source


def test_a_source_line_that_already_starts_with_the_mark_is_never_marked(builder: Any) -> None:
    # Otherwise strip_marks would remove a mark the source carried and break the identity.
    text = f"## Already marked\n{_PROSE}\n"
    assert builder.mark_headings(text) == (text, 0)


def _write_source(root: Path, docs: dict[str, str]) -> Path:
    src = root / "mldr_en_8k_slice"
    src.mkdir(parents=True)
    src.joinpath("corpus.jsonl").write_bytes(
        b"".join(json.dumps({"_id": d, "title": "", "text": t}).encode() + b"\n" for d, t in sorted(docs.items()))
    )
    src.joinpath("queries.json").write_text(json.dumps({"q1": "what is it"}))
    src.joinpath("qrels.json").write_text(json.dumps({"q1": {"doc-1": 1}}))
    src.joinpath("doc_ids.txt").write_text("\n".join(sorted(docs)) + "\n")
    src.joinpath("meta.json").write_text(json.dumps({"corpus": "mldr_en_8k_slice", "total": len(docs)}))
    return src


def test_build_writes_a_twin_slice_with_the_same_ids_queries_and_qrels(builder: Any, tmp_path: Path) -> None:
    docs = {"doc-1": f"History\n{_PROSE}\n", "doc-2": f"{_PROSE}\n"}
    src = _write_source(tmp_path, docs)
    out = tmp_path / "mldr_en_8k_md_slice"
    meta = builder.build(src, out, sample_size=5, seed=1)
    assert out.joinpath("doc_ids.txt").read_text() == src.joinpath("doc_ids.txt").read_text()
    assert out.joinpath("queries.json").read_bytes() == src.joinpath("queries.json").read_bytes()
    assert out.joinpath("qrels.json").read_bytes() == src.joinpath("qrels.json").read_bytes()
    rows = [json.loads(line) for line in out.joinpath("corpus.jsonl").read_text().splitlines()]
    assert [r["_id"] for r in rows] == ["doc-1", "doc-2"]
    assert rows[0]["text"] == f"## History\n{_PROSE}\n"
    assert rows[1]["text"] == docs["doc-2"]
    assert {builder.strip_marks(r["text"]) for r in rows} == set(docs.values())
    assert meta["corpus"] == "mldr_en_8k_md_slice"
    assert meta["source_corpus"] == "mldr_en_8k_slice"
    assert meta["marked_lines"] == 1 and meta["marked_docs"] == 1 and meta["total"] == 2
    assert meta["rule"]["max_chars"] == builder.MAX_HEADING_CHARS
    assert len(meta["source_corpus_sha256"]) == 64
    written = json.loads(out.joinpath("meta.json").read_text())
    assert written == meta
    sample = out.joinpath("heading-sample.txt").read_text().splitlines()
    assert sample and sample[0].startswith("doc-1\t## History\t"), "one marked line per row: doc id, heading, next line"


def test_build_is_idempotent_and_never_rewrites_an_existing_slice(builder: Any, tmp_path: Path) -> None:
    src = _write_source(tmp_path, {"doc-1": f"History\n{_PROSE}\n"})
    out = tmp_path / "mldr_en_8k_md_slice"
    builder.build(src, out, sample_size=5, seed=1)
    before = out.joinpath("corpus.jsonl").read_bytes()
    out.joinpath("corpus.jsonl").write_bytes(b"tampered\n")
    builder.build(src, out, sample_size=5, seed=1)
    assert out.joinpath("corpus.jsonl").read_bytes() == b"tampered\n", "an existing slice is left alone"
    assert before != b"tampered\n"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_build_mldr_markdown_slice.py -q -p no:cacheprovider`
Expected: every test errors at the fixture with `FileNotFoundError` on the script path (the module does not exist).

- [ ] **Step 3: Write the builder**

```python
#!/usr/bin/env python
# pyright: basic
"""Render the MLDR English slice with its bare-line section headings marked as markdown.

MLDR English is Wikipedia text with the markup stripped. The section headings survive as bare
lines: a short line with no terminal punctuation, followed by a paragraph of running prose. This
writes a TWIN slice with the same doc ids, queries and qrels and each such line prefixed ``## ``,
so a chunk strategy that cuts on markdown structure can be measured against its own unmarked cell
per query. Stripping every ``## `` prefix gives the source back byte for byte (tested), which is
what makes the pair isolate the markup.

The rule is a heuristic and its precision is measured, not assumed: ``heading-sample.txt`` beside
the slice holds a seeded sample of marked lines with the line that follows each, for review. A
reviewer copies it to ``heading-sample-judged.txt`` and prefixes a wrong line with ``X\t``; the
paired-effect script reports the share.

Heading LEVEL is not recoverable from stripped text; every heading is level two.

Config (env; scripts/ convention, no argparse):
  SEMDEX_MDSLICE_SOURCE  source slice dir (default /corpora/mldr-slices/mldr_en_8k_slice)
  SEMDEX_MDSLICE_OUT     output slice dir (default /corpora/mldr-slices/mldr_en_8k_md_slice)
  SEMDEX_MDSLICE_SAMPLE  marked lines sampled for review (default 200)
  SEMDEX_MDSLICE_SEED    sample RNG seed (default 20260927)
  SEMDEX_MDSLICE_MAX_CHARS / _MAX_SPACES / _MIN_NEXT_CHARS  rule parameters (defaults 79 / 8 / 120)
"""

from __future__ import annotations

import hashlib
import os
import random
import re
from pathlib import Path
from typing import Any

import orjson

MARK = "## "
# The rule's parameters. A heading is short, has few words, and is followed by a paragraph.
MAX_HEADING_CHARS = int(os.environ.get("SEMDEX_MDSLICE_MAX_CHARS", "79"))
MAX_HEADING_SPACES = int(os.environ.get("SEMDEX_MDSLICE_MAX_SPACES", "8"))
MIN_NEXT_LINE_CHARS = int(os.environ.get("SEMDEX_MDSLICE_MIN_NEXT_CHARS", "120"))
_TERMINAL_PUNCTUATION = ".,;:!?)"
_PROSE_ENDINGS = (".", '"', ")")
_FORBIDDEN_CHARS = ("=", "\u2192", "+")  # the arrow as an escape: ASCII source only
_DIGIT_RUN = re.compile(r"\d{3}")
_LIST_MARKERS = ("-", "*", "\u2022")

__all__ = ["MARK", "build", "is_heading", "mark_headings", "strip_marks"]


def is_heading(line: str, following: str | None) -> bool:
    """Is ``line`` a section heading, given the next non-empty line?"""
    if following is None or not line or len(line) > MAX_HEADING_CHARS:
        return False
    if line.startswith(MARK) or line.startswith(_LIST_MARKERS):
        return False
    if line[-1] in _TERMINAL_PUNCTUATION or line.count(" ") > MAX_HEADING_SPACES:
        return False
    if any(ch in line for ch in _FORBIDDEN_CHARS) or _DIGIT_RUN.search(line):
        return False
    return len(following) >= MIN_NEXT_LINE_CHARS and following.endswith(_PROSE_ENDINGS)


def _next_non_empty(lines: list[str], start: int) -> str | None:
    for candidate in lines[start:]:
        if candidate.strip():
            return candidate
    return None


def mark_headings(text: str) -> tuple[str, int]:
    """Prefix every heading line with ``## ``; return the text and how many lines were marked."""
    lines = text.split("\n")
    count = 0
    for i, line in enumerate(lines):
        if is_heading(line, _next_non_empty(lines, i + 1)):
            lines[i] = MARK + line
            count += 1
    return "\n".join(lines), count


def strip_marks(text: str) -> str:
    """The inverse of ``mark_headings`` for text the source never marked."""
    return "\n".join(line[len(MARK) :] if line.startswith(MARK) else line for line in text.split("\n"))


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _rule() -> dict[str, Any]:
    return {
        "max_chars": MAX_HEADING_CHARS,
        "max_spaces": MAX_HEADING_SPACES,
        "min_next_line_chars": MIN_NEXT_LINE_CHARS,
        "terminal_punctuation": _TERMINAL_PUNCTUATION,
        "prose_endings": "".join(_PROSE_ENDINGS),
        "forbidden_chars": "".join(_FORBIDDEN_CHARS),
        "digit_run": _DIGIT_RUN.pattern,
        "mark": MARK,
    }


def _sample_rows(text: str, doc_id: str) -> list[str]:
    """Every marked line of one document as ``doc_id<TAB>heading<TAB>next line`` for the review file."""
    lines = text.split("\n")
    rows = []
    for i, line in enumerate(lines):
        if line.startswith(MARK):
            following = _next_non_empty(lines, i + 1) or ""
            rows.append(f"{doc_id}\t{line}\t{following[:160]}")
    return rows


def build(source: Path, out: Path, *, sample_size: int, seed: int) -> dict[str, Any]:
    """Write the marked twin of ``source`` into ``out`` (idempotent); return the meta written."""
    if (out / "meta.json").exists():
        print(f"[mdslice] {out.name}: already built - skip (delete the dir to rebuild)", flush=True)
        return orjson.loads((out / "meta.json").read_bytes())
    source_bytes = (source / "corpus.jsonl").read_bytes()
    marked_rows: list[bytes] = []
    review: list[str] = []
    marked_lines = marked_docs = total = 0
    for raw in source_bytes.splitlines():
        row = orjson.loads(raw)
        text, count = mark_headings(row.get("text") or "")
        if strip_marks(text) != (row.get("text") or ""):
            raise ValueError(f"{row['_id']}: stripping the marks does not give the source back")
        total += 1
        marked_lines += count
        marked_docs += count > 0
        review.extend(_sample_rows(text, row["_id"]))
        marked_rows.append(orjson.dumps({"_id": row["_id"], "title": row.get("title", ""), "text": text}) + b"\n")
    out.mkdir(parents=True, exist_ok=True)
    _write_atomic(out / "corpus.jsonl", b"".join(marked_rows))
    for name in ("queries.json", "qrels.json", "doc_ids.txt"):
        _write_atomic(out / name, (source / name).read_bytes())
    sample = random.Random(seed).sample(review, min(sample_size, len(review)))  # noqa: S311 - reproducible review sample, not cryptographic
    _write_atomic(out / "heading-sample.txt", ("\n".join(sample) + "\n").encode() if sample else b"")
    meta = {
        "corpus": out.name,
        "source_corpus": source.name,
        "source_corpus_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "rule": _rule(),
        "total": total,
        "marked_lines": marked_lines,
        "marked_docs": marked_docs,
        "sample_size": len(sample),
        "sample_seed": seed,
    }
    _write_atomic(out / "meta.json", orjson.dumps(meta, option=orjson.OPT_INDENT_2))
    print(f"[mdslice] {out.name}: {total} docs, {marked_lines} headings in {marked_docs} docs -> {out}", flush=True)
    return meta


def main() -> None:
    source = Path(os.environ.get("SEMDEX_MDSLICE_SOURCE", "/corpora/mldr-slices/mldr_en_8k_slice"))
    out = Path(os.environ.get("SEMDEX_MDSLICE_OUT", "/corpora/mldr-slices/mldr_en_8k_md_slice"))
    build(
        source,
        out,
        sample_size=int(os.environ.get("SEMDEX_MDSLICE_SAMPLE", "200")),
        seed=int(os.environ.get("SEMDEX_MDSLICE_SEED", "20260927")),
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_build_mldr_markdown_slice.py -q -p no:cacheprovider`
Expected: `13 passed`.

- [ ] **Step 5: Lint and type-check the two files**

Run: `.venv/bin/ruff check scripts/build_mldr_markdown_slice.py tests/test_build_mldr_markdown_slice.py && .venv/bin/ruff format --check scripts/build_mldr_markdown_slice.py tests/test_build_mldr_markdown_slice.py && .venv/bin/pyright scripts/build_mldr_markdown_slice.py`
Expected: `All checks passed!`, `2 files already formatted`, `0 errors`.

- [ ] **Step 6: Commit**

Write the message to a file, then: `git add scripts/build_mldr_markdown_slice.py tests/test_build_mldr_markdown_slice.py && git commit -F <msgfile>` with subject `bench(chunking): [22] builder for the markdown twin of the MLDR English slice`.

---

### Task 2: Build the real twin slice and review the heading sample

**Files:**
- Create (data, outside the repo): `/corpora/mldr-slices/mldr_en_8k_md_slice/` and `heading-sample-judged.txt` beside it.

**Interfaces:**
- Consumes: `build()` from Task 1 via `main()`.
- Produces: the slice dir the sweep resolves by name; `heading-sample-judged.txt` in the format `X\t<row>` for a wrong row, `<row>` unchanged otherwise (Task 4 reads it).

**STOP conditions:**
- `marked_lines` in `meta.json` is below 40,000 or above 80,000 (the design measured 56,332 on this slice; a large departure means the rule or the text is not what the design described).
- more than 15 percent of the 200 sampled lines are wrong: report to the user before spending embed budget.

- [ ] **Step 1: Run the builder**

Run: `cd <semdex-checkout> && CACHE_ROOT=/embeddings .venv/bin/python scripts/build_mldr_markdown_slice.py`
Expected: one line `[mdslice] mldr_en_8k_md_slice: 8000 docs, <about 56,000> headings in <about 7,600> docs -> /corpora/mldr-slices/mldr_en_8k_md_slice`.

- [ ] **Step 2: Verify the twin's ids, queries and qrels are the source's**

Run:
```bash
cd /corpora/mldr-slices && for f in doc_ids.txt queries.json qrels.json; do cmp mldr_en_8k_slice/$f mldr_en_8k_md_slice/$f && echo "same $f"; done && wc -l mldr_en_8k_md_slice/corpus.jsonl mldr_en_8k_md_slice/heading-sample.txt
```
Expected: three `same` lines, `8000` corpus rows, `200` sample rows.

- [ ] **Step 3: Review the sample**

Read `/corpora/mldr-slices/mldr_en_8k_md_slice/heading-sample.txt` in full (200 rows: doc id, marked heading, next line). Copy it to `heading-sample-judged.txt` and prefix every row whose heading is NOT a section heading (a sentence fragment, a caption, a table cell, a name in a list) with `X<TAB>`. Count them:
```bash
grep -c $'^X\t' /corpora/mldr-slices/mldr_en_8k_md_slice/heading-sample-judged.txt || true
```
Record the count and three examples of wrong rows in `EXECUTION-USER-REVIEW.md` (gitignored) under today's date; Tasks 8 and 9 read this file for the share the page prints.

- [ ] **Step 4: Note the slice in the backlog line**

Edit the `[22]` line in `OPEN-WORK.md`: in `open:` append `slice built 2026-09-27 (<marked_lines> headings, sample <wrong>/200 judged wrong)`. Commit `OPEN-WORK.md` alone.

---

### Task 3: The sweep wrapper, chunk once, launch the two chains

**Files:**
- Create: `~/semdex-sweep-run/item22-markdown-structure.sh` (outside the repo, like its siblings)
- Modify: `~/semdex-sweep-run/scripts/` (refresh the four sweep scripts from `scripts/` if `cmp` finds a difference; today it finds none)

**Interfaces:**
- Consumes: the slice from Task 2; `scripts/sweep_chunk_profiles.py` (env `SWEEP_CORPORA`, `SWEEP_PROFILES`, `SWEEP_EMBEDDINGS`); `scripts/score_chunk_sweep.py` (env `SEMDEX_SCORE_CORPORA`, `SEMDEX_SCORE_PROFILES`, `SEMDEX_SCORE_EMBEDDINGS`, `SEMDEX_SCORE_OUT`).
- Produces: 24 marked vector cells plus the one missing unmarked cell `mldr_en_8k_slice__semantic-t256-o0-gpt2__ollama-bge-m3`; scores in `/embeddings/scores/mldr_md_chunk_scores.json` (marked) and `/embeddings/scores/mldr_chunk_scores.json` (the one unmarked cell); RC lines `RC_CHUNK`, `RC_GPU`, `RC_CPU`, `RC_SCORE`, and `ITEM22-DONE` in `~/semdex-sweep-run/logs/item22-markdown-structure-<stamp>.log`.

**Out of scope:**
- `scripts/sweep_chunk_profiles.py`, `scripts/preembed_vectors.py` - unchanged; the corpus name resolves by family.

**STOP conditions:**
- `RC_CHUNK` is non-zero or a `chunks.parquet` is missing at the gate: read `<log>.chunk`, do not start the chains.
- `.venv-sweep` cannot import `semdex.adapters.chunker` strategies `markdown`, `fast`, `semantic` (probe in Step 1).
- ollama `/api/ps` shows a model loaded by somebody else when the GPU chain starts (the card is shared; see the memory on VRAM squatters).

- [ ] **Step 1: Probe the frozen interpreter and the GPU host**

Run:
```bash
~/semdex-sweep-run/.venv-sweep/bin/python -c "from semdex.domain.enums import ChunkStrategy as C; print([C.MARKDOWN.value, C.FAST.value, C.SEMANTIC.value, C.RECURSIVE.value])"
curl -s -m 5 http://px-semdex-test-embeddings:11434/api/ps
cd <semdex-checkout> && for f in sweep_chunk_profiles.py preembed_vectors.py score_chunk_sweep.py; do cmp -s scripts/$f ~/semdex-sweep-run/scripts/$f && echo "same $f" || echo "DIFF $f"; done
```
Expected: `['markdown', 'fast', 'semantic', 'recursive']`; `{"models":[]}`; three `same` lines. On `DIFF`, copy the repo file over the run copy and write `git rev-parse HEAD` into `~/semdex-sweep-run/SOURCE_SHA-item22.txt`.

- [ ] **Step 2: Write the wrapper**

```bash
#!/usr/bin/env bash
# Backlog [22]: the markdown-structure effect. Chunk the marked MLDR English twin under the four
# strategies once (static models riding along), then a GPU chain and a CPU chain embed concurrently
# and never share a chunks.parquet.tmp. The GPU chain runs over BOTH corpora so the one unmarked
# cell that is missing (semantic / bge-m3) is filled; preembed skips every complete cell. Phase 3
# scores the marked corpus into its OWN score file (the MLDR file refuses a foreign corpus) and the
# one unmarked cell into the MLDR file. Frozen interpreter, RC per phase, setsid-detached by the
# caller. Same shape as item20-chunk-floor.sh.
set -uo pipefail

RUN=~/semdex-sweep-run
REPO=<semdex-checkout>
PY="$RUN/.venv-sweep/bin/python"
DRIVER="$RUN/scripts/sweep_chunk_profiles.py"
STAMP=$(date +%Y%m%d-%H%M)
LOG="$RUN/logs/item22-markdown-structure-$STAMP.log"

export CACHE_ROOT=/embeddings PYTHONUNBUFFERED=1
export FASTEMBED_CACHE_PATH=/embeddings/.fastembed_cache
export SEMDEX_BENCH_OLLAMA_URL=http://px-semdex-test-embeddings:11434
MARKED=mldr_en_8k_md_slice
UNMARKED=mldr_en_8k_slice
export SWEEP_PROFILES=markdown:256:0,recursive:256:0,fast:256:0,semantic:256:0
PROFILES_SCORE=markdown-t256-o0-gpt2,recursive-t256-o0-gpt2,fast-t256-o0-gpt2,semantic-t256-o0-gpt2
STATIC=model2vec:potion-retrieval-32M,model2vec:potion-base-8M
CPU=fastembed:bge-base
GPU=ollama:bge-m3,ollama:qwen3-embedding-4b,ollama:qwen3-embedding-8b

cd "$REPO" || exit 2
{
    echo "== item22-markdown-structure $(date -Is) interpreter=$PY source=$(cat "$RUN/SOURCE_SHA-item22.txt" 2>/dev/null || cat "$RUN/SOURCE_SHA-item20.txt")"
    echo "== profiles=$SWEEP_PROFILES marked=$MARKED unmarked=$UNMARKED"
    echo "== phase 1: chunk the marked corpus + static embedders $(date -Is)"
} >> "$LOG"
SWEEP_CORPORA=$MARKED SWEEP_EMBEDDINGS=$STATIC nice -n 19 ionice -c3 "$PY" "$DRIVER" >> "$LOG.chunk" 2>&1
echo "RC_CHUNK=$?" >> "$LOG"

# Gate: every marked chunk set must exist before the chains split, or they race on chunks.parquet.tmp.
missing=0
for s in markdown recursive fast semantic; do
    f="/embeddings/chunks/${MARKED}__${s}-t256-o0-gpt2/chunks.parquet"
    [ -s "$f" ] || { echo "MISSING $f" >> "$LOG"; missing=1; }
done
if [ "$missing" -ne 0 ]; then
    echo "ITEM22-ABORTED: chunk sets incomplete, chains not started $(date -Is)" >> "$LOG"
    exit 3
fi
echo "gate ok: all four marked chunk sets present $(date -Is)" >> "$LOG"

echo "== phase 2: gpu chain (both corpora) and cpu chain (marked) concurrently $(date -Is)" >> "$LOG"
(
    # 128: qwen3-8b at 256 x 256-token chunks has hit the adapter timeout under contention before.
    SWEEP_CORPORA=$MARKED,$UNMARKED SWEEP_EMBEDDINGS=$GPU SEMDEX_PREEMBED_EMBED_BATCH=128 \
        nice -n 19 ionice -c3 "$PY" "$DRIVER" >> "$LOG.gpu" 2>&1
    echo "RC_GPU=$?" >> "$LOG"
) &
(
    SWEEP_CORPORA=$MARKED SWEEP_EMBEDDINGS=$CPU nice -n 19 ionice -c3 "$PY" "$DRIVER" >> "$LOG.cpu" 2>&1
    echo "RC_CPU=$?" >> "$LOG"
) &
wait

echo "== phase 3: score $(date -Is)" >> "$LOG"
rc_score=0
SEMDEX_SCORE_CORPORA=$MARKED SEMDEX_SCORE_PROFILES=$PROFILES_SCORE \
    SEMDEX_SCORE_EMBEDDINGS=$STATIC,$CPU,$GPU SEMDEX_SCORE_OUT=/embeddings/scores/mldr_md_chunk_scores.json \
    nice -n 19 ionice -c3 "$PY" "$RUN/scripts/score_chunk_sweep.py" >> "$LOG.score" 2>&1 || rc_score=$?
SEMDEX_SCORE_CORPORA=$UNMARKED SEMDEX_SCORE_PROFILES=semantic-t256-o0-gpt2 \
    SEMDEX_SCORE_EMBEDDINGS=ollama:bge-m3 SEMDEX_SCORE_OUT=/embeddings/scores/mldr_chunk_scores.json \
    nice -n 19 ionice -c3 "$PY" "$RUN/scripts/score_chunk_sweep.py" >> "$LOG.score" 2>&1 || rc_score=$?
echo "RC_SCORE=$rc_score" >> "$LOG"
echo "ITEM22-DONE $(date -Is)" >> "$LOG"
```

Format and lint it as the repo's shell gate does: `shfmt -i 4 -ci -d item22-markdown-structure.sh` and `bashate -i E003 --max-line-length 120 item22-markdown-structure.sh` (the sibling wrappers pass both).

- [ ] **Step 3: Launch it detached and confirm it is running**

Run:
```bash
cd ~/semdex-sweep-run && chmod +x item22-markdown-structure.sh && setsid nohup ./item22-markdown-structure.sh > /dev/null 2>&1 < /dev/null & sleep 5; ls -t ~/semdex-sweep-run/logs/item22-markdown-structure-*.log | head -1 | xargs cat
```
Expected: the three `==` header lines, no RC line yet. Within ten minutes `ls -la /embeddings/chunks/mldr_en_8k_md_slice__markdown-t256-o0-gpt2/` shows a growing `chunks.parquet.tmp`.

- [ ] **Step 4: Arm a bounded watcher on the RC lines, then build Tasks 4 to 8 while it runs**

The chunk phase is about an hour (four strategies on 8,000 docs plus two static models); the CPU chain about six hours; the GPU chain about a day and a half. Poll the log's RC lines at those horizons, never a shorter loop. When `ITEM22-DONE` lands, check every RC is `0` and continue at Task 8. A non-zero RC means read the phase's own log (`<log>.chunk`, `.gpu`, `.cpu`, `.score`), root-cause, relaunch (every phase is idempotent and resumable).

---

### Task 4: The paired cross-corpus script, `scripts/score_markdown_structure_effect.py`

**Files:**
- Create: `scripts/score_markdown_structure_effect.py`
- Test: `tests/test_bench_markdown_structure_effect.py`

**Interfaces:**
- Consumes: `paired_ci(left, right)` from `scripts/_score_stats.py` (returns `mean_delta`, `ci_lo`, `ci_hi`, `wins`, `ties`, `losses`, `n_shared`, `resolved`); `measured_on_summary(rows)` and `MEASURED_ON` as `score_query_length_effect.py` imports them; per-query arrays at `<CACHE_ROOT>/scores/perquery/<cell>.npz` with `qids` and `ndcg`/`ndcg@10`; the audit `tests/benchmarks/raw/chunk-dimension-audit.json` (`chunk_sets` rows with `corpus`, `profile`, `rows`, `token_p50`); the marked chunk sets' `chunks.parquet` (`text` column); `heading-sample-judged.txt` from Task 2.
- Produces: `tests/benchmarks/raw/markdown-structure-effect.json` with keys `measured_on`, `note`, `slice` (meta of the marked slice plus `sample_judged_wrong`, `sample_judged`), `effects` (one row per strategy and embedder: `strategy`, `embedding`, `marked_cell`, `unmarked_cell`, the paired fields, `marked_rows`, `unmarked_rows`, `marked_token_p50`, `unmarked_token_p50`, `marked_heading_start_share`), `missing` (list of `{strategy, embedding, missing: [cell ids]}`). Public functions: `pair_cells(perquery_dir, *, strategy, label, marked, unmarked) -> dict | None`, `heading_start_share(parquet: Path) -> float`, `judged_wrong(path: Path) -> tuple[int, int]`, `main() -> int`.
- Env: `CACHE_ROOT` (default `/embeddings`), `MDSTRUCT_MARKED` (default `mldr_en_8k_md_slice`), `MDSTRUCT_UNMARKED` (default `mldr_en_8k_slice`), `MDSTRUCT_STRATEGIES` (default `markdown,recursive,fast,semantic`), `MDSTRUCT_EMBEDDINGS` (default the six labels), `MDSTRUCT_SLICE_DIR` (default `/corpora/mldr-slices/mldr_en_8k_md_slice`), `MDSTRUCT_AUDIT` (default `tests/benchmarks/raw/chunk-dimension-audit.json`), `OUT` (default `tests/benchmarks/raw/markdown-structure-effect.json`).

**Out of scope:**
- `scripts/export_bench_raw.py` - the exporter pairs within a corpus only; the cross-corpus axis lives here (design decision).
- `scripts/score_query_length_effect.py` - the model, unchanged.

**STOP conditions:**
- a pair's query id sets differ (the script must REFUSE by name, and the test pins it).
- `paired_ci` returns a shape other than the one listed.

- [ ] **Step 1: Write the failing tests**

```python
"""Marked against unmarked: the same queries, the same documents, only the heading markup differs.

The exporter pairs cells within one corpus. The script under test pairs each marked cell with its
unmarked twin across the two corpus ids and runs the shared paired bootstrap. These tests build
tiny per-query arrays with a known shape and require the script to report a shifted pair as
resolved, an identical pair as unresolved with delta zero, and a missing or mismatched twin by name.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "score_markdown_structure_effect.py"

pytestmark = pytest.mark.os_agnostic

_MARKED = "mldr_en_8k_md_slice"
_UNMARKED = "mldr_en_8k_slice"
_LABEL = "fake:embedder"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("score_markdown_structure_effect", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["score_markdown_structure_effect"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def effect() -> Any:
    return _load()


def _write_cell(perquery: Path, corpus: str, strategy: str, scores: dict[str, float]) -> None:
    perquery.mkdir(parents=True, exist_ok=True)
    qids = sorted(scores)
    with (perquery / f"{corpus}__{strategy}-t256-o0-gpt2__fake-embedder.npz").open("wb") as handle:
        np.savez(handle, qids=np.asarray(qids), ndcg=np.asarray([scores[q] for q in qids]))


def test_a_shifted_pair_resolves_in_favour_of_the_marked_cell(effect: Any, tmp_path: Path) -> None:
    n = 60
    _write_cell(tmp_path, _UNMARKED, "markdown", {f"q{i}": 0.5 for i in range(n)})
    _write_cell(tmp_path, _MARKED, "markdown", {f"q{i}": 0.6 for i in range(n)})
    row = effect.pair_cells(tmp_path, strategy="markdown", label=_LABEL, marked=_MARKED, unmarked=_UNMARKED)
    assert row["mean_delta"] == pytest.approx(0.1)
    assert row["resolved"] and row["wins"] == n and row["losses"] == 0
    assert row["marked_cell"] == f"{_MARKED}__markdown-t256-o0-gpt2__fake-embedder"
    assert row["unmarked_cell"] == f"{_UNMARKED}__markdown-t256-o0-gpt2__fake-embedder"


def test_an_identical_pair_is_delta_zero_and_unresolved(effect: Any, tmp_path: Path) -> None:
    scores = {f"q{i}": 0.3 + i / 100 for i in range(40)}
    _write_cell(tmp_path, _UNMARKED, "fast", scores)
    _write_cell(tmp_path, _MARKED, "fast", scores)
    row = effect.pair_cells(tmp_path, strategy="fast", label=_LABEL, marked=_MARKED, unmarked=_UNMARKED)
    assert row["mean_delta"] == 0.0 and not row["resolved"] and row["ties"] == 40


def test_a_missing_twin_yields_none_not_a_zero(effect: Any, tmp_path: Path) -> None:
    _write_cell(tmp_path, _MARKED, "semantic", {"q0": 0.5})
    assert effect.pair_cells(tmp_path, strategy="semantic", label=_LABEL, marked=_MARKED, unmarked=_UNMARKED) is None


def test_mismatched_query_ids_are_refused_by_name(effect: Any, tmp_path: Path) -> None:
    _write_cell(tmp_path, _UNMARKED, "recursive", {"q0": 0.5, "q1": 0.5})
    _write_cell(tmp_path, _MARKED, "recursive", {"q0": 0.5, "q2": 0.5})
    with pytest.raises(ValueError, match="recursive-t256-o0-gpt2__fake-embedder.*query ids differ"):
        effect.pair_cells(tmp_path, strategy="recursive", label=_LABEL, marked=_MARKED, unmarked=_UNMARKED)


def test_heading_start_share_counts_chunks_that_begin_with_the_mark(effect: Any, tmp_path: Path) -> None:
    table = pa.table({"text": ["## History\nprose", "plain prose", "## Synthesis\nmore", "x"]})
    pq.write_table(table, tmp_path / "chunks.parquet")
    assert effect.heading_start_share(tmp_path / "chunks.parquet") == pytest.approx(0.5)


def test_judged_wrong_counts_the_x_prefixed_rows(effect: Any, tmp_path: Path) -> None:
    judged = tmp_path / "heading-sample-judged.txt"
    judged.write_text("doc-1\t## History\tprose\nX\tdoc-2\t## Not one\tprose\ndoc-3\t## Synthesis\tprose\n")
    assert effect.judged_wrong(judged) == (1, 3)
    assert effect.judged_wrong(tmp_path / "absent.txt") == (0, 0)


def test_main_writes_effects_missing_pairs_and_carried_provenance(
    effect: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    perquery = cache / "scores" / "perquery"
    n = 30
    _write_cell(perquery, _UNMARKED, "markdown", {f"q{i}": 0.5 for i in range(n)})
    _write_cell(perquery, _MARKED, "markdown", {f"q{i}": 0.55 for i in range(n)})
    _write_cell(perquery, _MARKED, "fast", {f"q{i}": 0.5 for i in range(n)})  # no unmarked twin
    for corpus in (_MARKED, _UNMARKED):
        d = cache / "chunks" / f"{corpus}__markdown-t256-o0-gpt2"
        d.mkdir(parents=True)
        pq.write_table(
            pa.table({"text": ["## H\nx", "y"] if corpus == _MARKED else ["H\nx", "y"]}), d / "chunks.parquet"
        )
    stamp = {"host": "bench-box", "semdex": "1.2.3"}
    (cache / "scores" / "mldr_md_chunk_scores.json").write_text(
        json.dumps(
            {
                f"{_MARKED}__markdown-t256-o0-gpt2__fake-embedder": {
                    "corpus": _MARKED,
                    "embedding": _LABEL,
                    "measured_on": stamp,
                }
            }
        )
    )
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "chunk_sets": [
                    {"corpus": _MARKED, "profile": "markdown-t256-o0-gpt2", "rows": 12, "token_p50": 200},
                    {"corpus": _UNMARKED, "profile": "markdown-t256-o0-gpt2", "rows": 10, "token_p50": 210},
                ]
            }
        )
    )
    slice_dir = tmp_path / _MARKED
    slice_dir.mkdir()
    slice_dir.joinpath("meta.json").write_text(
        json.dumps({"corpus": _MARKED, "marked_lines": 7, "marked_docs": 3, "total": 4})
    )
    slice_dir.joinpath("heading-sample-judged.txt").write_text("a\t## A\tp\nX\tb\t## B\tp\n")
    out = tmp_path / "out.json"
    monkeypatch.setenv("CACHE_ROOT", str(cache))
    monkeypatch.setenv("MDSTRUCT_STRATEGIES", "markdown,fast")
    monkeypatch.setenv("MDSTRUCT_EMBEDDINGS", _LABEL)
    monkeypatch.setenv("MDSTRUCT_SLICE_DIR", str(slice_dir))
    monkeypatch.setenv("MDSTRUCT_AUDIT", str(audit))
    monkeypatch.setenv("OUT", str(out))
    assert effect.main() == 0
    payload = json.loads(out.read_text())
    assert payload["measured_on"]["runs"] == [stamp], "provenance comes from the scorer's stamp, not this machine"
    (row,) = payload["effects"]
    assert row["strategy"] == "markdown" and row["embedding"] == _LABEL
    assert row["mean_delta"] == pytest.approx(0.05) and row["resolved"]
    assert (row["marked_rows"], row["unmarked_rows"]) == (12, 10)
    assert (row["marked_token_p50"], row["unmarked_token_p50"]) == (200, 210)
    assert row["marked_heading_start_share"] == pytest.approx(0.5)
    assert payload["missing"] == [
        {"strategy": "fast", "embedding": _LABEL, "missing": [f"{_UNMARKED}__fast-t256-o0-gpt2__fake-embedder"]}
    ]
    assert payload["slice"]["marked_lines"] == 7
    assert (payload["slice"]["sample_judged_wrong"], payload["slice"]["sample_judged"]) == (1, 2)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_bench_markdown_structure_effect.py -q -p no:cacheprovider`
Expected: every test errors at the fixture (script missing).

- [ ] **Step 3: Write the script**

```python
#!/usr/bin/env python
# pyright: basic
# Benchmark harness on numpy (no strict stubs); strict mode would only add reportUnknown* noise.
# Same stance as score_query_length_effect.py.
"""Does marking the headings change what each chunk strategy retrieves? Marked against unmarked.

``mldr_en_8k_md_slice`` is ``mldr_en_8k_slice`` with its bare-line section headings prefixed
``## `` (scripts/build_mldr_markdown_slice.py): the same documents, the same queries and qrels.
The exporter pairs cells within ONE corpus, so this script pairs each marked cell with its
unmarked twin across the two corpus ids, per strategy and embedder, and runs the shared paired
bootstrap over the per-query nDCG@10 arrays the scorer wrote. Positive means the marked corpus won.

Beside each pair it carries what the markers did to the chunks (count and median size of both
cells from the chunk-dimension audit, and the share of marked chunks that begin at a heading) and
the heading rule's reviewed precision (``heading-sample-judged.txt`` beside the slice, rows the
reviewer prefixed ``X<TAB>`` are wrong). A missing twin is reported by name, never skipped in
silence; a pair whose query id sets differ is refused. Provenance is carried from the scorer's
stamps, never taken from this machine.

Env:
  CACHE_ROOT           cache dir (default /embeddings)
  MDSTRUCT_MARKED      marked corpus id (default mldr_en_8k_md_slice)
  MDSTRUCT_UNMARKED    unmarked twin (default mldr_en_8k_slice)
  MDSTRUCT_STRATEGIES  comma list (default markdown,recursive,fast,semantic)
  MDSTRUCT_EMBEDDINGS  comma list of embedder labels (default the six the strategy tables carry)
  MDSTRUCT_SLICE_DIR   marked slice dir (default /corpora/mldr-slices/mldr_en_8k_md_slice)
  MDSTRUCT_AUDIT       chunk-dimension audit (default tests/benchmarks/raw/chunk-dimension-audit.json)
  OUT                  results json (default tests/benchmarks/raw/markdown-structure-effect.json)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _provenance import MEASURED_ON
from _score_stats import paired_ci
from export_bench_raw import measured_on_summary

_ROOT = Path(__file__).resolve().parent.parent
_PROFILE = "{strategy}-t256-o0-gpt2"
_METRIC_KEYS = ("ndcg", "ndcg@10")
_MARK = "## "
_DEFAULT_EMBEDDINGS = (
    "model2vec:potion-retrieval-32M",
    "model2vec:potion-base-8M",
    "fastembed:bge-base",
    "ollama:bge-m3",
    "ollama:qwen3-embedding-4b",
    "ollama:qwen3-embedding-8b",
)

__all__ = ["heading_start_share", "judged_wrong", "main", "pair_cells"]


def _cache_root() -> Path:
    return Path(os.environ.get("CACHE_ROOT", "/embeddings"))


def _dirsafe(label: str) -> str:
    return label.replace(":", "-").replace("/", "-")


def _cell(corpus: str, strategy: str, label: str) -> str:
    return f"{corpus}__{_PROFILE.format(strategy=strategy)}__{_dirsafe(label)}"


def _per_query(perquery_dir: Path, cell: str) -> dict[str, float] | None:
    path = perquery_dir / f"{cell}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        key = next((k for k in _METRIC_KEYS if k in data), None)
        if key is None:
            raise KeyError(f"{path} holds no nDCG array: {list(data)}")
        return {str(q): float(v) for q, v in zip(data["qids"], data[key], strict=True)}


def _round_paired(paired: dict[str, Any]) -> dict[str, Any]:
    out = dict(paired)
    for key in ("mean_delta", "ci_lo", "ci_hi"):
        out[key] = round(float(out[key]), 4)
    return out


def pair_cells(perquery_dir: Path, *, strategy: str, label: str, marked: str, unmarked: str) -> dict[str, Any] | None:
    """The paired delta (marked minus unmarked) for one strategy and embedder; None when a twin is unscored."""
    marked_cell, unmarked_cell = _cell(marked, strategy, label), _cell(unmarked, strategy, label)
    left, right = _per_query(perquery_dir, marked_cell), _per_query(perquery_dir, unmarked_cell)
    if left is None or right is None:
        return None
    if set(left) != set(right):
        raise ValueError(f"{marked_cell} against {unmarked_cell}: query ids differ, the twins are not twins")
    return {
        "strategy": strategy,
        "embedding": label,
        "marked_cell": marked_cell,
        "unmarked_cell": unmarked_cell,
        **_round_paired(paired_ci(left, right)),
    }


def heading_start_share(parquet: Path) -> float:
    """The share of chunks whose text begins at a heading mark, read column-wise in row groups."""
    reader = pq.ParquetFile(parquet)
    starts = total = 0
    for batch in reader.iter_batches(columns=["text"], batch_size=65536):
        texts = batch.column("text").to_pylist()
        total += len(texts)
        starts += sum(1 for t in texts if t.startswith(_MARK))
    return round(starts / total, 4) if total else 0.0


def judged_wrong(path: Path) -> tuple[int, int]:
    """(rows the reviewer marked wrong, rows judged) from the review file; (0, 0) when there is none."""
    if not path.exists():
        return 0, 0
    rows = [line for line in path.read_text().splitlines() if line.strip()]
    return sum(1 for line in rows if line.startswith("X\t")), len(rows)


def _missing_cells(perquery_dir: Path, cells: tuple[str, str]) -> list[str]:
    return [cell for cell in cells if not (perquery_dir / f"{cell}.npz").exists()]


def _audit_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    doc = json.loads(path.read_text()) if path.exists() else {"chunk_sets": []}
    return {(r["corpus"], r["profile"]): r for r in doc["chunk_sets"]}


def _dimensions(
    row: dict[str, Any], audit: dict[tuple[str, str], dict[str, Any]], cache: Path, marked: str, unmarked: str
) -> dict[str, Any]:
    profile = _PROFILE.format(strategy=row["strategy"])
    m, u = audit.get((marked, profile), {}), audit.get((unmarked, profile), {})
    parquet = cache / "chunks" / f"{marked}__{profile}" / "chunks.parquet"
    return {
        "marked_rows": m.get("rows"),
        "unmarked_rows": u.get("rows"),
        "marked_token_p50": m.get("token_p50"),
        "unmarked_token_p50": u.get("token_p50"),
        "marked_heading_start_share": heading_start_share(parquet) if parquet.exists() else None,
    }


def _score_rows(cache: Path, corpora: tuple[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((cache / "scores").glob("*_scores.json")):
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            rows.extend(r for r in data.values() if isinstance(r, dict) and r.get("corpus") in corpora)
    return rows


def _slice_facts(slice_dir: Path) -> dict[str, Any]:
    meta_path = slice_dir / "meta.json"
    facts = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    wrong, judged = judged_wrong(slice_dir / "heading-sample-judged.txt")
    return {**facts, "sample_judged_wrong": wrong, "sample_judged": judged}


def main() -> int:
    cache = _cache_root()
    marked = os.environ.get("MDSTRUCT_MARKED", "mldr_en_8k_md_slice")
    unmarked = os.environ.get("MDSTRUCT_UNMARKED", "mldr_en_8k_slice")
    strategies = [s for s in os.environ.get("MDSTRUCT_STRATEGIES", "markdown,recursive,fast,semantic").split(",") if s]
    labels = [e for e in os.environ.get("MDSTRUCT_EMBEDDINGS", ",".join(_DEFAULT_EMBEDDINGS)).split(",") if e]
    slice_dir = Path(os.environ.get("MDSTRUCT_SLICE_DIR", f"/corpora/mldr-slices/{marked}"))
    audit = _audit_index(
        Path(os.environ.get("MDSTRUCT_AUDIT", _ROOT / "tests/benchmarks/raw/chunk-dimension-audit.json"))
    )
    out = Path(os.environ.get("OUT", _ROOT / "tests/benchmarks/raw/markdown-structure-effect.json"))
    perquery_dir = cache / "scores" / "perquery"

    effects: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for strategy in strategies:
        for label in labels:
            row = pair_cells(perquery_dir, strategy=strategy, label=label, marked=marked, unmarked=unmarked)
            if row is None:
                cells = (_cell(marked, strategy, label), _cell(unmarked, strategy, label))
                missing.append(
                    {"strategy": strategy, "embedding": label, "missing": _missing_cells(perquery_dir, cells)}
                )
                continue
            effects.append({**row, **_dimensions(row, audit, cache, marked, unmarked)})
    payload = {
        MEASURED_ON: measured_on_summary(_score_rows(cache, (marked, unmarked))),
        "note": (
            "Paired per-query nDCG@10 delta (marked minus unmarked) between a cell on the MLDR English "
            "slice with its section headings marked '## ' and the same strategy and embedder on the "
            "unmarked slice: identical documents, queries and qrels, only the markup differs. "
            "resolved=false means the 95 percent interval spans zero. missing names every pair a "
            "twin is unscored for."
        ),
        "slice": _slice_facts(slice_dir),
        "effects": effects,
        "missing": missing,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[mdstruct] {len(effects)} pairs, {len(missing)} missing -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_bench_markdown_structure_effect.py -q -p no:cacheprovider`
Expected: `7 passed`.

- [ ] **Step 5: Lint, format, type-check, commit**

Run: `.venv/bin/ruff check scripts/score_markdown_structure_effect.py tests/test_bench_markdown_structure_effect.py && .venv/bin/ruff format scripts/score_markdown_structure_effect.py tests/test_bench_markdown_structure_effect.py && .venv/bin/pyright scripts/score_markdown_structure_effect.py`
Expected: `All checks passed!`, `0 errors`. Commit with subject `bench(chunking): [22] pair each marked MLDR English cell with its unmarked twin`.

---

### Task 5: The table registrar in `scripts/gen_bench_tables.py`

**Files:**
- Modify: `scripts/gen_bench_tables.py` (add `_register_markdown_structure_tables` next to `_register_query_length_tables`, register `("markdown-structure-effect.json", _register_markdown_structure_tables)` in `_SIMPLE_SOURCES`)
- Modify: `docs/benchmarks/03-chunking.md` (one marker block, in the new section Task 9 writes; placed by Task 8 Step 5 directly after the `chunk_knob_strategy` END marker at line 2041)
- Test: `tests/test_bench_tables_markdown_structure.py`

**Interfaces:**
- Consumes: `_table(title, columns, rows, note)`, `_verdict_cell(paired)` (both exist in the module); the raw file shape from Task 4.
- Produces: table id `markdown_structure_effect` with columns `["Strategy", "Embedder", "Marked minus unmarked", "95% CI", "Wins/losses", "Chunks (marked / unmarked)", "Median tokens (marked / unmarked)", "Chunks starting at a heading"]`.

**STOP conditions:**
- `_verdict_cell` or `_table` signatures differ from `def _verdict_cell(paired: dict[str, Any]) -> str` and `def _table(title, columns, rows, note="")`.

- [ ] **Step 1: Write the failing test**

```python
"""The markdown-structure table renders from its raw file with one row per strategy and embedder."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "gen_bench_tables.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("gen_bench_tables_mdstruct", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator() -> Any:
    return _load()


def _effect(strategy: str, delta: float, resolved: bool) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "embedding": "fastembed:bge-base",
        "mean_delta": delta,
        "ci_lo": delta - 0.01,
        "ci_hi": delta + 0.01,
        "wins": 300,
        "losses": 200,
        "ties": 300,
        "n_shared": 800,
        "resolved": resolved,
        "marked_rows": 152000,
        "unmarked_rows": 149082,
        "marked_token_p50": 205,
        "unmarked_token_p50": 208,
        "marked_heading_start_share": 0.2712,
    }


def test_the_table_has_one_row_per_pair_and_prints_the_chunk_facts(generator: Any) -> None:
    tables: dict[str, Any] = {}
    doc = {"effects": [_effect("markdown", 0.0123, True), _effect("fast", -0.0004, False)], "missing": []}
    generator._register_markdown_structure_tables(tables, doc)
    table = tables["markdown_structure_effect"]
    assert table["columns"] == [
        "Strategy",
        "Embedder",
        "Marked minus unmarked",
        "95% CI",
        "Wins/losses",
        "Chunks (marked / unmarked)",
        "Median tokens (marked / unmarked)",
        "Chunks starting at a heading",
    ]
    assert table["rows"][0] == [
        "`markdown`",
        "`fastembed:bge-base`",
        "+0.0123 resolved",
        "[+0.0023, +0.0223]",
        "300/200",
        "152,000 / 149,082",
        "205 / 208",
        "27.1%",
    ]
    assert table["rows"][1][2] == "-0.0004 unresolved"
    assert "identical documents, queries and qrels" in table["note"]


def test_missing_pairs_are_named_in_the_note(generator: Any) -> None:
    tables: dict[str, Any] = {}
    doc = {
        "effects": [_effect("markdown", 0.01, True)],
        "missing": [
            {
                "strategy": "semantic",
                "embedding": "ollama:bge-m3",
                "missing": ["x__semantic-t256-o0-gpt2__ollama-bge-m3"],
            }
        ],
    }
    generator._register_markdown_structure_tables(tables, doc)
    assert "semantic / ollama:bge-m3" in tables["markdown_structure_effect"]["note"]


def test_no_effects_registers_no_table(generator: Any) -> None:
    tables: dict[str, Any] = {}
    generator._register_markdown_structure_tables(tables, {"effects": [], "missing": []})
    assert "markdown_structure_effect" not in tables
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_bench_tables_markdown_structure.py -q -p no:cacheprovider`
Expected: `AttributeError: ... has no attribute '_register_markdown_structure_tables'`.

- [ ] **Step 3: Add the registrar and register the source**

Insert after `_register_query_length_tables` in `scripts/gen_bench_tables.py`:

```python
def _register_markdown_structure_tables(tables: dict[str, Any], doc: dict[str, Any]) -> None:
    """Marked against unmarked MLDR English: what heading markup buys each strategy, per embedder."""
    effects = doc.get("effects") or []
    if not effects:
        return
    missing = doc.get("missing") or []
    absent = "; ".join(f"{m['strategy']} / {m['embedding']}" for m in missing)
    tables["markdown_structure_effect"] = _table(
        "Heading markup on MLDR English: marked against unmarked, paired per query",
        [
            "Strategy",
            "Embedder",
            "Marked minus unmarked",
            "95% CI",
            "Wins/losses",
            "Chunks (marked / unmarked)",
            "Median tokens (marked / unmarked)",
            "Chunks starting at a heading",
        ],
        [
            [
                f"`{row['strategy']}`",
                f"`{row['embedding']}`",
                _verdict_cell(row),
                f"[{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]",
                f"{row['wins']}/{row['losses']}",
                f"{row['marked_rows']:,} / {row['unmarked_rows']:,}",
                f"{row['marked_token_p50']} / {row['unmarked_token_p50']}",
                f"{100 * row['marked_heading_start_share']:.1f}%",
            ]
            for row in effects
        ],
        "The marked corpus is the MLDR English slice with every bare-line section heading prefixed "
        "`## `; the unmarked corpus is the slice as published: identical documents, queries and qrels, "
        "so the per-query difference isolates the markup. Positive favours the marked corpus. "
        "`fast` and `semantic` are controls: the first shares `markdown`'s splitter family without "
        "reading structure, the second ignores structure entirely."
        + (f" Pairs with an unscored twin: {absent}." if absent else ""),
    )
```

Add to `_SIMPLE_SOURCES` after the `query-length-effect.json` entry:

```python
(("markdown-structure-effect.json", _register_markdown_structure_tables),)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_bench_tables_markdown_structure.py -q -p no:cacheprovider`
Expected: `3 passed`.

- [ ] **Step 5: Check the whole tables suite still passes without the raw file**

Run: `.venv/bin/python -m pytest tests/test_bench_tables_current.py -q -p no:cacheprovider`
Expected: all pass (the raw file does not exist yet, so nothing is registered and no doc references the id yet).

- [ ] **Step 6: Commit**

Subject: `bench(chunking): [22] register the markdown-structure table`.

---

### Task 6: The forest plot in `scripts/gen_bench_charts.py`

**Files:**
- Modify: `scripts/gen_bench_charts.py` (add `_markdown_structure()` collector next to `_knob_effects`, a `"markdown_structure"` key in `collect_chart_data`, `render_markdown_structure(d)` and its call in `_render_all`)
- Test: extend `tests/test_bench_charts_current.py` with one test

**Interfaces:**
- Consumes: `_read(path)`, `_RAW`, `_style()`, `_c`, `_BLUE`, `_ORANGE`, `_muted()`, `_ink()`, `_save(fig, name)` (all exist); the raw file from Task 4.
- Produces: chart name `chunk_markdown_structure` (files `docs/benchmarks/img/chunk_markdown_structure.png` per theme via `_save`), collector shape `{"rows": [{"label", "delta", "lo", "hi", "resolved"}]}` sorted by delta.

**STOP conditions:**
- `_save` or `_style` do not exist with those names.

- [ ] **Step 1: Write the failing test** (append to `tests/test_bench_charts_current.py`)

```python
@pytest.mark.os_agnostic
def test_the_markdown_structure_chart_labels_strategy_and_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One row per pair, named by strategy and embedder, ordered by delta."""
    gen = _load_generator()
    raw = tmp_path / "raw"
    raw.mkdir()
    raw.joinpath("markdown-structure-effect.json").write_text(
        json.dumps(
            {
                "effects": [
                    {
                        "strategy": "fast",
                        "embedding": "fastembed:bge-base",
                        "mean_delta": 0.001,
                        "ci_lo": -0.002,
                        "ci_hi": 0.004,
                        "resolved": False,
                    },
                    {
                        "strategy": "markdown",
                        "embedding": "fastembed:bge-base",
                        "mean_delta": 0.012,
                        "ci_lo": 0.005,
                        "ci_hi": 0.019,
                        "resolved": True,
                    },
                ]
            }
        )
    )
    monkeypatch.setattr(gen, "_RAW", raw)
    d = gen._markdown_structure()
    assert [r["label"] for r in d["rows"]] == ["fast bge-base", "markdown bge-base"]
    assert d["rows"][1]["resolved"] is True
    assert "markdown_structure" in gen.collect_chart_data(), "the collector is wired into the manifest"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_bench_charts_current.py -q -p no:cacheprovider -k markdown_structure`
Expected: `AttributeError: ... '_markdown_structure'`.

- [ ] **Step 3: Add collector, renderer and wiring**

After `_knob_effects` in `scripts/gen_bench_charts.py`:

```python
def _markdown_structure() -> dict[str, Any]:
    """Marked against unmarked MLDR English, one paired delta per strategy and embedder."""
    doc = _read(_RAW / "markdown-structure-effect.json")
    if not doc:
        return {"rows": []}
    rows = [
        {
            "label": f"{e['strategy']} {e['embedding'].split(':')[1]}",
            "delta": e["mean_delta"],
            "lo": e["ci_lo"],
            "hi": e["ci_hi"],
            "resolved": e["resolved"],
        }
        for e in doc["effects"]
    ]
    return {"rows": sorted(rows, key=lambda r: r["delta"])}
```

In `collect_chart_data` add `"markdown_structure": _markdown_structure(),` after `"chunk_knob_effects"`.

After `_render_one_knob_axis`:

```python
def render_markdown_structure(d: dict[str, Any]) -> None:
    """Forest plot: the paired delta of heading markup per strategy and embedder against zero.

    Same form as the knob plots and for the same reason: the question is whether the difference
    resolves, and a bar chart of two means hides exactly that.
    """
    rows = d["rows"]
    if not rows:
        return
    plt = _style()
    fig, ax = plt.subplots(figsize=(7.6, max(2.2, 0.26 * len(rows) + 1.0)))
    ys = list(range(len(rows)))
    for y, row in zip(ys, rows, strict=True):
        resolved = row["resolved"]
        colour = _c(_BLUE if row["delta"] > 0 else _ORANGE) if resolved else _muted()
        ax.plot(
            [row["lo"], row["hi"]],
            [y, y],
            color=colour,
            lw=2.1 if resolved else 1.1,
            alpha=1.0 if resolved else 0.5,
            solid_capstyle="round",
        )
        ax.plot([row["delta"]], [y], "o", color=colour, ms=5.0 if resolved else 3.2, alpha=1.0 if resolved else 0.5)
    ax.axvline(0, color=_ink(), lw=1.1, alpha=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=7.5)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.tick_params(labelsize=7.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("paired nDCG@10 difference, 95% CI (positive favours the marked corpus)", fontsize=8)
    resolved_count = sum(1 for r in rows if r["resolved"])
    ax.set_title(
        f"heading markup on MLDR English: {resolved_count} of {len(rows)} pairs resolve; faded bars cross zero",
        color=_ink(),
        fontsize=10,
        loc="left",
        pad=8,
    )
    fig.tight_layout()
    _save(fig, "chunk_markdown_structure")
    plt.close(fig)
```

In `_render_all` add `render_markdown_structure(data["markdown_structure"])` after `render_chunk_knob_effects(...)`.

- [ ] **Step 4: Run the charts tests**

Run: `.venv/bin/python -m pytest tests/test_bench_charts_current.py -q -p no:cacheprovider`
Expected: the new test passes; `test_committed_charts_match_current_raw_data` FAILS because the manifest gained a key. Regenerate: `.venv/bin/python scripts/gen_bench_charts.py` (the raw file does not exist yet, so the new entry hashes an empty row list and no image is written), then the suite passes.

- [ ] **Step 5: Commit**

`git add scripts/gen_bench_charts.py tests/test_bench_charts_current.py docs/benchmarks/img/charts.manifest.json`, subject `bench(chunking): [22] forest plot for the markdown-structure pairs`.

---

### Task 7: The exporter learns the marked corpus

**Files:**
- Modify: `scripts/export_bench_raw.py:90-98` (the `chunk-sweep-mldr.json` spec)
- Test: `tests/test_bench_export_corpus_guard.py` (extend one test)

**Interfaces:**
- Produces: `chunk-sweep-mldr.json` sources `["mldr_chunk_scores.json", "mldr_md_chunk_scores.json"]`, corpora `["mldr_de_3k_slice", "mldr_en_8k_md_slice", "mldr_en_8k_slice"]`, and a note that names the marked twin.

**STOP conditions:**
- the export's unscored-cell guard names a marked cell when Task 8 runs it: the sweep is not finished, do not withhold, wait.

- [ ] **Step 1: Write the failing test** (append to `tests/test_bench_export_corpus_guard.py`)

```python
def test_the_mldr_export_publishes_the_marked_twin_from_its_own_source_file(exporter: Any) -> None:
    spec = exporter._EXPORTS["chunk-sweep-mldr.json"]
    assert "mldr_md_chunk_scores.json" in spec["sources"]
    assert "mldr_en_8k_md_slice" in spec["corpora"]
    assert "heading" in spec["note"], "the note must say the third corpus is the marked twin, not a new body"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_bench_export_corpus_guard.py -q -p no:cacheprovider -k marked_twin`
Expected: `AssertionError`.

- [ ] **Step 3: Widen the allowlist and the note together**

Replace the `chunk-sweep-mldr.json` entry:

```python
    "chunk-sweep-mldr.json": {
        # The marked twin is a THIRD corpus id on purpose: the scorer refuses to merge a foreign
        # corpus into mldr_chunk_scores.json, so its cells live in their own source file, and the
        # export publishes it beside the unmarked slice so the strategy tables pair within it.
        "sources": ["mldr_chunk_scores.json", "mldr_md_chunk_scores.json"],
        "corpora": ["mldr_de_3k_slice", "mldr_en_8k_md_slice", "mldr_en_8k_slice"],
        "note": (
            "Chunk-parameter sweep on long documents (MLDR: 18.5 chunks/doc en, 67.3 de at "
            "recursive-t256-ov0). This is the body the chunk axes can actually be judged on. "
            "mldr_en_8k_md_slice is the English slice with its section headings marked '## ' "
            "(same documents, queries and qrels); the cross-corpus pairing is in "
            "markdown-structure-effect.json."
        ),
        "fit_for_chunk_claims": True,
    },
```

- [ ] **Step 4: Run the export tests**

Run: `.venv/bin/python -m pytest tests/test_bench_export_corpus_guard.py tests/test_bench_export_unscored_guard.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Commit**

Subject: `bench(export): [22] the MLDR export publishes the marked twin from its own score file`.

---

### Task 8: After `ITEM22-DONE`: audit, effect, export, tables, charts

**Files:**
- Modify (generated): `tests/benchmarks/raw/chunk-dimension-audit.json`, `tests/benchmarks/raw/chunk-sweep-mldr.json`, `tests/benchmarks/raw/chunk-knob-effects.json` (and siblings the exporter rewrites), `tests/benchmarks/raw/markdown-structure-effect.json`, `docs/benchmarks/*.md` generated blocks, `docs/benchmarks/img/*`, `docs/benchmarks/tables.manifest.json` (or the name `gen_bench_tables._MANIFEST` points at), `docs/benchmarks/img/charts.manifest.json`.
- Modify: `docs/benchmarks/03-chunking.md` - add the marker block for the new table where Task 9 writes the section.

**STOP conditions:**
- any RC line in the item22 log is non-zero.
- the export names an unscored marked cell.
- `markdown-structure-effect.json` has a non-empty `missing` list: score the named cell first (the wrapper's phase 3 can be re-run; it skips cached cells).

- [ ] **Step 1: Confirm the sweep**

Run: `grep -E '^RC_|ITEM22' "$(ls -t ~/semdex-sweep-run/logs/item22-markdown-structure-*.log | head -1)"`
Expected: `RC_CHUNK=0`, `RC_GPU=0`, `RC_CPU=0`, `RC_SCORE=0`, `ITEM22-DONE`.

- [ ] **Step 2: Audit every chunk set (all corpora, or the fit filter drops the others)**

Run: `cd ~/semdex-sweep-run && ./audit-chunks.sh && grep RC_AUDIT logs/audit-*.log | tail -1`
Expected: `RC_AUDIT=0`; `tests/benchmarks/raw/chunk-dimension-audit.json` now holds four `mldr_en_8k_md_slice` rows.

- [ ] **Step 3: Run the paired script**

Run: `CACHE_ROOT=/embeddings .venv/bin/python scripts/score_markdown_structure_effect.py`
Expected: `[mdstruct] 24 pairs, 0 missing -> .../markdown-structure-effect.json`.

- [ ] **Step 4: Export**

Run: `cd ~/semdex-sweep-run && ./plan-a-export.sh; tail -3 logs/plan-a-export-*.log | tail -3`
Expected: `PLAN-A-EXPORT-OK`. (It re-runs the audit first; harmless.)

- [ ] **Step 5: Place the table marker and regenerate tables and charts**

In `docs/benchmarks/03-chunking.md`, directly after the END marker of the `chunk_knob_strategy`
block, insert a new subsection heading `### When the structure is there` followed by an empty line
and an empty BEGIN/END marker pair for the id `markdown_structure_effect`, written in exactly the
form of the `chunk_knob_strategy` pair (copy those two lines and swap the id). This plan does not
spell the two lines out on purpose: the generator and its tests scan every `.md` under `docs/`,
plans included, so a literal marker pair here would be spliced into and checked as a live block.

Then: `.venv/bin/python scripts/gen_bench_tables.py && .venv/bin/python scripts/gen_bench_charts.py`
Expected: `[tables] docs/benchmarks/03-chunking.md: 6 blocks`, `wrote 16 charts x 2 themes + manifest`.

- [ ] **Step 6: Run the generated-data gates**

Run: `.venv/bin/python -m pytest tests/test_bench_tables_current.py tests/test_bench_charts_current.py tests/test_bench_chunk_dimensions_integrity.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 7: Commit the data and generated blocks**

`git add tests/benchmarks/raw docs/benchmarks` with subject `bench(chunking): [22] the 24 marked cells, their paired effect, tables and chart`. (The prose is still stale at this point and the claim gate is expected to fail on the headline counts; Task 9 fixes it.)

---

### Task 9: The write-up and the claims

**Files:**
- Modify: `docs/benchmarks/03-chunking.md` (the new section under `### When the structure is there`; the paragraph at lines 2086-2090 that says "no corpus here carries markdown structure at all - a scan of the MLDR English documents found no headings to speak of"; the strategy summary bullet in the headline section; the caveat under the strategy table)
- Modify: `docs/benchmarks/07-selection.md:48-53` (item 6)
- Modify: `docs/benchmarks/08-gaps.md:490-491` (the "No corpus is markdown" bullet) and `:506-509` (the "To close it" sentence: drop "build one markdown-native corpus", add "measure heading levels and a non-Wikipedia markdown body")
- Modify: `tests/benchmarks/claims/03-chunking.toml` (`[sets].fit_corpora` gains `"mldr_en_8k_md_slice"`; a `[rowsets.mdstruct]` with `source = "markdown-structure-effect.json"`, `path = "effects"`; `[rowsets.mdstruct_slice]` with `path = "slice"`; one `[[claim]]` per figure), `07-selection.toml`, `08-gaps.toml` as figures appear there.

**Interfaces:**
- Consumes: `tests/benchmarks/raw/markdown-structure-effect.json` (the only source of every new figure), the regenerated table.

**STOP conditions:**
- `check_bench_claims.py` reports a quote missing after a reflow: the quote crossed a line break; re-lay the paragraph, never shorten the figure.

- [ ] **Step 1: Read the result before writing a word**

Run: `.venv/bin/python -c "import json; d=json.load(open('tests/benchmarks/raw/markdown-structure-effect.json')); [print(f\"{e['strategy']:10s} {e['embedding']:32s} {e['mean_delta']:+.4f} [{e['ci_lo']:+.4f},{e['ci_hi']:+.4f}] {e['wins']}/{e['losses']} {'RES' if e['resolved'] else '---'} rows {e['marked_rows']}/{e['unmarked_rows']} p50 {e['marked_token_p50']}/{e['unmarked_token_p50']} hs {e['marked_heading_start_share']}\") for e in d['effects']]; print(d['slice'])"`

Pre-registered readings (decide from the numbers, not the wording): if `markdown` and `recursive` resolve positive for most embedders and `fast`/`semantic` do not, the page says the structure-aware strategies gain from headings and the controls show the text change alone did nothing. If `fast` moves as much as `markdown`, the effect is the extra characters, not the structure, and the page says so. If nothing resolves, the page says heading markup at one level on Wikipedia-shaped text is not worth a setting.

- [ ] **Step 2: Write the section in `03-chunking.md`**

Under `### When the structure is there` (above the marker block placed in Task 8), write four paragraphs in this order, each figure taken from the JSON: (1) what the twin is (marked_lines headings in marked_docs of 8,000 documents, sample_judged_wrong of sample_judged reviewed rows judged wrong, one heading level); (2) the answer for `markdown` and `recursive` per embedder (count of resolved pairs, the range of deltas); (3) the two controls; (4) what the markers did to the chunks (rows and median tokens, heading-start share). Then correct the paragraph at lines 2086-2090 to: "Only one corpus here carries markdown structure, and it is a re-rendering: MLDR English's Wikipedia text keeps its section headings as bare lines, so the twin slice above marks them and measures the difference; the earlier scan that found 'no headings' looked for `#` markers. Every other `markdown` verdict on this page is a verdict about its fallback rules." Update the headline summary bullet for `strategy` and the caveat under the strategy table to mention the section.

- [ ] **Step 3: Update 07 item 6 and 08 gap 13**

07 item 6: append one sentence giving the marked result for the structure-aware strategies with its figure. 08: replace the "No corpus is markdown" bullet with what is still open (heading levels; a markdown body of another domain, filed as [24]); edit the "To close it" list accordingly.

- [ ] **Step 4: Add the claims**

For every figure written in Steps 2-3, add a `[[claim]]` in the page's TOML with `id`, `quote` (one physical line, unique), `published`, `rows`, `reduce`, `field`, `where` and `why`, following the existing entries in `tests/benchmarks/claims/03-chunking.toml` (for example `reduce = "value"`, `field = "mean_delta"`, `where = { strategy = "markdown", embedding = "fastembed:bge-base" }` on `rows = "mdstruct"`; `reduce = "count"` with `where = { strategy = "markdown", resolved = true }` for a resolved count; `rows = "mdstruct_slice"`, `field = "marked_lines"` for the heading count). Add `"mldr_en_8k_md_slice"` to `[sets].fit_corpora` and re-derive the headline counts the gate then reports as changed.

- [ ] **Step 5: Run the claim gate until it is quiet**

Run: `.venv/bin/python scripts/check_bench_claims.py`
Expected: `N examined, 0 failing` with N above 661. Fix each report by re-pointing the quote or correcting the figure, never by deleting the claim.

- [ ] **Step 6: Full gate, commit, push**

Run `make test` through the gate jig (read the RC from its log, not the tail), then commit `docs/benchmarks tests/benchmarks/claims` with subject `bench(chunking): [22] what heading markup buys the structure-aware strategies on MLDR English`, and `git push`. CI is expected to fail at steps=0 (backlog [15], billing); the local gate is the validation.

- [ ] **Step 7: Close the backlog line and decide [24]**

In `OPEN-WORK.md`: `[22]` becomes `- [x] ... | closed: 2026-<date>, <one-line result>`. `[24]` gets its `open:` field updated with the decision the result implies (a markdown-native body is worth running if the structure-aware strategies gained; otherwise it stays parked with the reason). Append the decision to `EXECUTION-USER-REVIEW.md`. Commit `OPEN-WORK.md` alone with subject `docs(backlog): close [22]; [24] decided on its result`, push.

---

## Self-review against the design

- Component A (builder, guarantees, sample, meta, idempotent): Task 1 and 2. The "list item" guarantee is pinned by `test_a_list_item_before_prose_is_not_a_heading`; "formula" and "See also" by the parametrized test; strip-back identity by its own test and by the builder's per-document assertion.
- Component B (sweep, two chains, chunk first, audit over all corpora, score four profiles for six embedders): Task 3 and Task 8 Steps 1-2. The missing unmarked `semantic`/`bge-m3` twin is covered by running the GPU chain over both corpora.
- Component C (paired script, refuse mismatched ids, report missing by name, provenance carried, chunk facts): Task 4.
- Component D (table, chart, 03 section and corrected sentence, 07 item 6, 08 entry, claims, fit_corpora): Tasks 5, 6, 7, 8, 9.
- Type consistency: `pair_cells` returns `marked_cell`/`unmarked_cell` and the paired fields; `_dimensions` adds `marked_rows`, `unmarked_rows`, `marked_token_p50`, `unmarked_token_p50`, `marked_heading_start_share`; the table registrar and the chart collector read exactly those keys.
- Placeholder scan: Task 9's figures come from the JSON at run time, which is inherent to a write-up, and Step 1 pre-registers how each outcome is read.
