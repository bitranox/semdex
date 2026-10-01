# Effect Floors Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use bitranox:process-agents-subagent-driven-development (recommended) or bitranox:process-plan-executor to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the effect tables print how many resolutions chance alone produces, count only material differences as findings, and judge each numeric ladder once by its end-to-end pair instead of tallying partners.

**Architecture:** One shared constant (the material floor, beside the bootstrap alpha in `scripts/_score_stats.py`) is stamped by the exporter into every effect row as `material` and into the payload header; the table generator reads the stamp and the header, adds two columns to the knob summary, splits its direction column by axis type through one general ladder grouping, and reports material steps in the ceiling table; the claim checker needs no code, since the stamped field is filterable as it stands. The design is `docs/plans/2026-09-27-effect-floors-design.md`.

**Tech Stack:** Python 3.10+ (scripts run under the project `.venv`), numpy, pytest; the `scripts/` env-config convention (no argparse); `# pyright: basic` in the two harness modules touched, strict elsewhere.

## Global Constraints

- Work on `main`, commit after every task, push after the gate; never a PR (repo rule).
- Every new figure in prose gets a `[[claim]]` in the page's file under `tests/benchmarks/claims/`; a quote sits on ONE physical line and is unique in the page's prose.
- ASCII punctuation only in files (no em-dash, no arrow character, no curly quotes).
- Never spell out a generated-table BEGIN/END marker pair in any file under `docs/` (plans included); name a block by its id.
- The material floor is `0.005` on `ndcg@10`, absolute; alpha is `0.05`. Both live in `scripts/_score_stats.py` and nowhere else; the exporter stamps them into `chunk-knob-effects.json`; the generator READS them from that file and never restates the values.
- `resolved` keeps its meaning and its count in every row: no existing claim's derivation changes.
- A ladder axis is one whose levels are all numeric (`int`/`float`, never `bool`); it is detected from the level values, not from a list.
- A brief's test code can carry a bug; fix it minimally and disclose it, never weaken an assertion.
- Never `pkill -f`; the [22] sweep runs on this box (two workers, log under `~/semdex-sweep-run/logs/`); run the re-export under `nice -n 19 ionice -c3` and judge it by its RC line.

---

## Ground truth read before planning (2026-09-27)

- `scripts/_score_stats.py` (`# pyright: basic`): `__all__ = ["bootstrap_ci", "paired_ci"]`; `_DEFAULT_BOOT = 10000`, `_DEFAULT_SEED = 20260806`; both functions take `alpha: float = 0.05`. `paired_ci` returns `mean_delta, ci_lo, ci_hi, wins, ties, losses, n_shared, resolved`.
- `scripts/export_bench_raw.py`: `from _score_stats import paired_ci` (line 57); `knob_effects(rows, metric="ndcg@10")` at line 484 builds rows through `_effect_row(axis, left, right, metric, paired)` at line 540, whose dict ends with `"resolved": paired["resolved"]`; `_write_knob_effects(out_dir)` at line 637 writes `{MEASURED_ON: ..., "note": ..., "effects": effects}` with `sort_keys=True`. `main()` runs every `_EXPORTS` entry unless `ONLY` names some, then always `_write_knob_effects(out_dir)`, then `_write_product_k`. `OUT_DIR` defaults to `tests/benchmarks/raw`; per-query arrays come from `SEMDEX_SCORE_PERQUERY` (default `/embeddings/scores/perquery`, present, 40 MB). A full export took 11 minutes on 2026-09-27 (10:30:48 to 10:41:46).
- `tests/test_bench_export_axes.py`: loads the exporter by path (`_load()` registers it in `sys.modules`), builds cells with `_cell(embedding, *, method=None, max_tokens=256)`, writes per-query arrays with `_write_per_query(directory, cell, scores, exporter)` (npz with `qids` and `exporter.NPZ_KEYS["ndcg@10"]`), and points the exporter at them with `monkeypatch.setenv("SEMDEX_SCORE_PERQUERY", str(tmp_path))`.
- `scripts/gen_bench_tables.py`: `_CHUNK_AXES` (line 54), `_table(title, columns, rows, note="")` (127), `level_label(level)` (245), `_favoured_level(effect)` (328), `_knob_summary_table(effects_doc, fit=None, axes=None)` (333 to 375), `_held_fixed_label(held, axis)`, `_oriented(effect)`, `_ladders_by_embedder(effects, fit)` (697, overlap only, key `(corpus, held, cap, embedding)`), `_rung_ceiling_table(effects, fit)` (723 to 775, columns `Corpus, Held fixed, Embedder, Steps resolved, Highest resolved step, Last step, Last step delta`), `_register_effect_tables` (778) registering `chunk_knob_summary`, `chunk_knob_summary_chunking`, `chunk_overlap_rungs`, `chunk_overlap_ceiling`; `_load(name)`, `_fit_corpora(audit)`; `main()` regenerates every block under `docs/` and writes `docs/benchmarks/generated-tables.manifest.json`. Module imports `itertools` already; check whether it imports `math` before adding it.
- `tests/test_bench_tables_current.py` regenerates and compares every block, requires every registered table non-empty and carrying a `note`, and pins that `chunk_knob_summary_chunking` carries only the four chunk axes while `chunk_knob_summary` carries `embedding` and `method` too.
- `tests/test_bench_tables_product_k.py` is the pattern for a generator unit test: `_load()` by path, a `generator` module fixture, synthetic rows.
- `scripts/check_bench_claims.py`: `orient_rows` (142) copies each row through `generator._oriented` and adds `favoured_level`, `unfavoured_level`, `favoured_end`, `levels`; every other row field, including a stamped `material`, is already filterable. `tests/benchmarks/claims/README.md` documents the oriented fields at its `## Oriented rowsets` section.
- Level types in the committed file: `max_tokens` and `overlap_tokens` are `int`; `breakpoint_model` mixes `str` and `None`; `embedding`, `method`, `strategy` are `str`. Every ladder in the file carries ALL pairs, so the end-to-end pair exists for each.
- Counts in the committed file (all corpora): overlap 1023 comparisons, 533 resolved, 395 at or above 0.005; strategy 332/121/116; max_tokens 184/115/111; breakpoint_model 200/43/43; embedding 1239/1053/1047; method 154/123/123. The pages show smaller denominators because `_fit_corpora` drops corpora that cannot carry a chunking claim.
- Pages carrying `chunk_knob_summary`: `docs/benchmarks/04-embedding.md`, `docs/benchmarks/07-selection.md`; carrying `chunk_knob_summary_chunking` and `chunk_overlap_ceiling`: `docs/benchmarks/03-chunking.md`. The gap paragraph is `docs/benchmarks/08-gaps.md` lines 494 to 497, and its closing sentence at 507 to 508 ("and put a chance floor and a material-difference floor into the effect tables").
- Tally sentences the write-up replaces: 03-chunking.md "Every one of GerDaLIR's 496 resolved overlap comparisons favours MORE overlap; 22 of MLDR's 34 favour LESS." (claims `headline-gerdalir-overlap-upward`, `headline-mldr-overlap-downward`, `headline-mldr-overlap-resolved`) and the summary bullet "On GerDaLIR, all 496 resolved comparisons favour MORE overlap." (claim `summary-gerdalir-upward`); 03-chunking.md "Chunk size resolves in 109 of its 152 comparisons: 91 favour the smaller chunk ... 18 favour the larger" (find its claims by `grep -n '"109"\|"91"\|"18"' tests/benchmarks/claims/03-chunking.toml`).

---

### Task 1: The material floor, stamped by the exporter

**Files:**
- Modify: `scripts/_score_stats.py:27-33` (constants and `__all__`), append one function
- Modify: `scripts/export_bench_raw.py:57` (import), `:484-489` (metric guard), `:540-565` (`_effect_row`), `:637-660` (`_write_knob_effects` payload)
- Test: `tests/test_bench_export_material.py` (create)
- Regenerate and commit: `tests/benchmarks/raw/chunk-knob-effects.json` (and any other raw file the export rewrites)

**Interfaces:**
- Consumes: `paired_ci` result dict; `_effect_row`, `_write_knob_effects` as read above.
- Produces: `_score_stats.DEFAULT_ALPHA: float = 0.05`, `_score_stats.MATERIAL_FLOOR: float = 0.005`, `_score_stats.MATERIAL_FLOOR_METRIC: str = "ndcg@10"`, `_score_stats.is_material(paired: dict[str, Any], *, floor: float = MATERIAL_FLOOR) -> bool`; every effect row carries `"material": bool`; the payload carries `"alpha": 0.05` and `"material_floor": {"metric": "ndcg@10", "value": 0.005}` beside `"effects"`.

**Out of scope** - do NOT touch, though they look related:
- `scripts/gen_bench_tables.py` - Task 2 reads the stamp; changing the generator here would make the committed pages stale mid-task.
- `scripts/score_query_power.py`, `scripts/_query_power.py` - they read `ci_lo/ci_hi/mean_delta` and are unaffected by an added field.
- `tests/benchmarks/claims/*.toml` - no existing claim's rows change; Task 3 adds claims.

**STOP conditions** - stop and report rather than improvise, if:
- `_effect_row` or `_write_knob_effects` does not match the shape described above;
- the re-export writes a `chunk-knob-effects.json` with a different `effects` count than 3132, or a different `resolved` count than 1988 (the stamp adds a field; it must move no number);
- the re-export runs past 30 minutes (2.7x the measured 11) or exits non-zero;
- a step's verification fails twice after one reasonable fix attempt.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bench_export_material.py`. It drives the real seam: `knob_effects` over per-query arrays on disk, and `_write_knob_effects` over a raw directory, and asserts on what they write.

```python
"""The exporter stamps every paired comparison with whether it is MATERIAL, not only resolved.

A 12,298-query corpus resolves a step of 0.0009 nDCG@10. Resolved says the query set can tell
the two configurations apart; material says the difference is worth acting on. The floor and
the alpha live in _score_stats and are written into the file so a table can print what the rows
were judged against, rather than restating the numbers.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "export_bench_raw.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("export_bench_raw_material", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_bench_raw_material"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exporter() -> Any:
    return _load()


def _cell(overlap: int) -> dict[str, Any]:
    axes = {"strategy": "recursive", "max_tokens": 256, "overlap_tokens": overlap, "breakpoint_model": None}
    return {"cell": f"c__recursive-t256-o{overlap}-gpt2__e", "corpus": "c", "embedding": "e", "axes": axes}


def _write_per_query(directory: Path, cell: str, scores: list[float], exporter: Any) -> None:
    np = exporter.np
    qids = np.array([f"q{i}" for i in range(len(scores))])
    np.savez(directory / f"{cell}.npz", qids=qids, **{exporter.NPZ_KEYS["ndcg@10"]: np.array(scores)})


def _effects_for_shift(exporter: Any, directory: Path, shift: float) -> list[dict[str, Any]]:
    """Two cells whose per-query scores differ by exactly ``shift`` on every query.

    A constant shift gives a degenerate bootstrap interval at the shift itself, so the pair is
    resolved whatever the shift's size; only the floor can tell 0.002 from 0.020.
    """
    base = [0.1 * (i % 7) for i in range(24)]
    cells = [_cell(0), _cell(51)]
    _write_per_query(directory, cells[0]["cell"], base, exporter)
    _write_per_query(directory, cells[1]["cell"], [v + shift for v in base], exporter)
    return exporter.knob_effects(cells)


def test_a_resolved_step_under_the_floor_is_stamped_immaterial(
    exporter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEMDEX_SCORE_PERQUERY", str(tmp_path))
    (effect,) = _effects_for_shift(exporter, tmp_path, 0.002)
    assert effect["resolved"] is True
    assert effect["material"] is False


def test_a_resolved_step_at_or_above_the_floor_is_stamped_material(
    exporter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEMDEX_SCORE_PERQUERY", str(tmp_path))
    (effect,) = _effects_for_shift(exporter, tmp_path, 0.020)
    assert effect["resolved"] is True
    assert effect["material"] is True


def test_an_unresolved_comparison_is_never_material(exporter: Any) -> None:
    paired = {"mean_delta": 0.5, "resolved": False}
    assert exporter.is_material(paired) is False


def test_the_effects_file_records_alpha_and_the_floor_it_was_judged_against(
    exporter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    perquery = tmp_path / "perquery"
    perquery.mkdir()
    monkeypatch.setenv("SEMDEX_SCORE_PERQUERY", str(perquery))
    out_dir = tmp_path / "raw"
    out_dir.mkdir()
    cells = [_cell(0), _cell(51)]
    base = [0.1 * (i % 7) for i in range(24)]
    _write_per_query(perquery, cells[0]["cell"], base, exporter)
    _write_per_query(perquery, cells[1]["cell"], [v + 0.02 for v in base], exporter)
    (out_dir / next(iter(exporter._EXPORTS))).write_text(json.dumps({"cells": cells}))
    exporter._write_knob_effects(out_dir)
    payload = json.loads((out_dir / "chunk-knob-effects.json").read_text())
    assert payload["alpha"] == 0.05
    assert payload["material_floor"] == {"metric": "ndcg@10", "value": 0.005}
    assert [e["material"] for e in payload["effects"]] == [True]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env -u VIRTUAL_ENV .venv/bin/python -m pytest tests/test_bench_export_material.py -q -p no:cacheprovider`
Expected: 4 failed. The two stamp tests fail with `KeyError: 'material'`, the third with `AttributeError: ... has no attribute 'is_material'`, the payload test with `KeyError: 'alpha'`.

- [ ] **Step 3: Add the constants and the predicate to `scripts/_score_stats.py`**

Replace lines 27 to 33 (from `__all__` through `_BOOT_BLOCK`) with:

```python
__all__ = ["DEFAULT_ALPHA", "MATERIAL_FLOOR", "MATERIAL_FLOOR_METRIC", "bootstrap_ci", "is_material", "paired_ci"]

_DEFAULT_BOOT = 10000
# Fixed so a re-run of the same scores reproduces the same interval; the docs quote these numbers
# and a wandering last digit reads as a measurement change.
_DEFAULT_SEED = 20260806
_BOOT_BLOCK = 1000  # resamples per batch, so the index array stays bounded for large query sets

# The interval's alpha and the floor under which a RESOLVED difference is not worth acting on.
# Both are written into chunk-knob-effects.json by the exporter, so a table prints what the rows
# were judged against instead of restating the numbers. 0.005 nDCG@10 is half a point: at 12,298
# queries the paired interval resolves a step of 0.0009, which no deployer would change a setting
# for, while every comparison on an 800-query corpus that resolves at all clears 0.005.
DEFAULT_ALPHA = 0.05
MATERIAL_FLOOR = 0.005
MATERIAL_FLOOR_METRIC = "ndcg@10"
```

Then change both `alpha: float = 0.05,` signature defaults (in `bootstrap_ci` and `paired_ci`) to `alpha: float = DEFAULT_ALPHA,`, and append at the end of the file:

```python
def is_material(paired: dict[str, Any], *, floor: float = MATERIAL_FLOOR) -> bool:
    """Whether a paired comparison is resolved AND moves the metric by at least the floor.

    Resolved says the query set can tell the two configurations apart. Material says the
    difference is large enough to act on. A large query set makes the first cheap and leaves
    the second untouched, which is why a count of resolutions alone overstates what was found.

    Args:
        paired: a ``paired_ci`` result, or any row carrying ``mean_delta`` and ``resolved``.
        floor: the smallest absolute difference that counts, in the metric's own units.

    Returns:
        True only for a resolved comparison whose absolute mean difference is at or above the floor.
    """
    return bool(paired["resolved"]) and abs(float(paired["mean_delta"])) >= floor
```

The module has `from __future__ import annotations` and imports only numpy; add `from typing import Any` beside `import numpy as np` (stdlib first, then third party, per ruff isort). Run `env -u VIRTUAL_ENV .venv/bin/python -m pyright scripts/_score_stats.py` and require 0 errors.

- [ ] **Step 4: Stamp the row and the payload in `scripts/export_bench_raw.py`**

Line 57 becomes:

```python
from _score_stats import DEFAULT_ALPHA, MATERIAL_FLOOR, MATERIAL_FLOOR_METRIC, is_material, paired_ci
```

In `knob_effects`, after the docstring and before `mark_dense_stand_ins(rows)`, add the guard (the floor is defined for one metric; stamping another metric's rows against it would be a silent lie):

```python
    if metric != MATERIAL_FLOOR_METRIC:
        raise ValueError(f"the material floor is defined for {MATERIAL_FLOOR_METRIC}, not {metric}")
```

In `_effect_row`, after the `"resolved": paired["resolved"],` line and its comment, add:

```python
        # Resolved AND at least MATERIAL_FLOOR apart. A 12,298-query corpus resolves a step of
        # 0.0009; a table that counts it as a finding is counting the query set, not the knob.
        "material": is_material(paired),
```

In `_write_knob_effects`, the payload becomes:

```python
    payload = {
        MEASURED_ON: measured_on_summary(rows),
        "alpha": DEFAULT_ALPHA,
        "material_floor": {"metric": MATERIAL_FLOOR_METRIC, "value": MATERIAL_FLOOR},
        "note": (
            "Paired per-query deltas for cell pairs differing in exactly ONE chunking axis. "
            "resolved=false means the interval at alpha spans zero, so the query set cannot "
            "separate the two configurations. material=true means resolved AND an absolute "
            "mean_delta at or above material_floor: a difference worth acting on, not only one "
            "the query set can see."
        ),
        "effects": effects,
    }
```

and the final print gains the material count:

```python
resolved = sum(1 for e in effects if e["resolved"])
material = sum(1 for e in effects if e["material"])
print(
    f"[export] chunk-knob-effects.json: {len(effects)} paired comparisons, {resolved} resolved, {material} material",
    flush=True,
)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `env -u VIRTUAL_ENV .venv/bin/python -m pytest tests/test_bench_export_material.py tests/test_bench_export_axes.py -q -p no:cacheprovider`
Expected: all passed (4 new plus the existing axes tests).

- [ ] **Step 6: Re-export, detached, under nice, judged by its RC line**

The full export took 11 minutes on this box; only `_write_knob_effects` needs to run, and `ONLY=none` skips every `_EXPORTS` entry while the effects writer still runs over the committed files:

```bash
mkdir -p ~/semdex-sweep-run/logs
LOG=~/semdex-sweep-run/logs/export-effects-$(date +%Y%m%d-%H%M).log
setsid nohup bash -c 'env -u VIRTUAL_ENV ONLY=none nice -n 19 ionice -c3 .venv/bin/python scripts/export_bench_raw.py; echo RC=$?' > "$LOG" 2>&1 < /dev/null &
echo "$LOG"
```

Poll with `grep -E 'knob-effects|RC=' "$LOG"` every few minutes (the sweep watcher cadence is fine). Expected within about 15 minutes:

```
[export] chunk-knob-effects.json: 8 pair(s) scoring identically on every shared query dropped as one measurement under two labels
[export] chunk-knob-effects.json: 3132 paired comparisons, 1988 resolved, <N> material
RC=0
```

Then verify the stamp moved no number:

```bash
python3 - <<'EOF'
import json
d = json.load(open("tests/benchmarks/raw/chunk-knob-effects.json"))
e = d["effects"]
print(len(e), sum(r["resolved"] for r in e), sum(r["material"] for r in e), d["alpha"], d["material_floor"])
assert len(e) == 3132 and sum(r["resolved"] for r in e) == 1988
assert all(r["material"] == (r["resolved"] and abs(r["mean_delta"]) >= 0.005) for r in e)
EOF
git diff --stat tests/benchmarks/raw/
```

Expected: `3132 1988 <N> 0.05 {'metric': 'ndcg@10', 'value': 0.005}` and a diff touching `chunk-knob-effects.json` only (`product-k.json` is not written while its scores are absent). If another raw file changed, STOP and report which.

- [ ] **Step 7: Commit**

Write the message to a file first, then commit with a pathspec:

```bash
cat > /tmp/claude-1000/msg-task1.txt <<'EOF'
bench(export): [31] stamp every paired comparison as material or not, and the floor it was judged against

The material floor (0.005 nDCG@10) and alpha live in _score_stats beside the bootstrap; the
exporter writes both into chunk-knob-effects.json and a per-row material flag. Resolved keeps
its meaning and count (3132 comparisons, 1988 resolved, unchanged).
EOF
git add tests/test_bench_export_material.py
git commit scripts/_score_stats.py scripts/export_bench_raw.py tests/test_bench_export_material.py tests/benchmarks/raw/chunk-knob-effects.json -F /tmp/claude-1000/msg-task1.txt
```

---

### Task 2: The tables read the stamp: chance ceiling, material count, ladders judged end to end

**Files:**
- Modify: `scripts/gen_bench_tables.py:333-375` (`_knob_summary_table`), `:697-720` (`_ladders_by_embedder` becomes a wrapper), `:723-775` (`_rung_ceiling_table`), plus three new helpers beside them
- Test: `tests/test_bench_tables_effect_floors.py` (create)
- Regenerate and commit: every `docs/**/*.md` block the generator rewrites and `docs/benchmarks/generated-tables.manifest.json`

**Interfaces:**
- Consumes: `effects_doc["alpha"]`, `effects_doc["material_floor"]["value"]`, `effect["material"]` from Task 1; `_oriented`, `_held_fixed_label`, `level_label`, `_favoured_level`, `_table` as they exist.
- Produces: `_is_ladder_axis(effects: list[dict[str, Any]], axis: str) -> bool`; `_ladders(effects_doc: dict[str, Any], axis: str, fit: set[str] | None) -> dict[tuple[str, str, str], dict[tuple[Any, Any], dict[str, Any]]]` keyed `(corpus, held_fixed_label, embedding)`, pairs keyed `(low_level, high_level)` and oriented; `_ladder_verdicts(ladders) -> dict[str, int]` with keys `more`, `less`, `undecided`, `unspanned`; `_ladder_direction(ladders) -> str`. Task 3's write-up calls `_ladders` and `_ladder_verdicts` to derive its figures.

**Out of scope** - do NOT touch, though they look related:
- `_rung_table` (pools embedders per step) and the per-axis `_knob_rows` tables - the item is about the summary's counts and direction and the ceiling table; the per-step tables print intervals, which already carry their own uncertainty.
- `scripts/gen_bench_charts.py` - the charts read `resolved` and intervals, neither changes.
- Any prose in `docs/benchmarks/*.md` - Task 3. Regenerating the blocks is part of this task; the sentences beside them are not.

**STOP conditions** - stop and report rather than improvise, if:
- `chunk-knob-effects.json` lacks `alpha`, `material_floor` or per-row `material` (Task 1 did not land);
- `python scripts/gen_bench_tables.py` rewrites a block other than `chunk_knob_summary`, `chunk_knob_summary_chunking` and `chunk_overlap_ceiling` (list them from `git diff --stat docs/` and report);
- a ladder in the real data has no end-to-end pair (`unspanned` > 0 anywhere): the ground truth says every ladder carries all pairs, so this means the data or the grouping changed;
- a step's verification fails twice after one reasonable fix attempt.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bench_tables_effect_floors.py`:

```python
"""The knob summary prints a chance ceiling and a material count, and judges a ladder end to end.

A per-level tally on a ladder counts PARTNERS: a middle rung beats everything below it and loses
to everything above it, so it collects the most favours while saying nothing about direction.
These tests build a three-rung ladder where the tally favours the middle rung and require the
table to say "more" from the single highest-against-lowest pair instead.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "gen_bench_tables.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("gen_bench_tables_effect_floors", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator() -> Any:
    return _load()


def _effect(
    axis: str,
    low: Any,
    high: Any,
    delta: float,
    *,
    resolved: bool = True,
    corpus: str = "c",
    embedding: str = "e",
    held: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One paired comparison, stored HIGH to low as the exporter does in half its rows."""
    held_fixed = held if held is not None else {"strategy": "recursive", "max_tokens": 256}
    return {
        "axis": axis,
        "metric": "ndcg@10",
        "corpus": corpus,
        "embedding": embedding,
        "dim": 768,
        "held_fixed": held_fixed,
        "from_level": high,
        "to_level": low,
        "from_cell": f"{corpus}__{axis}-{high}__{embedding}",
        "to_cell": f"{corpus}__{axis}-{low}__{embedding}",
        "mean_delta": -delta,
        "ci_lo": -delta - (0.001 if resolved else 0.1),
        "ci_hi": -delta + (0.001 if resolved else 0.1),
        "wins": 5,
        "ties": 0,
        "losses": 3,
        "n_shared": 8,
        "resolved": resolved,
        "material": resolved and abs(delta) >= 0.005,
    }


def _doc(effects: list[dict[str, Any]]) -> dict[str, Any]:
    return {"alpha": 0.05, "material_floor": {"metric": "ndcg@10", "value": 0.005}, "effects": effects}


def _three_rung_ladder() -> list[dict[str, Any]]:
    """0 -> 51 gains 0.010, 51 -> 128 loses 0.002 (resolved, immaterial), 0 -> 128 gains 0.006.

    The tally over resolved rows favours 51 twice and nothing else; top against bottom says more.
    """
    return [
        _effect("overlap_tokens", 0, 51, 0.010),
        _effect("overlap_tokens", 51, 128, -0.002),
        _effect("overlap_tokens", 0, 128, 0.006),
    ]


def _summary_row(generator: Any, doc: dict[str, Any], axis: str) -> list[str]:
    table = generator._knob_summary_table(doc)
    (row,) = [r for r in table["rows"] if r[0] == f"`{axis}`"]
    return row


def test_a_numeric_axis_is_a_ladder_and_a_categorical_one_is_not(generator: Any) -> None:
    effects = _three_rung_ladder() + [_effect("strategy", "fast", "recursive", 0.02)]
    assert generator._is_ladder_axis(effects, "overlap_tokens") is True
    assert generator._is_ladder_axis(effects, "strategy") is False


def test_a_ladder_is_judged_by_its_top_against_bottom_pair_not_by_partner_tallies(generator: Any) -> None:
    row = _summary_row(generator, _doc(_three_rung_ladder()), "overlap_tokens")
    direction = row[-1]
    assert direction.startswith("1 favour more, 0 favour less, 0 unresolved or immaterial")
    assert "favour 51" not in direction


def test_the_direction_column_names_the_ceiling_table_for_where_the_optimum_sits(generator: Any) -> None:
    table = generator._knob_summary_table(_doc(_three_rung_ladder()))
    assert "top against bottom" in table["note"]
    assert "ceiling" in table["note"].lower()


def test_the_chance_ceiling_is_comparisons_times_alpha_rounded_up(generator: Any) -> None:
    row = _summary_row(generator, _doc(_three_rung_ladder()), "overlap_tokens")
    assert row[1] == "3"
    assert row[3] == "1"  # ceil(3 * 0.05)
    many = [_effect("overlap_tokens", 0, 51, 0.01, corpus=f"c{i}") for i in range(1023)]
    assert _summary_row(generator, _doc(many), "overlap_tokens")[3] == "52"  # ceil(51.15)


def test_the_material_count_excludes_a_resolved_step_under_the_floor(generator: Any) -> None:
    row = _summary_row(generator, _doc(_three_rung_ladder()), "overlap_tokens")
    assert row[2].startswith("3 ")  # three resolved
    assert row[4] == "2"  # two material


def test_a_categorical_axis_tallies_material_resolutions_only(generator: Any) -> None:
    effects = [
        _effect("strategy", "fast", "recursive", 0.020),
        _effect("strategy", "fast", "semantic", 0.002),  # resolved, immaterial: not tallied
        _effect("strategy", "late", "recursive", 0.030, resolved=False),
    ]
    row = _summary_row(generator, _doc(effects), "strategy")
    assert row[-1] == "1 favour recursive"


def test_the_table_reads_alpha_and_the_floor_from_the_file_not_from_constants(generator: Any) -> None:
    doc = _doc(_three_rung_ladder())
    doc["alpha"] = 0.10
    doc["material_floor"]["value"] = 0.02
    table = generator._knob_summary_table(doc)
    (row,) = [r for r in table["rows"] if r[0] == "`overlap_tokens`"]
    assert row[3] == "1"  # ceil(3 * 0.10) is still 1, but the note must carry the file's values
    assert "0.02" in table["note"] and "0.10" in table["note"]


def test_a_ladder_grouping_keys_by_corpus_held_fixed_and_embedder(generator: Any) -> None:
    effects = _three_rung_ladder() + [_effect("overlap_tokens", 0, 51, 0.01, embedding="f")]
    ladders = generator._ladders(_doc(effects), "overlap_tokens", None)
    assert sorted(key[2] for key in ladders) == ["e", "f"]
    verdicts = generator._ladder_verdicts(ladders)
    assert verdicts == {"more": 2, "less": 0, "undecided": 0, "unspanned": 0}


def test_the_ceiling_table_reports_material_steps_and_tags_an_immaterial_last_step(generator: Any) -> None:
    doc = _doc(_three_rung_ladder())
    table = generator._rung_ceiling_table(doc, None)
    (row,) = table["rows"]
    steps_col = table["columns"].index("Steps resolved (material)")
    assert row[steps_col] == "2/2 (1 material)"
    assert row[table["columns"].index("Highest material step")] == "0 to 51"
    assert row[-1] == "-0.0020 resolved, immaterial"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `env -u VIRTUAL_ENV .venv/bin/python -m pytest tests/test_bench_tables_effect_floors.py -q -p no:cacheprovider`
Expected: every test fails; the first with `AttributeError: ... has no attribute '_is_ladder_axis'`, the summary ones on `KeyError: 'alpha'` or an `IndexError` on the shorter row, the ceiling one on `ValueError: 'Steps resolved (material)' is not in list`.

- [ ] **Step 3: Add the ladder helpers and rewrite the summary table in `scripts/gen_bench_tables.py`**

Add `import math` to the module's imports if it is not already there. Replace `_knob_summary_table` (lines 333 to 375) with:

```python
def _is_ladder_axis(effects: list[dict[str, Any]], axis: str) -> bool:
    """Whether an axis's levels are all numeric, so its comparisons form ordered ladders.

    Detected from the values rather than kept in a list: a new numeric knob (a minimum chunk
    size, say) would otherwise fall back silently to the per-level tally this rule replaces.
    ``bool`` is excluded because it is an ``int`` to ``isinstance`` and never a rung.
    """
    levels = {
        level for effect in effects if effect["axis"] == axis for level in (effect["from_level"], effect["to_level"])
    }
    return bool(levels) and all(isinstance(level, (int, float)) and not isinstance(level, bool) for level in levels)


def _ladders(
    effects_doc: dict[str, Any], axis: str, fit: set[str] | None
) -> dict[tuple[str, str, str], dict[tuple[Any, Any], dict[str, Any]]]:
    """One ladder per corpus, held-fixed set and embedder: every oriented pair on the axis, keyed low to high."""
    ladders: dict[tuple[str, str, str], dict[tuple[Any, Any], dict[str, Any]]] = {}
    for raw in effects_doc["effects"]:
        if raw["axis"] != axis or (fit is not None and raw["corpus"] not in fit):
            continue
        effect = _oriented(raw)
        key = (effect["corpus"], _held_fixed_label(effect["held_fixed"], axis), effect["embedding"])
        ladders.setdefault(key, {})[(effect["from_level"], effect["to_level"])] = effect
    return ladders


def _ladder_verdicts(ladders: dict[tuple[str, str, str], dict[tuple[Any, Any], dict[str, Any]]]) -> dict[str, int]:
    """Judge each ladder ONCE, by the pair of its highest and lowest level.

    A per-level tally counts partners: on a 20-rung ladder a middle rung has 19 and collects the
    most favours by beating everything below it, which says nothing about direction. The
    end-to-end pair is one comparison per ladder, so the counts here are ladders.
    ``undecided`` is a ladder whose end-to-end pair is unresolved or immaterial; ``unspanned`` is
    one whose end-to-end pair was never measured, which the committed data does not contain.
    """
    verdicts = {"more": 0, "less": 0, "undecided": 0, "unspanned": 0}
    for pairs in ladders.values():
        levels = sorted({level for pair in pairs for level in pair})
        span = pairs.get((levels[0], levels[-1]))
        if span is None:
            verdicts["unspanned"] += 1
        elif not span["material"]:
            verdicts["undecided"] += 1
        elif span["mean_delta"] > 0:
            verdicts["more"] += 1
        else:
            verdicts["less"] += 1
    return verdicts


def _ladder_direction(ladders: dict[tuple[str, str, str], dict[tuple[Any, Any], dict[str, Any]]]) -> str:
    verdicts = _ladder_verdicts(ladders)
    text = (
        f"{verdicts['more']} favour more, {verdicts['less']} favour less, "
        f"{verdicts['undecided']} unresolved or immaterial"
    )
    if verdicts["unspanned"]:
        text += f", {verdicts['unspanned']} without an end-to-end pair"
    return f"{text} (of {len(ladders)} ladders)"


def _knob_summary_table(
    effects_doc: dict[str, Any], fit: set[str] | None = None, axes: frozenset[str] | None = None
) -> dict[str, Any]:
    """One row per axis: comparisons, resolved, the chance ceiling, material, and which way.

    ``axes`` restricts the rows. The chunking page shows only the chunk axes: the embedding and
    retrieval-method rows belong to their own pages, and on the chunking page they would outweigh
    every chunking axis with nothing in the prose to explain them.
    """
    alpha = float(effects_doc["alpha"])
    floor = float(effects_doc["material_floor"]["value"])
    metric = str(effects_doc["material_floor"]["metric"])
    buckets: dict[str, dict[str, Any]] = {}
    for effect in effects_doc["effects"]:
        if fit is not None and effect["corpus"] not in fit:
            continue
        if axes is not None and effect["axis"] not in axes:
            continue
        bucket = buckets.setdefault(effect["axis"], {"n": 0, "resolved": 0, "material": 0, "winners": {}})
        bucket["n"] += 1
        if effect["resolved"]:
            bucket["resolved"] += 1
        if effect["material"]:
            bucket["material"] += 1
            level = level_label(_favoured_level(effect))
            bucket["winners"][level] = bucket["winners"].get(level, 0) + 1
    rows = []
    for axis, bucket in sorted(buckets.items()):
        share = f"{bucket['resolved'] / bucket['n'] * 100:.0f}%" if bucket["n"] else "n/a"
        if _is_ladder_axis(effects_doc["effects"], axis):
            direction = _ladder_direction(_ladders(effects_doc, axis, fit))
        else:
            direction = (
                ", ".join(
                    f"{count} favour {level}"
                    for level, count in sorted(bucket["winners"].items(), key=lambda kv: -kv[1])
                )
                or "none"
            )
        rows.append(
            [
                f"`{axis}`",
                str(bucket["n"]),
                f"{bucket['resolved']} ({share})",
                str(math.ceil(bucket["n"] * alpha)),
                str(bucket["material"]),
                direction,
            ]
        )
    return _table(
        "Do the chunking knobs move retrieval at all",
        [
            "Axis",
            "Paired comparisons",
            f"Resolved at {1 - alpha:.0%}",
            "Up to N by chance",
            f"Resolved and material (>= {floor:g} {metric})",
            "Which way",
        ],
        rows,
        "Each comparison pairs two cells differing in exactly one axis and bootstraps the "
        "per-query difference over the queries they share. Unresolved means the interval spans "
        f"zero: the query set cannot separate those two configurations. Up to N by chance is comparisons "
        f"times alpha {alpha:.2f}, rounded up: the resolutions that many true nulls would produce on their own, "
        f"a ceiling on how much of the resolved count is noise. Material means resolved AND an absolute "
        f"difference of at least {floor:g} {metric}; a 12,298-query corpus resolves steps far below that. "
        "For a numeric axis the direction is judged once per ladder (corpus, embedder, held-fixed set) by its "
        "top against bottom pair, so the counts are ladders; where the optimum sits within a ladder is the "
        "ceiling table's question. For a categorical axis the material comparisons are tallied per level. "
        "Restricted to corpora long enough to carry a chunking claim.",
    )
```

Note the direction test asserts the string STARTS with the counts; the `(of N ladders)` suffix follows.

- [ ] **Step 4: Make `_ladders_by_embedder` a wrapper and report material steps in the ceiling table**

Replace `_ladders_by_embedder` (lines 697 to 720) with:

```python
def _ladders_by_embedder(
    effects: dict[str, Any], fit: set[str] | None
) -> dict[tuple[str, str, int, str], dict[tuple[int, int], dict[str, Any]]]:
    """Overlap ladders keyed with their cap, one per EMBEDDER, for the ceiling table.

    ``_rung_table`` deliberately pools the embedders at a step; this keys them apart, because the
    question "has the ladder stopped paying" is asked of one embedder at a time. The grouping is
    ``_ladders``; this adds the cap the ceiling table prints and drops a ladder with no cap.
    """
    ladders: dict[tuple[str, str, int, str], dict[tuple[int, int], dict[str, Any]]] = {}
    for (corpus, held, embedding), pairs in _ladders(effects, "overlap_tokens", fit).items():
        cap = next(iter(pairs.values()))["held_fixed"].get("max_tokens")
        if cap is None:
            continue
        ladders[(corpus, held, int(cap), embedding)] = pairs
    return ladders


def _step_verdict(step: dict[str, Any]) -> str:
    """A step's delta with what it means: material, resolved but under the floor, or unresolved."""
    if step["material"]:
        state = "material"
    elif step["resolved"]:
        state = "resolved, immaterial"
    else:
        state = "unresolved"
    return f"{step['mean_delta']:+.4f} {state}"
```

In `_rung_ceiling_table`, replace the body of the `for` loop from `resolved = [...]` through `rows.append([...])` with:

```python
        resolved = [step for step in steps if step[2]["resolved"]]
        material = [step for step in steps if step[2]["material"]]
        top_low, top_high, top = steps[-1]
        rows.append(
            [
                corpus,
                held,
                f"`{embedding}`",
                f"{len(resolved)}/{len(steps)} ({len(material)} material)",
                f"{material[-1][0]} to {material[-1][1]}" if material else "none",
                f"{top_low} to {top_high} ({top_high / cap:.0%} of cap)",
                _step_verdict(top),
            ]
        )
```

and its columns and note become:

```text
[
    "Corpus",
    "Held fixed",
    "Embedder",
    "Steps resolved (material)",
    "Highest material step",
    "Last step",
    "Last step delta",
],
rows,
"One row per EMBEDDER rather than per step, because a mean over embedders cannot "
"distinguish a plateau from a split - embedders that have found their ceiling and "
"embedders still climbing average to a small gain that describes neither. A step is "
"material when it resolves AND clears the file's material floor; a resolved step under it "
"is a difference the query set can see and no deployer would act on. A last step that is "
"still material means the sweep stopped before the data did, so that embedder's optimum "
"lies above the range measured here.",
```

The function's docstring sentence "The decisive column is the LAST step: a ladder whose final step still resolves has been cut short" becomes "...whose final step is still material has been cut short...".

- [ ] **Step 5: Run the new tests and the generator's own tests**

Run: `env -u VIRTUAL_ENV .venv/bin/python -m pytest tests/test_bench_tables_effect_floors.py -q -p no:cacheprovider`
Expected: 9 passed.

Run: `env -u VIRTUAL_ENV .venv/bin/python -m pytest tests/test_bench_tables_current.py -q -p no:cacheprovider`
Expected: `test_every_generated_block_matches_a_fresh_render` and `test_the_manifest_is_current` FAIL (the pages hold the old columns); everything else passes.

- [ ] **Step 6: Regenerate the pages and confirm only the three blocks moved**

```bash
env -u VIRTUAL_ENV .venv/bin/python scripts/gen_bench_tables.py
git diff --stat docs/
git diff docs/ | grep -c '^[-+]<!--' ; echo '(marker lines themselves never change; this count must be 0)'
git diff docs/ | grep '^[-+]|' | grep -c 'Axis '
```

Expected: `docs/benchmarks/03-chunking.md`, `04-embedding.md`, `07-selection.md` and `generated-tables.manifest.json` changed; the header-row count is 6 (three tables, old and new header each). Then:

```bash
env -u VIRTUAL_ENV .venv/bin/python -m pytest tests/test_bench_tables_current.py tests/test_bench_claims_current.py -q -p no:cacheprovider
```

Expected: all passed. The claims pass because no prose changed and no `resolved` count moved. If a claim fails, STOP: something moved a number.

Read the regenerated `chunk_knob_summary_chunking` block in `docs/benchmarks/03-chunking.md` and confirm the `overlap_tokens` row's direction cell has `0 without an end-to-end pair` absent (every ladder spanned) and the `strategy` row still tallies levels. Copy the two ladder rows into the task report, Task 3 needs them.

- [ ] **Step 7: Commit**

```bash
cat > /tmp/claude-1000/msg-task2.txt <<'EOF'
bench(tables): [31] chance ceiling, material count and end-to-end ladder verdicts in the knob summary

The summary reads alpha and the floor from the effects file: a column for the resolutions that
many true nulls would produce, a column for resolved-and-material, and a direction judged once per
ladder by its top against bottom pair for numeric axes (the per-level tally counted partners on a
20-rung ladder). The ceiling table reports material steps and tags an immaterial last step. Pages
regenerated; prose follows.
EOF
git add tests/test_bench_tables_effect_floors.py
git commit scripts/gen_bench_tables.py tests/test_bench_tables_effect_floors.py docs/benchmarks/03-chunking.md docs/benchmarks/04-embedding.md docs/benchmarks/07-selection.md docs/benchmarks/generated-tables.manifest.json -F /tmp/claude-1000/msg-task2.txt
```

If `git diff --stat docs/` in Step 6 listed any other file, STOP before committing and report it.

---

### Task 3: The prose reads the new columns, the gap closes, the gate runs, the push lands

**Files:**
- Modify: `docs/benchmarks/03-chunking.md` (the paragraphs after the `chunk_knob_summary_chunking` block, about lines 33 to 50, and the overlap summary bullet near the `summary-gerdalir-upward` claim's quote), `docs/benchmarks/04-embedding.md` (the two paragraphs after its `chunk_knob_summary` block), `docs/benchmarks/07-selection.md` (the sentence introducing its block), `docs/benchmarks/08-gaps.md:494-497` and `:507-508`
- Modify: `tests/benchmarks/claims/03-chunking.toml` (re-point `headline-gerdalir-overlap-upward`, `headline-mldr-overlap-downward`, `headline-mldr-overlap-resolved`, `summary-gerdalir-upward`; add claims for every new figure), `tests/benchmarks/claims/04-embedding.toml` and `07-selection.toml` for any new figure there, `tests/benchmarks/claims/08-gaps.toml` if the removed paragraph carried claims (check with `grep -n '"49"\|"980"\|0.0005' tests/benchmarks/claims/08-gaps.toml`)
- Modify: `tests/benchmarks/claims/README.md` `## Oriented rowsets` section (document `material`)
- Update: `OPEN-WORK.md` line for [31] (close it), `TODO.md` if it lists the item

**Interfaces:**
- Consumes: `_ladders`, `_ladder_verdicts`, `_load`, `_fit_corpora` from Task 2; the regenerated tables.
- Produces: nothing downstream.

**Out of scope** - do NOT touch, though they look related:
- The `## Where the ladder stops paying` reading of the ceiling table beyond swapping "resolved" for "material" where the table now says material - the ceiling table's per-embedder findings were re-derived on 2026-09-11 and stand; check each sentence against the regenerated table and change only what the table changed.
- Any generated block - regenerated in Task 2; a hand edit inside the markers is undone by the next run.
- `docs/benchmarks/01-method.md` - the method page describes the bootstrap; if it needs a sentence on the floor, add ONE sentence naming the two columns and no number.

**STOP conditions** - stop and report rather than improvise, if:
- `python scripts/check_bench_claims.py` fails on a claim you did not edit;
- a figure you want to print cannot be derived by a `where` filter over `effects` rows (a ladder count is derivable ONLY as the end-to-end pair: filter `from_level`, `to_level`, `material = true`, `favoured_end`, corpus and `held_fixed.max_tokens`; if two ladder families share a corpus at different tops, state them separately);
- `make test` fails on a stage other than a claim or table you changed;
- a step's verification fails twice after one reasonable fix attempt.

- [ ] **Step 1: Derive the figures from the regenerated data, never from memory**

```bash
env -u VIRTUAL_ENV .venv/bin/python - <<'EOF'
import importlib.util, json
from pathlib import Path
spec = importlib.util.spec_from_file_location("g", Path("scripts/gen_bench_tables.py"))
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
doc = json.load(open("tests/benchmarks/raw/chunk-knob-effects.json"))
fit = g._fit_corpora(g._load("chunk-dimension-audit.json"))
print("fit corpora:", sorted(fit) if fit else fit)
for axis in ("overlap_tokens", "max_tokens"):
    ladders = g._ladders(doc, axis, fit)
    for corpus in sorted({k[0] for k in ladders}):
        sub = {k: v for k, v in ladders.items() if k[0] == corpus}
        tops = sorted({max(l for p in v for l in p) for v in sub.values()})
        print(axis, corpus, "ladders", len(sub), "tops", tops, g._ladder_verdicts(sub))
        for key, pairs in sorted(sub.items()):
            levels = sorted({l for p in pairs for l in p})
            span = pairs[(levels[0], levels[-1])]
            print("   ", key[1], key[2], levels[0], "to", levels[-1], f"{span['mean_delta']:+.4f}", "material" if span["material"] else ("resolved" if span["resolved"] else "unresolved"))
rows = [e for e in doc["effects"] if fit is None or e["corpus"] in fit]
for axis in sorted({e["axis"] for e in rows}):
    sub = [e for e in rows if e["axis"] == axis]
    print(axis, "n", len(sub), "resolved", sum(e["resolved"] for e in sub), "material", sum(e["material"] for e in sub))
EOF
```

Keep this output in the task report: every figure the prose prints comes from it or from the regenerated table.

- [ ] **Step 2: Rewrite the chunking page's reading of the headline table**

In `docs/benchmarks/03-chunking.md`, the paragraph beginning "No axis points the same way everywhere" keeps its size and whitespace sentences (their claims still derive) but the overlap sentences change. Replace the paragraph beginning "Overlap does not point one way, and what decides it is the QUERY. Every one of GerDaLIR's 496 resolved overlap comparisons favours MORE overlap; 22 of MLDR's 34 favour LESS." with a paragraph of this shape, the figures taken from Step 1 (the bracketed parts are the figures to fill, the sentence order is fixed so each quote stays on one line):

> Overlap does not point one way, and what decides it is the QUERY. Judged once per ladder by its
> end-to-end pair, [N] of GerDaLIR's [M] overlap ladders favour MORE overlap and none favours less;
> on MLDR [P] of [Q] favour LESS and [R] are undecided. The per-level tallies this page once printed
> counted partners on a 20-rung ladder, not wins. Binning GerDaLIR's own queries by length turns its
> answer into MLDR's: ...

and continue with the existing binning sentences unchanged. Add a sentence after the table's first paragraph that reads the two new columns, of the shape:

> Of the [533] overlap comparisons that resolve, up to [N] would resolve by chance alone and [K]
> are under the half-point floor, all of them on the 12,298-query corpus; the strategy and size
> axes lose [S] and [T] to the floor.

Each bracketed figure gets a claim:

```toml
[[claim]]
id = "headline-gerdalir-overlap-ladders-more"
quote = "[N] of GerDaLIR's [M] overlap ladders favour MORE"
published = "[N]"
rows = "effects"
reduce = "count"
where = { axis = "overlap_tokens", material = true, favoured_end = "high", corpus = "gerdalir_de_12k_slice", from_level = 0, to_level = 256 }
why = "GerDaLIR overlap ladders judged by their end-to-end pair (0 to 256, the top of every GerDaLIR ladder), material and favouring more"
```

If Step 1 shows GerDaLIR ladders with different tops, write one claim per top and make the sentence name them. The denominator `[M]` is the same filter without `material` and `favoured_end`. The MLDR ladders have tops per corpus and cap; derive each from Step 1 the same way. The chance figure is `math.ceil(n * 0.05)` and cannot be derived by a claim reducer; print it only inside the generated table and refer to it in prose as "the chance column" rather than as a number. The under-floor count is `reduce = "count"` with `where = { axis = "overlap_tokens", resolved = true, material = false, corpus = "@fit_corpora" }`.

Re-point the four existing claims named in **Files:** to the new quotes (or delete a claim whose sentence no longer exists and say so in the commit message). Run `env -u VIRTUAL_ENV .venv/bin/python scripts/check_bench_claims.py` after every edit; a failure names the claim.

- [ ] **Step 3: The embedding and selection pages**

`docs/benchmarks/04-embedding.md`: the paragraph after the block says "The embedding row resolves in four comparisons out of five". Add one sentence after it: "Its material count sits within [X] of its resolved count, so almost every model switch that the query set can see is also large enough to act on; the chunking axes are where the floor bites." with `[X]` the difference between the resolved and material counts on the `embedding` row of the regenerated `chunk_knob_summary` block, claimed as `reduce = "count"` with `where = { axis = "embedding", resolved = true, material = false }` (no corpus filter: that table is not restricted... CHECK the regenerated row's denominator against the file first; if it is restricted by `fit`, add `corpus = "@fit_corpora"`).

`docs/benchmarks/07-selection.md`: the introducing sentence "The paired comparisons make the ordering unusually clear, because the same test was applied to every axis." gains: "The material column is the one to read: it removes resolutions that the query set can see and no deployer would act on." No figure, no claim.

- [ ] **Step 4: Close the gap and document the field**

`docs/benchmarks/08-gaps.md`: delete the bullet at lines 494 to 497 ("No count subtracts the resolutions expected by chance ...") and, in the closing paragraph at 507 to 508, delete "; and put a chance floor and a material-difference floor into the effect tables" so the sentence ends after "build one markdown-native corpus." If `tests/benchmarks/claims/08-gaps.toml` carried a claim on "49" or "980" from that bullet, delete the claim too and say so in the commit message.

`tests/benchmarks/claims/README.md`, in `## Oriented rowsets`, after the `levels` bullet add:

```markdown
- `material` - carried from the file, not derived: `true` when the comparison resolves AND its
  absolute `mean_delta` is at or above the file's `material_floor` (0.005 nDCG@10 today). A claim
  about a finding filters on `material = true`; one about what the query set can merely see
  filters on `resolved = true`.
```

- [ ] **Step 5: Gate and verify**

```bash
env -u VIRTUAL_ENV .venv/bin/python scripts/check_bench_claims.py
env -u VIRTUAL_ENV .venv/bin/python -m pytest tests/test_bench_claims_current.py tests/test_bench_tables_current.py tests/test_bench_tables_effect_floors.py tests/test_bench_export_material.py -q -p no:cacheprovider
```

Expected: the checker prints its validated count and exits 0; all tests pass. Then the full gate, read from its RC line and its result envelope, never its tail:

```bash
env -u VIRTUAL_ENV make test > /tmp/claude-1000/gate-task3.log 2>&1; echo RC=$? >> /tmp/claude-1000/gate-task3.log
grep -E 'RC=|"result"|\(exit code' /tmp/claude-1000/gate-task3.log
```

Expected: `{"result":"pass",...}` and `RC=0`. Any `(exit code: N)` line names a failed stage; fix it, never exempt it.

- [ ] **Step 6: Close the backlog line, commit, push**

In `OPEN-WORK.md`, the [31] line becomes `- [x]` and gains `| closed: 2026-09-27, chance ceiling and material columns in the knob summary, ladders judged end to end, gap paragraph removed (docs/plans/2026-09-27-effect-floors-design.md)`. Mirror the state in `TODO.md` if it lists [31].

```bash
cat > /tmp/claude-1000/msg-task3.txt <<'EOF'
docs(benchmarks): [31] read the chance ceiling and the material floor; judge overlap ladders end to end

The chunking page states the overlap direction per ladder instead of per resolved comparison,
names how many resolutions the floor removes, and the gaps page drops the paragraph this closes.
Every new figure is claimed; the four tally claims are re-pointed. README documents the stamped
material field.
EOF
git commit docs/benchmarks/03-chunking.md docs/benchmarks/04-embedding.md docs/benchmarks/07-selection.md docs/benchmarks/08-gaps.md tests/benchmarks/claims/03-chunking.toml tests/benchmarks/claims/04-embedding.toml tests/benchmarks/claims/08-gaps.toml tests/benchmarks/claims/README.md OPEN-WORK.md -F /tmp/claude-1000/msg-task3.txt
git push origin main
git status -sb | head -1
```

Drop from the pathspec any file the task did not change (git refuses a pathspec that matches nothing that changed only with an error; read it). Expected: `## main...origin/main` with nothing ahead. CI is blocked account-side ([15]); the local gate is the verification.

---

## Self-review against the design

- Chance ceiling column: Task 2 (`math.ceil(n * alpha)`, alpha from the file). Covered.
- Material floor 0.005 absolute, stamped per row, floor and alpha in the payload: Task 1. Covered.
- Numeric axes judged top against bottom once per ladder; categorical tally over material rows: Task 2. Covered.
- Ordinality detected from values: `_is_ladder_axis`, Task 2, with a test on both kinds. Covered.
- Ceiling table reports material steps and tags an immaterial last step: Task 2. Covered.
- Claim checker: the design said `material` joins the derived fields; ground truth shows a stamped field is already filterable, so the checker has no code change and Task 3 documents the field in the README instead. This is the one correction to the design.
- Pages 03, 04, 07 and the gaps paragraph: Task 3. Covered.
- Out of scope per the design: FDR count, relative floor, charts. None planned.
- Types: `_ladders` returns `dict[tuple[str, str, str], dict[tuple[Any, Any], dict[str, Any]]]` in Task 2's Produces block, its code and Task 3's snippet; `_ladder_verdicts` returns the four-key dict in both. `is_material(paired, *, floor=...)` is keyword-only in Task 1 and called with one positional in the row stamp. Consistent.
