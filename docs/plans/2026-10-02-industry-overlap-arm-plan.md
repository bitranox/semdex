# [12] Industry-comparable overlap arm (option 2) + small-overlap cleanup - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use bitranox:process-agents-subagent-driven-development (recommended) or bitranox:process-plan-executor to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. On approval, copy this file to `docs/plans/2026-10-02-industry-overlap-arm-plan.md` (never quote a generated-table BEGIN/END marker pair in it).

**Goal:** Measure chunk overlap at the sizes and percentages industry guidance uses (cap512 at 0-25 percent, cap1024 at 0-20 percent). Prove the method on a pilot before any multi-hour run. Then delete the 2-6 percent overlap data everywhere and republish the overlap pages in percent of the cap.

**Architecture:** This reuses the existing preembed -> score -> export -> regenerate pipeline. There are two small code changes:
- an ollama `num_batch` pass-through in `scripts/preembed_vectors.py`;
- a percent-of-cap label in `scripts/gen_bench_tables.py`.

A pre-registered PILOT gate (Task 3) must pass, then be re-checked by an independent verifier and shown to the user, before the long run (Task 4). The deletion (Task 6) runs AFTER the new cells are exported, so the overlap pages are rewritten once (Task 7) and the cap512 overlap axis is never missing in between.

**Tech Stack:** Python 3.10+ scripts run with uv/.venv, chonkie (recursive + OverlapRefinery), fastembed, model2vec, ollama on the GPU CT (`SEMDEX_BENCH_OLLAMA_URL`), bmk `make test`.

## Context

- **What the user asked for (OPEN-WORK.md [12], 2026-10-02):** semdex chunking must be "comparable to industry standard". Guidance is phrased in percent: 10-20 percent common, Azure 512 tokens + 25 percent, NVIDIA 1024 with 15 percent best, LlamaIndex 1024/200. semdex measured that band only at cap256. cap512 has 0, 10 and 15 TOKENS (0/2/3 percent). cap1024 has nothing.
- **User decisions this session:**
  - Option 2: cap512 at 0/10/15/20/25 percent on MLDR en, MLDR de and GerDaLIR, with all six embedders. Plus cap1024 at 10/15/20 percent for the long-context embedders.
  - Delete the small-overlap data from disk AND from published raw files, tables and docs.
  - Validate the method on a few records first, checked properly, before running for many hours.

## Facts measured while planning (they shape the design)

- **Overlap adds length, not chunks.** chonkie `OverlapRefinery` (`src/semdex/adapters/chunker/chonkie.py:196`) APPENDS the next chunk's first N tokens:
  - The chunk COUNT does not change: GerDaLIR has 431,721 chunks at both o0 and o51. `export_bench_raw.py:331-371` voids any overlap set whose count differs from o0.
  - A chunk can reach cap + overlap: cap512-o128 is up to 640 gpt2 tokens.
- **Truncation:** bge-base and model2vec read 512 tokens, so they can cut off exactly the appended overlap. Already measured: bge-base clips 4-7 percent of chunks at cap512-o15. The exporter voids a cell above 5 percent of token mass clipped. ollama bge-m3 (8192) and qwen3 (40960) do not truncate.
- **ollama `num_batch`:** preembed passes none (`scripts/preembed_vectors.py:523`), so ollama keeps its 2048-token physical batch. A longer input is silently embedded from its first 2048 tokens (memory `ollama-num-batch-silently-truncates-long-inputs`). cap1024+205 is about 1229 gpt2 tokens; the margin on German is unverified.
- **Baselines:** cap1024 has no o0 cell, so without one the overlap rungs have nothing to compare to. GerDaLIR has no cap512 set at all. MLDR en/de have cap512-o0 for the six embedders, already scored.
- **Recipe:** every existing recursive set uses recipe `""` while the product ships `"markdown"` ([23]). The new arm keeps `""` so it compares directly with the cap256 ladder. Task 3 measures the difference as information for [23].
- **Rates and cost** (cap512 remeasure logs):

```
bge-base     4.2-5.7 chunks/s (CPU)   cap512 arm ~90 h, parallel to the GPU chain
qwen3-4b     12-15 chunks/s (GPU)     ~35 h
qwen3-8b     8.5-10.7 chunks/s (GPU)  ~50 h
bge-m3       ~52-65 chunks/s at cap256, never measured at 512   ~15 h
model2vec    seconds per cell
cap1024 GPU arm: about 3-4 more days, not measured
```

The GPU chain runs about 7-8 days in total. Task 3 re-measures every rate on real cap512/1024 chunks before launch.
- **What deleting o10/o15 hits** (consumer map, simulated in memory, nothing written). No product code breaks. It does break:
  - 42 of 731 claims;
  - 12 generated tables, the charts manifest, `explorer.html` and `/embeddings/catalog.*`.

  It also removes the BEIR and MIRACL overlap ladders entirely (their levels are only 0/10/15) and moves several headline numbers. Most visibly, the current MLDR German #1 config (`recursive cap256 ov10tok` + qwen3-8b, 0.7350) leaves the ranking.

## Global Constraints

- Six embedders: `fastembed:bge-base`, `model2vec:potion-base-8M`, `model2vec:potion-retrieval-32M`, `ollama:bge-m3`, `ollama:qwen3-embedding-4b`, `ollama:qwen3-embedding-8b`. The cap1024 arm uses the three ollama models only.
- cap512 rungs: overlap 0/51/77/102/128 tokens (0/10/15/20/25 percent). cap1024 rungs: 0/102/154/205 (0/10/15/20 percent). Tokenizer `gpt2`, chunker `recursive`, recipe `""`.
- Corpora: `mldr_en_8k_slice`, `mldr_de_3k_slice` (scores -> `/embeddings/scores/mldr_chunk_scores.json`) and `gerdalir_de_12k_slice` (scores -> `gerdalir_chunk_scores.json`). Never `gerdalir_new_rungs_scores.json`, which the export does not read.
- Long jobs:
  - run from the FROZEN copy (`/home/srvadmin/semdex-sweep-run/scripts/` + `.venv-sweep`), record `SOURCE_SHA-item12.txt`, and export `SEMDEX_SOURCE_SHA` from it: the frozen copy has no repository to ask, so without it every row is stamped `semdex_git_sha: unknown`;
  - wrap every Python step in `nice -n 19 ionice -c3`, launch the script with `setsid nohup`, and have it append `RC_*=` lines itself;
  - set `PYTHONUNBUFFERED=1`, `FASTEMBED_CACHE_PATH=/embeddings/.fastembed_cache` and ollama batch 128;
  - never put them in `run_in_background`.
- No GPU heavy install on CT 60400 while a chain runs. Before each GPU phase, check that `ollama ps` shows PROCESSOR 100% GPU.
- Public repo: no host names, CT ids or private paths in committed files. Run scripts stay in the private run dir.
- Gate: `make test` through compuse-toolbox `gate.py` with `env -u VIRTUAL_ENV BMK_PYTHON_CMD=$PWD/.venv/bin/python`. Then commit and push to main (no PR) and watch CI by the full sha.
- Deletion is irreversible, so back up first: copy every score JSON and npz that gets edited to `<file>.bak-20261002`, and take a ZFS snapshot of the embeddings dataset before the `rm` (Task 6).

---

### Task 1: ollama `num_batch` pass-through in preembed

**Files:**
- Modify: `scripts/preembed_vectors.py` (env docstring at :12-60; `_ensure_vectors` at :504-523)
- Test: `tests/test_preembed_num_batch.py` (new)

**Interfaces:**
- Produces: env `SEMDEX_PREEMBED_NUM_BATCH` (int, optional). It reaches `build_embedding(..., num_batch=)` (`src/semdex/composition/__init__.py:715`, already has the parameter) for ollama only.
- Produces: `_num_batch_from_env() -> int | None`.

**Out of scope:** `src/semdex/adapters/embedding/ollama.py`, which already plumbs `num_batch` and is regression-tested.

**STOP if:** `build_embedding` no longer accepts `num_batch`.

- [ ] **Step 1: Failing test.** Drive the real helper; the env is the external edge.

```python
import importlib.util, sys
from pathlib import Path
import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "preembed_vectors.py"
    spec = importlib.util.spec_from_file_location("preembed_vectors", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_num_batch_unset_keeps_server_default(monkeypatch):
    monkeypatch.delenv("SEMDEX_PREEMBED_NUM_BATCH", raising=False)
    assert _load()._num_batch_from_env() is None


def test_num_batch_read_as_int(monkeypatch):
    monkeypatch.setenv("SEMDEX_PREEMBED_NUM_BATCH", "4096")
    assert _load()._num_batch_from_env() == 4096


@pytest.mark.parametrize("bad", ["0", "-1", "abc"])
def test_num_batch_refuses_nonsense(monkeypatch, bad):
    monkeypatch.setenv("SEMDEX_PREEMBED_NUM_BATCH", bad)
    with pytest.raises(SystemExit):
        _load()._num_batch_from_env()
```

- [ ] **Step 2:** `.venv/bin/python -m pytest tests/test_preembed_num_batch.py -q`. Expected: FAIL (no attribute `_num_batch_from_env`).
- [ ] **Step 3: Implement.**

```python
def _num_batch_from_env() -> int | None:
    """ollama's physical batch; unset keeps the server default (2048), which silently truncates longer inputs."""
    raw = os.environ.get("SEMDEX_PREEMBED_NUM_BATCH")
    if raw is None:
        return None
    if not raw.isdigit() or int(raw) <= 0:
        raise SystemExit(f"SEMDEX_PREEMBED_NUM_BATCH must be a positive integer, got {raw!r}")
    return int(raw)
```

In `_ensure_vectors`, pass `num_batch=_num_batch_from_env() if backend is EmbeddingBackend.OLLAMA else None` to `build_embedding`. Document the env var in the module docstring list.
- [ ] **Step 4:** Re-run the test; expected PASS. RED-verify by reverting the helper body to `return None` with the compuse-toolbox `mutation_arm` jig: the int test must fail with its own assertion.
- [ ] **Step 5:** Commit `scripts/preembed_vectors.py tests/test_preembed_num_batch.py` ("feat(bench): pass ollama num_batch through preembed").

### Task 2: percent-of-cap in every overlap label, plus the document-level caveat

**Files:**
- Modify: `scripts/gen_bench_tables.py` (`_display_profile` :101-122; `_knob_rows` :266-304 level text; the `chunk_knob_overlap` note :948)
- Test: `tests/test_bench_tables_overlap_percent.py` (new)

**Interfaces:**
- Produces: `_overlap_percent(overlap_tokens: int, cap: int) -> int`, rounded half-up. Labels read `recursive cap512 ov102tok (20%)`. Overlap 0 prints `ov0tok (0%)`.

**Out of scope:** the cache directory names (`-o102-`), which are keys for hundreds of cells.

- [ ] **Step 1: Failing test.**

```python
from gen_bench_tables import _display_profile, _overlap_percent


def test_percent_is_of_the_cap():
    assert _overlap_percent(51, 512) == 10
    assert _overlap_percent(128, 512) == 25
    assert _overlap_percent(205, 1024) == 20
    assert _overlap_percent(26, 256) == 10


def test_label_carries_tokens_and_percent():
    axes = {"strategy": "recursive", "max_tokens": 512, "overlap_tokens": 102}
    assert _display_profile(axes) == "recursive cap512 ov102tok (20%)"
```

(Import the bare module name like sibling tests; never `from tests.`.)
- [ ] **Step 2:** Run it; expected FAIL (ImportError on `_overlap_percent`).
- [ ] **Step 3: Implement.**

```python
def _overlap_percent(overlap_tokens: int, cap: int) -> int:
    """Overlap as the percent of the cap that industry guidance quotes (10-25 percent)."""
    return int(overlap_tokens * 100 / cap + 0.5)
```

In `_display_profile`: `label += f" ov{overlap}tok ({_overlap_percent(overlap, size)}%)"` when `size` is an int. Make the `_knob_rows` overlap level text use the same helper. Replace the :948 note with: "Overlap is counted in gpt2 tokens and shown with its percent of the cap. Each verdict is document-level nDCG@10 (best chunk per document); `semdex search` returns chunks."
- [ ] **Step 4:** Run the test (PASS), then `.venv/bin/python scripts/gen_bench_tables.py` and `git diff --stat docs/`. Only label text may change. Run the claims gate `scripts/check_bench_claims.py`; for every claim whose quoted label changed, update its quote in `tests/benchmarks/claims/*.toml`.
- [ ] **Step 5:** Gate `make test`, commit, push, watch CI by sha (Tasks 1 and 2 ship together if convenient).

### Task 3: PILOT - validate the method before any long run (the gate)

Every check below has a PASS rule written BEFORE the run, and a failure STOPS the plan. The pilot runs the real frozen scripts against the real data, but only the cheap parts at full size:
- chunking (about 60 s per set);
- model2vec embedding and scoring (seconds per cell).

The expensive embedders run on a 1,000-chunk sample only. Work dir: `/home/srvadmin/semdex-sweep-run/pilot-item12/`. Scores go to `pilot-*.json` scratch files, never the real score files.

**Files:** create `/home/srvadmin/semdex-sweep-run/item12-pilot.sh` (private) and `pilot-item12/checks.py` (private; a short uv-run script that prints one PASS/FAIL line per check plus a JSON envelope).

- [ ] **Step 1: Freeze.** `git rev-parse HEAD > SOURCE_SHA-item12.txt`, then copy `scripts/` into `$RUN/scripts/`.
- [ ] **Step 2: Chunk everything at full size** with `SWEEP_EMBEDDINGS=model2vec:potion-base-8M` (driver `sweep_chunk_profiles.py`). Profiles:
  - `recursive:512:0,512:51,512:77,512:102,512:128` on GerDaLIR (o0 is new) and the four new rungs on MLDR en/de;
  - `recursive:1024:0,1024:102,1024:154,1024:205` on all three corpora.

  Also chunk `mldr_en_8k_slice` at `recursive:512:0` with `SEMDEX_PREEMBED_RECIPE=markdown`, for [23] information only. Embed the second model2vec model the same way.
- [ ] **Step 3: Run the checks** (`checks.py`).

| #   | Check                                    | PASS rule                                                                                                                                                                                                                                                                                                                                                                                                                                          |
|-----|------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| C1  | Config                                   | Every new `meta.json` has the intended `max_tokens`, `overlap`, `tokenizer`=gpt2 and `recipe`="", and the directory name matches.                                                                                                                                                                                                                                                                                                                  |
| C2  | Count invariant                          | `audit_chunk_dimensions.py` over the 3 corpora: every overlap set's count equals its o0 count at the same cap, and no set is flagged.                                                                                                                                                                                                                                                                                                              |
| C3  | Overlap really present                   | For 300 random adjacent pairs from the same doc, the gpt2 tokens at the end of chunk i equal the first `overlap` tokens of chunk i+1 in at least 99 percent of pairs. Realized length: p95 <= cap+overlap, max <= cap+overlap+8. Print 3 pairs for eyeballing.                                                                                                                                                                                     |
| C4  | o0 really has no overlap                 | The same probe on o0 sets finds a match in at most 1 percent of pairs. This is the control; it proves C3 can fail.                                                                                                                                                                                                                                                                                                                                 |
| C5  | Truncation                               | `audit_embedder_truncation.py` (AUDIT_SAMPLE=20000) for bge-base and both model2vec models on every new cap512 set. Report the token mass lost and the share of the appended suffix that survives. Pre-registered rule: any (set, embedder) above 5 percent mass lost is DROPPED from the long run (the exporter would void it anyway) and listed in `withheld-cells.json` with the reason. model2vec at cap1024 is withheld by rule (512 window). |
| C6  | No silent ollama truncation              | Take the 5 longest cap1024-o205 German chunks. Embed each with qwen3-8b at `num_batch=4096`, and again with the text cut at 2048 model tokens. The vectors must differ (cosine < 0.9999). Repeat at `num_batch` unset: if those vectors equal the cut ones, the default truncates and 4096 is required, so set `SEMDEX_PREEMBED_NUM_BATCH=4096` for the run.                                                                                       |
| C7  | Liveness pair                            | Re-embed `mldr_en_8k_slice__recursive-t512-o0-gpt2` with potion-base-8M into a scratch cache root and score it into `pilot-liveness.json`. Its vectors are byte-identical to the cached ones and its nDCG@10 equals the committed `chunk-sweep-mldr.json` value to 4 dp. This proves the harness reproduces a known number.                                                                                                                        |
| C8  | Plausibility on the full model2vec cells | Score every new model2vec cell into `pilot-scores.json`. Every nDCG@10 is in (0.05, 1). The rungs at one cap are NOT all identical (a constant is a broken instrument). Each rung is within +/-0.05 of its own o0. MLDR rows sit within 0.05 of the corresponding cap256 ladder trend. The perquery npz row count equals the number of queries.                                                                                                    |
| C9  | Rates and GPU health                     | 1,000 real cap512-o128 chunks and 500 cap1024-o205 chunks per expensive embedder (`SEMDEX_PREEMBED_MAX_DOCS`, scratch cache root). Record chunks/s. `ollama ps` shows 100% GPU, and `nvidia-smi` shows no squatter. Compute the per-chain ETA from these rates, not the planning estimate. Expensive vectors: dims correct, no NaN, norms in [0.9, 1.1] where the model normalises.                                                                |
| C10 | Export dry run                           | Run `export_bench_raw.py` with `OUT_DIR=pilot-raw/` and `gen_bench_tables.py` against a scratch copy. The new model2vec rows appear with `(10%)`-style labels, none of them is in the void list, and no unscored-cell refusal appears except cells already withheld.                                                                                                                                                                               |
| C11 | Recipe (info)                            | Chunk count and boundary share of `recipe=markdown` vs `""` on MLDR en cap512. Recorded in OPEN-WORK [23], never a stop.                                                                                                                                                                                                                                                                                                                           |

- [ ] **Step 4: Independent verification.** Dispatch an `opus` subagent with fresh context. It gets the check table, the pilot dir and the plan, NOT my outputs. It RE-RUNS `checks.py` and the C3/C7 probes itself and tries to disprove each PASS, demanding a concrete failing input per finding. Any finding blocks.
- [ ] **Step 5: User sign-off.** Write `pilot-item12/REPORT.md` (private): one line per check with its evidence, the measured ETA per chain, the dropped (set, embedder) list, and the eyeball samples. Show it to the user and WAIT for an explicit go. Do not launch Task 4 on the verifier's PASS alone.
- [ ] **Step 6:** Delete the scratch cache root and pilot score files once Task 4 is launched (the full-size chunk sets and model2vec cells are REAL outputs; keep them).

**STOP if:** any C1-C10 FAILs. Report the failing check with its output; do not adjust a threshold to make it pass.

### Task 4: the long run (after the user's go)

**Files:** create `/home/srvadmin/semdex-sweep-run/item12-overlap-arm.sh` (private), copied from `item20-chunk-floor.sh`'s skeleton (frozen `.venv-sweep`, per-phase logs, RC lines).

- [ ] **Step 1:** The script runs two chains in parallel (`&` + `wait`):
  - **GPU:** `ollama:bge-m3,ollama:qwen3-embedding-4b,ollama:qwen3-embedding-8b`, batch 128, `SEMDEX_PREEMBED_NUM_BATCH=4096` (the pilot's C6 found it required: bge-m3 at ollama's default processes 2048 of up to 2157 tokens). cap512 profiles first (all three corpora), then cap1024.
  - **CPU:** `fastembed:bge-base` on the cap512 profiles minus the C5-dropped sets (the pilot dropped four: GerDaLIR o128, MLDR de o128, MLDR en o102 and o128).
  - Then score per corpus file, all six embedders named explicitly in `SEMDEX_SCORE_EMBEDDINGS` (the default skips ollama), writing `RC_SCORE`. Then `audit_chunk_dimensions.py` and the truncation census, writing `RC_AUDIT`. Final line: `DONE`.
- [ ] **Step 2:** Launch with `setsid nohup bash item12-overlap-arm.sh > logs/item12-<ts>.log 2>&1 &`. Within 60 s, verify the first vectors.npy.tmp is growing and both chains are alive.
- [ ] **Step 3:** Arm the monitoring like the bp-sweep monitor:
  - a 30-minute check on the RC lines, process liveness and `.tmp` growth;
  - a pure-deadline backstop at 1.5x the C9 ETA.

  Update the OPEN-WORK [12] `next:` with the log path and ETA.
- [ ] **Step 4:** On completion, read the RC lines (not the process table). Every RC must be 0.

### Task 5: export the new cells and sanity-read them

- [ ] `export_bench_raw.py` (real `OUT_DIR`), then `gen_bench_tables.py`, `gen_bench_charts.py`, `gen_bench_explorer.py`.
- [ ] Read the new `chunk-knob-effects.json` rows with the from/to convention (memory: `from_level` is the HIGHER level). Print the per-embedder x per-rung matrix of resolved deltas for cap512 and cap1024. Never a mean only: look for a split between embedders.
- [ ] Do not commit yet. Task 6 changes the same raw files; commit both together in Task 7.

### Task 6: delete the 2-6 percent overlap data (disk + published)

The scope is the four profiles `recursive-t256-o10-gpt2`, `recursive-t256-o15-gpt2`, `recursive-t512-o10-gpt2` and `recursive-t512-o15-gpt2`, on all 7 corpora: 28 chunk sets, 148 vector cells, about 90 GB. The list comes from `ls /embeddings/{chunks,vectors} | grep -E -- '-t(256|512)-o1[05]-gpt2(__|$)'`, saved to `delete-list.txt` and counted (expect 28 + 148) before anything is removed.

- [ ] **Step 1: Back up.** Take a ZFS snapshot of the embeddings dataset (`zfs snapshot zpool-nvme/semdex-test-embeddings@pre-item12-delete`; check first with `zfs list zpool-nvme/semdex-test-embeddings` that the dataset is local here). Copy each score JSON that holds rows for these profiles (`beir_chunk_scores.json`, `chunk_sweep_scores.json`, `chunk_sweep_scores_qwen3.json`, `mldr_chunk_scores.json`, `nf_sci_chunk_scores.json`) to `.bak-20261002`.
- [ ] **Step 2: Remove the rows and files.** A uv-run script (private) does it:
  - loads each score JSON, removes cells whose profile is in the set, writes it back, re-loads and asserts zero remain and the row count dropped by exactly the expected number;
  - moves the matching `perquery/*.npz` (92 files) into a dated `deleted-item12/` dir;
  - then `rm -rf` every path in `delete-list.txt`.
- [ ] **Step 3: Regenerate everything that read them.**
  - `preembed_vectors.py` catalog mode to rebuild `catalog.json`/`.md`;
  - `audit_chunk_dimensions.py` over ALL corpora (main() overwrites with only what it scanned);
  - the truncation census, `export_bench_raw.py`, `gen_bench_tables.py`, `gen_bench_charts.py`, `gen_bench_explorer.py`.

  `grep -rc` the four profile names across `tests/benchmarks/raw`, `docs/` and `/embeddings/catalog.*`: every count must be 0, except in the synthetic test fixtures listed in the consumer map.
- [ ] **Step 4:** Destroy the ZFS snapshot only after Task 7 is pushed and CI is green, with the user's OK. Until then it is the undo.

### Task 7: rewrite the overlap prose and claims, ship

**Files:**
- `docs/benchmarks/03-chunking.md` (overlap sections, the lines at :56-58, :85-110, :333-335, :1408-1417)
- `docs/benchmarks/01-method.md:278`, `04-embedding.md:34`, `07-selection.md:35`, `08-gaps.md:416,450-454`
- `tests/benchmarks/claims/*.toml` (the 42 failing claims plus new ones for the new arm)

- [ ] Run `scripts/check_bench_claims.py`. For each failing claim:
  - delete it if its row is gone by design (the o10/o15 deltas, `*-matches-o0`);
  - otherwise re-derive the number from the raw files and update quote and prose together.

  A quoted phrase must sit on ONE physical line.
- [ ] Write the new overlap findings in percent (cap256 ladder, cap512 0-25 percent, cap1024 0-20 percent), each with the document-level-nDCG caveat in its own cell or sentence. Name the rankings that moved: the MLDR German #1 config changes because the old winner was a 10-token cell.
- [ ] Update 08-gaps: close the gap for overlap at industry sizes, and note that BEIR/MIRACL have no overlap ladder any more (their documents are about one chunk long, so overlap does nothing there).
- [ ] Gate `make test` (read `gate.py`'s own line). Commit raw + docs + claims together, push to main, then watch CI by the full sha with the `ci_wait` jig.
- [ ] OPEN-WORK.md:
  - close [12] with the result;
  - mark [40] partly done (GerDaLIR cap512-o0 now exists);
  - write C11's measurement into [23];
  - add a line for any C5-dropped cell.

## Verification (end to end)

- Pilot: `checks.py` prints C1-C10 PASS, the opus verifier found nothing, and the user said go.
- Run: every `RC_*=0` in `logs/item12-*.log`, and the final line is `DONE`.
- Data: `audit_chunk_dimensions.py` flags nothing. The new cells are in `chunk-sweep-mldr.json` / `chunk-sweep-gerdalir.json` and none of them is in `voided` unless withheld by C5. The four deleted profile names count 0 outside the test fixtures.
- Repo: `make test` green (claims gate, tables-current, charts-current), pushed, and CI green on the full sha.
