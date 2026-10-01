# Design: chunk size-guard + content sanitizer (junk / oversized chunks)

## Context

The semantic/late chunkers emit **oversized chunks**: chonkie groups at sentence granularity, so an
un-splittable "sentence" (a minified-JS/webpack blob scraped into a doc, 0 newlines) overflows the
`max_tokens` target. Audit 2026-07-22 (`mldr_de` semantic): median 51 tokens but **max 43090**, 1685
chunks >512 (0.55%), 151 >4096. The `recursive`/`whitespace` chunkers cap correctly (max 527 =
512+overlap); only the embedding-driven strategies overflow.

Two harms: (1) the embedder truncates a giant chunk to its max_seq (bge-base/small = 512), so ~98% of
its content is silently **lost at embed time**; (2) a semantic-vs-recursive chunk benchmark is biased
against semantic (only it has oversized chunks). The giant chunks are almost always **junk**
(machine content: minified JS/CSS, base64), which no query retrieves as relevant - so the practical
nDCG impact is likely small, but the size-contract violation is a real defect.

This design adds two independent pieces plus a remediation tool.

## Component A - Chunker size-guard (lossless split, always-on)

A pure post-chunk guard in the chunker layer (`adapters/chunker/`, shared by all strategies): any
`Chunk` with `token_count > max_tokens` is split into consecutive <=max_tokens **token windows**
(same overlap policy as the strategy), continuous ordinals, inherited `source`. **Lossless**:
`concat(sub-chunk texts) == original` (modulo overlap). This is exactly what `recursive` already does,
applied as the overflow fallback for `semantic`/`late` only - normal chunks pass through untouched.

- Config: `[chunker].enforce_max_tokens` (bool, default **true**). It is a safety net, not a tuning
  knob; on by default so no strategy can ever violate the size contract.
- It is a SPLIT, never a truncate - nothing is dropped. (Truncation is what happens today, at embed
  time; the guard prevents it.)

## Component B - Content sanitizer (Extract-decorator, opt-in)

A `SanitizingExtractor(inner: Extract, config)` in a new `adapters/sanitize/` package. It implements
the **same `Extract` port**, delegates to the inner extractor, then scans the returned text for
**machine-content** spans and (per mode) strips or just flags them. Wired in `build_extractor`
(composition) as a decorator when `[sanitize].enabled`, so it applies uniformly to every extractor
backend (text/markitdown/docling/xberg/mineru).

Data flow: `source -> inner Extract -> ExtractedDocument(text) -> detect machine-content spans ->
[dry_run: log a SanitizeReport, text unchanged] OR [strip: excise spans + log] -> cleaned
ExtractedDocument -> chunker (with size-guard) -> embed`.

Principles: **read-only** (transforms only the derived extracted text, never the source),
**opt-in** (`enabled=false` default), **never silently lose** (default mode `dry_run` = mark+log
only; every removal logged with bytes/tokens + reason via a frozen `SanitizeReport`).

### Detection (deterministic, no ML, fully CI-testable)

Segment the text into windows (by lines/blocks where newlines exist; a sliding `window_tokens`
window for no-newline blobs) and score each window with cheap heuristics:

- `chars_per_token` < `chars_per_token_max` (~2.5) - the strongest discriminator (prose ~4-5,
  minified/base64 ~1.5-2),
- `whitespace_ratio` < `whitespace_ratio_min` (~0.08) (prose ~0.15),
- `symbol_ratio` > `symbol_ratio_max` (~0.4) (code = many braces/symbols),
- a long run with no sentence terminator.

A window is machine-content if **>= `min_detectors`** (default 2) trip; adjacent junk windows merge
into a span; only spans **>= `min_span_tokens`** (default 128) are acted on (a short code snippet in
real prose is left alone).

## Config

`[chunker].enforce_max_tokens` = true.

`[sanitize]` (all thresholds are config, no magic numbers; documented in `defaultconfig.d/28-sanitize.toml`):
`enabled=false` | `mode="dry_run"` (or `strip`) | `window_tokens=256` | `min_span_tokens=128` |
`chars_per_token_max=2.5` | `whitespace_ratio_min=0.08` | `symbol_ratio_max=0.4` | `min_detectors=2`.

## Testing + validation

- **size-guard** (unit): a synthetic 43090-token chunk -> N chunks each <=max_tokens; `concat ==
  original`; ordinals continuous; provenance inherited. Normal chunks unchanged.
- **detectors** (unit, table-driven): minified-JS -> flagged; DE+EN prose -> NOT flagged (zero false
  positives); base64 -> flagged; short legit code snippet -> not stripped (< min_span).
- **decorator** (unit): dry_run leaves text unchanged + emits report; strip removes the span; report
  counts match.
- **e2e**: a doc with an embedded JS blob through extract->sanitize->chunk -> no chunk > max_tokens
  (size-guard), blob removed (strip) / flagged (dry_run).
- **validation** (`local_only` perf): nDCG on `mldr_de` sanitize off vs strip - must show
  non-regression/improvement BEFORE `[sanitize].enabled` is ever defaulted on ("measure, don't
  trust"). All sanitizer/guard code is pure -> full CI coverage (unlike the embedders).

## Remediation - surgical repair of existing oversized chunks (NOT days of CPU)

The existing cache has oversized chunks in the semantic sets (mainly `mldr_de`). A full re-chunk +
re-embed is the wrong tool (days on fastembed CPU). Instead a `scripts/repair_oversized_chunks.py`:

1. For each affected chunk set: build the NEW chunk list = each old chunk kept as-is, except an
   oversized one replaced by its size-guard splits. (chunking is cheap - seconds.)
2. For each embedder cell of that set: build the new `vectors.npy` by **copying** the unchanged
   chunks' existing vectors and embedding **only the new split pieces** (~a few thousand vs ~300k).
   -> ~minutes (GPU) to ~15 min (fastembed CPU) per cell, never days.

`restart2`'s slow fastembed vectors are NOT wasted - they are the "unchanged" vectors the repair
copies. Run the repair AFTER the go-forward guard/sanitizer land. Practical impact is likely small
(junk chunks are not retrieved), so repair is optional/for a clean semantic-vs-recursive comparison.

## Implementation plan (TDD, in an isolated worktree)

Build in a `git worktree` so the running bp-sweep (main `.venv`, editable) is untouched; merge only
when the sweeps can tolerate a chunker behaviour change (or after they finish).

1. `adapters/chunker/` size-guard (`enforce_max_tokens`, split helper) + config `[chunker]` field -
   tests first.
2. `domain` `SanitizeReport` value + `adapters/sanitize/` detectors (pure) + `SanitizingExtractor` -
   tests first.
3. `adapters/config/sanitize.py` `SanitizeConfig` + `defaultconfig.d/28-sanitize.toml`.
4. Composition wiring in `build_extractor`; import-linter clean; `uv run pyright` + ruff + pytest.
5. `scripts/repair_oversized_chunks.py` (surgical) + a `local_only` nDCG validation sweep.

Defaults on merge: size-guard ON, sanitizer `enabled=false` (opt-in) until the nDCG validation
justifies enabling it.
