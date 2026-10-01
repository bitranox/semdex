# Effect tables: a chance ceiling, a material floor, and ladders judged end to end

Design for backlog item [31]. Decided with the user on 2026-09-27, one decision at a time.

## The problem

`tests/benchmarks/raw/chunk-knob-effects.json` holds one paired bootstrap per pair of cells that
differ in exactly one chunking axis, and the effect tables on the chunking, embedding and selection
pages count how many of those comparisons "resolve" (the 95 percent interval excludes zero). Three
things about that count mislead a reader:

1. **Nothing says how many resolutions chance alone produces.** At alpha 0.05 a family of 1,023
   overlap comparisons carries up to 51 false resolutions, and the table prints 533 as if all were
   findings.
2. **A resolution can be real and worthless.** With 12,298 GerDaLIR queries the interval is narrow
   enough to resolve a step of 0.0009 nDCG@10. Of the 533 resolved overlap comparisons, 138 sit
   under half a point, and 136 of those are GerDaLIR steps. One "no optimum inside the range"
   reading rests on such a step.
3. **The per-level "favour" tally counts partners, not wins.** On a 20-rung ladder a middle rung has
   19 partners and collects 20 to 40 favours while the bottom rung, with the same 19, collects 14
   because it loses to everything above it. The column reads as a result and is mostly geometry.

## Decisions

**Chance floor: print the ceiling, do not correct.** A column "up to N by chance" beside each
resolved count, N being comparisons times alpha rounded up. It is a ceiling (true only if every
comparison were a null), so it tells the reader how much of the count could be noise, not which
rows are. Decided against controlling the false-discovery rate per axis: it needs bootstrap
p-values the export does not carry, it would change the `resolved` flag that every pinned claim
filters on, and at 12,298 queries nearly everything survives it anyway. The material floor is the
filter that removes the noisy rows; this column keeps the remainder honest.

**Material floor: 0.005 nDCG@10, absolute.** A resolved comparison whose absolute mean delta is
under the floor is "resolved but immaterial". At 0.005 the drop is exactly the GerDaLIR artefacts
(136 of 138 dropped rows are 12,298-query steps) and nothing on a small-query corpus. Decided
against 0.010, the one-point convention: it halves the overlap axis (533 to 257) and cuts strategy
from 121 to 69, which reopens the chunking page's headline verdicts. Decided against deriving the
floor from the small corpora's half-width: the value would move whenever cells are added. The
export carries no per-cell base score, so a relative floor would need a second lookup; not now.

**Ladder tally: top against bottom, once per ladder.** For an axis whose levels are all numeric
(overlap_tokens and max_tokens today), the direction cell reports, per ladder (corpus, embedder,
held-fixed set), the single pair of the highest and lowest level: "N ladders favour more, M favour
less, K unresolved or immaterial". A categorical axis (strategy, breakpoint model, embedding,
method) keeps the per-level tally, since every level there has the same partner count; the tally
counts material resolutions only, so the column agrees with the count beside it. Where the optimum
sits stays in the per-embedder ceiling table, which the cell points at. Ordinality is detected from
the level values rather than kept in a list, because a list goes stale the moment a new numeric
knob (a minimum chunk size, item [33]) arrives and a stale list silently falls back to the tally
this design removes.

## Components

* `scripts/_score_stats.py`: one named constant for the material floor beside the bootstrap's
  alpha, so the exporter, the table generator and the claim checker share a single definition.
* `scripts/export_bench_raw.py`: the effects payload gains top-level `alpha` and `material_floor`
  (keyed by metric) and every row a boolean `material`. No other row field changes, so every claim
  that filters on `resolved` keeps its count. Re-export runs locally against
  `/embeddings/scores/perquery`.
* `scripts/gen_bench_tables.py`: the knob summary gains "up to N by chance" and "resolved and
  material" and splits its direction column by axis type; the ceiling table reports steps resolved
  and material, the highest material step, and tags an immaterial last step. A general ladder
  grouping (corpus, embedder, held-fixed) serves both the direction cell and the ceiling table.
* `scripts/check_bench_claims.py`: `material` joins `favoured_level`, `unfavoured_level` and
  `favoured_end` as a field a claim's `where` can filter on.
* Pages 03-chunking, 04-embedding, 07-selection: the table's reading and the tally sentences,
  rewritten as ladder statements with fresh claims; 08-gaps closes its paragraph on this gap.

## Tests, red first

On a synthetic effects document: a three-rung ladder where the partner tally favours the middle
rung but top against bottom says "more"; the ceiling arithmetic including the round-up; a resolved
row under the floor flagged immaterial and excluded from the material count and the tally; a
categorical axis untouched by the ladder rule; the derived field reaching the claim checker. The
existing regenerate-and-compare test then pins the rendered pages.

## Out of scope

A false-discovery-rate count (the reviewer's instrument, filed if wanted), a relative floor, and
any chart: the item names tables.
