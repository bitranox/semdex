# Design: measure the structure-aware chunk strategies on documents that carry the structure

## Context

Two chunk strategies cut on markdown structure. `markdown` is the Rust CommonMark splitter, and
`recursive` uses chonkie's markdown recipe, whose first recursion level splits on `#` to
`######` before it falls back to paragraphs and sentences. semdex indexes markdown, so the
strategies named for its primary input are the ones a deployer reaches for first.

**Correction (2026-10-01):** the premise above is false for every measured cell. The benchmark
chunker (`scripts/preembed_vectors.py`) defaults `SEMDEX_PREEMBED_RECIPE` to `""`, so each
cached `recursive` chunk set ran chonkie's generic rules, which have no `#` level and never cut at
a heading. Only `markdown` read the structure in this measurement; `recursive` acted as a third
structure-blind control. The recipe semdex ships (`recipe = "markdown"`) is measured by a
separate arm, profile `recursive-t256-o0-gpt2-rmarkdown`.

No benchmark corpus carries that structure, so every verdict on the chunking page about those
two strategies is a verdict about their fallback rules. MLDR English is Wikipedia text with the
markup stripped, and the page reports that a scan found no headings to speak of. That scan looked
for `#` markers. The section headings are still there as bare lines: a strict heuristic (a short
line without terminal punctuation, followed by a paragraph of running prose) recovers 56,332 of
them across 7,608 of the slice's 8,000 documents, a median of 5 per document, with samples like
"History", "Early life", "Synthesis", "Robert Bilott investigation".

That makes a paired design possible that a new corpus cannot give: the SAME documents and the SAME
800 queries, once with the headings marked as markdown and once as they are, so the per-query
difference isolates what the structure buys each strategy. It also keeps the whole existing
chain: slice layout, sweep driver, scorer, exporter, tables, claims.

Decision (user, 2026-09-27): MLDR English re-rendered, not a new markdown corpus. A genuinely
markdown-native corpus with deep hierarchy (CRUMB legal_qa: statutes in markdown up to five
heading levels, 6,753 queries, 3.1 GB, cc-by-nc-4.0) stays a candidate for a later item; it
answers a different question (how strategies rank on such a corpus) and cannot isolate structure.

## What is measured

Three questions, in order of what the design can prove:

1. **Does marking the headings change what each strategy retrieves?** Per strategy and per
   embedder, the paired per-query nDCG@10 difference between the marked and the unmarked cell,
   with the same bootstrap interval the chunking page uses. Positive means the marked corpus won.
2. **Which strategy wins once the structure is there?** The exporter already pairs cells within
   one corpus on the strategy axis, so the marked corpus joins the strategy tables as a fourth
   corpus with no new code.
3. **What did the markers do to the chunks?** Chunk count, median size and the share of chunks
   that begin at a heading, marked against unmarked, from the chunk-dimension audit.

Strategies: `markdown` and `recursive` (the two that read structure), `fast` (the same Rust
splitter family as `markdown` with character rules only, the control for "the text changed by a
few characters"), and `semantic` (ignores structure entirely, the second control). One profile
each: cap 256, overlap 0, the page's reference setting. Six embedders, the same set the strategy
tables carry, so the new rows are comparable with the old.

Heading LEVEL is not measured. Every marked line becomes `## ` because the stripped text carries
no level; nested-structure cutting is out of scope and stays in the gaps page.

## Components

### A. Slice builder: `scripts/build_mldr_markdown_slice.py`

Reads `/corpora/mldr-slices/mldr_en_8k_slice` and writes `/corpora/mldr-slices/mldr_en_8k_md_slice`
with identical `doc_ids.txt`, `queries.json` and `qrels.json`, and `corpus.jsonl` re-rendered.
The name follows the slice-family rule (`mldr_` prefix, `_slice` suffix), so the sweep driver and
scorer resolve it with no change.

`mark_headings(text) -> tuple[str, int]` is a pure function in the script: a line is a heading
when it is 1 to 79 characters, ends in no terminal punctuation (`.,;:!?)`), has at most eight
spaces, contains none of `=`, `→`, `+` or a run of three or more digits, and the next non-empty
line is at least 120 characters and ends in `.`, `"` or `)`. A heading becomes `## <line>`. The
rule is the one measured above; its precision is measured, not assumed: the builder writes a
seeded sample of 200 marked lines with their following line to `heading-sample.txt` beside the
slice for inspection, and the write-up reports the share judged wrong.

Guarantees, each pinned by a test on synthetic documents: stripping every `## ` prefix gives back
the source text byte for byte; doc ids and query ids are the source's; a list item, a formula line
and a "See also" line followed by short lines are not marked; a heading followed by prose is.
`meta.json` records the source slice, its `corpus.jsonl` content hash, the rule's parameters, the
count of marked lines and documents. Idempotent like the other builders: an existing slice dir is
never rewritten.

### B. Sweep

`sweep_chunk_profiles.py` as it stands, `SWEEP_CORPORA=mldr_en_8k_md_slice`,
`SWEEP_PROFILES=markdown:256:0,recursive:256:0,fast:256:0,semantic:256:0`, split into the usual
two chains: the CPU chain (two static models, `bge-base`) and the GPU chain (`bge-m3`,
`qwen3-embedding-4b`, `qwen3-embedding-8b`), chunking first so the chains never race on a
`chunks.parquet.tmp`. Then `audit_chunk_dimensions.py` over all corpora (its fit filter needs the
marked corpus's `recursive-t256-o0-gpt2` row) and `score_chunk_sweep.py` over the four profiles
for all six embedders. Twenty-four cells.

Budget, from the rates measured 2026-09-11 on this box: the static models minutes, `bge-base`
about an hour and a half per profile, the two qwen3 models about three and a half hours per
profile each over HTTP. The CPU chain is an afternoon; the GPU chain is about a day and a half,
serial on the one card.

### C. Paired cross-corpus effect: `scripts/score_markdown_structure_effect.py`

The exporter pairs only within a corpus. This script, modelled on
`score_query_length_effect.py`, pairs each marked cell with its unmarked twin
(`mldr_en_8k_md_slice__<profile>__<embedder>` against `mldr_en_8k_slice__<profile>__<embedder>`),
refuses a pair whose query id sets differ, runs the shared `paired_ci` over the per-query arrays,
and writes `tests/benchmarks/raw/markdown-structure-effect.json`: one row per strategy and
embedder with delta, interval, wins, losses, ties, resolved, plus the chunk count and median
size of both cells from the dimension audit. Provenance is carried from the scorer's stamps on
the cells, never taken from this machine. A missing twin is reported by name, never skipped in
silence.

### D. Publication

`gen_bench_tables.py` registers `markdown_structure_effect` (one table, rows per strategy and
embedder); `gen_bench_charts.py` a forest plot beside the strategy plot. `03-chunking.md` gets a
section "When the structure is there" under the strategy section, its caveat and the summary
bullet rewritten; the "no headings to speak of" sentence is corrected to what the scan found.
`07-selection.md` item 6 and `08-gaps.md`'s markdown entry follow, the latter narrowed to heading
levels and non-Wikipedia markdown. Every new figure gets a claim; the marked corpus joins the
claim file's `fit_corpora` set only where the sentence includes it.

## Testing

* Builder: the guarantees above on synthetic text (RED first), and an end-to-end run on a
  ten-document excerpt of the real slice asserting the strip-back identity.
* Paired script: synthetic per-query arrays for two twins, one identical pair (delta zero,
  unresolved), one shifted pair (resolved), one with mismatched query ids (refused by name).
* Tables and charts: the existing "every generated block matches a fresh render" and manifest
  tests cover the new table once registered; the claim gate covers the prose.

## Out of scope

Heading levels; a markdown-native corpus of another domain (the CRUMB candidate above); overlap
and cap ladders on the marked corpus (one profile per strategy is the question's size; a ladder
is a follow-up if the first result resolves).
