# Design: score the product's delivered chunk list, not the deduplicated document list

## Context

Every chunking verdict in `docs/benchmarks/03-chunking.md` is nDCG@10 over DOCUMENTS:
`scripts/score_chunk_sweep.py` over-fetches chunks, deduplicates the ranking down to ten distinct
documents, and scores that. `semdex search` does something else: it returns `default_k = 5` chunks
(`10-index.toml`) with no deduplication, at `max_tokens = 256`. Under overlap, or wherever a document
has several chunks close to the query, adjacent chunks of one document score alike, so the product's
five slots can fill with one document's neighbours where the document metric sees one hit. The only
measurement so far that scores the product's unit is the span-integrity pass
(`scripts/score_span_integrity.py`, k=5, one embedder, three SQuAD-style corpora), and it stops at
the answer-span question.

Backlog item [27] asks for the missing half: the product's own unit on the document-judged
long-document corpora, and a comparison of chunk sizes at a FIXED retrieved-token budget, since a
consumer that reads k chunks of c tokens reads k times c tokens whatever the cap.

Decisions taken with the user on 2026-09-27:

* Build the scorer now, run it after the [22] sweep frees the box: the run is model-free (query
  vectors are cached per embedder) but its top-k streaming would contend with the sweep's CPU chain.
* Metric: nDCG@k over the raw chunk list where a repeated document earns zero gain, beside the
  dedup nDCG@k from the SAME top list, plus the mean count of distinct documents in the top k.
* Two token budgets: 1280 tokens (the shipped product answer, 5 x 256) with k = 20, 10, 5, 2 at
  caps 64, 128, 256, 512, and 2560 tokens with k = 40, 20, 10, 5.
* Scope: the k=5 verdict on every cap-256, zero-overlap cell of the three long-document corpora
  (MLDR English, MLDR German, GerDaLIR) across all strategies and the six embedders; the budget
  ladders on `recursive` zero-overlap cells of MLDR English (all four caps) and GerDaLIR (64, 128,
  256; the 512 rung reported as not measured).

## What is measured

For one cell (corpus, profile, embedder) and one query, the exact top-100 chunk list by cosine
from `_score_kernel.topk_stream`. From that list, for each k in the cell's rung set:

* `ndcg_delivered@k`: the list cut at k, each chunk's gain equal to its document's relevance the
  FIRST time that document appears and zero after. A slot spent on a document already delivered
  is a wasted slot. This is what a consumer of `semdex search` gets.
* `ndcg_documents@k`: the list deduplicated to k distinct documents in order, scored as
  `score_chunk_sweep` does. This is what every page so far has measured, at the product's k.
* `distinct_docs@k`: how many distinct documents the first k chunks name, mean over queries.
* The paired per-query difference `ndcg_documents@k - ndcg_delivered@k` with a bootstrap interval
  from `_score_stats`: the cost of not deduplicating, in the metric family the pages use.

The rung set for a cell is k=5 always, plus the budget rungs its cap qualifies for: cap 64 gets
20 and 40, cap 128 gets 10 and 20, cap 256 gets 5 and 10, cap 512 gets 2 and 5. Each row names its
budget (1280 or 2560) beside its k, so a table groups by budget without recomputing. A cap outside
the ladder gets k=5 only.

## Components

### A. Scorer: `scripts/score_product_k.py`

Env-config, no argparse, in the shape of its siblings. Per cell it reuses `score_chunk_sweep`'s
exported seams (`query_vectors`, `dirsafe`, the uri loader) and `topk_stream` with `fetch=100`.
It is model-free by design: a cell whose query-vector cache is absent is REFUSED by name, never
embedded on the spot, so the run never loads an embedder and can share a box with a sweep.
Output: `/embeddings/scores/product_k_scores.json` (respecting `SEMDEX_SCORE_OUT`), one row per
cell and rung with the three means, their intervals, the paired gap, and the per-cell provenance
stamp the stamp test requires; per-query arrays in an npz beside it.

Knobs, all env with a config-recorded default: `SEMDEX_PRODUCT_K` (5), `SEMDEX_PRODUCT_BUDGETS`
(`1280,2560`), `SEMDEX_PRODUCT_FETCH` (100), the corpus and profile selectors the siblings share.

### B. Export

A dedicated writer in `scripts/export_bench_raw.py` beside the knob-effects writer, not an
`_EXPORTS` entry: `export_row` is built for the chunk-sweep metric set and would print null
metrics for every product-k cell. It writes `product-k.json` from `product_k_scores.json` as FLAT
rows, one per cell and rung, because the claim checker's dotted getter walks dicts only and a
figure inside a nested list could never carry a claim. Corpora are the three long-document slices
behind the same foreign-corpus guard; the note says the file scores the product's delivered chunk
list rather than deduplicated documents and names the two budgets. The unscored-cell guard unions
the sources of `_EXPORTS` only, so a writer outside it changes no census and the cap-256 subset
scored here leaves nothing unscored: the chunk-sweep exports already cover every cell.

### C. Tables and chart

`scripts/gen_bench_tables.py`, two registrars on the one raw file:

* `product_k_verdict`: one row per corpus, strategy and embedder at cap 256, zero overlap:
  documents nDCG@5, delivered nDCG@5, the paired gap with interval and resolved flag, mean distinct
  documents in five slots.
* `product_k_budget`: one row per corpus, embedder and cap: the k at each budget and the delivered
  nDCG@k there; GerDaLIR's 512 rung prints as not measured.

`scripts/gen_bench_charts.py`, one chart `chunk_product_k_budget`: delivered nDCG@k against cap on
MLDR English, one line per embedder, the two budgets as two panels. The gap stays a table.

### D. Publication

A new subsection on the chunking page beside "What document-level scoring counts but does not
deliver", that page's "A reader consumes the retrieved chunk" row in its What-to-set table citing
the verdict (the settings table lives on the chunking page; the selection page has no k row), and
every quoted figure a claim in `tests/benchmarks/claims/03-chunking.toml`.

## Testing

* The two metric functions on hand-built rankings: the all-one-document list (delivered gain
  collapses to the first slot, documents nDCG unaffected), interleaved documents, a relevant
  document first seen past k, an empty qrels entry.
* The rung table: every cap maps to the k the budgets imply, an off-ladder cap to k=5 only.
* The scorer refuses a cell with no query-vector cache by name and scores a fixture cell whose
  vectors and query cache are written by the test.
* The export spec (source, corpora, note) and the two registrars on a fixture raw file, the chart
  collector on a fixture, the provenance stamp test extended to the new scorer.

## Out of scope

Overlap cells at k=5 (the verdict is at the shipped zero overlap; overlap's cost in delivered
slots is a follow-up if the gap is material), the BEIR and MIRACL corpora (about one chunk per
document, so delivered equals documents by construction), and a change to `semdex search` itself.
