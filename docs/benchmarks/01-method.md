# Method

How these numbers are produced, and what they can and cannot support. Every other page in this
directory assumes what is written here.

## Metrics

| Metric    | What it asks                                                                       | Range            |
|-----------|------------------------------------------------------------------------------------|------------------|
| nDCG@10   | Of the ten results returned, how good are they and how high did the good ones rank | 0 to 1           |
| Recall@10 | What fraction of all relevant documents made it into the top ten                   | 0 to 1           |
| MRR       | How far down the list the first relevant result appeared                           | 0 to 1           |
| P@1       | Was the very first result relevant                                                 | 0 or 1 per query |

nDCG@10 is the headline because it is the only one of the four that rewards both finding the
relevant documents and ordering them well. The others are reported beside it because they fail in
different directions: a configuration can win on Recall@10 by dragging relevant documents into
positions 8 to 10 while losing on P@1.

### How much the four columns actually tell you

That four columns imply four perspectives was an assumption on this page for a long time, hedged
with an argument about the long-document corpora. It is measurable across the configurations
already swept, so here it is measured.

<!-- BEGIN GENERATED metric_redundancy (scripts/gen_bench_tables.py) -->
| Corpus               | Cells | Relevant docs/query | Effective views | Near-duplicate pairs |
|----------------------|-------|---------------------|-----------------|----------------------|
| miracl_de_100k_slice | 48    | 2.65                | 1.01            | 6/6                  |
| mldr_en_8k_slice     | 32    | 1.00                | 1.02            | 6/6                  |
| mldr_de_3k_slice     | 32    | 1.00                | 1.03            | 6/6                  |
| fiqa                 | 4     | 2.63                | 1.04 (thin)     | 3/6                  |
| miracl_en_100k_slice | 48    | 2.91                | 1.07            | 3/6                  |
| scifact              | 32    | 1.13                | 1.09            | 3/6                  |
| nfcorpus             | 47    | 38.19               | 1.12            | 2/6                  |
| cqadupstack          | 11    | 1.91                | 1.51 (thin)     | 2/6                  |


`Effective views` is the participation ratio of the four metrics' rank-correlation matrix across the swept configurations: 4.00 would mean four independent perspectives and 1.00 means one. A pair counts as near-duplicate at |rho| >= 0.95. Marked `(thin)` below 20 cells, where a rank correlation is not stable enough to read.
<!-- END GENERATED metric_redundancy -->

**On every corpus with enough cells to say, it is one view.** The effective count runs 1.01 to
1.12, where four independent metrics would give 4.00. The one reading above 1.2 comes from
cqadupstack, which has eleven cells and is too thin to carry a rank correlation.

**The mechanism this page used to cite is real, and exact.** Both MLDR slices ship precisely 1.00
relevant documents per query, and both collapse completely: all six metric pairs sit above the
near-duplicate threshold. With one relevant document, Recall@10 can only be 0 or 1 and nDCG@10 has
a single non-zero gain whose position is what MRR already reports.

**But it is not the whole explanation.** nfcorpus averages 38 relevant documents per query, the
opposite extreme, and still returns 1.12 views. Judgment density explains the total collapse on
MLDR; it does not explain why the metrics barely separate anywhere else. The four columns are
closer to one view than to four regardless of the corpus.

<!-- BEGIN GENERATED metric_disagreement (scripts/gen_bench_tables.py) -->
| Metric pair          | Lowest | Highest | Where it disagrees most |
|----------------------|--------|---------|-------------------------|
| ndcg@10 vs recall@10 | 1.9%   | 6.2%    | nfcorpus                |
| mrr vs p@1           | 0.8%   | 9.0%    | scifact                 |
| ndcg@10 vs mrr       | 1.0%   | 9.2%    | nfcorpus                |
| ndcg@10 vs p@1       | 1.7%   | 11.5%   | nfcorpus                |
| recall@10 vs mrr     | 2.8%   | 13.8%   | nfcorpus                |
| recall@10 vs p@1     | 3.5%   | 16.3%   | nfcorpus                |


Over every pair of swept configurations within a corpus, the share where the two metrics rank them oppositely. Pairs tied under either metric are excluded rather than counted as agreement, since a tie is the metric declining to choose. Thin corpora are left out.
<!-- END GENERATED metric_disagreement -->

The pairs split the way the definitions suggest: `nDCG@10` with `Recall@10` (did you find them),
and `MRR` with `P@1` (was the top hit right). Those two groupings are the tightest, and the
disagreements that do occur are across the groups. The intuition stated above this section - that
a configuration can win on Recall@10 while losing on P@1 - is the pair that separates most, and it
does so on 16.3 percent of comparisons at the very most.

**What to do with that.** Read `nDCG@10`. On the long-document corpora the other three change the
answer in fewer than one comparison in twenty; even at the widest, on nfcorpus, better than five in
six comparisons agree. The other columns are worth keeping as a cheap check that nothing pathological
happened, not as independent evidence, and a difference that shows up in only one of them is more
likely noise than a finding.

## Uncertainty, and why so many comparisons are ties

Every score is a mean over a query set. Per-query nDCG@10 has a standard deviation around 0.4
(0.24 to 0.47 across the cells), so the precision of a mean is set by how many queries the corpus
ships:

<!-- BEGIN GENERATED ci_precision (scripts/gen_bench_tables.py) -->
| Corpus              | Queries | Mean 95% half-width | Smallest separable difference |
|---------------------|---------|---------------------|-------------------------------|
| cqadupstack         | 876     | +/-0.024            | 0.048                         |
| mldr_en_8k_md_slice | 800     | +/-0.022            | 0.045                         |
| mldr_en_8k_slice    | 800     | +/-0.022            | 0.044                         |
| fiqa                | 648     | +/-0.026            | 0.052                         |
| nfcorpus            | 323     | +/-0.034            | 0.067                         |
| scifact             | 300     | +/-0.043            | 0.086                         |
| mldr_de_3k_slice    | 200     | +/-0.058            | 0.117                         |


A difference smaller than the half-width is not a finding, it is noise. This is why several previously published verdicts, decided on gaps of 0.01 or less, are reported here as unresolved.
<!-- END GENERATED ci_precision -->

Two consequences run through every page:

1. **A ranking table's neighbouring rows are usually not separated.** Rows are ordered by mean,
   and the mean is the best single estimate, but an ordering is not a finding.
2. **A winner claim needs a paired comparison, not two overlapping intervals.** Both
   configurations answer the same queries, and query difficulty is most of the variance and is
   common to both. Bootstrapping the per-query *difference* removes it, and routinely resolves a
   comparison that the two marginal intervals leave looking ambiguous. Comparing intervals by eye
   is the weaker test, and it is not used here.

Where a comparison does not resolve, the tables say so rather than printing the larger number in
bold.

The chunking, embedding and selection pages read that resolved count two further ways: an "Up to
N by chance" column is comparisons times alpha, rounded up, the resolutions that many true nulls
would produce on average; and a material column narrows it to the resolutions whose
absolute effect also clears the file's material floor, so a real but tiny difference does not
count as a finding either.

The CI regression gate is the same rule applied to the same slice across time rather than across
configurations: two runs of a fixed slice answer identical queries, so `scripts/benchmark_compare.py`
differences them per query instead of comparing two means. That is what makes 20 queries enough to
gate on at all, and it is why the baseline stores per-query scores and not just averages.

### A tie is two different things, and they need opposite responses

The tables above say "not resolved" for two situations that look identical and are not. One is a
real difference too small to see at this query count, which more data would settle. The other is
two configurations that are the same, which no query count settles because there is nothing to
find. Told apart, the first is a shopping list and the second is an answer.

A paired half-width shrinks as one over the square root of the query count, so an observed effect
`d` at half-width `h` needs about `n x (h/d)^2` queries to clear it. Applied to every unresolved
comparison, that turns "underpowered" into a number.

<!-- BEGIN GENERATED query_power (scripts/gen_bench_tables.py) -->
| Corpus           | Queries | Ties    | Median effect | Median half-width | Queries needed | Never |
|------------------|---------|---------|---------------|-------------------|----------------|-------|
| scifact          | 300     | 51/96   | 0.0161        | 0.0231            | 797            | 2     |
| mldr_de_3k_slice | 200     | 173/373 | 0.0352        | 0.0352            | 1,236          | 17    |
| nfcorpus         | 323     | 39/96   | 0.0157        | 0.0132            | 1,377          | 4     |
| mldr_en_8k_slice | 800     | 180/373 | 0.0119        | 0.0140            | 3,718          | 18    |
| fiqa             | 648     | 1/6     | 0.2035        | 0.0247            | 7,597          | 0     |
| cqadupstack      | 876     | 12/18   | 0.0017        | 0.0061            | 11,279         | 1     |


`Queries needed` is the median over that corpus's unresolved comparisons of `n x (half-width / effect)^2`, the count at which the interval would clear the observed effect. `Never` counts comparisons whose effect is indistinguishable from zero: those are not underpowered, they are answered, and the answer is that the two configurations are the same. Both assume the OBSERVED effect is the true one, which for a comparison that came out a tie is optimistic, so read the counts as a floor.
<!-- END GENERATED query_power -->

**The corpus with the fewest queries is not the one starved of evidence.** The German slice ships
200 queries, the smallest set on the page, and its median tie needs about 1,236 - roughly six times
what it has, which is an ordinary dataset size. `cqadupstack` ships 876 queries, over four times as
many, and its median tie needs 11,279. The reason is in the effect column: German effects are the
largest here at 0.0352 median (fiqa's 0.2035 rests on one tie) and cqadupstack's are the smallest
at 0.0017, twenty times smaller. What decides whether a query set is adequate is its size against
the effects it has to see, never its size alone.

**The same comparisons in English, with four times the queries, resolve slightly less often.**
`mldr_en` and `mldr_de` are the same 373 comparisons over the same profiles. English has 800
queries and intervals two and a half times tighter, and still returns 48.3 percent ties against
German's 46.4. Its effects are three times smaller, and that cancels the extra data out.

<!-- BEGIN GENERATED query_power_ladder (scripts/gen_bench_tables.py) -->
| Corpus           | Ties | 500 | 1,000 | 2,000 | 5,000 | 20,000 | 100,000 |
|------------------|------|-----|-------|-------|-------|--------|---------|
| scifact          | 51   | 9   | 29    | 36    | 42    | 46     | 47      |
| mldr_de_3k_slice | 173  | 46  | 73    | 100   | 121   | 136    | 149     |
| nfcorpus         | 39   | 5   | 15    | 20    | 28    | 32     | 33      |
| mldr_en_8k_slice | 180  | 0   | 13    | 52    | 95    | 127    | 143     |
| fiqa             | 1    | 0   | 0     | 0     | 0     | 1      | 1       |
| cqadupstack      | 12   | 0   | 0     | 2     | 2     | 8      | 10      |


Cumulative: each column counts the corpus's currently unresolved comparisons that would separate at that many queries. A row that barely moves across the whole ladder is a corpus whose ties are near-zero effects rather than a corpus starved of data.
<!-- END GENERATED query_power_ladder -->

**Some ties are already answered.** Seventeen German comparisons and eighteen English ones have an
effect indistinguishable from zero. Those are not waiting on data; they are saying the two
configurations perform the same, and no corpus changes that. Counting them as "unknown" overstates
how much a bigger query set would buy.

### Does the prediction hold up

The table above is arithmetic on an assumption, so it was checked rather than trusted. The English
slice runs the same comparisons at 800 queries; subsampling it to 200, predicting from that, and
scoring against the full set the prediction never saw is a holdout that costs nothing.

<!-- BEGIN GENERATED query_power_validation (scripts/gen_bench_tables.py) -->
| Check                     | Expected       | Measured                    | Reading                                  |
|---------------------------|----------------|-----------------------------|------------------------------------------|
| Half-width scaling        | 2.00x          | 2.01x (1.85 to 2.08)        | the 1/sqrt(n) mechanism holds            |
| Ties predicted to resolve | matches actual | 75 predicted, 60 actual     | optimistic by 1.24x                      |
| Which ties resolve        | 1.00 precision | 0.47 precision, 0.59 recall | barely better than chance per comparison |


The English slice runs the same comparisons at 800 queries. Subsampling it to 200 - the size of the German set - predicting from that, and scoring against the full set the prediction never saw, over 8 subsamples. The count is inflated because the effect estimated from a small sample is noisy, and the comparisons that look nearly resolvable are the ones most likely to be overstated.
<!-- END GENERATED query_power_validation -->

**The mechanism is sound and the numbers are optimistic.** The half-width really does fall as one
over the square root of the query count, within a few percent. But the predicted COUNT of ties that
a larger corpus would settle runs about a quarter high, because each required figure assumes the
observed effect is the true one and a small sample overstates exactly the effects that look nearly
resolvable.

**And it cannot tell you WHICH ties.** Under half the comparisons it predicts will resolve actually
do, and it misses two in five of the ones that do. Read the ladder as a rough count of how much a
larger corpus buys, discounted by about a quarter, and not as a list of which questions get
answered.

## Corpus fitness: the entry requirement for a chunking claim

A chunker can only be judged on documents long enough to chunk. If a corpus yields about one chunk
per document, then every configuration produces that same one chunk, the retrieval scores are
identical by construction, and any ranking between them is noise.

<!-- BEGIN GENERATED corpus_fitness (scripts/gen_bench_tables.py) -->
| Corpus                | Documents | Chunks    | Chunks/doc | Median chunk tokens | Can carry a chunk claim |
|-----------------------|-----------|-----------|------------|---------------------|-------------------------|
| mldr_de_3k_slice      | 3,000     | 201,824   | 67.28      | 215                 | yes                     |
| gerdalir_de_12k_slice | 12,000    | 431,721   | 35.98      | 220                 | yes                     |
| mldr_en_8k_md_slice   | 8,000     | 148,189   | 18.52      | 214                 | yes                     |
| mldr_en_8k_slice      | 8,000     | 148,008   | 18.50      | 214                 | yes                     |
| nfcorpus              | 3,633     | 6,922     | 1.91       | 212                 | NO                      |
| scifact               | 5,183     | 9,406     | 1.81       | 213                 | NO                      |
| cqadupstack           | 32,176    | 52,987    | 1.65       | 169                 | NO                      |
| fiqa                  | 57,600    | 72,884    | 1.26       | 123                 | NO                      |
| miracl_de_100k_slice  | 100,000   | 119,453   | 1.20       | 121                 | NO                      |
| miracl_en_100k_slice  | 100,000   | 103,663   | 1.04       | 74                  | NO                      |
| msmarco_1000000       | 1,000,000 | 1,000,587 | 1.00       | 67                  | NO                      |
| msmarco_250000        | 250,000   | 250,149   | 1.00       | 67                  | NO                      |
| msmarco_50000         | 50,000    | 50,031    | 1.00       | 66                  | NO                      |


Measured at `recursive cap256 ov0tok`. A corpus that yields about one chunk per document cannot distinguish chunk configurations at all, because every profile produces the same single chunk. The threshold of 3 chunks/doc is the point below which the profiles in this sweep stop differing meaningfully.
<!-- END GENERATED corpus_fitness -->

This is why the chunking page is measured on MLDR. The MIRACL and MSMARCO slices remain in the
repository and are used heavily, but as an **embedder** comparison and as real-vector fodder for
the store benchmarks, never as evidence about chunk size.

## Verifying the axes instead of trusting their names

A profile string records the knobs a chunk set was built with. It does not prove any of them took
effect. An adapter can accept a knob and ignore it, and two differently named profiles can hold
byte-identical chunk boundaries. So every axis is checked against the chunk data itself, by
comparing sets that differ in exactly one knob and hashing the chunk boundary sequence:

<!-- BEGIN GENERATED chunk_axis_status (scripts/gen_bench_tables.py) -->
| Axis               | Strategy   | Verdict     | Groups compared | Per-group outcome       |
|--------------------|------------|-------------|-----------------|-------------------------|
| `breakpoint_model` | semantic   | real        | 4               | real 4                  |
| `max_tokens`       | recursive  | real        | 22              | real 22                 |
| `max_tokens`       | semantic   | real        | 17              | real 17                 |
| `max_tokens`       | whitespace | real        | 2               | real 2                  |
| `overlap_tokens`   | fast       | real        | 3               | real 3                  |
| `overlap_tokens`   | markdown   | real        | 3               | real 3                  |
| `overlap_tokens`   | recursive  | conditional | 15              | content-only 12, real 3 |
| `strategy`         | mixed      | real        | 27              | real 27                 |


Measured from the chunk sets, not read off the profile names. `inert` means the chunk boundaries were byte-identical across the axis levels, `content-only` means the boundaries did not move but the chunk text did, `weak` means under 5 percent of chunks moved, `conditional` means the axis bites under some held-fixed values and not others.
<!-- END GENERATED chunk_axis_status -->

Three findings from that check shape how the chunking page must be read, and they are stated
there in full: overlap is counted in tokens rather than percent and applies to one strategy only,
`max_tokens` is a target rather than a cap for the semantic chunker, and `token_count` is not
comparable between strategies.

## Naming

Cache directories keep their original short tags, because several hundred cells are keyed by them.
Everything a reader sees is renamed so it needs no legend:

| Cache tag                | Shown as                              | Why                                                                                                                                                                                                                                 |
|--------------------------|---------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `o0` / `o10` / `o15`     | `ov0tok` / `ov10tok` / `ov15tok`      | The number is TOKENS. `o10` reads as 10 percent, and it is not.                                                                                                                                                                     |
| `t256` under `recursive` | `cap256`                              | A hard cap that is enforced.                                                                                                                                                                                                        |
| `t256` under `semantic`  | `hint256`                             | A target rather than a cap: chonkie groups at sentence granularity and overflows it. The `enforce_max_tokens` guard (default true) splits the overflow, and the cached sets have been repaired with it, so they now hold their cap. |
| `bp<model>`              | `breakpoint-<model>`                  | Unreadable in a chart legend.                                                                                                                                                                                                       |
| `semantic` with no `bp`  | `breakpoint-default(potion-base-32M)` | The default is invisible in the tag, and it is English-distilled, which is load-bearing for German results.                                                                                                                         |
| `gpt2`                   | `count-gpt2`                          | It is the token COUNTER, not any embedder's tokenizer.                                                                                                                                                                              |

## Hardware

| Role                      | Machine                                             | Notes                                                    |
|---------------------------|-----------------------------------------------------|----------------------------------------------------------|
| Scoring, store benchmarks | `lxc-pydev`, Intel i9-13900K, 20 cores, AVX2, 63 GB | lancedb needs AVX2, so store numbers must be taken here. |
| GPU embedding             | CT 60400 on `proxmox06`, RTX 4070 Ti SUPER 16 GB    | Serves ollama. One model resident at a time.             |

Latency figures are only meaningful with the machine attached, so every raw file records host,
CPU, numpy and BLAS build, and thread count.

## Measurement tiers

| Tier    | What it is                                                     | Where it lives                              | What it can support                                                                                                    |
|---------|----------------------------------------------------------------|---------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| Gated   | 150 documents, 20 queries, re-run in CI weekly and on dispatch | `tests/benchmarks/baseline.json`            | Catching a regression, by comparing the same 20 queries pairwise against the baseline rather than comparing two means. |
| Swept   | The full offline grid over the pre-computed vector cache       | `tests/benchmarks/raw/chunk-sweep-*.json`   | The selection advice on these pages. Carries intervals.                                                                |
| One-off | A single reference run on a named machine                      | the remaining `tests/benchmarks/raw/*.json` | Order of magnitude. Not a third decimal, and not a regression signal.                                                  |

## Where the numbers live, and why the tables are generated

Every table on these pages is rendered from committed JSON in `tests/benchmarks/raw/` by
`scripts/gen_bench_tables.py`, between `BEGIN GENERATED` markers. Nothing is hand-transcribed.
This is not tidiness: the previous documentation stated the store quality lever as 0.71/0.81/0.83
where the raw data says 0.6534/0.7933/0.8257, and claimed one store "held 100% recall" where the
same repository's own numbers said 0.783. A hand-copied number drifts silently, and only one table
in the whole previous set was ever checked against its source.

To refresh after new measurements:

```bash
python scripts/export_bench_raw.py     # cache -> tests/benchmarks/raw/
python scripts/gen_bench_tables.py     # raw -> the tables on these pages
python scripts/gen_bench_charts.py     # raw -> docs/benchmarks/img/
```

`python scripts/gen_bench_tables.py --check` exits non-zero if any published table has drifted
from its data, which is what CI runs.

The export also counts what nobody scored. It diffs the vector cache against every score file that
publishes a cell's corpus, prints each vector cell with no score by name, and exits non-zero on one
unless `tests/benchmarks/raw/withheld-cells.json` records that cell with a reason. Every earlier
stage reports the effort it made; this is the one count taken from the corpus it was handed, and
it exists because measured cells sat unscored four times while the pages read as complete.

The figures quoted in the PROSE around those tables are hand-transcribed, because a sentence is
not a rendering of anything. Each one is instead paired with its derivation in a claim file under
`tests/benchmarks/claims/`, and `python scripts/check_bench_claims.py` recomputes every derivation
from the same raw JSON and exits non-zero on a mismatch. A claim records what the number MEANS -
which rows were reduced, and how - so reading the claim file beside its page is also how you see
which figures on that page are guarded at all. A figure nobody wrote a claim for is not checked.

## Reproducing the measurements

All of these reuse the pre-computed vector cache under `/embeddings`, so none of them re-embeds a
corpus. Timings are from the machine above.

```bash
# What each axis actually did (80 chunk sets, streamed) - about 30 seconds
python scripts/audit_chunk_dimensions.py

# Score cached cells. CPU embedders need no GPU; the profile list is mandatory for MLDR
SEMDEX_SCORE_CORPORA=mldr_en_8k_slice,mldr_de_3k_slice \
SEMDEX_SCORE_PROFILES=recursive-t256-o0-gpt2,...,semantic-t512-o0-gpt2 \
SEMDEX_SCORE_EMBEDDINGS=fastembed:bge-small,fastembed:bge-base \
SEMDEX_SCORE_OUT=/embeddings/scores/mldr_chunk_scores.json \
  python scripts/score_chunk_sweep.py                      # about 12 minutes for 64 cells

# Exact against ANN on real embeddings, by dimension
SOURCE=cache DIMS=256,384,512,768,1024,2048,2560,4096 SCALES=25000,50000,100000 \
QUERIES=100 REPEATS=3 WORK=/embeddings/tmp-dimx OUT=dim-crossover-real.json \
  python scripts/bench_dim_crossover.py
```

Scoring is exact brute-force cosine over the cached vectors, streamed in blocks, so it needs no
store and returns what an exact store would. Only the evaluation queries are embedded, and those
are cached across profiles.
