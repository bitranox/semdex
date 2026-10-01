# Gaps

What is not measured, ranked by how badly a reader is misled if nobody says so. A benchmark set
that only lists what it measured invites the reader to assume the rest was checked and found
unimportant.

Each entry states the gap, why it matters, and what would close it.

## 1. The retrieval-method measurements are narrow

Lexical retrieval, fusion and cross-encoder reranking are all measured now (see
[Retrieval method](06-retrieval-method.md)), which is what this entry used to say was missing.
What remains is how narrow that measurement is, and it is narrow enough to matter:

* **One chunk profile**, `recursive cap256 ov0tok`, on two long-document corpora. The interaction
  between chunk size and lexical matching is untested, and a chunk size that suits dense recall
  need not suit BM25.
* **One reranker at one depth.** `bge-reranker-base` over the top 20. The larger
  `bge-reranker-v2-m3` was measured at 3.3 pairs per second against 13.5 on this CPU and was
  skipped on cost, not on quality, so "a bigger reranker would not have helped either" is NOT
  established.
* **Document-level reranking through one representative chunk.** A cross-encoder able to read a
  whole MLDR document, or judgements made per chunk rather than per document, would be measuring
  a different thing.
* **RRF's damping constant** is left at the conventional 60 and never swept.

The reranking result in particular is a negative one on strong baselines, and negative results
generalise worse than positive ones. It says this reranker did not help this first stage on these
corpora; it does not say reranking is a bad idea.

*To close it:* extend the lexical and rerank sweeps across the chunk profiles, add a larger
reranker on a GPU, and sweep the fusion constant.

## 2. The composed store measurement is one profile at one scale

Quality is now measured THROUGH the stores rather than only through an exact scan, with latency
from the same run (see [Vector store](05-vector-store.md)). What that leaves open is coverage:

* two corpora, one chunk profile, two embedding models, at 148,000 and 201,824 chunks. The
  crossover behaviour at a million rows is still only measured as latency and recall, never as
  quality.
* the composed quality-through-store table is `lancedb` only, at the shipped
  `nprobes=10 refine_factor=5` setting. The
  [tuning sweep](05-vector-store.md#tuning-what-each-store-can-reach-not-what-it-does-by-default)
  scores all three stores at many parameter settings on one cell, so the gap left here is a
  composed table for the two server stores, not the question.
* the judgements are document-level and MLDR carries roughly one relevant document per query, so
  the finding that approximation costs little may be softer on a corpus with many relevant
  documents per query, where losing tail documents would cost more.

There is also an open correctness question underneath the coverage one. On the gated nfcorpus
slice the four stores that scan every row should return identical rankings, and two of them do not:
`json`, `sqlite_vec` and `lancedb` agree on all 20 queries, while `pgvector` differs on 4 and
`mariadb` on 2. Each store's difference points one way, which float noise would not do.

It is also widening. Five weeks apart on a pinned fixture, corpus and embedding model, the three
agreeing stores returned byte-identical scores both times, while `pgvector` moved from 0.6230 to
0.6303 and `mariadb` from 0.6182 to 0.6054. A fixed precision offset would not drift, so something
version- or state-dependent in the two server stores is involved. Until that is explained, their
quality numbers carry an unexplained offset of about 0.01 to 0.02 nDCG@10 against the exact scan.

*To close it:* the same run at MSMARCO scale, and against pgvector, after resolving the exact-store
disagreement.

## 3. The CI regression gate compares runs per query, but only where it has per-query data

The gated slice is 150 documents and 20 queries. Comparing two means over 20 queries carries a
95 percent half-width of +/-0.099 on nDCG@10, bootstrapped over the gate's own baseline scores
(median across the 26 cells that carry them), against a tolerance of 0.03. That gate was blind to
real drops several times its threshold and could fire on noise, which is the worst of both.

The slice size is not the thing that was wrong. Both runs answer the *same* queries, so the
difference can be taken query by query, which removes query difficulty - nearly all of that
variance. The comparator now does that: where both the baseline and the current run carry
per-query scores it compares the shared queries pairwise, and reports a regression when the mean
difference falls past 0.005 and more queries got worse than better. Same 20 queries, a change
20 times smaller resolved.

The 0.005 is a threshold rather than a noise floor, which is the point: two runs of a fixed slice
that behave identically produce a per-query difference of exactly zero, so what limits detection
is how small a real shift is worth reporting, not how much the measurement wobbles.

What remains open:

* extraction cells have no per-query scores, because extraction fidelity is not measured per
  query. They still gate on the old two-means comparison at 0.03.
* a baseline captured before per-query recording existed also falls back to that path. The
  fallback is deliberate - without it, adding the pairing would have quietly stopped the gate
  from checking anything - but a cell only gains the resolution once its baseline is recaptured.
* pairing detects that behaviour changed. It does not say the change is bad: a deliberate
  improvement to chunking will register as a regression on some cells and the baseline has to be
  updated with judgement, not reflexively.
* the slice still measures one corpus at 150 documents, so it speaks for nfcorpus and nothing
  else.

*To close it:* per-query recording for the extraction fixtures, if a sensible per-fixture unit
exists, and a second corpus in the gated slice.

## 4. Chunk quality is measured at span level now, on Wikipedia-style prose only

Span-level evaluation exists (see [Does the boundary cut the
answer?](03-chunking.md#does-the-boundary-cut-the-answer)), which is what this entry used to say
was missing. It changed one conclusion and qualified another.

Overlap does prevent the failure it exists for, at every chunk size and every overlap level
tested, and not one question in the grid got worse when overlap was added - while the same knob
moves end-to-end nDCG only for passage-length queries, and against short ones. The two results are
compatible: whether overlap helps find the document is a question about the queries, and it keeps
the answer intact once found regardless. The size guidance also gained a counter-pressure, since
larger chunks split fewer answers while smaller chunks retrieve better.

The measurement also refuted the sharper form of this entry's own claim. Composed with retrieval,
up to 51 percent of queries scored as document hits deliver no whole answer - but only about a
quarter of that is a boundary cutting the answer. The rest is an answer sitting intact in a chunk
the ranking did not retrieve, which is the embedder's business and not chunking's. Reporting the
total as chunking's blind spot would have been the easy and wrong story.

What is still narrow:

* **Four corpora, all Wikipedia-style prose, three languages.** Answer spans are short factoid
  strings. A corpus whose answers are long or procedural would split far more often, so these
  rates are a floor rather than a general figure.
* **One retrieval configuration** in the composed half: k=5 with `fastembed:bge-base`, and
  `recursive` only. Since most of the shortfall is ranking, that half says as much about the
  embedder as about chunking, and a stronger embedder would move it. It is also the one
  measurement on the chunking page that scores the way `semdex search` retrieves - top-k CHUNKS
  with no per-document dedup - and it now reaches 512 tokens: on the two German sets the
  delivered-answer rate rises 3.9 to 7.4 points from 256 to 512 while the document hit falls
  1.3 to 3.6, so the reader's floor and the ranking's floor are measured pulling apart at that
  step.
* **Containment is all-or-nothing.** An answer one character outside a chunk scores the same as
  one split down the middle, and a chunk holding the answer without the context needed to make
  sense of it counts as a success.

*To close it:* a corpus with long-form answers, the composed measurement repeated across the
embedder axis so the ranking component can be attributed rather than merely named, and a fixed
retrieved-token budget (twenty 64-token chunks against five 256-token ones is what a context
window sees) beside the fixed k.

## 5. The ANN frontier is measured on one cell only

The sweep this entry asked for exists (see
[Tuning](05-vector-store.md#tuning-what-each-store-can-reach-not-what-it-does-by-default)), and it
mattered more than expected. lancedb's default keeps 0.567 of the exact top-10; a refine factor
takes that to 0.998 and makes its nDCG@10 equal the exact scan to four decimals. The "approximation
costs up to 5.6 percent" figure this documentation published was the price of an untuned default,
not a property of ANN.

Two of this entry's own premises were also wrong. lancedb's recall barely responds to `nprobe`
(0.561 at 5 probes, 0.567 at 80) - the refine factor is the knob that matters. And `m` /
`ef_construction` could not have been swept when this was written, because no adapter read them:
they were declared, documented and resolved, then silently discarded. They had to be wired before
the sweep could say anything about them.

What is still narrow:

* **One cell**, 148,008 chunks at 768 dimensions on English long documents, at one query depth of
  200 chunk hits. Frontier shapes and the crossover between stores can all move with scale,
  dimension and depth.
* **mariadb's flat response is unexplained.** Recall sits at 0.677 to 0.684 across `ef_search`
  10 to 200 while latency rises from 63 ms to 76 ms. The session variable was verified set on the
  connection the queries run on, so this is not a plumbing fault in semdex, but the cause is not
  established.
* the composed quality-through-store table has since been re-measured at the shipped setting, so
  it and the frontier now describe the same configuration.

*To close it:* the same sweep at MSMARCO scale and at a second dimension, and a root cause for
mariadb.

## 6. Extraction fidelity is measured on real documents now, on scans only

All five backends are scored against OmniDocBench (see [Fidelity on real
documents](02-extraction.md#fidelity-on-real-documents)), which is what this entry used to say was
missing. It changed the ranking: `mineru` recovers 97.7 percent of the annotated text and `olmocr`
84.4, against 76.0 for `xberg` and 69.0 for `docling`, where the fixture grid had every supported
format at 100 percent.

It also produced the finding a fixture grid structurally cannot: `xberg` scores 0.86 on English
print and 0.10 on handwritten notes. And `markitdown` scores zero on every page, which the
text-layer control shows is the tool having no OCR rather than a broken harness.

What is still narrow:

* **Every page is a scan.** OmniDocBench is images by construction, so this measures OCR and
  layout reading. It says nothing about a PDF that already carries a text layer, which is the
  case the coverage grid covers and where `markitdown` is perfectly capable. The two halves
  answer different questions and neither replaces the other.
* **113 pages across ten document types**, so a per-type figure rests on about a dozen pages and
  `historical_document` on five.
* **One resolution for the vision backends** (150 DPI). Both would likely score higher at the 200
  semdex ships, which this 16 GB card could not sustain - the encoder allocates outside vLLM's
  reservation and ran out of spare VRAM mid-page.
* `mineru`'s number in the COVERAGE grid is still the one taken on a GPU host that the CPU matrix
  reports as skipped; only the real-document table above re-measured it.

*To close it:* a text-layer corpus with ground truth, so the non-OCR path gets the same treatment,
and a second resolution for the vision backends.

## 7. Chunker throughput is repeated and swept now; extractor and embedding rates are not

Chunking throughput carries a median over repeats, the spread between the extremes, the machine's
load beside it, and a worker sweep (see [What it costs to chunk](03-chunking.md#what-it-costs-to-chunk)).
That fixed a real defect rather than adding polish. Four of the six strategies had been timed
over passes shorter than 0.05 seconds and one rounded to 0.0, so a rate was divided out of a
duration the clock could not resolve. Repeating them properly moved all four, in proportion to
how short the original pass was: `recursive` was understated **fourfold** (13,749 -> 54,870
docs/s), `whitespace` roughly twofold. `semantic` and `late`, whose passes ran 0.53 s and 113 s,
agree within 7 percent across both methods and so serve as the control that the new harness is
not simply inflating everything.

It also answered the concurrency question for chunking, and the answer is negative: every
strategy got SLOWER with more worker threads. That is a deployment-relevant result which no
amount of single-pass timing could have produced.

What is still a single run on a shared box:

* **Extractor latency.** The per-format table in [Extraction](02-extraction.md#latency) is one
  document per cell, one pass, and says so. The orders of magnitude between backends are safe;
  the digits are not.
* **Embedding rates.** `gpu_docs_per_s_warm` is a single warm figure per model, with no repeat
  count and no spread.
* **Store latency under concurrent query load.** The ANN tables measure a quiet store answering
  one query at a time. What happens when a server answers fifty is unmeasured.
* **Processes, not threads.** The worker sweep says threads do not help. Whether a process pool
  scales linearly is the obvious follow-up and is not measured.
* **One machine throughout.** Every absolute number here belongs to one box. Ratios travel;
  digits do not.

*To close it:* the same repeat-and-spread treatment for the extractor and embedding timings, and
a concurrent-query load test against at least one ANN store.

## 8. qwen3's instruction prefix is measured now, and semdex could not send one

Both models are scored with and without the model card's instruction, paired over the same 323
queries (see [The instruction prefix qwen3 was never
given](04-embedding.md#the-instruction-prefix-qwen3-was-never-given)). It is worth +0.0459
nDCG@10 on the 4B and +0.0363 on the 8B, both resolved, which is 12.9 and 9.6 percent relative.

Measuring it turned up the larger problem. semdex could not apply an instruction at all through
the two providers that serve qwen3: `ollama` and `openai` both built a query as
`passages_fn([text])[0]`, so a query and a passage reached the server as byte-identical text. The
`EmbeddingProvider` port has separate query and passage methods and its docstring gives the
instruction asymmetry as the reason, and `sentence_transformers` even had `query_prefix` and
`passage_prefix` parameters, reachable from nothing. So this was not only an unmeasured caveat,
it was a capability the product did not have. `[embedding].query_prefix` and `passage_prefix`
exist now, plumbed to the three providers that can honour them and REFUSED by the ones that apply
their own, because a knob that is silently ignored is the defect this repo already shipped once.

Prefixing the passages as well was measured as a control rather than assumed away: it does not
resolve on the 4B and resolves NEGATIVE on the 8B. The card's "No need to add instruction for
retrieval documents" is therefore right, and usefully so - the gain is query-side, so switching it
on does not invalidate a single stored vector.

What is still open:

* **Every other qwen3 number on this page is still the plain configuration.** The embedder
  rankings, the German comparison and the dimension tables were all taken without the prefix, so
  they understate qwen3 by roughly this margin. Re-running them is a separate job.
* **One corpus, one language, one task description.** NFCorpus is English and medical; the
  instruction used is the card's generic web-search wording. A task description written for the
  corpus is the obvious next lever and is unmeasured.
* **E5 and BGE want their own prefixes** and semdex can now send them, but what they are worth
  here has not been measured.

*To close it:* re-score the qwen3 cells in the embedder ranking with the prefix on, so the
published comparison reflects the configuration the model is meant to run in.

## 9. Store and embedder memory is measured now; the server-backed stores are not

Resident memory is measured for the three embedded stores across a 1,000 to 100,000 chunk ladder
and for every in-process embedding provider (see [Memory: will it fit in the
container](05-vector-store.md#memory-will-it-fit-in-the-container)). The 8 GB question has an
answer: `sqlite_vec` is FLAT - 45.1 MB at a thousand chunks and 45.8 MB at a hundred thousand, so
its ceiling is disk rather than memory - while the embedded `json` store runs about 19 kB per
chunk and would fill 8 GB somewhere near 440,000 chunks.

Two things fell out of it that the disk table could not show. `lancedb` gets both LIGHTER and
FASTER crossing 100,000 rows (320 MB to 164 MB, 15.1 ms to 5.6 ms), because that is where
`lance_index_threshold` builds the IVF index and search stops being a flat scan. And building an
index costs several times what serving one does, which is the number that decides whether a
machine can create an index at all rather than merely answer from it.

The harness itself needed two properties to be true, and both were found by measuring rather than
by reasoning: peak RSS never falls, so every cell needs its own interpreter or each inherits the
high-water mark of the one before; and the fixture that FEEDS a store is bigger than the store, so
ingest and serving need separate processes too. A first version had both wrong and produced
ordered, plausible numbers that were mostly measuring the benchmark.

What is still missing:

* **`pgvector` and `mariadb` hold their vectors in a database process**, so the number that
  matters is the server's memory and not the client's. Measuring the client would report a small
  figure that means nothing. That is a different measurement and it is not done.
* **One machine, one allocator, one Python build.** Absolute megabytes belong to this box; the
  shape of each curve is what travels.
* **One access pattern, no concurrency.** These are high-water marks for a single querying loop.
* **Nothing about the MCP server process** as a whole, which holds a store, an embedder and the
  server framework at once.

*To close it:* the server-side memory of the two SQL backends under load, read from the database
process rather than the client, and a figure for a running `semdex serve`.

## 10. Energy is measured now; hardware, labour and wall power are not

Energy per document is measured for every provider that runs locally, and priced against the
hosted APIs' list prices (see [What it costs to run](07-selection.md#what-it-costs-to-run)). It is
a measurement rather than a model: the CPU package energy counter and the GPU's own power reading,
with idle windows INTERLEAVED between the work windows because this box's idle draw swings by
tens of watts and a baseline taken once absorbs whatever a neighbour was doing.

The headline is a one-to-two order of magnitude gap. Electricity runs 0.10 to 1.26 EUR per million
documents; the hosted APIs list 6.96 to 79.40 USD for the same million. Nothing about the tariff,
the exchange rate or the tokenizer spread changes that direction. The honest reading is narrower
than "self-hosting is cheaper": it compares electricity to an invoice, and says nothing about the
hardware or the hours.

Two results came out of the measurement discipline rather than the numbers. `placeholder` and
`model2vec` draw less than the machine's idle variation, so they report NO figure - at those rates
energy is not a differentiator, and a number quoted there would have been arithmetic on drift. And
the tokenizer caveat this entry used to carry is now quantified on the actual corpus: `gpt2` and
`bge-m3` differ by 14 percent on the same documents, which is why the energy table is denominated
in documents and the cloud column is a range.

What is still missing:

* **No hardware, amortisation or labour.** The marginal energy of the work against a list price.
  A total cost of ownership model is a different exercise and is not attempted.
* **Component power, not wall power.** The CPU figure is the package and the GPU figure is the
  card; neither includes RAM, disk, fans or PSU losses. The node hangs on a metered plug reporting
  214 W at the wall, so this is measurable and simply has not been done.
* **Marginal, not total.** Idle is 97 to 123 W for the CPU package and 44 to 51 W for the card on
  this box regardless. For a machine that exists only to run semdex, idle dominates every row;
  the raw file carries the total alongside.
* **One tariff, one corpus, one machine**, and list prices checked on a single date, with one
  vendor publishing no per-token price at all.

*To close it:* wall power from the node's metered plug during a run, so PSU and platform losses
are included, and an amortised hardware figure to sit beside the electricity.

## 11. The four metrics are one view, measured, and more broadly than this entry claimed

Measured across all 254 swept configurations, per corpus (see [How much the four columns actually
tell you](01-method.md#how-much-the-four-columns-actually-tell-you)). This entry used to argue
from the definitions that the four were closer to one and a half views on the long-document
bodies. The measurement says one, and says it everywhere: the effective view count runs 1.01 to
1.12 on every corpus with enough cells to carry a rank correlation, where four independent metrics
would give 4.00.

The mechanism this entry named is real and turned out to be exact. Both MLDR slices ship precisely
1.00 relevant documents per query, and both collapse completely - all six metric pairs above the
near-duplicate threshold. What the argument did not anticipate is that the collapse is not
CAUSED by that: nfcorpus averages 38.19 relevant documents per query, the opposite extreme, and
still returns 1.12 views. Judgment density explains the total collapse on MLDR and not the
near-collapse everywhere else.

The four do split into the two groups their definitions imply - `nDCG@10` with `Recall@10`, `MRR`
with `P@1` - and every disagreement that occurs is across those groups rather than within them.
The widest pair, `Recall@10` against `P@1`, still picks the same winner on 83.7 percent of
comparisons at its worst.

What is still open:

* **Two corpora are too thin to read.** cqadupstack has 11 swept cells and fiqa 4. cqadupstack
  returns the highest view count on the page (1.51) and is exactly the case where the two groups
  might genuinely separate, so it is the one worth re-sweeping rather than the ones that agree.
* **Correlation across CELLS, not queries.** This asks whether the metrics rank CONFIGURATIONS
  alike, which is the question a benchmark table answers. Whether they rank QUERIES alike within
  one configuration is a different question and is not measured.
* **The swept cells are not a random sample of configurations.** They are the profiles this repo
  chose to compare, clustered around similar settings, and metrics agree more readily on
  configurations that are close together.
* **Nothing here says the metrics are WRONG**, only that on this evidence three of them rarely
  change an answer the fourth already gave.

*To close it:* a wider sweep on cqadupstack, where the groups look most likely to separate, and a
per-query correlation to answer the other half of the question.

## 12. MLDR German has the smallest query set and the cheapest ties; cqadupstack is the underpowered one

This entry used to say the German results rest on small query sets, that several comparisons are
unresolved for that reason alone, and that no further computation changes it. The middle claim does
not survive being measured (see [A tie is two different
things](01-method.md#a-tie-is-two-different-things-and-they-need-opposite-responses)).

German does ship the fewest queries, 200, and it does have the widest intervals. But a query set is
adequate or not against the EFFECTS it has to see, and the German effects are the largest on the
page - 0.0352 median against cqadupstack's 0.0017, with fiqa's 0.2035 resting on one tie. Costing
every tie at `n x (half-width/effect)^2`
puts German's median at about 1,236 queries, roughly six times what it has and an ordinary dataset
size, while cqadupstack needs 11,279 despite already shipping 876. Of the six corpora here, German
ties are the second cheapest to settle and cqadupstack's are the most expensive by a factor of nine.

The control settles it. `mldr_en` runs the SAME 373 comparisons over the same profiles at four
times the queries, with intervals two and a half times tighter, and comes out slightly MORE
unresolved: 48.3 percent against 46.4. Its effects are three times smaller and that cancels the
extra data. Query count alone does not explain the German tie rate, because the corpus with four
times the queries does no better.

Seventeen German comparisons are also not underpowered at all: their effect is indistinguishable
from zero, which is an answer rather than a gap. Those configurations perform the same.

What is still true, and what is newly open:

* **The intervals are genuinely wide.** 200 queries gives a half-width around 0.059, so a real
  difference under about 0.12 cannot be separated on that corpus. That part of the entry stands.
* **The required counts are optimistic by about a quarter, measured.** A holdout on the English
  slice - subsample its 800 queries to 200, predict, then score against the full set - confirms
  the 1/sqrt(n) mechanism to within a few percent but shows the predicted COUNT of resolving ties
  running about 1.24x high. Worse, it identifies WHICH ties poorly: precision 0.47, recall 0.59.
  So the ladder is a rough budget, not a list of questions that get answered.
* **cqadupstack is now the entry that needs writing.** It has the most queries and the least to
  show for them, and 12 of its 18 comparisons are ties. Whether that is a corpus whose profiles
  genuinely do not differ, or a corpus badly matched to the axes being swept, is unmeasured.
* **A larger German corpus is in the sweep now, and it does not close this entry.** GerDaLIR
  carries 12,298 German queries, 61 times MLDR German's 200, and
  resolves 496 of its 632 overlap comparisons against MLDR German's 7 of 174. It is a DIFFERENT
  body of documents rather than more queries against the same one, so it cannot check the
  ladder's prediction for MLDR German's own ties, and it disagrees with MLDR German about
  overlap besides (see
  [Chunking](03-chunking.md)). "More German queries" and "more queries on this German corpus" are
  not the same request, and only the first has been met.

*To close it:* run MLDR German's OWN comparisons on a query set of about 1,500, which GerDaLIR
does not do. The ladder says 100 of the 173 ties separate at 2,000 queries and 73 at 1,000;
discounted by the 1.24x optimism measured on the English holdout, expect nearer 80 and 59.
Anything far below that would mean the German effects are inflated by the small slice rather than
real, which is the reading this entry still cannot rule out.

## 13. The overlap answer is about the queries, and so is the size answer

`whitespace`, `markdown`, `fast` and now `late` have cells on the long-document corpora, and the
overlap axis reaches the cap. Three things changed in the chunking page's conclusions on
2026-09-10, all of them corrections rather than extensions, and a fourth on 2026-09-27 turned
the last of them from a gap into a measurement.

**The overlap reversal has a mechanism now, and it is the query, not the corpus.** GerDaLIR's
queries are citing passages (median 104 words); MLDR's are questions (median 13). Binning the
stored per-query scores by query length turns GerDaLIR's answer into MLDR's on its own short
queries, and cutting every GerDaLIR query to its first 20 words and re-scoring the same cells
shrinks the gain for all six embedders and reverses it for three: +0.0312 becomes -0.0047 for
`potion-retrieval-32M` at 0 to 128 tokens, +0.0324 becomes -0.0120 for `potion-base-8M`, +0.0055
becomes -0.0176 for `bge-base`; the three multilingual `ollama` models keep about three tenths of
it or resolve nowhere (`bge-m3` +0.0136 becomes +0.0039, resolved). The "document structure"
hypothesis this entry carried is not needed. The recommendation is keyed on query shape, which a
deployer knows.

**One set of cells is withheld on an audit's judgement, and a second was.** The `whitespace`
cells on the German corpora are 72 to 93 percent clipped by the CPU embedders' 512-token windows;
the truncation census flags them and the export drops them with the reason recorded in the file.
The four cap512 overlap sets on MLDR had been re-cut by an earlier size-guard (chunk count up 16
to 22 percent, 10 to 16 percent crumb chunks) and supplied the seven largest "overlap hurts"
comparisons; the chunk-dimension audit flagged them and the export dropped them. They have since
been re-cut with the current chunker, re-embedded and re-scored: their chunk counts match the
overlap-0 sets, the audit passes them, and every table on the page carries them again.

**Chunk size has a measured floor now, and it splits the way overlap does.** The whitespace
ladder at 64, 96, 128 and 256 words had shown two static embedders gaining down to the smallest
rung on MLDR, which made "whitespace loses everywhere" the size verdict and 256 tokens where the
recursive sweep stopped rather than an optimum. `recursive` at 64 and 128 tokens on MLDR English
and GerDaLIR for all six embedders (2,137,255 chunks at 64 on GerDaLIR) settles it: on MLDR's
questions 128 beats 256 for every embedder and 64 beats 128, resolved, for three; on GerDaLIR's
passage queries every one of the 18 steps favours the larger chunk, 64 to 128 costing 0.065 to
0.099 nDCG@10 and 128 to 256 a further 0.021 to 0.059. "Smaller wins" was MLDR's answer, and the
largest resolved chunking effect anywhere is now the other way, 0.158 for `potion-base-8M` on
GerDaLIR from 64 up to 256.

What is still open:

* **The size axis has one open end per corpus.** GerDaLIR has no rung above 256 tokens, so where
  the passage-query gain stops is unmeasured; MLDR German has `recursive` at 256 against 512 only.
  The delivered-answer rate (gap 4) falls as chunks shrink, so the floor for a reader of the
  chunks is a different number from the floor for document retrieval.
* **The size verdict and the overlap ladder are one knob at two strides.** cap256 with 256 tokens
  of overlap is a 512-token window at stride 256; cap512 is the same window at stride 512. No
  corpus has both cells, so "dilution" is asserted, not shown. cap512-o0 on GerDaLIR and
  cap256-o256 on MLDR English (static models, minutes) separate them.
* **The truncated-query test has one cut, and it splits the embedders.** All six now carry the
  20-word re-score: three reverse, the three multilingual `ollama` models keep about three
  tenths of their gain or resolve nowhere. One truncation length cannot say where between 20 and
  104 words the gain returns, nor whether the split is the models' training language or their
  window; a second cut (say 50 words) and an English long-query corpus would.
* **The truncation audit has no query arm.** About a fifth of GerDaLIR's queries (a one-off
  count, not yet audited) are longer than `bge-base`'s 512-token window, so the default embedder
  scored them from a prefix; the audit measures chunks only.
* **GerDaLIR is swept at one breakpoint model.** It contributes nothing to that axis, so the
  breakpoint conclusion rests entirely on MLDR, whose queries are short.
* **The breakpoint-model ranking is confounded with granularity.** `bge-m3` cuts 428,748 chunks
  and wins 17-0; `e5-large` cuts 278,215 and wins twice. Nothing holds chunk count fixed while
  varying the model.
* **Markdown structure is measured once, and shallowly.** A twin of MLDR English with its section
  headings marked `## ` isolates what the markup does (on the chunking page it cost `markdown`,
  resolved for the two qwen3 models), but it carries one heading level, recovered by a
  heuristic, on Wikipedia text with short sections and short questions, and it tests one
  structure-aware strategy, not two: every `recursive` cell was chunked with chonkie's generic
  rules (`recipe = ""`), not the `recipe = "markdown"` semdex ships. Nested heading levels are
  not measured, and neither is a markdown body of another domain, such as statutes or software
  documentation, where sections run long and headings carry the words a query uses.
* **The within-twin strategy comparison is not published.** Which strategy wins once the
  structure is there was a goal of the twin, but the twin is a rendering of MLDR English, not a
  corpus of its own, so its strategy comparisons stay out of every count; and within it,
  `markdown` against `recursive` is structure against generic rules, not against the markdown
  recipe.
* **Every chunk verdict is under dense retrieval.** The hybrid method the retrieval page
  recommends has one chunk profile; BM25's length normalisation reacts to chunk size.
* **`late` has two embedders**, and its chunk-count metadata in the cached sets is wrong on most
  German chunks (chonkie reports a token count of 1 on long documents; the adapter now recounts).
  Its truncation was measured at 0.02 percent of token mass, so the four scores stand.
* **Overlap is meaningless on three of the six.** `whitespace` takes no overlap parameter;
  `semantic` and `late` accept one and ignore it. That is verified from the chunk boundaries
  rather than read off the adapter source.

*To close it:* sweep `recursive` at 512 on GerDaLIR and at 128 and 64 on MLDR German with all
six embedders; run the two stride cells; add a second truncation length to the query-length test; add
a query arm to the truncation audit; run `recursive` with the markdown recipe on both twins;
measure heading levels and a non-Wikipedia markdown body.

## 14. No multi-vector or late-interaction retrieval

ColBERT-style token-level matching changes the chunking question fundamentally, since boundaries
matter far less when matching happens per token. No adapter exists, so the entire chunking analysis
is scoped to single-vector retrieval.

## 15. Only bulk loading is measured

Every store measurement loads N vectors once and then queries. Nothing measures incremental
ingest, deletion, or index rebuild after churn, which is usually the dominant operational cost of
a live index.

## 16. Determinism is not accuracy

Given cached vectors and cached query vectors, re-running the scorer returns identical numbers, so
"run-to-run variance" does not apply to the quality figures; the uncertainty that matters is
query-sampling uncertainty, which the intervals report. The corollary is worth stating: a
systematic error in the cached vectors would reproduce perfectly forever. That is why the chunk
sets are audited against the text they came from rather than trusted.
