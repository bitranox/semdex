# Selection

Putting the four choices together. The pages before this measure one axis at a time; this one is
about which axis to spend effort on, and what a whole stack looks like for a given situation.

## Spend your effort in this order

The paired comparisons make the ordering unusually clear, because the same test was applied to
every axis. The material column is the one to read: it removes resolutions that the query set
can see and no deployer would act on.

<!-- BEGIN GENERATED chunk_knob_summary (scripts/gen_bench_tables.py) -->
| Axis               | Paired comparisons | Resolved at 95% | Up to N by chance | Resolved and material (>= 0.005 ndcg@10) | Which way                                                                                                                                                                                                                                                                  |
|--------------------|--------------------|-----------------|-------------------|------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `breakpoint_model` | 205                | 43 (21%)        | 11                | 43                                       | 17 favour bge-m3, 16 favour jina-v3, 7 favour qwen3-0.6b, 2 favour e5-large, 1 favour default                                                                                                                                                                              |
| `embedding`        | 1064               | 903 (85%)       | 54                | 897                                      | 274 favour ollama:qwen3-embedding-8b, 240 favour ollama:qwen3-embedding-4b, 145 favour ollama:bge-m3, 100 favour model2vec:potion-retrieval-32M, 90 favour fastembed:bge-base, 27 favour fastembed:bge-small, 18 favour model2vec:potion-base-8M, 3 favour openai:e5-large |
| `max_tokens`       | 152                | 109 (72%)       | 8                 | 106                                      | 6 favour more, 54 favour less, 39 unresolved or immaterial (of 99 ladders)                                                                                                                                                                                                 |
| `method`           | 154                | 123 (80%)       | 8                 | 123                                      | 26 favour hybrid, 25 favour bm25, 20 favour hybrid@20, 14 favour bm25@20, 12 favour bm25@20+rerank, 10 favour dense, 10 favour dense@20, 6 favour hybrid@20+rerank                                                                                                         |
| `overlap_tokens`   | 980                | 530 (54%)       | 49                | 394                                      | 9 favour more, 5 favour less, 54 unresolved or immaterial (of 68 ladders)                                                                                                                                                                                                  |
| `strategy`         | 313                | 114 (36%)       | 16                | 109                                      | 49 favour recursive, 24 favour markdown, 21 favour fast, 11 favour semantic, 4 favour late                                                                                                                                                                                 |


Each comparison pairs two cells differing in exactly one axis and bootstraps the per-query difference over the queries they share. Unresolved means the interval spans zero: the query set cannot separate those two configurations. Up to N by chance is comparisons times alpha 0.05, rounded up: the resolutions that many true nulls would produce on average, a ceiling on how much of the resolved count is noise. Material means resolved AND an absolute difference of at least 0.005 ndcg@10; a 12,298-query corpus resolves steps far below that. For a numeric axis the direction is judged once per ladder (corpus, embedder, held-fixed set) by its top against bottom pair, so the counts are ladders; where the optimum sits within a ladder is the ceiling table's question. For a categorical axis the material comparisons are tallied per level. Restricted to corpora long enough to carry a chunking claim.
<!-- END GENERATED chunk_knob_summary -->

1. **Add lexical retrieval.** Fusing BM25 with the dense results is the largest effect measured
   anywhere here: of the ten dense-to-hybrid comparisons, nine resolve and every resolved one is
   positive, up to +0.249 nDCG@10 against roughly 0.16 for the best chunking knob. Those ten are
   one level pair inside the 154 on the `method` row above, which crosses all nine retrieval
   levels pairwise and so counts shortlist depth and reranking as well (a single ranker read at
   two shortlist depths is one list, and is not counted twice). It needs no GPU and no
   re-embedding. Read [Retrieval method](06-retrieval-method.md) first, because it also shows when
   fusion LOSES.
2. **The embedding model.** It resolves in the large majority of comparisons, and its effects
   dwarf the chunking knobs': the largest resolved one is 0.51 nDCG@10, about three times the 0.16
   of the best chunking knob. It is also the most expensive decision to revisit, because changing
   it means re-embedding everything.
3. **Chunk size.** The chunking knob that resolves most often, and it follows the queries. On
   short questions smaller wins: 128 tokens over 256 for every embedder, and 64 over 128 for
   three of six. On passage-length queries larger wins: every step from 64 up to 256, for every
   embedder, by up to 0.158 nDCG@10. A reader of the chunks pays for small chunks in delivered
   answers, so pick the size on the length of your queries and on the metric your consumer sees.
4. **The extractor**, if your documents are not already text. A backend that cannot read your
   format, or that scrambles the reading order of a scan, sets a ceiling nothing downstream can
   lift. This does not appear in the table above because it is not a retrieval-quality axis; it
   is a can-you-read-the-document axis.
5. **The store**, once the corpus is large enough for search latency to be a problem. Below that,
   it changes nothing about quality.
6. **Chunking strategy**, which matters less than its realized chunk size. `recursive` is
   undefeated on German legal text; on English prose at the same nominal cap `semantic` mostly ties
   it, beating it for one static model and losing to both qwen3 models, and it wins clearly only
   where its 46-token fragments face 461-token chunks; `whitespace` lost only
   at 256 words, which hold 1.5 to 3 times the text of a 256-token chunk, and ties `recursive` at
   a matched size. Marking the section headings of MLDR English as markdown does not promote the
   splitter named for them: it cost `markdown` up to 0.018 nDCG@10, resolved for the two qwen3
   models. `recursive` was measured there only on chonkie's generic rules, not on the markdown
   recipe semdex ships, so this says nothing about the default.
7. **Overlap**, last here only because it has no default that travels across query shapes. Short
   questions: 0, it costs the default embedder a resolved 0.02 nDCG@10. Passage-length queries:
   25 to 30 percent of the cap, worth +0.005 (`bge-base`) to +0.023 (static models) on GerDaLIR.
   The chunking page measured this by cutting GerDaLIR's queries to 20 words and watching the
   gain reverse for three embedders and shrink to a sliver for the other three, so it is the
   queries that decide, not the documents or the language.

Overlap used to head this page's list of ways to waste effort, on the grounds that it never
measurably helped. That held on every corpus measured at the time and broke on the next one
added, and the reason turned out to be the next corpus's queries, which are paragraphs where the
earlier ones were sentences. If your users ask questions, leave it at 0 and spend nothing
deciding. If they paste a paragraph or ask for documents like this one, sweeping it is the
cheapest resolved gain on this page: no model change, no re-chunking of anything else, and
[Chunking](03-chunking.md) carries the ladder.

## Stacks

Each of these is a starting point, not a verdict. The settings are what the measurements support;
the notes say what would change them.

### Local, English, ordinary documents, no GPU

```toml
[extractor]
backend = "text"          # or "markitdown" via Docker for Office and HTML

[chunker]
strategy = "recursive"

[index]
max_tokens = 256
chunk_overlap = 0        # right for question queries; 64 to 77 for passage-length queries

[embedding]
provider = "fastembed"
model = "bge-base"        # 768-dim; bge-small if memory is tight

[vector_store]
backend = "sqlite_vec"    # exact; switch to lancedb only when latency becomes a problem
```

### Multilingual or non-English

Same shape, with two changes. Use a multilingual embedding model rather than a bge English model:
`qwen3-embedding:8b` leads both languages measured, `qwen3-embedding:4b` is close behind at a
smaller dimension, and `bge-m3` at 1024 is the cheapest of the three to store and search. Do not
pick on an English ranking alone - `e5-large` all but ties the leader on English and comes last on
German. See [embedding](04-embedding.md).
If you use the semantic chunking strategy, set `[chunker].semantic_model` explicitly: chonkie's
default breakpoint model is English-distilled, so on other languages it is choosing boundaries in
text it cannot read.
Overlap is not decided by the language either way. The two German corpora measured here disagree
with each other about it and MLDR German agrees with MLDR English; what separates them is the
length of their queries, so set it from your query shape rather than from the language.

### Scanned documents

`[extractor].backend = "xberg"` for CPU OCR, or a vision-LLM OCR engine if you have a GPU. Expect
to lose content either way: the best engine measured passes 44 percent of the content tests on
degraded scans, and reading order is where the classic engines fall furthest behind. Everything
downstream inherits that loss, so this is where the GPU budget goes first if you have scans.

### Large corpus, latency-sensitive

Keep the ingest settings above and change the store to `lancedb` (or `pgvector` if the data belongs
in Postgres, subject to its 2000-dimension index cap). At the settings semdex ships, the index
keeps 0.96 to 0.99 of the exact top ten and costs between 0 and 1.1 percent of nDCG@10 for a
6.5 to 12 times speedup - see the [vector store](05-vector-store.md) page. Above about 100,000
chunks the exact scan is no longer an interactive latency, so this is usually the right trade.

**Do not leave lancedb at its own driver default.** Its recall barely responds to `nprobes`; what
it needs is a refine factor, which is what `ann_recall = "balanced"` now sets. Selecting `fast`
turns that off and returns to keeping barely half the exact top ten.

### Fastest possible ingest, quality secondary

`model2vec` embeddings are roughly a thousand times faster than fastembed on CPU, 12,065 against
12.3 documents per second in the energy measurement below, and `potion-retrieval-32M` is the
better of the two model2vec models by a wide and consistently resolved margin. It still loses to
the fastembed models on quality; the table on the [embedding](04-embedding.md) page shows by how
much, so the trade can be made deliberately.

## Where a GPU actually helps

| Stage      | Does a GPU help                          | Notes                                                          |
|------------|------------------------------------------|----------------------------------------------------------------|
| Extraction | Only for OCR                             | Vision-LLM OCR needs one; ordinary format conversion does not. |
| Chunking   | Only for the embedding-driven strategies | `recursive` is CPU-bound string work.                          |
| Embedding  | Yes, substantially                       | This is where the time goes on a large ingest.                 |
| Search     | No                                       | Every store here searches on CPU.                              |

If you have one GPU and a large corpus, it belongs on embedding. If you have scans, it belongs on
OCR first, because content lost at extraction cannot be recovered later.

## What it costs to run

Every other page here ranks components by what they retrieve or how fast. None of them attached a
price, because a locally served model has no per-token bill. It does draw power, though, and that
is measurable on this hardware rather than modelled: the CPU package exposes an energy counter and
the GPU reports its own draw.

<!-- BEGIN GENERATED energy_per_document (scripts/gen_bench_tables.py) -->
| Provider              | Model                                                         | Runs on | J/doc             | Docs/s   | EUR per 1M docs | Signal/noise     |
|-----------------------|---------------------------------------------------------------|---------|-------------------|----------|-----------------|------------------|
| placeholder           | `in-memory`                                                   | CPU     | below noise floor | 8,749.7  | -               | 1.1 (unresolved) |
| model2vec             | `minishlab/potion-base-8M`                                    | CPU     | below noise floor | 12,064.7 | -               | 1.4 (unresolved) |
| ollama                | `bge-m3:latest`                                               | GPU     | 1.2457            | 36.6     | 0.1038          | 45.0             |
| sentence_transformers | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | CPU     | 1.5026            | 47.8     | 0.1252          | 4.6              |
| fastembed             | `BAAI/bge-small-en-v1.5`                                      | CPU     | 5.3369            | 12.3     | 0.4447          | 2.1              |
| ollama                | `qwen3-embedding:4b`                                          | GPU     | 9.0936            | 13.8     | 0.7578          | 27.6             |
| ollama                | `qwen3-embedding:8b`                                          | GPU     | 15.0693           | 9.9      | 1.2558          | 21.6             |


MARGINAL energy: what the work adds over the same machine sitting idle, with idle windows INTERLEAVED between the work windows rather than measured once beforehand. CPU figures come from the package RAPL counter and GPU figures from integrating card power, so neither includes the rest of the machine. `Signal/noise` is that marginal draw against the idle swing it had to be pulled out of; below 2.0 the row is arithmetic rather than a measurement and reports no figure. Electricity at 0.3 EUR/kWh. Denominated in DOCUMENTS, never tokens - see the note below the next table.
<!-- END GENERATED energy_per_document -->

**Two providers report no figure, and that is the result.** `placeholder` and `model2vec` add less
power than this shared machine's idle draw varies by, so their energy cost cannot be separated from
the neighbours. The useful reading is not a number but its absence: at these rates energy is not a
reason to choose between them, and anything that turns on a fraction of a joule per document is
below what this measurement can see.

**The best model on the page is the most expensive to run, by about 12x.** `qwen3-embedding:8b`
costs 15.07 J per document against `bge-m3`'s 1.25 on the same card. That is the price of the
quality it wins elsewhere in these benchmarks, and it is worth knowing before an ingest of any size.

**A GPU is not automatically the efficient choice.** `bge-m3` on the card is the cheapest row here,
but `sentence_transformers` on the CPU is within 21 percent of it per document and needs no GPU at
all. The card wins decisively on throughput, not on energy per unit of work.

### Against the hosted APIs

<!-- BEGIN GENERATED cloud_cost_per_million (scripts/gen_bench_tables.py) -->
| Provider | Model                    | USD per 1M tokens | USD per 1M documents |
|----------|--------------------------|-------------------|----------------------|
| openai   | `text-embedding-3-small` | 0.02              | 6.96 to 7.94         |
| openai   | `text-embedding-3-large` | 0.13              | 45.24 to 51.61       |
| gemini   | `gemini-embedding-001`   | 0.15              | 52.20 to 59.55       |
| gemini   | `gemini-embedding-2`     | 0.20              | 69.60 to 79.40       |
| cohere   | `embed-v4.0`             | not published     | -                    |


List prices checked 2026-08-12, each recorded with its source in the raw file. The document column is a RANGE because it needs a tokenizer and the answer depends on which: `gpt2` counts 348 tokens for the mean document on this corpus and `BAAI/bge-m3` counts 397, a 14 percent spread, and a vendor bills on its own tokenizer which is neither of these. That spread is why the energy table above is denominated in documents rather than tokens. Local figures are in EUR and these in USD; the gap between them is large enough that no exchange rate flips it.
<!-- END GENERATED cloud_cost_per_million -->

**Electricity is cheaper than the list prices by a factor between about 5.5 and roughly 670**:
5.5x for the cheapest API against the most expensive local option, and 670x for the most expensive
API against the cheapest local one, both at the `gpt2` token count. The comparison is not close
enough for the exchange rate, the tariff, or the tokenizer spread to change its direction.

That is a narrower claim than "self-hosting is cheaper", and the difference matters. It compares
electricity against an invoice. It says nothing about the card, the machine, the rack, or the hours
spent keeping any of it running, and those are exactly the costs a hosted API exists to remove.

### What this does not establish

* **No hardware cost, no amortisation, no labour.** This is the marginal energy of the work and
  the vendors' list price. A capital and operational model is a different exercise and is not
  attempted here.
* **Marginal, not total.** These figures exclude the idle draw, which on this box is 97 to 123 W
  for the CPU package and 44 to 51 W for the card whether or not anything is embedding. For a box
  that exists only to run semdex, the idle draw dominates every figure in the table; the raw file
  carries that total alongside.
* **Component power only.** The CPU figure is the package, the GPU figure is the card. Neither
  includes RAM, disk, fans, PSU losses, or the network. A wall measurement would be higher, and
  the node's metered plug makes that measurable later.
* **One machine, one corpus, one tariff.** English medical abstracts averaging 1,639 bytes on one
  i9 and one RTX 4070 Ti SUPER, at a stated electricity price.
* **List prices, not negotiated ones**, checked on one date, and one vendor publishes no per-token
  price at all.

## Before you trust any of this

Every component comparison above was measured with dense-only retrieval. Adding a lexical stage
changes the totals, which is why it is now the first item on the list; the component ORDERING
should survive it, but that is an assumption, not a measurement.

Do not add a cross-encoder reranker on faith. Of the twelve comparisons measured here, eight
resolve, and every one of those eight is worse. The only baseline it improved was the weakest one
measured, and that single improvement is the one case that does not resolve - see
[Retrieval method](06-retrieval-method.md), which also shows the reranker itself is working
correctly. Its cost is real and its benefit is conditional on a weak first stage.

The recommendations are sound relative to each other; they are not a claim that this is the best a
retrieval system can do. See [Gaps](08-gaps.md).
