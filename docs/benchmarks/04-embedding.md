# Embedding

Which model turns chunks into vectors. This is the choice that matters most, and it is not close.

## The model matters more than the chunking

The same paired machinery that measures the chunking knobs measures the model: two cells over the
same corpus and the same chunk profile, differing only in which model embedded them.

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

The embedding row resolves in four comparisons out of five, more often than any chunking axis, and
switching model changes retrieval measurably almost every time. If you have one change to spend,
spend it here.

Its material count sits within 6 of its resolved count, so almost every model switch that the
query set can see is also large enough to act on; the overlap axis is where the floor bites.

The gap is in the effect sizes rather than the resolve rates. The chunking axes resolve far more
often than they used to, because a corpus carrying 12,298 queries joined the sweep and a query set
that size settles arguments a 200-query one cannot. They have not caught up: the largest resolved
chunking effect is 0.158 nDCG@10 against 0.508 for the largest resolved model effect, and two
chunking axes, size and overlap, now point in opposite directions on different corpora. See
[Chunking](03-chunking.md).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/chunk_knob_embedding-dark.png" />
  <img src="img/chunk_knob_embedding.png" alt="Forest plot of the paired nDCG@10 difference between embedding models over the same corpus and chunk profile. Most comparisons resolve, and the largest effect is about three times the largest any chunking knob produces." width="820" />
</picture>

<!-- BEGIN GENERATED embedder_effects (scripts/gen_bench_tables.py) -->
| Corpus                | Dim  | Held fixed              | Change (low to high)                                        | Delta nDCG@10 | 95% CI             | Win/loss  | Verdict      |
|-----------------------|------|-------------------------|-------------------------------------------------------------|---------------|--------------------|-----------|--------------|
| gerdalir_de_12k_slice | 1024 | recursive cap256 ov0tok | ollama:bge-m3 to ollama:qwen3-embedding-4b                  | +0.0285       | [+0.0239, +0.0331] | 2754/2042 | resolved     |
| gerdalir_de_12k_slice | 1024 | recursive cap256 ov0tok | ollama:bge-m3 to ollama:qwen3-embedding-8b                  | +0.0352       | [+0.0305, +0.0397] | 2833/2016 | resolved     |
| gerdalir_de_12k_slice | 256  | recursive cap256 ov0tok | model2vec:potion-base-8M to model2vec:potion-retrieval-32M  | +0.0737       | [+0.0693, +0.0782] | 2381/707  | resolved     |
| gerdalir_de_12k_slice | 256  | recursive cap256 ov0tok | model2vec:potion-base-8M to ollama:bge-m3                   | +0.2202       | [+0.2132, +0.2273] | 5033/986  | resolved     |
| gerdalir_de_12k_slice | 256  | recursive cap256 ov0tok | model2vec:potion-base-8M to ollama:qwen3-embedding-4b       | +0.2487       | [+0.2413, +0.2560] | 5557/993  | resolved     |
| gerdalir_de_12k_slice | 256  | recursive cap256 ov0tok | model2vec:potion-base-8M to ollama:qwen3-embedding-8b       | +0.2554       | [+0.2480, +0.2628] | 5641/1041 | resolved     |
| gerdalir_de_12k_slice | 2560 | recursive cap256 ov0tok | ollama:qwen3-embedding-4b to ollama:qwen3-embedding-8b      | +0.0067       | [+0.0034, +0.0100] | 2124/1868 | resolved     |
| gerdalir_de_12k_slice | 512  | recursive cap256 ov0tok | model2vec:potion-retrieval-32M to ollama:bge-m3             | +0.1465       | [+0.1400, +0.1531] | 4188/1470 | resolved     |
| gerdalir_de_12k_slice | 512  | recursive cap256 ov0tok | model2vec:potion-retrieval-32M to ollama:qwen3-embedding-4b | +0.1750       | [+0.1680, +0.1819] | 4711/1443 | resolved     |
| gerdalir_de_12k_slice | 512  | recursive cap256 ov0tok | model2vec:potion-retrieval-32M to ollama:qwen3-embedding-8b | +0.1817       | [+0.1747, +0.1887] | 4800/1479 | resolved     |
| gerdalir_de_12k_slice | 768  | recursive cap256 ov0tok | fastembed:bge-base to model2vec:potion-base-8M              | +0.0159       | [+0.0095, +0.0222] | 2298/2028 | resolved     |
| gerdalir_de_12k_slice | 768  | recursive cap256 ov0tok | fastembed:bge-base to model2vec:potion-retrieval-32M        | +0.0896       | [+0.0831, +0.0963] | 3231/1576 | resolved     |
| gerdalir_de_12k_slice | 768  | recursive cap256 ov0tok | fastembed:bge-base to ollama:bge-m3                         | +0.2361       | [+0.2294, +0.2429] | 5433/855  | resolved     |
| gerdalir_de_12k_slice | 768  | recursive cap256 ov0tok | fastembed:bge-base to ollama:qwen3-embedding-4b             | +0.2646       | [+0.2576, +0.2716] | 5875/802  | resolved     |
| gerdalir_de_12k_slice | 768  | recursive cap256 ov0tok | fastembed:bge-base to ollama:qwen3-embedding-8b             | +0.2713       | [+0.2644, +0.2783] | 5965/831  | resolved     |
| mldr_de_3k_slice      | 1024 | recursive cap256 ov0tok | ollama:bge-m3 to ollama:qwen3-embedding-4b                  | +0.0362       | [+0.0033, +0.0708] | 29/21     | resolved     |
| mldr_de_3k_slice      | 1024 | recursive cap256 ov0tok | ollama:bge-m3 to ollama:qwen3-embedding-8b                  | +0.0365       | [+0.0039, +0.0710] | 29/18     | resolved     |
| mldr_de_3k_slice      | 256  | recursive cap256 ov0tok | model2vec:potion-base-8M to model2vec:potion-retrieval-32M  | +0.1057       | [+0.0606, +0.1514] | 43/13     | resolved     |
| mldr_de_3k_slice      | 256  | recursive cap256 ov0tok | model2vec:potion-base-8M to ollama:bge-m3                   | +0.4561       | [+0.3964, +0.5160] | 120/3     | resolved     |
| mldr_de_3k_slice      | 256  | recursive cap256 ov0tok | model2vec:potion-base-8M to ollama:qwen3-embedding-4b       | +0.4923       | [+0.4299, +0.5543] | 125/1     | resolved     |
| mldr_de_3k_slice      | 256  | recursive cap256 ov0tok | model2vec:potion-base-8M to ollama:qwen3-embedding-8b       | +0.4927       | [+0.4318, +0.5548] | 125/1     | resolved     |
| mldr_de_3k_slice      | 2560 | recursive cap256 ov0tok | ollama:qwen3-embedding-4b to ollama:qwen3-embedding-8b      | +0.0003       | [-0.0239, +0.0256] | 17/19     | not resolved |
| mldr_de_3k_slice      | 384  | recursive cap256 ov0tok | fastembed:bge-small to model2vec:potion-base-8M             | -0.2580       | [-0.3220, -0.1940] | 14/83     | resolved     |
| mldr_de_3k_slice      | 384  | recursive cap256 ov0tok | fastembed:bge-small to model2vec:potion-retrieval-32M       | -0.1523       | [-0.2169, -0.0900] | 25/65     | resolved     |
| mldr_de_3k_slice      | 384  | recursive cap256 ov0tok | fastembed:bge-small to ollama:bge-m3                        | +0.1981       | [+0.1492, +0.2485] | 72/9      | resolved     |
| mldr_de_3k_slice      | 384  | recursive cap256 ov0tok | fastembed:bge-small to ollama:qwen3-embedding-4b            | +0.2343       | [+0.1810, +0.2902] | 74/8      | resolved     |
| mldr_de_3k_slice      | 384  | recursive cap256 ov0tok | fastembed:bge-small to ollama:qwen3-embedding-8b            | +0.2347       | [+0.1813, +0.2888] | 74/8      | resolved     |
| mldr_de_3k_slice      | 512  | recursive cap256 ov0tok | model2vec:potion-retrieval-32M to ollama:bge-m3             | +0.3504       | [+0.2927, +0.4096] | 100/5     | resolved     |
| mldr_de_3k_slice      | 512  | recursive cap256 ov0tok | model2vec:potion-retrieval-32M to ollama:qwen3-embedding-4b | +0.3866       | [+0.3261, +0.4506] | 103/4     | resolved     |
| mldr_de_3k_slice      | 512  | recursive cap256 ov0tok | model2vec:potion-retrieval-32M to ollama:qwen3-embedding-8b | +0.3869       | [+0.3277, +0.4490] | 103/4     | resolved     |
| mldr_de_3k_slice      | 768  | recursive cap256 ov0tok | fastembed:bge-base to fastembed:bge-small                   | +0.0352       | [-0.0021, +0.0736] | 35/22     | not resolved |
| mldr_de_3k_slice      | 768  | recursive cap256 ov0tok | fastembed:bge-base to model2vec:potion-base-8M              | -0.2228       | [-0.2856, -0.1604] | 15/76     | resolved     |
| mldr_de_3k_slice      | 768  | recursive cap256 ov0tok | fastembed:bge-base to model2vec:potion-retrieval-32M        | -0.1171       | [-0.1851, -0.0500] | 31/58     | resolved     |
| mldr_de_3k_slice      | 768  | recursive cap256 ov0tok | fastembed:bge-base to ollama:bge-m3                         | +0.2333       | [+0.1791, +0.2896] | 75/10     | resolved     |
| mldr_de_3k_slice      | 768  | recursive cap256 ov0tok | fastembed:bge-base to ollama:qwen3-embedding-4b             | +0.2695       | [+0.2127, +0.3288] | 78/8      | resolved     |
| mldr_de_3k_slice      | 768  | recursive cap256 ov0tok | fastembed:bge-base to ollama:qwen3-embedding-8b             | +0.2698       | [+0.2133, +0.3286] | 80/7      | resolved     |
| mldr_en_8k_slice      | 1024 | recursive cap256 ov0tok | ollama:bge-m3 to ollama:qwen3-embedding-4b                  | +0.0122       | [-0.0015, +0.0263] | 74/57     | not resolved |
| mldr_en_8k_slice      | 1024 | recursive cap256 ov0tok | ollama:bge-m3 to ollama:qwen3-embedding-8b                  | +0.0199       | [+0.0066, +0.0336] | 75/51     | resolved     |
| mldr_en_8k_slice      | 256  | recursive cap256 ov0tok | model2vec:potion-base-8M to model2vec:potion-retrieval-32M  | +0.0825       | [+0.0653, +0.0999] | 176/46    | resolved     |
| mldr_en_8k_slice      | 256  | recursive cap256 ov0tok | model2vec:potion-base-8M to ollama:bge-m3                   | +0.1630       | [+0.1385, +0.1881] | 229/41    | resolved     |
| mldr_en_8k_slice      | 256  | recursive cap256 ov0tok | model2vec:potion-base-8M to ollama:qwen3-embedding-4b       | +0.1752       | [+0.1524, +0.1988] | 240/24    | resolved     |
| mldr_en_8k_slice      | 256  | recursive cap256 ov0tok | model2vec:potion-base-8M to ollama:qwen3-embedding-8b       | +0.1829       | [+0.1588, +0.2078] | 244/29    | resolved     |
| mldr_en_8k_slice      | 2560 | recursive cap256 ov0tok | ollama:qwen3-embedding-4b to ollama:qwen3-embedding-8b      | +0.0077       | [-0.0019, +0.0176] | 52/46     | not resolved |
| mldr_en_8k_slice      | 384  | recursive cap256 ov0tok | fastembed:bge-small to model2vec:potion-base-8M             | -0.1097       | [-0.1324, -0.0870] | 57/205    | resolved     |
| mldr_en_8k_slice      | 384  | recursive cap256 ov0tok | fastembed:bge-small to model2vec:potion-retrieval-32M       | -0.0272       | [-0.0456, -0.0084] | 83/140    | resolved     |
| mldr_en_8k_slice      | 384  | recursive cap256 ov0tok | fastembed:bge-small to ollama:bge-m3                        | +0.0533       | [+0.0373, +0.0701] | 119/48    | resolved     |
| mldr_en_8k_slice      | 384  | recursive cap256 ov0tok | fastembed:bge-small to ollama:qwen3-embedding-4b            | +0.0655       | [+0.0486, +0.0830] | 133/43    | resolved     |
| mldr_en_8k_slice      | 384  | recursive cap256 ov0tok | fastembed:bge-small to ollama:qwen3-embedding-8b            | +0.0732       | [+0.0560, +0.0904] | 141/35    | resolved     |
| mldr_en_8k_slice      | 512  | recursive cap256 ov0tok | model2vec:potion-retrieval-32M to ollama:bge-m3             | +0.0805       | [+0.0604, +0.1006] | 163/53    | resolved     |
| mldr_en_8k_slice      | 512  | recursive cap256 ov0tok | model2vec:potion-retrieval-32M to ollama:qwen3-embedding-4b | +0.0928       | [+0.0743, +0.1121] | 170/44    | resolved     |
| mldr_en_8k_slice      | 512  | recursive cap256 ov0tok | model2vec:potion-retrieval-32M to ollama:qwen3-embedding-8b | +0.1004       | [+0.0804, +0.1212] | 177/45    | resolved     |
| mldr_en_8k_slice      | 768  | recursive cap256 ov0tok | fastembed:bge-base to fastembed:bge-small                   | -0.0254       | [-0.0408, -0.0105] | 68/103    | resolved     |
| mldr_en_8k_slice      | 768  | recursive cap256 ov0tok | fastembed:bge-base to model2vec:potion-base-8M              | -0.1351       | [-0.1578, -0.1124] | 44/216    | resolved     |
| mldr_en_8k_slice      | 768  | recursive cap256 ov0tok | fastembed:bge-base to model2vec:potion-retrieval-32M        | -0.0526       | [-0.0723, -0.0332] | 73/154    | resolved     |
| mldr_en_8k_slice      | 768  | recursive cap256 ov0tok | fastembed:bge-base to ollama:bge-m3                         | +0.0279       | [+0.0124, +0.0436] | 95/57     | resolved     |
| mldr_en_8k_slice      | 768  | recursive cap256 ov0tok | fastembed:bge-base to ollama:qwen3-embedding-4b             | +0.0401       | [+0.0254, +0.0553] | 107/47    | resolved     |
| mldr_en_8k_slice      | 768  | recursive cap256 ov0tok | fastembed:bge-base to ollama:qwen3-embedding-8b             | +0.0478       | [+0.0321, +0.0633] | 111/45    | resolved     |


Two cells over the same corpus and the same chunk profile, differing only in the model. Compare the resolve rate here against the chunking axes above: the model choice is settled far more often than any chunking knob. Shown at the recommended chunk profile; the other profiles agree and are in the raw file. Only corpora that can carry a chunking claim are shown; the rest yield about one chunk per document, where every profile produces the same chunk. The full set is in `tests/benchmarks/raw/chunk-knob-effects.json`.
<!-- END GENERATED embedder_effects -->

## The providers

| Provider                       | Runs                          | Cost                | Notes                                                                                                          |
|--------------------------------|-------------------------------|---------------------|----------------------------------------------------------------------------------------------------------------|
| `fastembed`                    | Embedded, ONNX, CPU           | None beyond CPU     | The default. `bge-small` (384d) and `bge-base` (768d).                                                         |
| `model2vec`                    | Embedded, static vectors, CPU | None beyond CPU     | Roughly a thousand times faster than fastembed on CPU; `potion-base-8M` (256d), `potion-retrieval-32M` (512d). |
| `ollama`                       | Separate server, GPU          | A GPU, or very slow | `bge-m3` (1024d), `qwen3-embedding` at 2560d and 4096d.                                                        |
| `sentence-transformers`        | In process, GPU or CPU        | Heavy dependency    | Widest model choice.                                                                                           |
| `openai` / `gemini` / `cohere` | Remote API                    | Per token           | Not measured here.                                                                                             |
| `placeholder`                  | Deterministic token hash      | None                | Offline fallback only. Never use it in production: it is lexical, so it misses any paraphrase.                 |

### What each provider costs in memory

<!-- BEGIN GENERATED embedder_memory (scripts/gen_bench_tables.py) -->
| Provider              | Model                                                         | Dim | Resident | Peak while loading |
|-----------------------|---------------------------------------------------------------|-----|----------|--------------------|
| placeholder           | `in-memory`                                                   | 256 | 32 MB    | 32 MB              |
| model2vec             | `minishlab/potion-base-8M`                                    | 256 | 111 MB   | 129 MB             |
| fastembed             | `BAAI/bge-small-en-v1.5`                                      | 384 | 269 MB   | 269 MB             |
| sentence_transformers | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | 1,434 MB | 1,434 MB           |


One process per provider, holding the model and one batch of 64 passages. `ollama`, `openai`, `gemini` and `cohere` are absent by nature rather than by omission: they hold no model in this process at all, which is the entire reason to choose one of them on a small machine. Their memory is their server's.
<!-- END GENERATED embedder_memory -->

The spread across the in-process providers is a factor of 46, and it is the constraint that
decides the choice on a small machine before quality does. `sentence_transformers` pulls torch and
holds 1,434 MB; `model2vec` holds 111 MB, under a tenth of that. The server-backed providers hold
no model here at all, which is the whole reason to reach for one when the box is small - the
memory moves to the server rather than disappearing. The chart is in
[Memory](05-vector-store.md#memory-will-it-fit-in-the-container).

## Ranked, on long documents

<!-- BEGIN GENERATED embedder_ranking_mldr (scripts/gen_bench_tables.py) -->
| Corpus     | Model                          | Dim  | Best profile for this model                          | nDCG@10 [95% CI]      | Recall@10 |
|------------|--------------------------------|------|------------------------------------------------------|-----------------------|-----------|
| mldr_en_8k | ollama:qwen3-embedding-8b      | 4096 | recursive cap64 ov0tok                               | 0.9242 [0.908, 0.940] | 0.9613    |
| mldr_en_8k | ollama:qwen3-embedding-4b      | 2560 | recursive cap64 ov0tok                               | 0.9171 [0.901, 0.933] | 0.9637    |
| mldr_en_8k | ollama:bge-m3                  | 1024 | recursive cap64 ov0tok                               | 0.9140 [0.897, 0.930] | 0.9613    |
| mldr_en_8k | fastembed:bge-base             | 768  | recursive cap64 ov0tok                               | 0.9017 [0.883, 0.919] | 0.9475    |
| mldr_en_8k | model2vec:potion-retrieval-32M | 512  | recursive cap64 ov0tok                               | 0.8304 [0.808, 0.852] | 0.9012    |
| mldr_en_8k | fastembed:bge-small            | 384  | semantic hint256 breakpoint-default(potion-base-32M) | 0.8294 [0.806, 0.852] | 0.8950    |
| mldr_en_8k | model2vec:potion-base-8M       | 256  | recursive cap64 ov0tok                               | 0.7836 [0.759, 0.808] | 0.8675    |
| mldr_de_3k | ollama:qwen3-embedding-8b      | 4096 | recursive cap256 ov10tok                             | 0.7350 [0.678, 0.790] | 0.8050    |
| mldr_de_3k | ollama:qwen3-embedding-4b      | 2560 | fast hint256 ov26tok                                 | 0.7306 [0.674, 0.785] | 0.8150    |
| mldr_de_3k | ollama:bge-m3                  | 1024 | fast hint256 ov51tok                                 | 0.7245 [0.668, 0.779] | 0.8100    |
| mldr_de_3k | fastembed:bge-small            | 384  | recursive cap256 ov0tok                              | 0.4949 [0.432, 0.559] | 0.5900    |
| mldr_de_3k | fastembed:bge-base             | 768  | fast hint256 ov0tok                                  | 0.4675 [0.402, 0.533] | 0.5350    |
| mldr_de_3k | model2vec:potion-retrieval-32M | 512  | markdown hint256 ov51tok                             | 0.3601 [0.299, 0.422] | 0.4250    |
| mldr_de_3k | model2vec:potion-base-8M       | 256  | whitespace hint64                                    | 0.2573 [0.203, 0.312] | 0.3350    |


One row per model, at whichever profile that model scored highest on, so a model is not penalised for a setting that suits another. Dimension rides along with the model and cannot be separated from it here.
<!-- END GENERATED embedder_ranking_mldr -->

## Dimension is not the same question as model

Every model in the table above sits at its own dimension, so a ranking of models is also a ranking
of dimensions and the two cannot be separated from it. That is a real confound, and the honest
statement is that these rows compare *models as shipped*, not dimensions.

What can be said from the store measurements is what a dimension COSTS, which is measured
directly on identical text in [Vector store](05-vector-store.md): exact-search latency and disk
both scale about linearly with dimension, so at 100,000 rows a 4096-dimensional model costs
15 times the storage and 19 times the search time of a 256-dimensional one.

Separating quality from dimension needs a model whose output can be truncated (a Matryoshka-trained
model such as Qwen3-Embedding) scored at several widths, so the model is held fixed while the
dimension moves. That is not measured yet; see [Gaps](08-gaps.md).

## Language

The plain rankings above cover English and German long documents separately, and they do not agree
on the ordering, which is the point: pick on the language you actually have.

For non-English or mixed-language corpora the multilingual models are measured on the same long
documents, at whichever chunk profile suits each best:

<!-- BEGIN GENERATED embedder_ranking_multilingual (scripts/gen_bench_tables.py) -->
| Corpus     | Model                     | Dim  | Best profile for this model                          | nDCG@10 [95% CI]      | Recall@10 |
|------------|---------------------------|------|------------------------------------------------------|-----------------------|-----------|
| mldr_en_8k | ollama:qwen3-embedding-8b | 4096 | semantic hint256 breakpoint-bge-m3                   | 0.8839 [0.865, 0.903] | 0.9387    |
| mldr_en_8k | openai:e5-large           | 1024 | semantic hint256 breakpoint-qwen3-0.6b               | 0.8808 [0.861, 0.900] | 0.9363    |
| mldr_en_8k | ollama:qwen3-embedding-4b | 2560 | semantic hint256 breakpoint-qwen3-0.6b               | 0.8747 [0.855, 0.894] | 0.9325    |
| mldr_en_8k | ollama:bge-m3             | 1024 | semantic hint256 breakpoint-potion-multilingual-128M | 0.8656 [0.845, 0.885] | 0.9250    |
| mldr_de_3k | ollama:qwen3-embedding-8b | 4096 | semantic hint256 breakpoint-bge-m3                   | 0.7300 [0.673, 0.785] | 0.8050    |
| mldr_de_3k | ollama:qwen3-embedding-4b | 2560 | semantic hint256 breakpoint-jina-v3                  | 0.7178 [0.662, 0.772] | 0.8100    |
| mldr_de_3k | ollama:bge-m3             | 1024 | semantic hint256 breakpoint-jina-v3                  | 0.6738 [0.614, 0.733] | 0.7600    |
| mldr_de_3k | openai:e5-large           | 1024 | semantic hint256 breakpoint-bge-m3                   | 0.6660 [0.605, 0.724] | 0.7550    |


One row per model, at whichever profile that model scored highest on, so a model is not penalised for a setting that suits another. Dimension rides along with the model and cannot be separated from it here.
<!-- END GENERATED embedder_ranking_multilingual -->

Two things to take from it. `qwen3-embedding-8b` leads on both languages and is the strongest
model measured anywhere in this set. And `e5-large` at 1024 dimensions all but matches it on
English, 0.881 against 0.884, then falls to last on German - so a model picked on an English
ranking can be the worst choice on the language you actually have. Rank on your own language.

These models all need a server: the qwen3 pair and `bge-m3` run on ollama, `e5-large` behind a
sentence-transformers shim. None of them is a drop-in replacement for the embedded CPU providers
above; they are the answer when retrieval quality justifies running infrastructure.

## The instruction prefix qwen3 was never given

Qwen3-Embedding is trained to receive its queries wrapped in an instruction and its passages
bare. Every qwen3 figure above was taken with neither. That is at least self-consistent, and it
is also not the configuration the model's published numbers assume, so the rows understated it by
an amount nobody here had measured.

<!-- BEGIN GENERATED qwen3_instruction (scripts/gen_bench_tables.py) -->
| Model              | Configuration                  | nDCG@10 [95% CI]      | Recall@10 | MRR    |
|--------------------|--------------------------------|-----------------------|-----------|--------|
| qwen3-embedding:4b | no prefix (as published)       | 0.3571 [0.322, 0.393] | 0.1771    | 0.5479 |
| qwen3-embedding:4b | instruction on the query       | 0.4030 [0.367, 0.440] | 0.1965    | 0.6213 |
| qwen3-embedding:4b | instruction on query + passage | 0.4055 [0.370, 0.442] | 0.1992    | 0.6227 |
| qwen3-embedding:8b | no prefix (as published)       | 0.3792 [0.343, 0.416] | 0.1893    | 0.5716 |
| qwen3-embedding:8b | instruction on the query       | 0.4155 [0.379, 0.453] | 0.1983    | 0.6272 |
| qwen3-embedding:8b | instruction on query + passage | 0.4078 [0.372, 0.445] | 0.1976    | 0.6254 |


`beir/nfcorpus/test`, 3,633 documents, 323 queries. The prefix is the model card's own: `Instruct: <task>\nQuery:` with the task "Given a web search query, retrieve relevant passages that answer the query". Passages are identical between the first two rows of each model, because a query prefix does not touch them.
<!-- END GENERATED qwen3_instruction -->

<!-- BEGIN GENERATED qwen3_instruction_paired (scripts/gen_bench_tables.py) -->
| Model              | Comparison                                                 | Delta nDCG@10 | 95% CI             | Better/worse | Verdict      |
|--------------------|------------------------------------------------------------|---------------|--------------------|--------------|--------------|
| qwen3-embedding:4b | instruction on the query vs no prefix (as published)       | +0.0459       | [+0.0319, +0.0605] | 139/74       | resolved     |
| qwen3-embedding:4b | instruction on query + passage vs no prefix (as published) | +0.0484       | [+0.0327, +0.0646] | 146/72       | resolved     |
| qwen3-embedding:4b | instruction on query + passage vs instruction on the query | +0.0025       | [-0.0037, +0.0086] | 85/88        | not resolved |
| qwen3-embedding:8b | instruction on the query vs no prefix (as published)       | +0.0363       | [+0.0239, +0.0492] | 146/66       | resolved     |
| qwen3-embedding:8b | instruction on query + passage vs no prefix (as published) | +0.0286       | [+0.0144, +0.0429] | 147/81       | resolved     |
| qwen3-embedding:8b | instruction on query + passage vs instruction on the query | -0.0077       | [-0.0154, -0.0001] | 76/105       | resolved     |


Differences per query rather than between two means, because the arms differ by far less than the query-to-query spread. `Better/worse` counts queries; the remainder tied.
<!-- END GENERATED qwen3_instruction_paired -->

**The instruction is worth having, and the effect resolves on both models.** Adding it to the
query alone moves nDCG@10 by +0.0459 on the 4B and +0.0363 on the 8B, both resolved on a paired
test over the same 323 queries. In relative terms that is 12.9 and 9.6 percent. It is one config
line and it costs nothing at query time: a few tokens on a query, and the corpus untouched.

**It changes no vectors you have already computed.** The model card's wording is "No need to add
instruction for retrieval documents", and that has a practical consequence worth stating plainly:
turning this on does NOT require a reindex. The passage vectors are identical with and without a
query prefix, which is exactly why this measurement was cheap enough to run at all.

**Prefixing the passages as well does not help, and on the larger model it hurts.** On the 4B the
`both` arm posts the highest raw nDCG@10 of the three, which invites the conclusion that the
asymmetry is a formality and you may as well prefix everything. Differenced against the `query`
arm over the same queries it does not resolve there, and on the 8B it resolves NEGATIVE. So the
card's "No need to add instruction for retrieval documents" is not merely permission to skip the
passage prefix; on the bigger model it is a warning. Raw means would have said the opposite on
the 4B, which is what a paired test is for.

This is the practically important half of the result. If passages needed the prefix too, turning
this on would mean re-embedding the whole corpus. They do not, so it is a query-side change with
a resolved gain and no reindex.

### What this does not establish

* **One corpus, English, one task description.** NFCorpus is medical and the instruction used is
  the model card's generic web-search wording. A task description written for the actual corpus
  is the obvious next lever and is not measured.
* **The rankings above were not re-run.** Every qwen3 row elsewhere on this page is still the
  plain configuration, so those comparisons understate qwen3 by roughly this margin. They are
  not corrected here, because re-running them is a different job from measuring the effect.
* **Only qwen3.** E5 and BGE want their own prefixes and semdex can now send them, but what they
  are worth is unmeasured.

## Practical notes

**ollama truncates silently unless `num_batch` is raised.** ollama's physical batch defaults to
2048 tokens while it serves most embedding models with a 4096-token context. It cannot process an
input larger than that batch, and its recovery is to re-run the input truncated to 2048 and answer
HTTP 200. A longer chunk is therefore embedded from its first 2048 tokens with no error anywhere.
Set `[embedding].num_batch` to at least the model's context. This matters most with the semantic
strategy, which does not enforce a chunk size at all.

**An instruction-tuned model needs its prefix sent.** ollama and the OpenAI-compatible endpoint
apply no template of their own, so a model trained to receive an instruction gets one only if
`[embedding].query_prefix` is set. See the section above for what it is worth on qwen3. The
providers that handle this natively (fastembed, gemini, cohere) REJECT the setting rather than
ignore it, so a prefix that cannot be applied is an error instead of a silent no-op.

**Do not mix models in one collection.** Vectors from different models are not comparable, and the
collection records the model id precisely so that mixing is refused rather than silently returning
nonsense.

**A model change means a full re-index.** Every vector has to be recomputed, so this is the
expensive decision to get right first, and the cheapest to get right before ingest rather than
after.
