# Raw benchmark data

Measured benchmark readings preserved so new reports and rankings can be written
without re-running the (sometimes multi-hour) measurements. These files are data,
not a CI gate.

## What lives where

- **The gated 150-doc slice** (per-corpus quality, store matrix, embedding-model
  matrix, extractor grid, chunker cells) is kept in `../baseline.json`. That file is
  the regression baseline `scripts/benchmark_compare.py` checks against, keyed by
  `<dimension>/<variant>@<corpus>`. Read it for any 150-doc/20-query reading.
- **Ungated one-off scale references** live here, because they are too slow to gate
  in CI and are not part of `baseline.json`:

| File                           | What it holds                                                                                                           |
|--------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| `msmarco-scale.json`           | Store backends vs MSMARCO scale: exact stores at 50K/250K, the 3 ANN stores at 50K/250K/1M.                             |
| `ann-scale-50k-250k.json`      | The 3 ANN stores (lancedb/pgvector/mariadb) at 50K + 250K, real data (source for the merge).                            |
| `ann-1m.json`                  | The 3 ANN stores at 1M docs, real data (source for the 1M rows of msmarco-scale.json).                                  |
| `chunker-throughput.json`      | Pure chunking throughput (docs/s, chunks/s) per strategy, chunking only.                                                |
| `ann-frontier.json`            | Recall-latency frontier per ANN store: query-time and build-time knob sweeps against the exact top-10.                  |
| `extraction-omnidocbench.json` | Extraction fidelity of all five backends on 113 annotated OmniDocBench pages, by document type and language.            |
| `span-integrity.json`          | Does a chunk boundary cut the answer: SQuAD-style span containment per profile, plus the retrieval-composed blind spot. |
| `dim-crossover.json`           | sqlite_vec (exact) vs lancedb (ANN) p50/p95/disk swept over embedding dim x corpus size.                                |
| `embedding-qwen3-gpu.json`     | qwen3-embedding 4b/8b on a GPU host: quality (also in baseline), served dim, warm GPU docs/s.                           |
| `query-power-validation.json`  | Holdout test of the queries-needed formula: predict from a 200-query subsample, score against 800.                      |
| `query-power.json`             | Queries needed to resolve each corpus's tied comparisons, and which are unresolvable at any size.                       |
| `metric-redundancy.json`       | Effective independent views carried by the four reported metrics, per corpus, with judgment density.                    |
| `energy.json`                  | Marginal energy per document for the in-process embedding providers, from the CPU package RAPL counter.                 |
| `energy-gpu.json`              | The same for the ollama-served models, from integrated GPU card power.                                                  |
| `cloud-prices.json`            | Hosted embedding API list prices with sources, plus measured tokens/document per tokenizer.                             |
| `memory.json`                  | Resident memory per store (1k-100k chunks, serving and ingest) and per in-process embedder.                             |
| `qwen3-instruction.json`       | qwen3-embedding 4b with and without its instruction prefix: three arms, paired per query.                               |
| `qwen3-instruction-8b.json`    | The same for qwen3-embedding 8b.                                                                                        |
| `embedding-german-miracl.json` | All 10 registry models on `miracl/de/dev` (native German): the multilingual ranking.                                    |

## Field notes

`msmarco-scale.json` (one object per store x scale):

- `ndcg@10`, `recall@10`, `mrr@10` - retrieval quality against BEIR judgments.
- `search_p50_ms`, `search_p95_ms` - query latency.
- `upsert_s`, `embed_s`, `chunk_s` - one-time ingest costs (embedding is shared
  across stores at a scale; only `upsert_s` is per-store).
- `db_mb` / `db_bytes` - on-disk size (server stores measured via `du` inside the
  container; includes the fixed empty-server baseline).
- `note` - present on the `mariadb` rows (corrected after the `DISTANCE=cosine`
  vector-index fix; the pre-fix rows full-scanned) and the `json` rows (in-RAM O(n)
  exact scan, the measured reason it is unfeasible at scale).

`chunker-throughput.json` (one object per corpus x strategy): `docs`, `chunks`,
`seconds`, `docs_per_s`, `chunks_per_s`. Chunking only - no embedding, no store.

`measured_on` - which machine and which code produced a reading. On a cell it is the
stamp its scorer wrote at the moment it measured that cell: `measured_utc`,
`semdex_git_sha`, `host`, `cpu`, `python`, `numpy`, `blas`, `openblas_num_threads`.
At file level it summarises what the cells hold - `recorded`, `not_recorded`, and
`runs`, the distinct stamps in measurement order.

Read it per cell, not per file. A sweep resumes across days and hosts and skips cells
it has already measured, so one file legitimately mixes runs: `store-quality.json`
holds lancedb rows, which need AVX2, beside rows a node without AVX2 can produce.

`not_recorded` counts cells measured before the scorers stamped anything. There is no
way to recover the machine those ran on, so it stays blank rather than borrowing the
machine that exported them - which is what these files used to do, describing the
export host and date on latency numbers measured weeks earlier. Nothing in a file now
depends on the environment that exported it, so re-exporting unchanged sources
reproduces it byte for byte. When a file landed is in `git log`, not in the file.

## Reproducing

The measurement recipe (env knobs, slice, comparator flow) is the `bench-rerun`
skill. The scale runs use `scripts/bench_msmarco_scale.py` (point `MSMARCO_DIR` at a
BEIR-format corpus dir); the throughput uses `scripts/bench_chunker_throughput.py`.
Both are standalone harness scripts, not part of the shipped package.

## Charts

The ranking charts in `docs/benchmarks/img/*.png` are generated FROM this data by
`scripts/gen_bench_charts.py` (needs `semdex[charts]`: matplotlib), so they cannot
drift. After changing any raw file or a charted `baseline.json` cell, re-run it and
commit the refreshed PNGs + `docs/benchmarks/img/charts.manifest.json`. `test_bench_charts_current`
re-derives the plotted values (no matplotlib) and fails if a committed chart is stale.
