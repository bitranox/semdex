---
name: bench-rerun
description: Use when semdex components were updated (extractor/store container images, fastembed or other embedding models, chunker libraries, ollama models) or on a periodic re-test, and the benchmarks must be re-run, compared against the committed baseline, and the bench_*.md rankings refreshed.
---

# Re-run the semdex benchmarks after component updates

Everything runs against the **fixed slice** the baseline was seeded with - a different slice
makes the comparison apples-to-oranges. One-off scale references (the MSMARCO tables in
`bench_database.md`) are NOT part of this loop; re-measure those deliberately.

## 1. Update what actually changed

```bash
# Containers: docker run reuses the cached tag - a stale image silently tests nothing new.
docker pull <image>            # then record: docker inspect --format '{{index .RepoDigests 0}}' <image>
# Python deps: upgrade IN THE BENCHMARK VENV (the one with all extras, see
# COMPONENT_SETUP.md "Benchmark prerequisites"), not some other venv.
uv pip install -U <package>
# ollama models: ollama pull <model> on your server (endpoint via SEMDEX_BENCH_OLLAMA_URL
# from .env - never hardcode a host anywhere in the repo).
```

## 2. Re-run at the fixed slice - ONE pytest session

Result cells accumulate into one `benchmark-results.json` per pytest **process**; a second
invocation overwrites it. Run all needed files in a single session:

```bash
SEMDEX_BENCH_CORPORA=nfcorpus SEMDEX_BENCH_MAX_DOCS=150 SEMDEX_BENCH_MAX_QUERIES=20 \
  pytest -p no:cacheprovider -o addopts="" -s \
  tests/test_e2e_matrix.py tests/test_e2e_extractor_matrix.py::test_extractor_fixture_matrix
```

Scope a partial re-check to the changed axis with the env filters
(`SEMDEX_BENCH_CHUNKERS/_EMBEDDINGS/_STORES/_EXTRACTORS/_EMBED_MODELS`); untouched cells show
as non-gating "not_run". Cells whose container/server is down record `skip` - the comparator
flags `ok -> skip`, so fix the service rather than accepting it.

## 3. Compare - and root-cause before touching the baseline

```bash
python scripts/benchmark_compare.py benchmark-results.json
```

On a regression, **never `--update` to make red green.** Attribute it first: status flip ->
diff the recorded image digest vs the baseline's; quality drop -> what changed in the model
(re-run once to confirm - fastembed is deterministic); adapter error -> upstream API change,
fix the adapter + its tests. A semdex-side fix ends with the slice green; an upstream
regression ends with a pin + upstream issue; a genuine accepted change moves on to step 4.

## 4. Adopt and re-rank

```bash
python scripts/benchmark_compare.py benchmark-results.json --update
```

Then update, in the same commit: the affected ranked tables AND their prose in
Nothing by hand. Run `python scripts/export_bench_raw.py` then
`python scripts/gen_bench_tables.py` and `python scripts/gen_bench_charts.py`: every published
table and chart under `docs/benchmarks/` is generated from `tests/benchmarks/raw/`, and
`pytest tests/test_bench_tables_current.py tests/test_bench_charts_current.py` fails if a
published number no longer matches its data. Only the PROSE around a table needs a human, if a verdict
changed; the new component versions/digests named in the commit message; a `CHANGELOG.md`
entry; a `pyproject.toml` constraint if a floor/pin changed. The quality-table numbers come
from the 4-corpus run (`SEMDEX_BENCH_CORPORA=nfcorpus,cqadupstack,scifact,fiqa`) - re-run that
before re-ranking chunker/embedding tables.

## 5. Gate before pushing

```bash
grep -rnE "192\.168\.|[0-9]{1,3}(\.[0-9]{1,3}){3}|\.local\." docs/ tests/benchmarks/ .claude/skills/ || echo leak-clean
make test
```

Done when: every flagged cell has a written root cause, `make test` is green, the baseline
refresh is its own reviewable commit naming the version bump that justifies it, and the doc
tables cite the new numbers.
