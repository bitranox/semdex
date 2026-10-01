# ANN store tuning + embedding batch config knobs (#44) - design

**Goal:** expose the ANN store accuracy/speed knobs (today hardcoded / driver-default) and the
indexing embedding batch size as normal `lib_layered_config` keys, so recall-vs-latency and
ingest throughput are tunable without code changes and without a bespoke preset engine.

**Non-goals:** changing any default behavior (this ships non-breaking), a per-environment
mechanism (that is what `lib_layered_config` profiles already are - orthogonal to these knobs),
and re-deriving chunk/embedder defaults (that is #42's job).

## Background: what is exposed vs hardcoded today

- `[embedding]` already exposes `provider, model, endpoint, timeout, retries, api_key, threads`.
  The one embedding gap is the indexing **batch size**: `indexing.py:81` embeds every chunk of a
  source in one `embed_passages(...)` call, with no cap.
- `[vector_store]` exposes `backend, lance_index_threshold, compact_after_records,
  compact_after_seconds, default_partition`. The ANN **index build + query params are hardcoded**:
  - lancedb: `create_index("vector", IvfPq(distance_type="cosine"))`; query is
    `search(v).metric("cosine").limit(k)` with no `.nprobes()/.refine_factor()`.
  - pgvector: `CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)` (default `m`/
    `ef_construction`); query is a plain `ORDER BY <=> LIMIT k` (default `hnsw.ef_search`).
  - mariadb: MHNSW index at defaults; query uses `VEC_DISTANCE` (default `mhnsw_ef_search`).
  - sqlite_vec + json are EXACT (brute-force KNN) - no ANN knob applies.

## Config surface

```toml
[vector_store]
# Portable recall/latency preset. Maps per backend to a QUERY-TIME param (no reindex, live).
# "balanced" == today's driver-default behavior (non-breaking). Ignored by exact stores
# (sqlite_vec, json).
ann_recall = "balanced"          # fast | balanced | accurate  (StrEnum)

# Optional raw per-backend overrides (power users). A set value WINS over the ann_recall preset
# for that backend. Build-time params (m / ef_construction / num_partitions) need a reindex.
[vector_store.lancedb]
nprobes = 20                     # query-time
refine_factor = 0                # query-time (0 = off)
num_partitions = 0               # build-time; 0 = auto (lancedb tunes from row count)

[vector_store.pgvector]
ef_search = 40                   # query-time (SET hnsw.ef_search)
m = 16                           # build-time
ef_construction = 64             # build-time

[vector_store.mariadb]
ef_search = 40                   # query-time (mhnsw_ef_search)
m = 16                           # build-time (MHNSW M)

[embedding]
batch = 0                        # passages per embed_passages call at index time; 0 = no cap
                                 # (today's behavior). ollama loaders already sub-batch internally.
```

All keys are ordinary `lib_layered_config` keys: env-overridable
(`SEMDEX___VECTOR_STORE__ANN_RECALL=accurate`,
`SEMDEX___VECTOR_STORE__PGVECTOR__EF_SEARCH=200`, `SEMDEX___EMBEDDING__BATCH=128`),
provenance-tracked, and per-environment via a `lib_layered_config` profile if a deployment wants
that (bonus, not part of this design).

## ann_recall -> native query param (mapping; values to be calibrated)

`balanced` MUST equal the current driver default so the change is non-breaking; `fast` /
`accurate` deviate. The exact numbers are calibrated by the recall/latency bench (see Testing),
not guessed - these are starting points only.

```
| ann_recall | lancedb nprobes | lancedb refine | pgvector/mariadb ef_search | exact stores |
|------------|-----------------|----------------|----------------------------|--------------|
| fast       | 10              | 0              | 20                         | n/a          |
| balanced   | driver default  | 0              | 40 (driver default)        | n/a          |
| accurate   | 40              | 10             | 200                        | n/a          |
```

## Config models (changed)

- New `AnnRecall(StrEnum)` = `fast | balanced | accurate` in `domain/enums.py`.
- `VectorStoreConfig` (adapters/config) gains `ann_recall: AnnRecall = AnnRecall.BALANCED` and
  optional per-backend Pydantic sub-models `LancedbParams`, `PgvectorParams`, `MariadbParams`
  (all fields default to a sentinel meaning "unset -> use the preset / driver default"; validated
  as non-negative ints).
- `EmbeddingConfig` gains `batch: int = 0` (0 = uncapped; validated >= 0).
- A resolver `resolve_ann_params(backend, ann_recall, raw_overrides) -> AnnParams` (a small typed
  object with `nprobes/refine_factor/ef_search/m/ef_construction/num_partitions`, each optional)
  centralizes preset-to-native mapping + override precedence. Pydantic in, typed object out - no
  dicts (per the strict data-architecture rule).

## Plumbing

- `build_vector_store(backend, store_dir, *, dsn, lance_index_threshold, ann_params=None)`:
  pass the resolved `AnnParams` into each store constructor.
- Query-time application (live, no reindex):
  - lancedb `query()`: `search(v).metric("cosine").nprobes(p).refine_factor(r).limit(k)` when set.
  - pgvector `query()`: `SET LOCAL hnsw.ef_search = <n>` in the query txn before the SELECT.
  - mariadb `query()`: set `mhnsw_ef_search` (session var) before the SELECT.
- Build-time application (reindex):
  - pgvector `ensure_collection()`: `CREATE INDEX ... USING hnsw (...) WITH (m=<m>,
    ef_construction=<e>)`.
  - mariadb `ensure_collection()`: MHNSW `M=<m>`.
  - lancedb `compact()`: `IvfPq(distance_type="cosine", num_partitions=<n>)` when set.
- Embedding batch: thread `batch` into the index use case (`indexing.py`) - split
  `[piece.text for piece in chunks]` into batches of `batch` before `embed_passages`, concatenate
  the vectors. `batch = 0` keeps the single-call path (unchanged).

## Data flow

`config TOML/env -> VectorStoreConfig + EmbeddingConfig (Pydantic) -> composition:
resolve_ann_params() + batch -> build_vector_store(ann_params) / index use case (batch) ->
adapter applies query-time param per search and build-time param at ensure_collection/compact`.

## Non-breaking guarantee

`ann_recall = "balanced"` with all raw overrides unset, and `batch = 0`, emit exactly today's
DDL and query - asserted by tests. Changing `ann_recall` alone never triggers a reindex (it is
query-time only); build-time overrides do, and their docs say so.

## Error handling

- Invalid `ann_recall` -> Pydantic validation error at config load (fail fast, explicit choice).
- Raw override out of range (negative) -> validation error.
- A param set for an exact store (sqlite_vec/json) -> ignored with a one-line debug log (the store
  has no ANN index); `ann_recall` is simply inert there.
- A backend driver that rejects a param -> surfaces as the store's existing error type.

## Testing

- Config parse: `ann_recall` enum + per-backend sub-models + env overrides + provenance.
- Adapter unit tests (fakes / captured SQL): `ann_recall=fast|accurate` sets the expected native
  param (assert the `SET hnsw.ef_search` / `.nprobes()` call); `balanced` + unset == current
  DDL/query (the non-breaking lock).
- Embedding batch: `batch=2` over 5 chunks makes 3 `embed_passages` calls; `batch=0` makes 1;
  vectors identical either way (fake provider).
- Perf calibration (integration, deferred to a quiet box): run the ANN recall/latency bench
  (bench_database.md / #12 methodology) at fast/balanced/accurate on a fixed slice to CALIBRATE
  the preset numbers - magic numbers are set from measurement, not guessed. Ship the mechanism +
  balanced (safe) first; finalize fast/accurate numbers after the bench.
- Docs: `15-vectorstore.toml` + `30-embedding.toml` documented comprehensively (default, effect,
  when to set, consequences, per-scenario recommendation), per the repo's TOML-doc convention;
  update `bench_database.md` with the calibrated recall/latency-per-preset table.

## Scope / phasing (YAGNI)

- Phase A (v1): `ann_recall` preset (query-time, all ANN backends) + query-time raw overrides
  (nprobes/refine_factor/ef_search) + `[embedding].batch`. High value, live, non-breaking.
- Phase B: build-time raw overrides (m/ef_construction/num_partitions) - lower demand, need a
  reindex; ship once someone needs them.
- Phase C: calibrate the fast/accurate numbers from the bench and publish the per-preset
  recall/latency table.

## Decisions (resolved in brainstorm)

1. Preset shape: a `fast/balanced/accurate` StrEnum, not a numeric `search_ef` - portable across
   backends whose native params differ in name and scale.
2. Scope: a global `[vector_store].ann_recall` default WITH an optional per-dataset override in
   `DatasetConfig`, mirroring how `backend`/`partition` already work per dataset; fan-out search
   then queries each dataset at its own recall. A config-driven knob is appropriate here because
   semdex is an APP whose operational policy lives in config - unlike an in-process library, where
   an explicit per-call parameter would be preferred over a config-layer default.
