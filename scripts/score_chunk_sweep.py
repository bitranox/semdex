#!/usr/bin/env python
# pyright: basic
"""Score the chunk-param sweep cells (#42): for each cached (corpus x profile x embedding)
cell, embed the eval queries and report nDCG@10 / Recall@10 / MRR / P@1.

Reuses the pre-computed vector cache, so it never re-embeds passages - only the (tiny) eval-query
set, and even that is cached across profiles. Ranking is exact brute-force cosine over the cell's
cached vectors, streamed a block at a time by :mod:`_score_kernel`, so it needs no store, matches
what an exact store would return, and holds memory flat regardless of cell size (the 1M x 4096
msmarco cell is 16 GB and used to be materialised twice). Per-query scores are retained and
turned into bootstrap confidence intervals, because a bare mean over 200 queries cannot support
the third-decimal comparisons these tables get read for.

Writes a JSON results file and prints a per-corpus markdown table sorted by nDCG@10. Idempotent:
a scored cell is skipped unless SEMDEX_SCORE_FORCE=1, and results persist after every cell so a
re-run continues where it stopped.

Env:
  CACHE_ROOT               cache dir (default /embeddings)
  SEMDEX_SCORE_CORPORA     comma list (default miracl_en_100k_slice,miracl_de_100k_slice)
  SEMDEX_SCORE_PROFILES    comma list of profile tags (default: every profile present in the cache
                           FOR THE REQUESTED CORPORA)
  SEMDEX_SCORE_EMBEDDINGS  comma list of embedding labels (default the 4 CPU embedders; add the
                           ollama qwen3 cells once the GPU sweep is idle)
  SEMDEX_BENCH_OLLAMA_URL  endpoint for ollama (qwen3, bge-m3) query embedding
  SEMDEX_BENCH_OPENAI_URL  endpoint for the openai-compatible shim (e5-large)
  SEMDEX_SCORE_K           top-k (default 10)
  SEMDEX_SCORE_OUT         results json (default <cache>/scores/chunk_sweep_scores.json,
                           which is the MIRACL export's source - set it for any other corpus)
  SEMDEX_SCORE_FORCE       =1 to re-score cells already in the results file
  SEMDEX_SCORE_TIMEOUT     per-request timeout for non-fastembed backends (default 600)
  SEMDEX_SCORE_KEEP_ALIVE  ollama model residency (default 10m; one load per embedder, not per cell)
  SEMDEX_SCORE_PERQUERY    dir for per-query score arrays (default <cache>/scores/perquery)
  SEMDEX_SCORE_QVECS       dir for the query-vector cache (default <cache>/scores/qvecs)
  SEMDEX_SCORE_BLOCK_BYTES streaming block budget in bytes (default 256 MiB)
  SEMDEX_SCORE_QUERY_WORDS score with every query cut to its first N words (a QUERY-LENGTH
                           control: the corpus vectors are reused, only the query vectors change).
                           Refused unless SEMDEX_SCORE_OUT and SEMDEX_SCORE_PERQUERY are both set,
                           because the rows carry the same cell keys as the real measurement and
                           would overwrite it in the default files. Rows record `query_words`.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import the sibling driver helpers

from _provenance import stamped
from _score_kernel import topk_stream
from _score_stats import bootstrap_ci
from preembed_vectors import _EMBED_MODELS, _cache_root, _ndcg, _read_json

from semdex.composition import build_embedding
from semdex.domain.enums import EmbeddingBackend

# CPU-only embedders: scoring these embeds only the eval queries on CPU, so it does not contend
# with the GPU (qwen3) embedding pass of a still-running sweep. Add the ollama cells via
# SEMDEX_SCORE_EMBEDDINGS once the GPU is idle.
_CPU_EMBEDDERS = [
    "model2vec:potion-base-8M",
    "model2vec:potion-retrieval-32M",
    "fastembed:bge-small",
    "fastembed:bge-base",
]

_DEFAULT_BLOCK_BYTES = 256 * 1024 * 1024

# Metric name -> npz key. The metric names contain "@", which is not a valid identifier, and an
# npz entry is addressed by name; both the writer here and the exporter's reader use this map.
NPZ_KEYS = {"ndcg@10": "ndcg", "recall@10": "recall", "mrr": "mrr", "p@1": "p1"}


def _dirsafe(label: str) -> str:
    return label.replace(":", "-")


def _load_uris(parquet: Path) -> list[str]:
    """Chunk source-uris in vectors.npy row order (parquet is written in the same order)."""
    uris: list[str] = []
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=8192, columns=["source_uri"]):
        uris.extend(batch.to_pydict()["source_uri"])
    return uris


def _metrics(ranked: list[str], rels: dict[str, int], k: int) -> dict[str, float]:
    """nDCG@k, Recall@k, MRR, P@1 for one query's deduped doc ranking."""
    relevant = {doc for doc, rel in rels.items() if rel > 0}
    topk = ranked[:k]
    reciprocal = 0.0
    for position, doc in enumerate(topk):
        if doc in relevant:
            reciprocal = 1.0 / (position + 1)
            break
    return {
        "ndcg@10": _ndcg(ranked, rels, k),
        "recall@10": len([d for d in topk if d in relevant]) / len(relevant) if relevant else 0.0,
        "mrr": reciprocal,
        "p@1": 1.0 if topk and topk[0] in relevant else 0.0,
    }


def _build_provider(label: str, model_id: str, endpoint: str | None) -> Any:
    """Build the embedding provider for query embedding only.

    allow_fallback=False: this module forces HF_HUB_OFFLINE (via the preembed import), so a
    missing fastembed cache would otherwise degrade to the placeholder. Its dim differs from the
    cell's, so the ranking matmul would die on an opaque shape mismatch; fail at the load with a
    clear EmbeddingError naming the real cause instead.
    """
    backend = _EMBED_MODELS[label][0]
    # A cold multi-GB ollama load does not fit the adapter's 60s default. keep_alive holds the
    # model across the whole embedder pass (the loop is embedder-outer) rather than reloading per
    # cell, and it is released explicitly before the next embedder so two models never share the
    # 16G card, which stalls the load and times every request out.
    timeout = (
        float(os.environ.get("SEMDEX_SCORE_TIMEOUT", "600")) if backend is not EmbeddingBackend.FASTEMBED else None
    )
    keep_alive = os.environ.get("SEMDEX_SCORE_KEEP_ALIVE", "10m") if backend is EmbeddingBackend.OLLAMA else None
    return build_embedding(
        backend,
        model=model_id,
        endpoint=endpoint,
        timeout=timeout,
        keep_alive=keep_alive,
        allow_fallback=False,
    )


def _release_ollama(label: str, model_id: str, endpoint: str | None) -> None:
    """Evict an ollama model from VRAM before the next embedder loads its own."""
    if _EMBED_MODELS[label][0] is not EmbeddingBackend.OLLAMA:
        return
    try:
        provider = build_embedding(
            EmbeddingBackend.OLLAMA, model=model_id, endpoint=endpoint, keep_alive="0", allow_fallback=False
        )
        provider.embed_query("release")  # the request carries keep_alive=0, which unloads the model
        print(f"[score] released {model_id} from the GPU", flush=True)
    except Exception as exc:  # a failed release must never fail the run; the model expires anyway
        print(f"[score] could not release {model_id}: {type(exc).__name__}: {exc}", flush=True)


def _qvec_path(corpus: str, label: str, queries: dict[str, str]) -> Path:
    """Cache path keyed by the query set's content, so a changed slice cannot hit a stale cache."""
    digest = hashlib.sha256(json.dumps(queries, sort_keys=True).encode()).hexdigest()[:12]
    root = Path(os.environ.get("SEMDEX_SCORE_QVECS", str(_cache_root() / "scores" / "qvecs")))
    return root / f"{corpus}__{_dirsafe(label)}__{digest}.npz"


def _query_vectors(
    corpus: str, label: str, model_id: str, endpoint: str | None, queries: dict[str, str]
) -> tuple[list[str], np.ndarray]:
    """Query vectors for one (corpus, embedder), embedded once and reused across profiles.

    Every profile of a corpus shares its query set, so without this the same 800 mldr_en queries
    were embedded once per profile - 16 GPU passes over 1600 distinct texts for the qwen3 grid.

    Queries are embedded ONE AT A TIME through ``embed_query``, deliberately. The port separates
    query from passage embedding because many models apply a different instruction prefix, so
    routing queries through the batch ``embed_passages`` call would silently embed them as
    passages and quietly change what every asymmetric model scores.
    """
    path = _qvec_path(corpus, label, queries)
    qids = sorted(queries)
    if path.exists():
        with np.load(path) as data:
            cached_ids = [str(q) for q in data["qids"]]
            if cached_ids == qids:
                print(f"[score] query vectors cached for {corpus} / {label}", flush=True)
                return qids, data["vectors"]
    provider = _build_provider(label, model_id, endpoint)
    vectors = np.asarray([provider.embed_query(queries[qid]) for qid in qids], dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    # Write through a file HANDLE: np.savez appends ".npz" to any path that does not already end
    # in it, so a ".tmp" path silently becomes ".tmp.npz" and the rename then fails on a missing
    # file. Writing to a handle leaves the name exactly as given.
    with tmp.open("wb") as handle:
        np.savez(handle, qids=np.asarray(qids), vectors=vectors)
    tmp.replace(path)  # atomic: a killed run never leaves a half-written cache entry
    print(f"[score] embedded {len(qids)} queries for {corpus} / {label}", flush=True)
    return qids, vectors


def _rank_documents(uris: list[str], rows: np.ndarray, k: int) -> list[str]:
    """Deduplicate a chunk ranking down to k distinct documents, preserving order."""
    seen: list[str] = []
    for row in rows:
        uri = uris[int(row)]
        if uri not in seen:
            seen.append(uri)
            if len(seen) >= k:
                break
    return seen


def _score_cell(corpus: str, profile: str, label: str, k: int, endpoint: str | None) -> dict[str, Any] | None:
    cell = f"{corpus}__{profile}__{_dirsafe(label)}"
    vroot = _cache_root() / "vectors" / cell
    meta = _read_json(vroot / "meta.json")
    if not meta or "count" not in meta:
        return None  # not embedded yet
    model_id, dim = meta["model_id"], int(meta["dim"])
    chunks_dir = _cache_root() / "chunks" / f"{corpus}__{profile}"
    queries = _read_json(chunks_dir / "queries.json") or {}
    qrels = _read_json(chunks_dir / "qrels.json") or {}
    words = query_words()
    if words is not None:
        queries = truncate_queries(queries, words)

    # The query-vector cache is keyed by query CONTENT, so a truncated set never hits the full one.
    qids, query_matrix = _query_vectors(corpus, label, model_id, endpoint, queries)
    uris = _load_uris(chunks_dir / "chunks.parquet")
    rows = int(np.load(vroot / "vectors.npy", mmap_mode="r").shape[0])  # header read, no data
    if len(uris) != rows:
        raise SystemExit(f"{cell}: uri/vector row mismatch ({len(uris)} vs {rows})")
    block_bytes = int(os.environ.get("SEMDEX_SCORE_BLOCK_BYTES", _DEFAULT_BLOCK_BYTES))
    # Over-fetch chunks so dedup still yields k distinct documents.
    index, _ = topk_stream(vroot / "vectors.npy", query_matrix, fetch=k * 20, block_bytes=block_bytes)

    per_query = {qid: _metrics(_rank_documents(uris, index[i], k), qrels.get(qid, {}), k) for i, qid in enumerate(qids)}
    row = _summarize(cell, corpus=corpus, profile=profile, label=label, dim=dim, per_query=per_query)
    if words is not None:
        row["query_words"] = words
    return row


def _summarize(
    cell: str, *, corpus: str, profile: str, label: str, dim: int, per_query: dict[str, dict[str, float]]
) -> dict[str, Any]:
    """Means plus a bootstrap interval per metric, and the per-query arrays written out beside them."""
    row: dict[str, Any] = {
        "corpus": corpus,
        "profile": profile,
        "embedding": label,
        "dim": dim,
        "n_queries": len(per_query),
    }
    for metric in ("ndcg@10", "recall@10", "mrr", "p@1"):
        stats = bootstrap_ci([scores[metric] for scores in per_query.values()])
        row[metric] = stats["mean"]
        row[f"{metric}_ci_lo"] = stats["ci_lo"]
        row[f"{metric}_ci_hi"] = stats["ci_hi"]
        row[f"{metric}_sd"] = stats["sd"]
    row["perquery_sha256"] = _write_per_query(cell, per_query)
    return row


def _write_per_query(cell: str, per_query: dict[str, dict[str, float]]) -> str:
    """Persist per-query scores so any interval or paired comparison can be re-derived later.

    Kept out of the committed results file (about 13 KB per cell, 405 cells) and referenced by
    hash instead, so a reviewer can reproduce a confidence interval without git carrying 400
    float arrays.
    """
    root = Path(os.environ.get("SEMDEX_SCORE_PERQUERY", str(_cache_root() / "scores" / "perquery")))
    root.mkdir(parents=True, exist_ok=True)
    qids = sorted(per_query)

    def column(metric: str) -> np.ndarray:
        return np.asarray([per_query[q][metric] for q in qids], dtype=np.float32)

    path = root / f"{cell}.npz"
    # Explicit keyword arguments rather than **arrays: the metric names contain "@", so the npz
    # keys are the identifier-safe aliases in NPZ_KEYS, and unpacking an untyped dict into savez
    # would widen into its allow_pickle parameter.
    np.savez(
        path,
        qids=np.asarray(qids),
        ndcg=column("ndcg@10"),
        recall=column("recall@10"),
        mrr=column("mrr"),
        p1=column("p@1"),
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _discover_profiles(corpora: list[str]) -> list[str]:
    """Every profile present in the cache FOR THE REQUESTED CORPORA.

    This used to glob miracl_*_100k_slice only, so an MLDR run without an explicit
    SEMDEX_SCORE_PROFILES silently scored MIRACL profile names against MLDR corpora and reported
    every one of the 96 cells as not-embedded.
    """
    profiles: set[str] = set()
    for corpus in corpora:
        for entry in (_cache_root() / "vectors").glob(f"{corpus}__*"):
            parts = entry.name.split("__")
            if len(parts) == 3:
                profiles.add(parts[1])
    return sorted(profiles)


def _print_tables(results: dict[str, Any]) -> None:
    rows = list(results.values())
    for corpus in sorted({r["corpus"] for r in rows}):
        ranked = sorted((r for r in rows if r["corpus"] == corpus), key=lambda r: -r["ndcg@10"])
        print(f"\n### {corpus}\n")
        print("| profile | embedding | nDCG@10 | 95% CI | Recall@10 | MRR | P@1 | n |")
        print("|---------|-----------|---------|--------|-----------|-----|-----|---|")
        for r in ranked:
            interval = f"[{r.get('ndcg@10_ci_lo', float('nan')):.3f}, {r.get('ndcg@10_ci_hi', float('nan')):.3f}]"
            print(
                f"| {r['profile']} | {r['embedding']} | {r['ndcg@10']:.4f} | {interval} | "
                f"{r['recall@10']:.4f} | {r['mrr']:.4f} | {r['p@1']:.4f} | {r.get('n_queries', 0)} |"
            )


def _endpoints() -> dict[Any, str | None]:
    """Per-BACKEND endpoints: an openai-compat cell (e5-large behind the ST shim) must not be
    handed the ollama URL. One shared endpoint silently pointed every cell at ollama."""
    return {
        EmbeddingBackend.OLLAMA: os.environ.get("SEMDEX_BENCH_OLLAMA_URL"),
        EmbeddingBackend.OPENAI: os.environ.get("SEMDEX_BENCH_OPENAI_URL"),
    }


def _reject_corpus_mixing(out: Path, results: dict[str, Any], corpora: list[str]) -> None:
    """Refuse to merge this run into a results file measured on a different corpus.

    The default output is the file the MIRACL export publishes, and this function reads that file
    and merges into it. Scoring any other corpus without setting SEMDEX_SCORE_OUT therefore lands
    foreign cells there, to be published under a note calling the body VOID for chunk-parameter
    claims - 44 MLDR cells nearly went out that way. The export refuses such a file too, but by
    then the results file is already polluted; this refuses while it is still clean.

    There is deliberately no override. The one file that looked like a legitimate merge, the BEIR
    results, turned out to be a note-versus-contents mismatch rather than a precedent for one, and
    a flag whose only purpose is to switch off a safety check is what somebody sets to get past a
    block and then copies into a script. A real merge is a code change, made deliberately, with the
    matching export's corpora list and note updated alongside it.
    """
    present = sorted({str(r["corpus"]) for r in results.values() if isinstance(r, dict) and r.get("corpus")})
    if not present:
        return
    strangers = sorted(c for c in set(corpora) if c not in set(present))
    if not strangers:
        return
    raise ValueError(
        f"{out} already holds {len(results)} cells measured on {present}, and this run would add "
        f"{strangers}. Point SEMDEX_SCORE_OUT at that corpus's own results file (the default is "
        "the MIRACL one). No environment variable switches this off: a deliberate merge is a code "
        "change here, with the matching export's corpora list and its note updated alongside it."
    )


def query_words() -> int | None:
    """The query-length variant requested, or None for the queries as shipped."""
    raw = os.environ.get("SEMDEX_SCORE_QUERY_WORDS")
    return int(raw) if raw else None


def truncate_queries(queries: dict[str, str], words: int) -> dict[str, str]:
    """Every query cut to its first ``words`` whitespace-delimited words; shorter ones unchanged."""
    if words < 1:
        raise ValueError(f"query truncation needs at least 1 word, got {words}")
    return {qid: " ".join(text.split()[:words]) for qid, text in queries.items()}


def variant_guard(words: int | None, *, out_set: bool, perquery_set: bool) -> None:
    """Refuse a query variant that would land in the default results file or per-query dir.

    A variant row carries the same (corpus, profile, embedder) key as the real measurement, so in
    the default locations it would overwrite the published cell and its per-query array with a
    truncated-query one, and nothing downstream could tell.
    """
    if words is None:
        return
    for name, is_set in (("SEMDEX_SCORE_OUT", out_set), ("SEMDEX_SCORE_PERQUERY", perquery_set)):
        if not is_set:
            raise SystemExit(
                f"SEMDEX_SCORE_QUERY_WORDS={words} needs an explicit {name}: no variant in the default files"
            )


def main() -> None:
    cache = _cache_root()
    variant_guard(
        query_words(),
        out_set=bool(os.environ.get("SEMDEX_SCORE_OUT")),
        perquery_set=bool(os.environ.get("SEMDEX_SCORE_PERQUERY")),
    )
    corpora = os.environ.get("SEMDEX_SCORE_CORPORA", "miracl_en_100k_slice,miracl_de_100k_slice").split(",")
    profiles_env = os.environ.get("SEMDEX_SCORE_PROFILES")
    profiles = profiles_env.split(",") if profiles_env else _discover_profiles(corpora)
    embeddings = os.environ.get("SEMDEX_SCORE_EMBEDDINGS", ",".join(_CPU_EMBEDDERS)).split(",")
    k = int(os.environ.get("SEMDEX_SCORE_K", "10"))
    endpoints = _endpoints()
    force = os.environ.get("SEMDEX_SCORE_FORCE") == "1"
    out = Path(os.environ.get("SEMDEX_SCORE_OUT", str(cache / "scores" / "chunk_sweep_scores.json")))
    out.parent.mkdir(parents=True, exist_ok=True)

    results = _read_json(out) or {}
    _reject_corpus_mixing(out, results, corpora)
    # Embedder-OUTER: an ollama model is then loaded once per run instead of once per profile.
    for label in embeddings:
        endpoint = endpoints.get(_EMBED_MODELS[label][0])
        last_model = _run_embedder(
            label, corpora=corpora, profiles=profiles, k=k, endpoint=endpoint, force=force, results=results, out=out
        )
        if last_model:
            _release_ollama(label, last_model, endpoint)
    _print_tables(results)


def _run_embedder(
    label: str,
    *,
    corpora: list[str],
    profiles: list[str],
    k: int,
    endpoint: str | None,
    force: bool,
    results: dict[str, Any],
    out: Path,
) -> str | None:
    """Score every (corpus, profile) for one embedder. Returns its model id, for the release."""
    model_id: str | None = None
    for corpus in corpora:
        for profile in profiles:
            cell = f"{corpus}__{profile}__{_dirsafe(label)}"
            if cell in results and not force:
                print(f"[score] skip (cached) {cell}", flush=True)
                continue
            row = _score_cell(corpus, profile, label, k, endpoint)
            if row is None:
                print(f"[score] not-embedded {cell}", flush=True)
                continue
            results[cell] = stamped(row)
            model_id = (_read_json(_cache_root() / "vectors" / cell / "meta.json") or {}).get("model_id")
            print(
                f"[score] {cell}: nDCG@10={row['ndcg@10']:.4f} "
                f"[{row['ndcg@10_ci_lo']:.3f}, {row['ndcg@10_ci_hi']:.3f}] "
                f"R@10={row['recall@10']:.4f} MRR={row['mrr']:.4f} P@1={row['p@1']:.4f} (n={row['n_queries']})",
                flush=True,
            )
            out.write_text(json.dumps(results, indent=2, sort_keys=True))  # persist incrementally
    return model_id


# --- shared harness API ---------------------------------------------------------------
# Cell naming, per-query IR metrics and the cached query-vector loader, shared by every sweep.
# Public aliases so a strictly-checked sibling can import them without reportPrivateUsage;
# the underscore names stay for the scripts that already import them.
dirsafe = _dirsafe
metrics = _metrics
query_vectors = _query_vectors
qvec_path = _qvec_path
load_uris = _load_uris
__all__ = ["dirsafe", "load_uris", "metrics", "query_vectors", "qvec_path"]


if __name__ == "__main__":
    main()
