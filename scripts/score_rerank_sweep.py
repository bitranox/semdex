#!/usr/bin/env python
# pyright: basic
"""Rerank each retrieval method's candidates with a cross-encoder, and measure what it adds.

The hybrid page closes half of the first gap; this closes the other half. A cross-encoder reads
the query and a passage TOGETHER, so it can judge relevance a bi-encoder cannot: the bi-encoder
must commit each side to a vector before it has seen the other. It is far too slow to score a
corpus, which is why it runs as a second stage over a first stage's shortlist, and why the
question worth measuring is what it adds ON TOP of dense, BM25 and their fusion.

Reranking happens at the DOCUMENT level, on exactly the ranking each method already produced: the
top ``DEPTH`` documents, each represented by the chunk that earned it its place. That keeps the
comparison honest - the reranked run and its baseline differ only by the reranking step, not by a
different candidate-generation path - and it matches what the metrics score, which is documents.

Cross-encoder scores depend only on (query, chunk), never on which method proposed the pair or
which embedder produced it, so every pair is scored ONCE and reused across methods and embedders.
On these corpora that is roughly a third of the work a naive pass would do.

Env:
  CACHE_ROOT               cache dir (default /embeddings)
  SEMDEX_RERANK_MODEL      cross-encoder (default BAAI/bge-reranker-base)
  SEMDEX_RERANK_CORPORA    comma list (default the two MLDR slices)
  SEMDEX_RERANK_PROFILES   comma list (default recursive-t256-o0-gpt2)
  SEMDEX_RERANK_EMBEDDINGS comma list (default fastembed:bge-base)
  SEMDEX_RERANK_DEPTH      documents reranked per query (default 20)
  SEMDEX_RERANK_K          reported top-k (default 10)
  SEMDEX_RERANK_BATCH      cross-encoder batch size (default 32)
  SEMDEX_RERANK_THREADS    torch CPU threads (default: all)
  SEMDEX_RERANK_MAXLEN     cross-encoder max sequence length (default 512)
  SEMDEX_RERANK_OUT        results json (default <cache>/scores/rerank_scores.json)
  SEMDEX_RERANK_FORCE      =1 to re-score cells already present
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fusion import reciprocal_rank_fusion
from _optional_ml import CrossEncoderModel, load_cross_encoder, set_torch_threads
from _provenance import stamped
from _score_kernel import topk_stream
from _score_stats import bootstrap_ci
from preembed_vectors import _cache_root, _read_json
from score_chunk_sweep import _dirsafe, _metrics, _query_vectors
from score_hybrid_sweep import _chunk_columns, bm25_rankings, build_bm25, language_of

_METRICS = ("ndcg@10", "recall@10", "mrr", "p@1")
_BASE_METHODS = ("dense", "bm25", "hybrid")


def load_reranker(name: str) -> CrossEncoderModel:
    threads = int(os.environ.get("SEMDEX_RERANK_THREADS", "0"))
    if threads:
        set_torch_threads(threads)
    return load_cross_encoder(name, max_length=int(os.environ.get("SEMDEX_RERANK_MAXLEN", "512")))


def _documents_with_source(rows: Any, uris: list[str], limit: int) -> list[tuple[str, int]]:
    """Each distinct document in ranking order, with the chunk row that earned it its place.

    The representative chunk is the highest-ranked one for that document, which is the passage the
    retrieval stage actually matched, and therefore the passage the reranker should judge.
    """
    seen: dict[str, int] = {}
    for row in rows:
        uri = uris[int(row)]
        if uri not in seen:
            seen[uri] = int(row)
            if len(seen) >= limit:
                break
    return list(seen.items())


class ScoreCache:
    """(query id, chunk row) -> cross-encoder score, computed once per corpus."""

    def __init__(self, model: CrossEncoderModel, batch: int) -> None:
        self._model = model
        self._batch = batch
        self._scores: dict[tuple[str, int], float] = {}
        self.computed = 0

    def ensure(self, wanted: list[tuple[str, int]], queries: dict[str, str], texts: list[str]) -> None:
        missing = [key for key in dict.fromkeys(wanted) if key not in self._scores]
        if not missing:
            return
        pairs = [(queries[qid], texts[row]) for qid, row in missing]
        started = time.perf_counter()
        values = self._model.predict(pairs, batch_size=self._batch, show_progress_bar=False)
        elapsed = time.perf_counter() - started
        for key, value in zip(missing, values, strict=True):
            self._scores[key] = float(value)
        self.computed += len(missing)
        print(
            f"    reranked {len(missing):,} pairs in {elapsed:.0f}s ({len(missing) / max(elapsed, 1e-9):.1f}/s)",
            flush=True,
        )

    def order(self, qid: str, docs: list[tuple[str, int]]) -> list[str]:
        return [uri for uri, _row in sorted(docs, key=lambda d: -self._scores[(qid, d[1])])]


def _summarize(per_query: dict[str, dict[str, float]]) -> dict[str, Any]:
    row: dict[str, Any] = {"n_queries": len(per_query)}
    for metric in _METRICS:
        stats = bootstrap_ci([scores[metric] for scores in per_query.values()])
        row[metric] = stats["mean"]
        row[f"{metric}_ci_lo"] = stats["ci_lo"]
        row[f"{metric}_ci_hi"] = stats["ci_hi"]
        row[f"{metric}_sd"] = stats["sd"]
    return row


def _write_per_query(cell: str, per_query: dict[str, dict[str, float]]) -> str:
    root = Path(os.environ.get("SEMDEX_SCORE_PERQUERY", str(_cache_root() / "scores" / "perquery")))
    root.mkdir(parents=True, exist_ok=True)
    qids = sorted(per_query)

    def column(metric: str) -> np.ndarray:
        return np.asarray([per_query[q][metric] for q in qids], dtype=np.float32)

    path = root / f"{cell}.npz"
    np.savez(
        path,
        qids=np.asarray(qids),
        ndcg=column("ndcg@10"),
        recall=column("recall@10"),
        mrr=column("mrr"),
        p1=column("p@1"),
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidates(context: dict[str, Any], label: str) -> tuple[list[str], dict[str, dict[str, list[tuple[str, int]]]]]:
    """Per query, each method's top-DEPTH documents with their representative chunk."""
    corpus, profile = context["corpus"], context["profile"]
    vroot = _cache_root() / "vectors" / f"{corpus}__{profile}__{_dirsafe(label)}"
    meta = _read_json(vroot / "meta.json")
    if not meta or "count" not in meta:
        return [], {}
    qids, query_matrix = _query_vectors(corpus, label, meta["model_id"], None, context["queries"])
    dense_index, _ = topk_stream(vroot / "vectors.npy", query_matrix, fetch=context["fetch"])
    uris, depth = context["uris"], context["depth"]

    per_query: dict[str, dict[str, list[tuple[str, int]]]] = {}
    for position, qid in enumerate(qids):
        dense = _documents_with_source(dense_index[position], uris, depth)
        lexical = _documents_with_source(context["bm25_rows"][position], uris, depth)
        # dense last, so it wins a document both systems returned: its chunk is the one the
        # embedding actually matched, and the reranker should judge that passage.
        source = dict([*lexical, *dense])
        fused = reciprocal_rank_fusion([[u for u, _ in dense], [u for u, _ in lexical]])[:depth]
        per_query[qid] = {
            "dense": dense,
            "bm25": lexical,
            "hybrid": [(uri, source[uri]) for uri in fused],
        }
    return qids, per_query


def score_cell(context: dict[str, Any], label: str, cache: ScoreCache) -> dict[str, dict[str, Any]]:
    qids, candidates = _candidates(context, label)
    if not qids:
        return {}
    wanted = [(qid, row) for qid in qids for method in _BASE_METHODS for _uri, row in candidates[qid][method]]
    cache.ensure(wanted, context["queries"], context["texts"])

    k, qrels, depth = context["k"], context["qrels"], context["depth"]
    rows: dict[str, dict[str, Any]] = {}
    # Both the reranked run AND its matched baseline come from the SAME candidate list at the SAME
    # depth, so the pair differs by exactly one thing: the reranking. Comparing against the
    # depth-50 numbers on the retrieval-method page would confound the reranker with the shortlist
    # size, and the method names carry the depth so the two runs can never be mistaken for each
    # other.
    plans = [(f"{m}@{depth}", m, False) for m in _BASE_METHODS]
    plans += [(f"{m}@{depth}+rerank", m, True) for m in _BASE_METHODS]
    for name, method, reranked in plans:
        per_query = {
            qid: _metrics(
                (cache.order(qid, candidates[qid][method]) if reranked else [u for u, _ in candidates[qid][method]])[
                    :k
                ],
                qrels.get(qid, {}),
                k,
            )
            for qid in qids
        }
        cell = f"{context['corpus']}__{context['profile']}__{_dirsafe(label)}__{name}"
        rows[name] = {
            "corpus": context["corpus"],
            "profile": context["profile"],
            "embedding": label,
            "method": name,
            "dim": 0,
            "reranker": context["model_name"] if reranked else "none",
            "rerank_depth": context["depth"],
            **_summarize(per_query),
            "perquery_sha256": _write_per_query(cell, per_query),
        }
    return rows


def _context_for(corpus: str, profile: str, config: dict[str, Any]) -> dict[str, Any] | None:
    chunks_dir = _cache_root() / "chunks" / f"{corpus}__{profile}"
    if not (chunks_dir / "chunks.parquet").exists():
        return None
    queries = _read_json(chunks_dir / "queries.json") or {}
    texts, uris = _chunk_columns(chunks_dir / "chunks.parquet")
    language = language_of(corpus)
    print(f"[rerank] {corpus}__{profile}: BM25 index over {len(texts):,} chunks ({language}) ...", flush=True)
    index, analyzer = build_bm25(texts, language)
    qids = sorted(queries)
    rows = bm25_rankings(index, analyzer, [queries[q] for q in qids], config["fetch"], len(texts))
    return {
        "corpus": corpus,
        "profile": profile,
        "queries": queries,
        "qrels": _read_json(chunks_dir / "qrels.json") or {},
        "texts": texts,
        "uris": uris,
        "bm25_rows": rows,
        **{key: config[key] for key in ("k", "depth", "fetch", "model_name")},
    }


def main() -> None:
    cache_root = _cache_root()
    depth = int(os.environ.get("SEMDEX_RERANK_DEPTH", "20"))
    config = {
        "k": int(os.environ.get("SEMDEX_RERANK_K", "10")),
        "depth": depth,
        # Over-fetch chunks so DEPTH distinct documents survive deduplication.
        "fetch": depth * 20,
        "model_name": os.environ.get("SEMDEX_RERANK_MODEL", "BAAI/bge-reranker-base"),
    }
    corpora = os.environ.get("SEMDEX_RERANK_CORPORA", "mldr_en_8k_slice,mldr_de_3k_slice").split(",")
    profiles = os.environ.get("SEMDEX_RERANK_PROFILES", "recursive-t256-o0-gpt2").split(",")
    embeddings = os.environ.get("SEMDEX_RERANK_EMBEDDINGS", "fastembed:bge-base").split(",")
    force = os.environ.get("SEMDEX_RERANK_FORCE") == "1"
    out = Path(os.environ.get("SEMDEX_RERANK_OUT", str(cache_root / "scores" / "rerank_scores.json")))
    out.parent.mkdir(parents=True, exist_ok=True)
    results = _read_json(out) or {}

    print(f"[rerank] loading {config['model_name']} ...", flush=True)
    model = load_reranker(str(config["model_name"]))

    for corpus in corpora:
        for profile in profiles:
            context = _context_for(corpus, profile, config)
            if context is None:
                print(f"[rerank] no chunk set for {corpus}__{profile}", flush=True)
                continue
            # One cache per corpus: a (query, chunk) score is the same whichever method or
            # embedder proposed the pair.
            cache = ScoreCache(model, int(os.environ.get("SEMDEX_RERANK_BATCH", "32")))
            for label in embeddings:
                key = f"{corpus}__{profile}__{_dirsafe(label)}"
                if f"{key}__hybrid@{config['depth']}+rerank" in results and not force:
                    print(f"[rerank] skip (cached) {key}", flush=True)
                    continue
                rows = score_cell(context, label, cache)
                if not rows:
                    print(f"[rerank] not-embedded {key}", flush=True)
                    continue
                results.update({f"{key}__{name}": stamped(row) for name, row in rows.items()})
                print(
                    f"[rerank] {key}: "
                    + "  ".join(f"{name}={row['ndcg@10']:.4f}" for name, row in sorted(rows.items())),
                    flush=True,
                )
                out.write_text(json.dumps(results, indent=2, sort_keys=True))
            print(f"[rerank] {corpus}: {cache.computed:,} unique pairs scored", flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
