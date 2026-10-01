#!/usr/bin/env python
# pyright: basic
"""Score dense, BM25 and hybrid retrieval on the same queries, chunks and judgements.

Every other number in this benchmark set is single-vector dense retrieval with cosine similarity,
which makes every recommendation in the docs a "best DENSE-ONLY configuration". That is the
largest caveat on the whole set (gaps page, entry 1), and this closes it.

The three systems see identical input: the same cached chunk set, the same query set, the same
relevance judgements, and for the dense side the same cached vectors and cached query vectors the
rest of the sweep uses. So the comparison isolates the retrieval method and nothing else, and the
per-query scores are retained, which lets the difference be bootstrapped as a paired comparison
rather than eyeballed between two means.

BM25 depends only on the chunk set, never on the embedding model, so its index is built once per
chunk set and reused across embedders - and its score is identical for every embedder, which is a
useful self-check when reading the output.

Fusion is Reciprocal Rank Fusion (see :mod:`_fusion` for why ranks rather than scores).

Env:
  CACHE_ROOT               cache dir (default /embeddings)
  SEMDEX_HYBRID_CORPORA    comma list (default the two MLDR slices)
  SEMDEX_HYBRID_PROFILES   comma list (default recursive-t256-o0-gpt2)
  SEMDEX_HYBRID_EMBEDDINGS comma list (default the 4 CPU embedders)
  SEMDEX_HYBRID_K          reported top-k (default 10)
  SEMDEX_HYBRID_DEPTH      candidates per system before fusion (default 50)
  SEMDEX_HYBRID_RRF_K      RRF damping constant (default 60)
  SEMDEX_HYBRID_LANGUAGE   override the BM25 analyzer language (default: from the corpus name)
  SEMDEX_HYBRID_OUT        results json (default <cache>/scores/hybrid_scores.json)
  SEMDEX_HYBRID_FORCE      =1 to re-score cells already present
  SEMDEX_BENCH_OLLAMA_URL / SEMDEX_BENCH_OPENAI_URL  only needed if a query cache is cold
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fusion import RRF_K, reciprocal_rank_fusion
from _provenance import stamped
from _score_kernel import topk_stream
from _score_stats import bootstrap_ci
from preembed_vectors import _cache_root, _read_json
from score_chunk_sweep import _dirsafe, _metrics, _query_vectors

_METRICS = ("ndcg@10", "recall@10", "mrr", "p@1")
_CPU_EMBEDDERS = [
    "model2vec:potion-base-8M",
    "model2vec:potion-retrieval-32M",
    "fastembed:bge-small",
    "fastembed:bge-base",
]


def _chunk_columns(parquet: Path) -> tuple[list[str], list[str]]:
    """Chunk texts and their source uris, in vectors.npy row order."""
    texts: list[str] = []
    uris: list[str] = []
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=8192, columns=["text", "source_uri"]):
        data = batch.to_pydict()
        texts.extend(data["text"])
        uris.extend(data["source_uri"])
    return texts, uris


# Corpus name -> snowball stemmer language. A lexical index is only as good as its analyzer, and
# an English analyzer on German text is the same defect this project already documents in
# chonkie's English-distilled default breakpoint model: it would understate BM25 on German and
# quietly bias the dense-versus-lexical comparison in dense's favour.
_LANGUAGE_BY_SUFFIX = {"_de": "german", "_en": "english"}
_DEFAULT_LANGUAGE = "english"


def language_of(corpus: str) -> str:
    """The analyzer language for a corpus, from its name.

    ``mldr_de_3k_slice`` and ``miracl_de_100k_slice`` are German; the rest of the cached corpora
    are English. An unrecognised name falls back to English and says so at the call site, rather
    than silently analysing German with an English stemmer.
    """
    for marker, language in _LANGUAGE_BY_SUFFIX.items():
        if marker in corpus:
            return language
    return _DEFAULT_LANGUAGE


def build_bm25(texts: list[str], language: str) -> tuple[Any, Any]:
    """A BM25 index over the chunk texts, stemmed and stopworded in the corpus language.

    Stemming is on because a lexical system that cannot match "cats" to "cat" understates what
    lexical retrieval contributes, which would bias the very comparison this script exists to make.
    """
    import bm25s
    import Stemmer

    stemmer = Stemmer.Stemmer(language)
    stopwords = "de" if language == "german" else "en"
    tokens = bm25s.tokenize(texts, stopwords=stopwords, stemmer=stemmer, show_progress=False)
    index = bm25s.BM25()
    index.index(tokens, show_progress=False)
    return index, (stemmer, stopwords)


def bm25_rankings(index: Any, analyzer: Any, queries: list[str], depth: int, n_chunks: int) -> np.ndarray:
    """Retrieve with the SAME analyzer the index was built with; a mismatch retrieves nothing."""
    import bm25s

    stemmer, stopwords = analyzer
    tokens = bm25s.tokenize(queries, stopwords=stopwords, stemmer=stemmer, show_progress=False)
    results, _scores = index.retrieve(tokens, k=min(depth, n_chunks), show_progress=False)
    return np.asarray(results)


def _to_documents(rows: Any, uris: list[str], limit: int) -> list[str]:
    """Deduplicate a chunk ranking down to distinct documents, order preserved."""
    seen: list[str] = []
    for row in rows:
        uri = uris[int(row)]
        if uri not in seen:
            seen.append(uri)
            if len(seen) >= limit:
                break
    return seen


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
    import hashlib

    root = Path(os.environ.get("SEMDEX_SCORE_PERQUERY", str(_cache_root() / "scores" / "perquery")))
    root.mkdir(parents=True, exist_ok=True)
    qids = sorted(per_query)
    path = root / f"{cell}.npz"

    def column(metric: str) -> np.ndarray:
        return np.asarray([per_query[q][metric] for q in qids], dtype=np.float32)

    # Explicit keywords, not a **dict: unpacking an untyped mapping into savez widens into its
    # allow_pickle parameter. The npz keys are the identifier-safe aliases the metric names lack.
    np.savez(
        path,
        qids=np.asarray(qids),
        ndcg=column("ndcg@10"),
        recall=column("recall@10"),
        mrr=column("mrr"),
        p1=column("p@1"),
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_cell(context: dict[str, Any], label: str) -> dict[str, dict[str, Any]] | None:
    """Score one (corpus, profile, embedder) three ways. Returns rows keyed by method."""
    corpus, profile = context["corpus"], context["profile"]
    vroot = _cache_root() / "vectors" / f"{corpus}__{profile}__{_dirsafe(label)}"
    meta = _read_json(vroot / "meta.json")
    if not meta or "count" not in meta:
        return None

    qids, query_matrix = _query_vectors(
        corpus, label, meta["model_id"], context["endpoints"].get(label), context["queries"]
    )
    dense_index, _ = topk_stream(vroot / "vectors.npy", query_matrix, fetch=context["depth"])
    uris, qrels, k, depth = context["uris"], context["qrels"], context["k"], context["depth"]

    per_query: dict[str, dict[str, dict[str, float]]] = {"dense": {}, "bm25": {}, "hybrid": {}}
    for position, qid in enumerate(qids):
        rels = qrels.get(qid, {})
        dense_docs = _to_documents(dense_index[position], uris, depth)
        lexical_docs = _to_documents(context["bm25_rows"][position], uris, depth)
        fused = reciprocal_rank_fusion([dense_docs, lexical_docs], k=context["rrf_k"])
        per_query["dense"][qid] = _metrics(dense_docs[:k], rels, k)
        per_query["bm25"][qid] = _metrics(lexical_docs[:k], rels, k)
        per_query["hybrid"][qid] = _metrics(fused[:k], rels, k)

    rows: dict[str, dict[str, Any]] = {}
    for method, scores in per_query.items():
        cell = f"{corpus}__{profile}__{_dirsafe(label)}__{method}"
        rows[method] = {
            "corpus": corpus,
            "profile": profile,
            # The embedder label is carried on the bm25 row too, even though BM25 ignores it. It
            # makes dense / bm25 / hybrid for one cell differ in EXACTLY one field, which is what
            # lets them be compared pairwise; and the bm25 row repeating identically across
            # embedders is a free check that the lexical side really is embedder-independent.
            "embedding": label,
            "method": method,
            "dim": int(meta["dim"]),
            **_summarize(scores),
            "perquery_sha256": _write_per_query(cell, scores),
        }
    return rows


def _context_for(corpus: str, profile: str, config: dict[str, Any]) -> dict[str, Any] | None:
    chunks_dir = _cache_root() / "chunks" / f"{corpus}__{profile}"
    parquet = chunks_dir / "chunks.parquet"
    if not parquet.exists():
        return None
    queries = _read_json(chunks_dir / "queries.json") or {}
    qrels = _read_json(chunks_dir / "qrels.json") or {}
    texts, uris = _chunk_columns(parquet)
    language = os.environ.get("SEMDEX_HYBRID_LANGUAGE") or language_of(corpus)
    print(
        f"[hybrid] {corpus}__{profile}: indexing {len(texts):,} chunks for BM25 ({language} analyzer) ...",
        flush=True,
    )
    index, analyzer = build_bm25(texts, language)
    qids = sorted(queries)
    rows = bm25_rankings(index, analyzer, [queries[q] for q in qids], config["depth"], len(texts))
    del texts  # the index holds what it needs; the raw texts are large
    return {
        "corpus": corpus,
        "profile": profile,
        "queries": queries,
        "qrels": qrels,
        "uris": uris,
        "bm25_rows": rows,
        "endpoints": config["endpoints"],
        "k": config["k"],
        "depth": config["depth"],
        "rrf_k": config["rrf_k"],
    }


def _endpoints() -> dict[str, str | None]:
    from preembed_vectors import _EMBED_MODELS

    from semdex.domain.enums import EmbeddingBackend

    ollama = os.environ.get("SEMDEX_BENCH_OLLAMA_URL")
    openai = os.environ.get("SEMDEX_BENCH_OPENAI_URL")
    resolved: dict[str, str | None] = {}
    for label, spec in _EMBED_MODELS.items():
        backend = spec[0]
        if backend is EmbeddingBackend.OLLAMA:
            resolved[label] = ollama
        elif backend is EmbeddingBackend.OPENAI:
            resolved[label] = openai
        else:
            resolved[label] = None
    return resolved


def main() -> None:
    cache = _cache_root()
    config = {
        "k": int(os.environ.get("SEMDEX_HYBRID_K", "10")),
        "depth": int(os.environ.get("SEMDEX_HYBRID_DEPTH", "50")),
        "rrf_k": int(os.environ.get("SEMDEX_HYBRID_RRF_K", str(RRF_K))),
        "endpoints": _endpoints(),
    }
    corpora = os.environ.get("SEMDEX_HYBRID_CORPORA", "mldr_en_8k_slice,mldr_de_3k_slice").split(",")
    profiles = os.environ.get("SEMDEX_HYBRID_PROFILES", "recursive-t256-o0-gpt2").split(",")
    embeddings = os.environ.get("SEMDEX_HYBRID_EMBEDDINGS", ",".join(_CPU_EMBEDDERS)).split(",")
    force = os.environ.get("SEMDEX_HYBRID_FORCE") == "1"
    out = Path(os.environ.get("SEMDEX_HYBRID_OUT", str(cache / "scores" / "hybrid_scores.json")))
    out.parent.mkdir(parents=True, exist_ok=True)
    results = _read_json(out) or {}

    for corpus in corpora:
        for profile in profiles:
            context = _context_for(corpus, profile, config)
            if context is None:
                print(f"[hybrid] no chunk set for {corpus}__{profile}", flush=True)
                continue
            for label in embeddings:
                key = f"{corpus}__{profile}__{_dirsafe(label)}"
                if f"{key}__hybrid" in results and not force:
                    print(f"[hybrid] skip (cached) {key}", flush=True)
                    continue
                rows = score_cell(context, label)
                if rows is None:
                    print(f"[hybrid] not-embedded {key}", flush=True)
                    continue
                for method, row in rows.items():
                    results[f"{key}__{method}"] = stamped(row)
                print(
                    f"[hybrid] {key}: dense={rows['dense']['ndcg@10']:.4f} "
                    f"bm25={rows['bm25']['ndcg@10']:.4f} hybrid={rows['hybrid']['ndcg@10']:.4f}",
                    flush=True,
                )
                out.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
