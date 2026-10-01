#!/usr/bin/env python
"""German multilingual embedding reference: all registry models on a MIRACL-de slice.

Native-German queries + human judgments (miracl/de/dev via ir_datasets). Streams the 15.9M
corpus once, keeping the judged docs for N queries + a distractor fill (bounded memory), then
ranks every provider by nDCG@10/Recall@10/MRR/P@1 - the same metrics as the English model
matrix, so the two tables are comparable. ollama models need SEMDEX_BENCH_OLLAMA_URL; the rest
run locally (fastembed/model2vec CPU, sentence-transformers CPU).

Env: N_QUERIES (default 30), FILL_DOCS (total slice size, default 2000), OUT (json).
"""

# pyright: basic
# One-off benchmark harness built on numpy + ir_datasets, neither of which ships type stubs;
# strict mode would flood this with reportUnknown* noise for no safety gain (matches how the
# tests/fixtures generator is kept out of strict checking).

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import ir_datasets

from semdex.composition import build_embedding
from semdex.domain.enums import EmbeddingBackend

NDCG_K = 10

# (label, backend, model_id) - mirrors tests/test_e2e_matrix.py::_EMBED_MODELS.
MODELS = [
    ("fastembed:bge-small", EmbeddingBackend.FASTEMBED, "BAAI/bge-small-en-v1.5"),
    ("fastembed:bge-base", EmbeddingBackend.FASTEMBED, "BAAI/bge-base-en-v1.5"),
    ("model2vec:potion-base-8M", EmbeddingBackend.MODEL2VEC, "minishlab/potion-base-8M"),
    ("model2vec:potion-retrieval-32M", EmbeddingBackend.MODEL2VEC, "minishlab/potion-retrieval-32M"),
    ("st:all-MiniLM-L6-v2", EmbeddingBackend.SENTENCE_TRANSFORMERS, "sentence-transformers/all-MiniLM-L6-v2"),
    ("ollama:nomic-embed-text", EmbeddingBackend.OLLAMA, "nomic-embed-text"),
    ("ollama:all-minilm", EmbeddingBackend.OLLAMA, "all-minilm"),
    ("ollama:mxbai-embed-large", EmbeddingBackend.OLLAMA, "mxbai-embed-large"),
    ("ollama:qwen3-embedding-4b", EmbeddingBackend.OLLAMA, "qwen3-embedding:4b"),
    ("ollama:qwen3-embedding-8b", EmbeddingBackend.OLLAMA, "qwen3-embedding:8b"),
]


def _dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def _ndcg(ranked, rel, k):
    gains = [float(rel.get(d, 0)) for d in ranked[:k]]
    ideal = sorted((float(v) for v in rel.values()), reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg else 0.0


def _recall(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def _mrr(ranked, relevant):
    for i, d in enumerate(ranked):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


def load_slice(n_queries: int, fill_docs: int):
    """MIRACL-de dev: N queries with judged docs, streamed corpus filled to fill_docs total."""
    ds = ir_datasets.load("miracl/de/dev")
    qrels: dict[str, dict[str, int]] = {}
    for qr in ds.qrels_iter():
        if qr.relevance > 0:
            qrels.setdefault(qr.query_id, {})[qr.doc_id] = qr.relevance
    kept_qids = list(qrels)[:n_queries]
    qrels = {q: qrels[q] for q in kept_qids}
    queries = {q.query_id: q.text for q in ds.queries_iter() if q.query_id in qrels}
    qrels = {q: r for q, r in qrels.items() if q in queries}
    judged = {d for r in qrels.values() for d in r}

    docs: dict[str, str] = {}
    pending = set(judged)
    fill_budget = max(0, fill_docs - len(judged))
    filled = 0
    t0 = time.perf_counter()
    for i, d in enumerate(ds.docs_iter()):
        did = d.doc_id
        if did in pending:
            pending.discard(did)
        elif filled < fill_budget:
            filled += 1
        elif pending:
            continue  # fill done, still hunting judged stragglers
        else:
            break
        docs[did] = (getattr(d, "title", "") + " " + d.text).strip()
        if i % 500000 == 0:
            print(f"  streamed {i} corpus docs, kept {len(docs)}, {len(pending)} judged left", flush=True)
    print(
        f"slice: {len(queries)} queries, {len(docs)} docs, judged found {len(judged) - len(pending)}/{len(judged)} "
        f"in {time.perf_counter() - t0:.0f}s",
        flush=True,
    )
    return queries, qrels, docs


def cosine_rank(qvec, doc_ids, doc_mat):
    import numpy as np

    q = np.asarray(qvec, dtype="float32")
    q /= np.linalg.norm(q) or 1.0
    sims = doc_mat @ q
    order = np.argsort(-sims)
    return [doc_ids[i] for i in order[:NDCG_K]]


def evaluate(model_label, backend, *, model_id, queries, qrels, docs, endpoint):
    import numpy as np

    embedding = build_embedding(backend, model=model_id, endpoint=endpoint, allow_fallback=False)
    doc_ids = list(docs)
    texts = [docs[d] for d in doc_ids]
    # Embed in sub-batches: one giant /api/embed call for a slow model (qwen3) can exceed the
    # ollama adapter's 60s HTTP timeout. Small batches keep each call well under it.
    embed_batch = int(os.environ.get("EMBED_BATCH", "128"))
    t0 = time.perf_counter()
    vecs = []
    for i in range(0, len(texts), embed_batch):
        vecs.extend(embedding.embed_passages(texts[i : i + embed_batch]))
    embed_s = time.perf_counter() - t0
    mat = np.asarray(vecs, dtype="float32")
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12

    ndcgs, recalls, rrs, p1s = [], [], [], []
    for qid, text in queries.items():
        rel = qrels[qid]
        relevant = set(rel)
        qvec = embedding.embed_query(text)
        ranked = cosine_rank(qvec, doc_ids, mat)
        ndcgs.append(_ndcg(ranked, rel, NDCG_K))
        recalls.append(_recall(ranked, relevant, NDCG_K))
        rrs.append(_mrr(ranked, relevant))
        p1s.append(1.0 if ranked and ranked[0] in relevant else 0.0)
    n = len(queries)
    return {
        "label": model_label,
        "dim": embedding.dim,
        "ndcg@10": round(sum(ndcgs) / n, 4),
        "recall@10": round(sum(recalls) / n, 4),
        "mrr": round(sum(rrs) / n, 4),
        "p@1": round(sum(p1s) / n, 4),
        "embed_s": round(embed_s, 1),
        "docs_per_s": round(len(doc_ids) / embed_s, 1) if embed_s else 0.0,
    }


def main():
    import json

    n_queries = int(os.environ.get("N_QUERIES", "30"))
    fill_docs = int(os.environ.get("FILL_DOCS", "2000"))
    out = Path(os.environ.get("OUT", "german-bench.json"))
    endpoint = os.environ.get("SEMDEX_BENCH_OLLAMA_URL")
    only = {s.strip() for s in os.environ.get("ONLY_LABELS", "").split(",") if s.strip()}

    # Cache the streamed slice: building it means iterating the 15.9M-doc corpus (~20 min),
    # so a re-run (e.g. to redo a timed-out model) reuses it instead of re-streaming.
    slice_cache = Path(os.environ.get("SLICE_CACHE", str(out) + ".slice.json"))
    if slice_cache.exists():
        s = json.loads(slice_cache.read_text())
        queries, qrels, docs = s["queries"], s["qrels"], s["docs"]
        print(f"loaded cached slice: {len(queries)} queries, {len(docs)} docs", flush=True)
    else:
        queries, qrels, docs = load_slice(n_queries, fill_docs)
        slice_cache.write_text(json.dumps({"queries": queries, "qrels": qrels, "docs": docs}))
        print(f"cached slice to {slice_cache}", flush=True)

    results = []
    for label, backend, model_id in MODELS:
        if only and label not in only:
            continue
        ep = endpoint if backend is EmbeddingBackend.OLLAMA else None
        if backend is EmbeddingBackend.OLLAMA and not endpoint:
            print(f"SKIP {label} (no SEMDEX_BENCH_OLLAMA_URL)", flush=True)
            continue
        try:
            row = evaluate(label, backend, model_id=model_id, queries=queries, qrels=qrels, docs=docs, endpoint=ep)
            results.append(row)
            print(
                f"  {label:34s} dim={row['dim']:>5} nDCG={row['ndcg@10']:.4f} "
                f"R@10={row['recall@10']:.4f} MRR={row['mrr']:.4f} P@1={row['p@1']:.4f} "
                f"({row['docs_per_s']} docs/s)",
                flush=True,
            )
        except Exception as exc:
            print(f"  {label} FAILED: {type(exc).__name__}: {exc}", flush=True)
            results.append({"label": label, "error": f"{type(exc).__name__}: {exc}"})
        out.write_text(
            json.dumps(
                {"corpus": "miracl/de/dev", "n_queries": len(queries), "n_docs": len(docs), "results": results},
                indent=2,
            )
        )
    results_ok = [r for r in results if "ndcg@10" in r]
    results_ok.sort(key=lambda r: -r["ndcg@10"])
    print("\n=== RANKING (nDCG@10, German) ===", flush=True)
    for r in results_ok:
        print(f"  {r['ndcg@10']:.4f}  {r['label']}  (dim {r['dim']})", flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
