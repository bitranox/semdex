#!/usr/bin/env python
"""What the Qwen3 instruction prefix is worth, measured pairwise on the same queries.

Qwen3-Embedding is trained to receive its queries wrapped in an instruction and its passages
bare. Every qwen3 figure this repo published was taken with neither, which is at least
self-consistent but is not the configuration the model's own numbers assume. This measures the
difference instead of leaving it as a caveat.

Three arms, so the asymmetry is tested rather than trusted:

  plain      no prefix anywhere. What the published rows measured.
  query      the instruction on the query only. What the model card documents.
  both       the instruction on the query AND the passage. The control: the card says
             "No need to add instruction for retrieval documents", and this is what says
             whether that is a real asymmetry or just a default nobody questioned.

Passages are embedded ONCE for the two arms that leave them bare, because those arms send the
server byte-identical text - a property of the adapter rather than an assumption, so the cost is
two passage passes and three query passes rather than six.

Comparisons are PAIRED over the same query ids with a bootstrap interval, because the arms differ
by far less than the query-to-query spread and comparing two means cannot resolve that.

Env: MODEL (default qwen3-embedding:4b), ENDPOINT, DOCS, QUERIES, EMBED_BATCH, NUM_BATCH, OUT.
"""

# pyright: basic
# One-off benchmark harness on untyped deps (ir_datasets); strict mode would only add
# reportUnknown* noise (matches the other bench scripts).

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _score_stats import bootstrap_ci, paired_ci

from semdex.composition import build_embedding
from semdex.domain.enums import EmbeddingBackend

# The model card's own default task description, quoted rather than paraphrased. The format is
# f"Instruct: {task}\nQuery:{query}" - note there is NO space after "Query:", which is easy to
# add by reflex and would send the model text it was not trained on.
TASK = "Given a web search query, retrieve relevant passages that answer the query"
QUERY_PREFIX = f"Instruct: {TASK}\nQuery:"

K = 10


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg(ranked: list[str], rel: dict[str, int], k: int = K) -> float:
    gains = [float(rel.get(d, 0)) for d in ranked[:k]]
    ideal = sorted((float(v) for v in rel.values()), reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg else 0.0


def recall(ranked: list[str], relevant: set[str], k: int = K) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def rr(ranked: list[str], relevant: set[str]) -> float:
    for i, d in enumerate(ranked):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


def load_slice(n_queries: int, n_docs: int) -> tuple[dict[str, str], dict[str, dict[str, int]], dict[str, str]]:
    """NFCorpus test: judged documents first, then filled to ``n_docs`` with distractors.

    Judged documents are taken unconditionally. Without them a query has nothing to find and the
    run measures the fill rather than the model.
    """
    import ir_datasets

    ds = ir_datasets.load("beir/nfcorpus/test")
    qrels: dict[str, dict[str, int]] = {}
    for qr in ds.qrels_iter():
        if qr.relevance > 0:
            qrels.setdefault(qr.query_id, {})[qr.doc_id] = qr.relevance
    kept = list(qrels)[:n_queries]
    qrels = {q: qrels[q] for q in kept}
    queries = {q.query_id: q.text for q in ds.queries_iter() if q.query_id in qrels}
    qrels = {q: r for q, r in qrels.items() if q in queries}
    judged = {d for r in qrels.values() for d in r}

    docs: dict[str, str] = {}
    filled = 0
    budget = max(0, n_docs - len(judged))
    for d in ds.docs_iter():
        if d.doc_id in judged:
            pass
        elif filled < budget:
            filled += 1
        else:
            continue
        docs[d.doc_id] = (getattr(d, "title", "") + " " + d.text).strip()
    print(f"slice: {len(queries)} queries, {len(docs)} docs ({len(judged)} judged)", flush=True)
    return queries, qrels, docs


def provider_for(query_prefix: str, passage_prefix: str) -> Any:
    """An ollama provider through the REAL composition path, not a hand-rolled request.

    Going through build_embedding is the point: it is what a deployment uses, so a prefix that
    fails to reach the wire fails here too rather than being papered over by the benchmark.
    """
    return build_embedding(
        EmbeddingBackend.OLLAMA,
        model=os.environ.get("MODEL", "qwen3-embedding:4b"),
        endpoint=os.environ.get("ENDPOINT", "http://px-semdex-test-embeddings:11434"),
        # ollama truncates any input past its physical batch SILENTLY, at HTTP 200. The
        # instruction makes every query longer, so this run is exactly the shape that trips it.
        num_batch=int(os.environ.get("NUM_BATCH", "4096")),
        timeout=float(os.environ.get("TIMEOUT", "120")),
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
        allow_fallback=False,
    )


def embed_all(provider: Any, texts: list[str], label: str) -> Any:
    """Embed in sub-batches: one giant call to a slow model exceeds the adapter's HTTP timeout."""
    import numpy as np

    batch = int(os.environ.get("EMBED_BATCH", "128"))
    started = time.perf_counter()
    out: list[Any] = []
    for i in range(0, len(texts), batch):
        out.extend(provider.embed_passages(texts[i : i + batch]))
        if i and i % (batch * 8) == 0:
            print(f"  {label}: {i}/{len(texts)}", flush=True)
    matrix = np.asarray(out, dtype="float32")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    print(f"  {label}: {len(texts)} in {time.perf_counter() - started:.0f}s", flush=True)
    return matrix


def score_arm(
    provider: Any,
    doc_ids: list[str],
    doc_matrix: Any,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
) -> dict[str, dict[str, float]]:
    """Per-query scores, keyed by query id so the arms can be differenced pairwise."""
    import numpy as np

    per_query: dict[str, dict[str, float]] = {}
    for qid, text in queries.items():
        vector = np.asarray(provider.embed_query(text), dtype="float32")
        vector /= np.linalg.norm(vector) or 1.0
        order = np.argsort(-(doc_matrix @ vector))[:100]
        ranked = [doc_ids[i] for i in order]
        rel = qrels[qid]
        relevant = set(rel)
        per_query[qid] = {
            "ndcg@10": ndcg(ranked, rel),
            "recall@10": recall(ranked, relevant),
            "mrr": rr(ranked, relevant),
        }
    return per_query


def summarize(per_query: dict[str, dict[str, float]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric in ("ndcg@10", "recall@10", "mrr"):
        stats = bootstrap_ci([v[metric] for v in per_query.values()])
        out[metric] = round(stats["mean"], 4)
        out[f"{metric}_ci_lo"] = round(stats["ci_lo"], 4)
        out[f"{metric}_ci_hi"] = round(stats["ci_hi"], 4)
    out["queries"] = len(per_query)
    return out


def main() -> None:
    queries, qrels, docs = load_slice(int(os.environ.get("QUERIES", "323")), int(os.environ.get("DOCS", "3633")))
    doc_ids = list(docs)
    texts = [docs[d] for d in doc_ids]

    bare = provider_for("", "")
    prefixed_query = provider_for(QUERY_PREFIX, "")
    prefixed_both = provider_for(QUERY_PREFIX, QUERY_PREFIX)

    # The two bare-passage arms send byte-identical passage text, so they share one pass. Only
    # the "both" arm needs its own, and that is the whole reason this measurement is cheap.
    plain_docs = embed_all(bare, texts, "passages (bare)")
    prefixed_docs = embed_all(prefixed_both, texts, "passages (prefixed)")

    arms = {
        "plain": score_arm(bare, doc_ids, plain_docs, queries, qrels),
        "query": score_arm(prefixed_query, doc_ids, plain_docs, queries, qrels),
        "both": score_arm(prefixed_both, doc_ids, prefixed_docs, queries, qrels),
    }
    payload: dict[str, Any] = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": os.environ.get("MODEL", "qwen3-embedding:4b"),
        "corpus": "beir/nfcorpus/test",
        "task_description": TASK,
        "query_prefix": QUERY_PREFIX,
        "docs": len(doc_ids),
        "arms": {name: summarize(scores) for name, scores in arms.items()},
        "paired": {},
    }
    # both_vs_query is the one that tests the model card's own instruction rather than confirming
    # it. The card says a passage needs no instruction; if prefixing it anyway resolves as an
    # improvement, the card is wrong here, and if it does not resolve, the asymmetry costs nothing
    # and the simpler configuration wins. Either way it has to be measured, not assumed - and
    # eyeballing two means that differ by 0.005 is exactly the comparison a paired test exists for.
    for name, base in (("query", "plain"), ("both", "plain"), ("both", "query")):
        for metric in ("ndcg@10", "recall@10"):
            left = {q: v[metric] for q, v in arms[name].items()}
            right = {q: v[metric] for q, v in arms[base].items()}
            verdict = paired_ci(left, right)
            payload["paired"][f"{name}_vs_{base}/{metric}"] = {
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in verdict.items()
            }
    # Persist the per-query scores so a later question can be answered without re-embedding the
    # corpus, which is the same reason the CI gate stores them.
    payload["per_query"] = {
        name: {q: {m: round(v, 6) for m, v in s.items()} for q, s in scores.items()} for name, scores in arms.items()
    }
    out_path = Path(os.environ.get("OUT", "qwen3-instruction.json"))
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["arms"], indent=2), flush=True)
    print(json.dumps(payload["paired"], indent=2), flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
