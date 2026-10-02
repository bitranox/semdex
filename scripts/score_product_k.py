#!/usr/bin/env python3
"""Score the chunk list ``semdex search`` delivers, not the deduplicated document list.

Every chunking verdict in docs/benchmarks/03-chunking.md is nDCG@10 over DOCUMENTS: the chunk-sweep
scorer over-fetches chunks, deduplicates to ten distinct documents, and scores that. The product
returns ``default_k`` chunks with no deduplication, so five slots can fill with one document's
neighbouring chunks where the document metric sees one hit. This scorer measures both views from
the SAME top list per query, at the product's k and at fixed retrieved-token budgets, so a chunk
size can be compared at equal tokens read.

Model-free by design: it reads the query-vector cache the chunk-sweep scorer wrote and REFUSES a
cell whose cache is absent, so it never loads an embedder and can share a box with a sweep.

Env (all optional):
  SEMDEX_PRODUCT_CORPORA     comma list (default gerdalir_de_12k_slice,mldr_de_3k_slice,mldr_en_8k_slice)
  SEMDEX_PRODUCT_PROFILES    comma list of profile tags (default: every cap-256 zero-overlap profile
                             present for the corpus, plus recursive-t64/t128/t512-o0-gpt2)
  SEMDEX_PRODUCT_EMBEDDINGS  comma list of embedder labels (default: the six sweep embedders)
  SEMDEX_PRODUCT_K           the product's k (default 5, the shipped [index].default_k)
  SEMDEX_PRODUCT_BUDGETS     comma list of retrieved-token budgets (default 1280,2560)
  SEMDEX_PRODUCT_FETCH       chunks streamed per query (default 100; raised to the largest rung k)
  SEMDEX_PRODUCT_FORCE       1 to re-score cells already in the results file
  SEMDEX_SCORE_OUT           results json (default <cache>/scores/product_k_scores.json)
  SEMDEX_PRODUCT_K_PERQUERY  per-query npz dir (default <cache>/scores/perquery-product-k)
  SEMDEX_SCORE_QVECS         query-vector cache dir (default <cache>/scores/qvecs)
  SEMDEX_SCORE_BLOCK_BYTES   streaming block size for topk_stream
  CACHE_ROOT                 the vector cache root (default /embeddings)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import the sibling driver helpers

from _perquery import write_per_query
from _provenance import stamped
from _score_kernel import topk_stream
from _score_stats import bootstrap_ci, paired_ci
from audit_chunk_dimensions import parse_profile
from preembed_vectors import _cache_root, _ndcg, _read_json  # pyright: ignore[reportPrivateUsage]
from score_chunk_sweep import dirsafe, load_uris, qvec_path

LADDER_CAPS = (64, 128, 256, 512)
DEFAULT_PRODUCT_K = 5
DEFAULT_BUDGETS = (1280, 2560)
DEFAULT_FETCH = 100
DEFAULT_CORPORA = ("gerdalir_de_12k_slice", "mldr_de_3k_slice", "mldr_en_8k_slice")
DEFAULT_EMBEDDINGS = (
    "model2vec:potion-retrieval-32M",
    "model2vec:potion-base-8M",
    "fastembed:bge-base",
    "ollama:bge-m3",
    "ollama:qwen3-embedding-4b",
    "ollama:qwen3-embedding-8b",
)
LADDER_PROFILES = ("recursive-t64-o0-gpt2", "recursive-t128-o0-gpt2", "recursive-t512-o0-gpt2")
# A repeated document's later slot in the delivered view. Never a document id, so it earns no gain.
REPEAT = ""
_DEFAULT_BLOCK_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class Rung:
    """One k at which a cell is scored, and why: a budget rung, the product k, or both."""

    k: int
    budgets: tuple[int, ...]
    product: bool


def delivered_view(chunk_docs: list[str], k: int) -> list[str]:
    """The first k chunks as the product hands them back: a repeated document becomes REPEAT."""
    seen: set[str] = set()
    view: list[str] = []
    for doc in chunk_docs[:k]:
        view.append(REPEAT if doc in seen else doc)
        seen.add(doc)
    return view


def _documents_view(chunk_docs: list[str], k: int) -> list[str]:
    """The ranking deduplicated to k distinct documents, in order (the chunk-sweep scorer's unit)."""
    seen: list[str] = []
    for doc in chunk_docs:
        if doc not in seen:
            seen.append(doc)
            if len(seen) >= k:
                break
    return seen


def ndcg_delivered(chunk_docs: list[str], rels: dict[str, int], k: int) -> float:
    """nDCG@k over the delivered view: a slot spent on a document already delivered gains nothing."""
    return _ndcg(delivered_view(chunk_docs, k), rels, k)


def ndcg_documents(chunk_docs: list[str], rels: dict[str, int], k: int) -> float:
    """nDCG@k over k distinct documents, as every other chunking table scores."""
    return _ndcg(_documents_view(chunk_docs, k), rels, k)


def distinct_docs(chunk_docs: list[str], k: int) -> int:
    """How many distinct documents the first k chunks name."""
    return len(set(chunk_docs[:k]))


def rungs_for_cap(cap: int, *, product_k: int, budgets: list[int]) -> list[Rung]:
    """The ks a cell of this cap is scored at, sorted by k: the product k plus each budget's k."""
    by_k: dict[int, list[int]] = {product_k: []}
    if cap in LADDER_CAPS:
        for budget in budgets:
            k = budget // cap
            if k >= 1:
                by_k.setdefault(k, []).append(budget)
    return [Rung(k=k, budgets=tuple(sorted(by_k[k])), product=(k == product_k)) for k in sorted(by_k)]


def cached_query_vectors(corpus: str, label: str, queries: dict[str, str]) -> tuple[list[str], np.ndarray]:
    """The query vectors the chunk-sweep scorer cached for this (corpus, embedder); refused if absent.

    Reading the npz directly rather than through ``score_chunk_sweep.query_vectors`` is the point:
    that seam EMBEDS on a miss, which would load a model on a box that may be running a sweep.
    """
    path = qvec_path(corpus, label, queries)
    qids = sorted(queries)
    if not path.exists():
        raise SystemExit(
            f"{corpus}__{dirsafe(label)}: no query-vector cache at {path}; score this corpus with "
            "scripts/score_chunk_sweep.py first (this scorer never embeds)"
        )
    with np.load(path) as data:
        cached = [str(q) for q in data["qids"]]
        if cached != qids:
            raise SystemExit(f"{corpus}__{dirsafe(label)}: query-vector cache at {path} holds a different query set")
        return qids, np.asarray(data["vectors"], dtype=np.float32)


def _per_query_at(chunk_docs: list[str], rels: dict[str, int], k: int) -> dict[str, float]:
    return {
        "delivered": ndcg_delivered(chunk_docs, rels, k),
        "documents": ndcg_documents(chunk_docs, rels, k),
        "distinct": float(distinct_docs(chunk_docs, k)),
    }


def _summarize_rung(rung: Rung, per_query: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Means with intervals for the three views, and the paired documents-minus-delivered gap."""
    out: dict[str, Any] = {"k": rung.k, "budgets": list(rung.budgets), "product": rung.product}
    for view, key in (("delivered", "ndcg_delivered"), ("documents", "ndcg_documents"), ("distinct", "distinct_docs")):
        stats = bootstrap_ci([scores[view] for scores in per_query.values()])
        out[key] = stats["mean"]
        out[f"{key}_ci_lo"] = stats["ci_lo"]
        out[f"{key}_ci_hi"] = stats["ci_hi"]
        out[f"{key}_sd"] = stats["sd"]
    gap = paired_ci(
        {qid: scores["documents"] for qid, scores in per_query.items()},
        {qid: scores["delivered"] for qid, scores in per_query.items()},
    )
    for key in ("mean_delta", "ci_lo", "ci_hi", "wins", "losses", "ties", "n_shared", "resolved"):
        out[f"gap_{key}"] = gap[key]
    return out


def _write_per_query(
    cell: str, qids: list[str], rungs: list[Rung], per_rung: dict[int, dict[str, dict[str, float]]]
) -> str:
    """Per-query arrays beside the cache, one column per rung, referenced from the row by hash."""
    # Its own directory and its own variable: these files share their names with the dense
    # scorer's, so one directory for both lets either run replace the other's arrays.
    root = Path(os.environ.get("SEMDEX_PRODUCT_K_PERQUERY", str(_cache_root() / "scores" / "perquery-product-k")))
    ks = [rung.k for rung in rungs]

    def matrix(view: str) -> np.ndarray:
        return np.asarray([[per_rung[k][qid][view] for k in ks] for qid in qids], dtype=np.float32)

    arrays = {
        "qids": np.asarray(qids),
        "ks": np.asarray(ks, dtype=np.int32),
        "delivered": matrix("delivered"),
        "documents": matrix("documents"),
        "distinct": matrix("distinct"),
    }
    return write_per_query(root, cell, arrays)


def score_cell(
    corpus: str, profile: str, label: str, *, product_k: int, budgets: list[int], fetch: int
) -> dict[str, Any] | None:
    """Both views at every rung of one cell, from ONE streamed top list per query."""
    cell = f"{corpus}__{profile}__{dirsafe(label)}"
    vroot = _cache_root() / "vectors" / cell
    meta = _read_json(vroot / "meta.json")
    if not meta or "count" not in meta:
        return None  # not embedded yet
    chunks_dir = _cache_root() / "chunks" / f"{corpus}__{profile}"
    queries = _read_json(chunks_dir / "queries.json") or {}
    qrels = _read_json(chunks_dir / "qrels.json") or {}
    qids, query_matrix = cached_query_vectors(corpus, label, queries)
    uris = load_uris(chunks_dir / "chunks.parquet")
    rows = int(np.load(vroot / "vectors.npy", mmap_mode="r").shape[0])  # header read, no data
    if len(uris) != rows:
        raise SystemExit(f"{cell}: uri/vector row mismatch ({len(uris)} vs {rows})")
    axes = parse_profile(profile)
    if "max_tokens" not in axes:
        raise SystemExit(f"{cell}: {axes.get('parse_error', 'profile has no max_tokens axis')}")
    rungs = rungs_for_cap(int(axes["max_tokens"]), product_k=product_k, budgets=budgets)
    fetch = max(fetch, *(rung.k for rung in rungs))
    block_bytes = int(os.environ.get("SEMDEX_SCORE_BLOCK_BYTES", _DEFAULT_BLOCK_BYTES))
    index, _ = topk_stream(vroot / "vectors.npy", query_matrix, fetch=fetch, block_bytes=block_bytes)
    ranked = {qid: [uris[int(r)] for r in index[i]] for i, qid in enumerate(qids)}
    per_rung = {rung.k: {qid: _per_query_at(ranked[qid], qrels.get(qid, {}), rung.k) for qid in qids} for rung in rungs}
    row: dict[str, Any] = {
        "corpus": corpus,
        "profile": profile,
        "embedding": label,
        "dim": int(meta["dim"]),
        "n_queries": len(qids),
        "product_k": product_k,
        "fetch": fetch,
        "rungs": [_summarize_rung(rung, per_rung[rung.k]) for rung in rungs],
    }
    row["perquery_sha256"] = _write_per_query(cell, qids, rungs, per_rung)
    return row


def _discover_profiles(corpus: str) -> list[str]:
    """Every cap-256 zero-overlap profile present for the corpus, plus the recursive ladder rungs.

    Only recipe-free profiles: a recipe cell measures another rule set under the same strategy name,
    so it would sit beside its recipe-free twin as a second, indistinguishable row.
    """
    found: set[str] = set()
    for entry in (_cache_root() / "vectors").glob(f"{corpus}__*"):
        parts = entry.name.split("__")
        if len(parts) != 3:
            continue
        axes = parse_profile(parts[1])
        if axes.get("recipe") or "recipe_unparsed" in axes:
            continue
        if axes.get("max_tokens") == 256 and axes.get("overlap_tokens") == 0:
            found.add(parts[1])
        if parts[1] in LADDER_PROFILES:
            found.add(parts[1])
    return sorted(found)


def _env_list(name: str, default: tuple[str, ...]) -> list[str]:
    raw = os.environ.get(name)
    return [item for item in raw.split(",") if item] if raw else list(default)


def main() -> None:
    cache = _cache_root()
    corpora = _env_list("SEMDEX_PRODUCT_CORPORA", DEFAULT_CORPORA)
    embeddings = _env_list("SEMDEX_PRODUCT_EMBEDDINGS", DEFAULT_EMBEDDINGS)
    profiles_env = os.environ.get("SEMDEX_PRODUCT_PROFILES")
    product_k = int(os.environ.get("SEMDEX_PRODUCT_K", str(DEFAULT_PRODUCT_K)))
    budgets = [int(b) for b in _env_list("SEMDEX_PRODUCT_BUDGETS", tuple(str(b) for b in DEFAULT_BUDGETS))]
    fetch = int(os.environ.get("SEMDEX_PRODUCT_FETCH", str(DEFAULT_FETCH)))
    force = os.environ.get("SEMDEX_PRODUCT_FORCE") == "1"
    out = Path(os.environ.get("SEMDEX_SCORE_OUT", str(cache / "scores" / "product_k_scores.json")))
    out.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = _read_json(out) or {}
    for corpus in corpora:
        profiles = profiles_env.split(",") if profiles_env else _discover_profiles(corpus)
        for label in embeddings:
            for profile in profiles:
                cell = f"{corpus}__{profile}__{dirsafe(label)}"
                if cell in results and not force:
                    print(f"[product-k] skip (cached) {cell}", flush=True)
                    continue
                row = score_cell(corpus, profile, label, product_k=product_k, budgets=budgets, fetch=fetch)
                if row is None:
                    print(f"[product-k] not-embedded {cell}", flush=True)
                    continue
                results[cell] = stamped(row)
                product = next(r for r in row["rungs"] if r["product"])
                print(
                    f"[product-k] {cell}: documents@{product_k}={product['ndcg_documents']:.4f} "
                    f"delivered@{product_k}={product['ndcg_delivered']:.4f} "
                    f"gap={product['gap_mean_delta']:+.4f} "
                    f"[{product['gap_ci_lo']:+.4f}, {product['gap_ci_hi']:+.4f}] "
                    f"distinct={product['distinct_docs']:.2f} (n={row['n_queries']})",
                    flush=True,
                )
                out.write_text(json.dumps(results, indent=2, sort_keys=True))  # persist incrementally
    print(f"[product-k] {len(results)} cells in {out}", flush=True)


if __name__ == "__main__":
    main()
