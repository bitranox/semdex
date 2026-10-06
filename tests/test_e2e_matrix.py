# pyright: basic
"""Retrieval-quality + performance E2E benchmark matrix (local_only).

Measures REAL retrieval quality (nDCG@10 / Recall@10 / MRR / P@1) per
(chunker x embedding) combination and index/search performance per vector-store
backend, on labeled retrieval corpora loaded via ir_datasets (BEIR NFCorpus /
CQADupStack) or a HuggingFace loader (FreshStack). Corpora are DOWNLOADED at eval
time (cached under ~/.ir_datasets), never committed. Metrics are computed inline
(no pytrec_eval/beir/torch). Results print as tables and append to a
``benchmark-report.md`` artifact.

This file wraps the untyped ir_datasets/datasets libraries, so it opts down to
pyright basic mode. Needs ``semdex[bench,chunk,chunk-semantic]`` + a real
embedding (fastembed is a core dep); DB backends use Docker via the
``service_container`` fixture. Env knobs:
  SEMDEX_BENCH_CORPORA        comma list of nfcorpus,cqadupstack,scifact,fiqa,freshstack (default nfcorpus)
  SEMDEX_BENCH_MAX_DOCS       cap the corpus (keeps all relevant docs); 0/unset = full
  SEMDEX_BENCH_MAX_QUERIES    cap the query set; 0/unset = full
  SEMDEX_BENCH_FRESHSTACK_HF  HuggingFace dataset id for FreshStack (its loader is opt-in)
  SEMDEX_BENCH_REPORT         report path (default benchmark-report.md)
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import NamedTuple

import pytest
from _bench_corpus import corpus_extractor, corpus_sources, ranked_doc_ids
from _benchmark_report import ResultStatus, record

from semdex.application.use_cases import index_sources, search
from semdex.composition import build_chunker, build_embedding, build_vector_store
from semdex.domain.enums import ChunkStrategy, EmbeddingBackend, StoreBackend

# integration (not local_only): the store/quality/embedding matrices spin Docker DBs, download
# up to 10 embedding models, and run the slow `late` chunker - minutes, too long for `make test`.
# They run via `make ti` (`-m integration`) or directly by path with SEMDEX_BENCH_* env. The other
# local_only tests stay in `make test` (they use local resources but finish in seconds).
pytestmark = [pytest.mark.integration, pytest.mark.os_agnostic]

_REPORT = Path(os.environ.get("SEMDEX_BENCH_REPORT", "benchmark-report.md"))
_NDCG_K = 10
_SEARCH_K = 100  # chunk hits fetched, then deduped to documents


# --------------------------------------------------------------------------- metrics


def _dcg(gains: list[float]) -> float:
    return sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))


def _ndcg_at_k(ranked_docs: list[str], relevance: Mapping[str, int], k: int) -> float:
    gains = [float(relevance.get(doc, 0)) for doc in ranked_docs[:k]]
    ideal = sorted((float(v) for v in relevance.values()), reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg > 0 else 0.0


def _recall_at_k(ranked_docs: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked_docs[:k]) & relevant) / len(relevant)


def _mrr(ranked_docs: list[str], relevant: set[str]) -> float:
    for rank, doc in enumerate(ranked_docs):
        if doc in relevant:
            return 1.0 / (rank + 1)
    return 0.0


def _precision_at_1(ranked_docs: list[str], relevant: set[str]) -> float:
    return 1.0 if ranked_docs and ranked_docs[0] in relevant else 0.0


# --------------------------------------------------------------------------- corpora

# A corpus is (docs: id->text, queries: id->text, qrels: qid->{doc_id: relevance}).
Corpus = tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]]]


def _cap(value: str) -> int | None:
    n = int(os.environ.get(value, "0"))
    return n or None


def _load_ir_dataset(dataset_id: str) -> Corpus:
    ir_datasets = pytest.importorskip("ir_datasets")
    dataset = ir_datasets.load(dataset_id)
    qrels: dict[str, dict[str, int]] = {}
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:
            qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = int(qrel.relevance)
    queries = {q.query_id: q.default_text() for q in dataset.queries_iter() if q.query_id in qrels}
    docs = {d.doc_id: d.default_text() for d in dataset.docs_iter()}
    return _subsample(docs, queries, qrels)


def _load_freshstack() -> Corpus:
    hf_id = os.environ.get("SEMDEX_BENCH_FRESHSTACK_HF")
    if not hf_id:
        pytest.skip("FreshStack needs SEMDEX_BENCH_FRESHSTACK_HF set to its HuggingFace dataset id")
    pytest.importorskip("datasets")
    # FreshStack's on-disk layout is confirmed per its dataset card at implementation time;
    # until then this loader is opt-in and skips, so the default nfcorpus/cqadupstack runs stand.
    pytest.skip("FreshStack loader not yet wired for id " + hf_id)


def _subsample(docs: dict[str, str], queries: dict[str, str], qrels: dict[str, dict[str, int]]) -> Corpus:
    max_queries = _cap("SEMDEX_BENCH_MAX_QUERIES")
    if max_queries:
        keep = list(queries)[:max_queries]
        queries = {qid: queries[qid] for qid in keep}
        qrels = {qid: qrels[qid] for qid in keep if qid in qrels}
    max_docs = _cap("SEMDEX_BENCH_MAX_DOCS")
    if max_docs:
        relevant = {doc for rel in qrels.values() for doc in rel}
        kept = {doc: docs[doc] for doc in relevant if doc in docs}  # always keep judged docs
        for doc, text in docs.items():
            if len(kept) >= max_docs:
                break
            kept.setdefault(doc, text)
        docs = kept
        qrels = {qid: {d: r for d, r in rel.items() if d in docs} for qid, rel in qrels.items()}
        qrels = {qid: rel for qid, rel in qrels.items() if rel}
        queries = {qid: queries[qid] for qid in qrels}
    return docs, queries, qrels


_CORPORA: dict[str, Callable[[], Corpus]] = {
    "nfcorpus": lambda: _load_ir_dataset("beir/nfcorpus/test"),
    "cqadupstack": lambda: _load_ir_dataset("beir/cqadupstack/programmers"),
    "scifact": lambda: _load_ir_dataset("beir/scifact/test"),
    "fiqa": lambda: _load_ir_dataset("beir/fiqa/test"),
    "freshstack": _load_freshstack,
}


def _selected_corpora() -> list[str]:
    ids = os.environ.get("SEMDEX_BENCH_CORPORA", "nfcorpus").split(",")
    return [cid.strip() for cid in ids if cid.strip() in _CORPORA]


def _env_filter(env: str, members: tuple):
    """Restrict an enum tuple to the comma-listed values in *env* (empty = keep all) - for fast local runs."""
    raw = os.environ.get(env, "").strip()
    if not raw:
        return members
    wanted = {name.strip() for name in raw.split(",")}
    return tuple(member for member in members if member.value in wanted)


# --------------------------------------------------------------------------- harness


class EvalScores(NamedTuple):
    """What one indexed-and-searched corpus produced.

    ``aggregate`` is the flat metric map the report tables render. ``per_query`` is the same
    quality metrics before averaging, keyed ``{metric: {query id: score}}``, which is what lets
    the comparator difference a run against the baseline query by query rather than comparing
    two means. They are separate fields rather than one map because they are different shapes;
    folding the second into the first is what made this function lie about its return type.
    """

    aggregate: dict[str, float]
    per_query: dict[str, dict[str, float]]


def _index_and_eval(*, chunker, embedding, store, corpus: Corpus, max_tokens: int = 256) -> EvalScores:
    """Index the corpus with the given ports and score the queries."""
    docs, queries, qrels = corpus
    extractor = corpus_extractor(docs)
    started = time.perf_counter()
    index_sources(
        extract=extractor,
        chunk=chunker,
        embedding=embedding,
        store=store,
        collection="bench",
        sources=corpus_sources(docs),
        max_tokens=max_tokens,
    )
    index_seconds = time.perf_counter() - started

    ndcgs, recalls, rrs, p1s, latencies = [], [], [], [], []
    # Keyed by query id, not just accumulated: the comparator pairs a run against the baseline
    # query by query, which is what lets a 20-query slice detect a regression at all.
    per_query: dict[str, dict[str, float]] = {}
    for qid, text in queries.items():
        rel = qrels[qid]
        relevant = set(rel)
        t0 = time.perf_counter()
        hits = search(embedding=embedding, store=store, collection="bench", query=text, k=_SEARCH_K)
        latencies.append(time.perf_counter() - t0)
        ranked = ranked_doc_ids(hits, _NDCG_K)
        scores = {
            "ndcg@10": _ndcg_at_k(ranked, rel, _NDCG_K),
            "recall@10": _recall_at_k(ranked, relevant, _NDCG_K),
            "mrr": _mrr(ranked, relevant),
            "p@1": _precision_at_1(ranked, relevant),
        }
        per_query[qid] = scores
        ndcgs.append(scores["ndcg@10"])
        recalls.append(scores["recall@10"])
        rrs.append(scores["mrr"])
        p1s.append(scores["p@1"])

    latencies.sort()
    aggregate = {
        "ndcg": _mean(ndcgs),
        "recall": _mean(recalls),
        "mrr": _mean(rrs),
        "p1": _mean(p1s),
        "index_s": index_seconds,
        "search_p50_ms": _percentile(latencies, 0.50) * 1000,
        "search_p95_ms": _percentile(latencies, 0.95) * 1000,
    }
    return EvalScores(aggregate=aggregate, per_query=_by_metric(per_query))


def _by_metric(per_query: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Turn {query: {metric: score}} into {metric: {query: score}}, the shape the gate stores."""
    metrics: dict[str, dict[str, float]] = {}
    for qid, scores in per_query.items():
        for metric, value in scores.items():
            metrics.setdefault(metric, {})[qid] = value
    return metrics


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def _append_report(title: str, header: str, rows: list[str]) -> None:
    with _REPORT.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {title}\n\n{header}\n")
        for row in rows:
            fh.write(row + "\n")


# --------------------------------------------------------------------------- quality matrix

_QUALITY_CHUNKERS = (
    ChunkStrategy.WHITESPACE,
    ChunkStrategy.RECURSIVE,
    ChunkStrategy.MARKDOWN,
    ChunkStrategy.FAST,
    ChunkStrategy.SEMANTIC,
    ChunkStrategy.LATE,
)
_QUALITY_EMBEDDINGS = (
    EmbeddingBackend.PLACEHOLDER,
    EmbeddingBackend.FASTEMBED,
    EmbeddingBackend.MODEL2VEC,
)


def _build_chunker_or_skip(strategy: ChunkStrategy):
    # SEMANTIC (SDPM) uses model2vec; LATE uses sentence-transformers. Skip the cell
    # if that backend is absent (both come with semdex[chunk-semantic]/[embed]).
    if strategy is ChunkStrategy.SEMANTIC:
        pytest.importorskip("model2vec")
    elif strategy is ChunkStrategy.LATE:
        pytest.importorskip("sentence_transformers")
    return build_chunker(strategy, recipe="")  # recipe="" stays offline


@pytest.mark.parametrize("corpus_id", _selected_corpora() or ["nfcorpus"])
def test_quality_matrix(corpus_id: str, tmp_path: Path) -> None:
    """nDCG@10/Recall/MRR per chunker x embedding on the exact json store; real beats lexical."""
    corpus = _CORPORA[corpus_id]()
    results: dict[tuple[str, str], dict[str, float]] = {}
    rows: list[str] = []
    for strategy in _env_filter("SEMDEX_BENCH_CHUNKERS", _QUALITY_CHUNKERS):
        for provider in _env_filter("SEMDEX_BENCH_EMBEDDINGS", _QUALITY_EMBEDDINGS):
            if strategy in (ChunkStrategy.SEMANTIC, ChunkStrategy.LATE) and provider is EmbeddingBackend.PLACEHOLDER:
                continue  # embedding-driven chunker on a lexical embedding is meaningless
            store_dir = tmp_path / f"{strategy.value}_{provider.value}"
            scores = _index_and_eval(
                chunker=_build_chunker_or_skip(strategy),
                embedding=build_embedding(provider),
                store=build_vector_store(StoreBackend.JSON, store_dir),
                corpus=corpus,
            )
            metrics = scores.aggregate
            results[(strategy.value, provider.value)] = metrics
            record(
                "quality",
                f"{strategy.value}+{provider.value}",
                ResultStatus.OK,
                {
                    "ndcg@10": metrics["ndcg"],
                    "recall@10": metrics["recall"],
                    "mrr": metrics["mrr"],
                    "p@1": metrics["p1"],
                    "index_s": metrics["index_s"],
                },
                corpus=corpus_id,
                per_query=scores.per_query,
            )
            rows.append(
                f"| {strategy.value:10s} | {provider.value:11s} | {metrics['ndcg']:.4f} | "
                f"{metrics['recall']:.4f} | {metrics['mrr']:.4f} | {metrics['p1']:.4f} | {metrics['index_s']:.1f} |"
            )

    print(f"\n[{corpus_id}] quality matrix (json, exact):  chunker | embedding | nDCG@10 | Recall@10 | MRR | P@1")
    for row in rows:
        print("  " + row)
    _append_report(
        f"Retrieval quality - {corpus_id}",
        "| chunker | embedding | nDCG@10 | Recall@10 | MRR | P@1 | index_s |\n|---|---|---|---|---|---|---|",
        rows,
    )

    best_real = max(m["ndcg"] for (c, e), m in results.items() if e != "placeholder")
    best_placeholder = max((m["ndcg"] for (c, e), m in results.items() if e == "placeholder"), default=0.0)
    assert all(m["ndcg"] >= 0.0 for m in results.values())
    assert best_real > 0.05, f"pipeline looks broken: best real nDCG={best_real:.4f}"
    assert best_real >= best_placeholder, (
        f"real embeddings should beat the lexical placeholder: real={best_real:.4f} placeholder={best_placeholder:.4f}"
    )


# --------------------------------------------------------------------------- store perf matrix


def _pg_conninfo(port: int) -> str:
    return f"host=127.0.0.1 port={port} user=postgres password=semdex dbname=postgres"


def _pg_ready(port: int) -> bool:
    import psycopg

    try:
        psycopg.connect(_pg_conninfo(port), connect_timeout=2).close()
    except Exception:
        return False
    return True


def _maria_ready(port: int) -> bool:
    import pymysql

    try:
        pymysql.connect(
            host="127.0.0.1", port=port, user="root", password="semdex", database="semdex", connect_timeout=2
        ).close()
    except Exception:
        return False
    return True


def _store_dsn(backend: StoreBackend, service_container: Callable[..., int]) -> str | None:
    if backend is StoreBackend.PGVECTOR:
        pytest.importorskip("psycopg")
        pytest.importorskip("pgvector")
        return _pg_conninfo(
            service_container(
                image="pgvector/pgvector:pg16",
                container_port=5432,
                env={"POSTGRES_PASSWORD": "semdex"},
                ready=_pg_ready,
            )
        )
    if backend is StoreBackend.MARIADB:
        pytest.importorskip("pymysql")
        port = service_container(
            image="mariadb:11.8",
            container_port=3306,
            env={"MARIADB_ROOT_PASSWORD": "semdex", "MARIADB_DATABASE": "semdex"},
            ready=_maria_ready,
        )
        return f"mysql://root:semdex@127.0.0.1:{port}/semdex"
    if backend is StoreBackend.SQLITE_VEC:
        pytest.importorskip("sqlite_vec")
    if backend is StoreBackend.LANCEDB:
        pytest.importorskip("lancedb")
    return None


_STORE_BACKENDS = (
    StoreBackend.JSON,
    StoreBackend.SQLITE_VEC,
    StoreBackend.LANCEDB,
    StoreBackend.PGVECTOR,
    StoreBackend.MARIADB,
)


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def test_store_matrix(tmp_path: Path, service_container: Callable[..., int]) -> None:
    """Every store backend on recursive+fastembed: index/search perf + nDCG (exact agree, ANN within delta)."""
    corpus_id = (_selected_corpora() or ["nfcorpus"])[0]
    corpus = _CORPORA[corpus_id]()
    embedding = build_embedding(EmbeddingBackend.FASTEMBED)
    exact_ndcgs: list[float] = []
    rows: list[str] = []
    lance_ndcg: float | None = None
    for backend in _env_filter("SEMDEX_BENCH_STORES", _STORE_BACKENDS):
        dsn = _store_dsn(backend, service_container)
        store_dir = tmp_path / backend.value
        store_dir.mkdir(exist_ok=True)
        scores = _index_and_eval(
            chunker=build_chunker(ChunkStrategy.RECURSIVE, recipe=""),
            embedding=embedding,
            store=build_vector_store(backend, store_dir, dsn=dsn),
            corpus=corpus,
        )
        metrics = scores.aggregate
        size = _dir_size(store_dir) if dsn is None else 0
        record(
            "store",
            backend.value,
            ResultStatus.OK,
            {
                "ndcg@10": metrics["ndcg"],
                "index_s": metrics["index_s"],
                "search_p50_ms": metrics["search_p50_ms"],
                "search_p95_ms": metrics["search_p95_ms"],
                "store_mb": size / 1_000_000,
            },
            corpus=corpus_id,
            per_query=scores.per_query,
        )
        rows.append(
            f"| {backend.value:10s} | {metrics['ndcg']:.4f} | {metrics['index_s']:.1f} | "
            f"{metrics['search_p50_ms']:.1f} | {metrics['search_p95_ms']:.1f} | {size} |"
        )
        if backend is StoreBackend.LANCEDB:
            lance_ndcg = metrics["ndcg"]
        else:
            exact_ndcgs.append(metrics["ndcg"])

    print("\nstore matrix (recursive+fastembed):  store | nDCG@10 | index_s | p50_ms | p95_ms | bytes")
    for row in rows:
        print("  " + row)
    _append_report(
        "Store performance / scale",
        "| store | nDCG@10 | index_s | search_p50_ms | search_p95_ms | store_bytes |\n|---|---|---|---|---|---|",
        rows,
    )

    # Exact-search backends rank identically -> their nDCG agrees within float noise.
    assert max(exact_ndcgs) - min(exact_ndcgs) < 0.02, f"exact stores disagree: {exact_ndcgs}"
    if lance_ndcg is not None:
        assert lance_ndcg >= min(exact_ndcgs) - 0.10, "lancedb ANN recall dropped too far below exact"


# --------------------------------------------------------------------------- embedding-model matrix

# The embedding-MODEL dimension (distinct from the provider): a fixed markdown chunker +
# json store, varying the (provider, model) pair. Different models mean different vector
# dimensions and different retrieval quality even within one provider. ollama models need a
# reachable server via SEMDEX_BENCH_OLLAMA_URL (never committed - the endpoint is injected,
# not hardcoded); sentence-transformers needs its extra; fastembed/model2vec download their
# model on first use. A cell whose backend/model is unavailable is recorded (skip/fail) and
# the sweep continues, like the extractor grid.
_EMBED_MODELS: tuple[tuple[str, EmbeddingBackend, str], ...] = (
    ("fastembed:bge-small", EmbeddingBackend.FASTEMBED, "BAAI/bge-small-en-v1.5"),
    ("fastembed:bge-base", EmbeddingBackend.FASTEMBED, "BAAI/bge-base-en-v1.5"),
    ("model2vec:potion-base-8M", EmbeddingBackend.MODEL2VEC, "minishlab/potion-base-8M"),
    ("model2vec:potion-retrieval-32M", EmbeddingBackend.MODEL2VEC, "minishlab/potion-retrieval-32M"),
    ("st:all-MiniLM-L6-v2", EmbeddingBackend.SENTENCE_TRANSFORMERS, "sentence-transformers/all-MiniLM-L6-v2"),
    ("ollama:nomic-embed-text", EmbeddingBackend.OLLAMA, "nomic-embed-text"),
    ("ollama:all-minilm", EmbeddingBackend.OLLAMA, "all-minilm"),
    ("ollama:mxbai-embed-large", EmbeddingBackend.OLLAMA, "mxbai-embed-large"),
    # Qwen3-Embedding: the top MTEB-multilingual open family; large Matryoshka dims
    # (the adapter discovers the served dimension by probing, so no dim is pinned here).
    # GPU-tier - real cells need an ollama server with these tags pulled.
    ("ollama:qwen3-embedding-4b", EmbeddingBackend.OLLAMA, "qwen3-embedding:4b"),
    ("ollama:qwen3-embedding-8b", EmbeddingBackend.OLLAMA, "qwen3-embedding:8b"),
)


def _env_filter_labels(env: str, models: tuple[tuple[str, EmbeddingBackend, str], ...]):
    """Restrict the model registry to the comma-listed labels in *env* (empty = keep all)."""
    raw = os.environ.get(env, "").strip()
    if not raw:
        return models
    wanted = {name.strip() for name in raw.split(",")}
    return tuple(entry for entry in models if entry[0] in wanted)


def _embedding_backend_available(provider: EmbeddingBackend) -> bool:
    """Whether the provider can run here without aborting the whole sweep."""
    if provider is EmbeddingBackend.SENTENCE_TRANSFORMERS:
        import importlib.util

        return importlib.util.find_spec("sentence_transformers") is not None
    if provider is EmbeddingBackend.OLLAMA:
        return bool(os.environ.get("SEMDEX_BENCH_OLLAMA_URL"))
    return True  # fastembed / model2vec download their model on demand


def test_embedding_model_matrix(tmp_path: Path) -> None:
    """The embedding-MODEL dimension: nDCG@10/Recall/MRR/P@1 per (provider, model) on a fixed chunker + json store."""
    corpus_id = (_selected_corpora() or ["nfcorpus"])[0]
    corpus = _CORPORA[corpus_id]()
    chunker = build_chunker(ChunkStrategy.MARKDOWN, recipe="")
    ollama_url = os.environ.get("SEMDEX_BENCH_OLLAMA_URL")
    rows: list[str] = []
    ran = 0
    for label, provider, model in _env_filter_labels("SEMDEX_BENCH_EMBED_MODELS", _EMBED_MODELS):
        if not _embedding_backend_available(provider):
            record("embed_model", label, ResultStatus.SKIP, corpus=corpus_id)
            rows.append(f"| {label:40s} | skip | - | - | - |")
            continue
        endpoint = ollama_url if provider is EmbeddingBackend.OLLAMA else None
        store_dir = tmp_path / label.replace("/", "_").replace(":", "_")
        store_dir.mkdir(exist_ok=True)
        try:
            embedding = build_embedding(provider, model=model, endpoint=endpoint)
            scores = _index_and_eval(
                chunker=chunker,
                embedding=embedding,
                store=build_vector_store(StoreBackend.JSON, store_dir),
                corpus=corpus,
            )
        except Exception:  # a model download / server error is a recorded cell, not a sweep abort
            record("embed_model", label, ResultStatus.FAIL, corpus=corpus_id)
            rows.append(f"| {label:40s} | fail | - | - | - |")
            continue
        metrics = scores.aggregate
        ran += 1
        record(
            "embed_model",
            label,
            ResultStatus.OK,
            {"ndcg@10": metrics["ndcg"], "recall@10": metrics["recall"], "mrr": metrics["mrr"], "p@1": metrics["p1"]},
            corpus=corpus_id,
            per_query=scores.per_query,
        )
        m = metrics
        rows.append(f"| {label:40s} | {m['ndcg']:.3f} | {m['recall']:.3f} | {m['mrr']:.3f} | {m['p1']:.3f} |")

    print(f"\n[{corpus_id}] embedding-model matrix (markdown chunker, json store):  provider:model | nDCG@10 ...")
    for row in rows:
        print("  " + row)
    _append_report(
        f"Embedding-model matrix - {corpus_id}",
        "| provider:model | nDCG@10 | Recall@10 | MRR | P@1 |\n|---|---|---|---|---|",
        rows,
    )

    if ran == 0:
        pytest.skip("no embedding model could run here (offline / no providers / no ollama server)")
