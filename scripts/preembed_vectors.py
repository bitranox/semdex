#!/usr/bin/env python
# pyright: basic
"""Pre-embed the corpus x embedding-model matrix and persist the VECTORS to a cache.

Phase 3 of the semdex benchmark rig. This embeds each (corpus, embedding) cell ONCE
and writes the float32 vectors (plus the chunk texts, shared per corpus) to a cache
dir, so any later store/retrieval re-evaluation loads vectors from disk with NO
re-embedding. The expensive artifact (the vectors) is persisted; a store is cheap to
(re)populate from them on demand, so we do NOT pre-populate the 5 store backends -
that would be corpus x embedding x store (5x the compute + disk) for no durable gain.

All config is via environment variables (the scripts/ convention - no argparse):

  CACHE_ROOT            cache dir (default /embeddings)
  SEMDEX_PREEMBED_CORPORA     comma list (default nfcorpus,scifact,fiqa,cqadupstack);
                              a msmarco_<scale> id (e.g. msmarco_50000) loads msmarco at that scale;
                              a miracl_<lang>_100k_slice id (miracl_en_100k_slice / miracl_de_100k_slice)
                              loads a pre-built FROZEN MIRACL slice (scripts/build_miracl_slices.py)
  SEMDEX_PREEMBED_EMBEDDINGS  comma list of _EMBED_MODELS labels (default fastembed:bge-small)
  SEMDEX_PREEMBED_CHUNKER     chunk strategy id (default recursive)
  SEMDEX_PREEMBED_MAX_TOKENS  chunk size in tokens (default 256)
  SEMDEX_PREEMBED_OVERLAP     chunk overlap in tokens (default 0)
  SEMDEX_PREEMBED_TOKENIZER   chunk tokenizer (default gpt2)
  SEMDEX_PREEMBED_RECIPE      chonkie recipe (default "" = offline; the lib default is "markdown")
  SEMDEX_PREEMBED_SEMANTIC_MODEL  semantic-chunker breakpoint embedding model (default "" = chonkie's
                              default potion-base-32M, English-distilled). Set a multilingual model
                              (e.g. minishlab/potion-multilingual-128M) for non-English corpora. Only
                              affects the semantic strategy; encoded into the profile id (-bp<model>)
                              so it never collides with the default-breakpoint semantic cells.
  SEMDEX_PREEMBED_EMBED_BATCH passages per embed_passages call (default 2048; ollama: 256).
                              This is the GPU-throughput lever for ollama cells, NOT
                              OLLAMA_NUM_PARALLEL: _embed_batch is SERIAL (one request, wait,
                              next), so ollama never sees concurrent requests for NUM_PARALLEL to
                              parallelize. A bigger batch does more matmul per request and closes
                              the idle gaps: measured 256 -> ~38 docs/s vs 128 -> ~21 docs/s
                              (~1.8x, GPU util 69->100%) on qwen3-embedding:4b (RTX 4070 Ti SUPER),
                              which is why 256 is the ollama default (matches the #45 throughput
                              bench). Keep batch/docs_per_sec under the adapter's 60s httpx timeout
                              (512 @ ~38/s = ~13s, safe).
  SEMDEX_PREEMBED_NUM_BATCH   ollama's physical batch in tokens (default unset = the server's 2048).
                              ollama embeds a longer input from its first num_batch tokens and still
                              answers 200, so set it to at least the longest chunk (cap + overlap) in
                              model tokens for long chunk sets. Ignored for non-ollama backends.
  SEMDEX_PREEMBED_EMBED_THREADS cap the CPU embedder's (fastembed/onnxruntime) intra-op threads;
                              unset = onnxruntime's default (one spinning thread per core, which
                              thrashes on a shared/contended box). Set to ~half the cores there.
  HF_HUB_OFFLINE              default "1" (set here): use the local HF cache, NO hub network
                              calls. A cached asset otherwise still does a blocking hub
                              freshness check that stalls for minutes on a slow HF CDN. Set
                              HF_HUB_OFFLINE=0 to allow downloading a new, uncached model.

  The full chunk config (strategy/max_tokens/overlap/tokenizer/recipe) is encoded in each
  cell's cache dir as a "profile" (e.g. recursive-t256-o0-gpt2) AND stored in its meta.json,
  so vectors for DIFFERENT chunk settings never collide - they land in separate directories.
  SEMDEX_PREEMBED_MAX_DOCS    optional per-corpus doc cap (smoke runs; judged docs kept)
  IR_DATASETS_HOME            ir_datasets cache (default /corpora/ir_datasets)
  SEMDEX_MSMARCO_DIR          on-disk msmarco BEIR dir (default /corpora/bench-msmarco/msmarco)
  SEMDEX_MSMARCO_SPLIT        msmarco qrels split (default dev)
  SEMDEX_MSMARCO_MAX_Q        msmarco eval queries kept (default 1000)
  SEMDEX_BENCH_OLLAMA_URL     endpoint for ollama embedding cells (qwen3, bge-m3)
  SEMDEX_BENCH_OPENAI_URL     endpoint (/v1) for openai-compatible embedding cells (the
                              sentence-transformers shim: e5-large)

  MODE=embed (default) | catalog | loadstore

  loadstore mode (populate a FRESH store from a cached cell + score nDCG - the
  store-from-cache path a benchmark uses; proves the cache end to end):
    SEMDEX_LOADSTORE_CELL      cell id (e.g. nfcorpus__recursive__fastembed-bge-small)
    SEMDEX_BENCH_STORE         json | sqlite_vec | lancedb | pgvector | mariadb
    SEMDEX_BENCH_STORE_DIR     dir for the embedded backends
    SEMDEX_BENCH_PGVECTOR_DSN / SEMDEX_BENCH_MARIADB_DSN   for the server backends
    SEMDEX_BENCH_COLLECTION    collection name (default the cell id, dir-safe)
    SEMDEX_LOADSTORE_K         top-k for scoring (default 10)

Every heavy job is idempotent + streaming: a cell whose meta.json marker is complete is
skipped; vectors are filled into a memmap in batches, never all in RAM.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# The benchmark rig runs against a PRE-CACHED HuggingFace store, so force the hub OFFLINE:
# the chunker/embedder builders then use the local cache with NO network calls. Otherwise a
# cached asset still triggers a blocking hub freshness-check HTTP request, and a slow/stalled
# HF CDN wedges each chunker build in poll() for minutes at ~15% CPU (observed wedging the #42
# sweep). Everything the matrix needs (gpt2 tokenizer, potion/bge models, chonkie recipes) is
# cached; to add a NEW uncached model, run once with HF_HUB_OFFLINE=0 to download it first.
# setdefault, so an explicit HF_HUB_OFFLINE=0 in the environment still wins.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# fastembed's own model cache defaults under /tmp, which a reboot wipes. Combined with the
# offline mode above, the next run then cannot load a model that IS cached on the data volume:
# it either dies mid-sweep (bge-base failed 16 cells into an MLDR pass) or, where a fallback is
# allowed, silently degrades to the 256-dim placeholder and overwrites good cells. Point it at
# the cache that lives beside the vectors, next to CACHE_ROOT, so no run has to remember a flag.
os.environ.setdefault(
    "FASTEMBED_CACHE_PATH", str(Path(os.environ.get("CACHE_ROOT", "/embeddings")) / ".fastembed_cache")
)

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from semdex.composition import build_chunker, build_embedding, build_vector_store
from semdex.domain.enums import ChunkStrategy, EmbeddingBackend, StoreBackend
from semdex.domain.models import Chunk, Collection, ExtractedDocument, SourceRef

# --- registries (copied from tests/test_e2e_matrix.py, which are test-only/private) ---

# (label, backend, model_id). The ollama cells get their endpoint from SEMDEX_BENCH_OLLAMA_URL;
# the openai (sentence-transformers shim) cells from SEMDEX_BENCH_OPENAI_URL.
_EMBED_MODELS: dict[str, tuple[EmbeddingBackend, str]] = {
    "fastembed:bge-small": (EmbeddingBackend.FASTEMBED, "BAAI/bge-small-en-v1.5"),
    "fastembed:bge-base": (EmbeddingBackend.FASTEMBED, "BAAI/bge-base-en-v1.5"),
    "model2vec:potion-base-8M": (EmbeddingBackend.MODEL2VEC, "minishlab/potion-base-8M"),
    "model2vec:potion-retrieval-32M": (EmbeddingBackend.MODEL2VEC, "minishlab/potion-retrieval-32M"),
    "ollama:qwen3-embedding-4b": (EmbeddingBackend.OLLAMA, "qwen3-embedding:4b"),
    "ollama:qwen3-embedding-8b": (EmbeddingBackend.OLLAMA, "qwen3-embedding:8b"),
    # multilingual retrieval embedders for the breakpoint sweep (de/en long-doc ranking):
    "ollama:bge-m3": (EmbeddingBackend.OLLAMA, "bge-m3"),
    "openai:e5-large": (EmbeddingBackend.OPENAI, "intfloat/multilingual-e5-large"),
}

# corpus id -> ir_datasets id (BEIR small corpora; msmarco scales are handled separately).
_CORPORA: dict[str, str] = {
    "nfcorpus": "beir/nfcorpus/test",
    "scifact": "beir/scifact/test",
    "fiqa": "beir/fiqa/test",
    "cqadupstack": "beir/cqadupstack/programmers",
}

_STORE_BACKENDS = {b.value: b for b in StoreBackend}
_CHUNK_ROWGROUP = 4096  # chunks buffered before a parquet row-group flush (bounded memory)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _dirsafe(label: str) -> str:
    """A filesystem-safe form of an embedding label (``fastembed:bge-small`` -> ``fastembed-bge-small``)."""
    return label.replace(":", "-").replace("/", "-")


def _cache_root() -> Path:
    return Path(os.environ.get("CACHE_ROOT", "/embeddings"))


def _chunker_id() -> str:
    return os.environ.get("SEMDEX_PREEMBED_CHUNKER", "recursive")


def _max_tokens() -> int:
    return int(os.environ.get("SEMDEX_PREEMBED_MAX_TOKENS", "256"))


def _overlap() -> int:
    return int(os.environ.get("SEMDEX_PREEMBED_OVERLAP", "0"))


def _tokenizer() -> str:
    return os.environ.get("SEMDEX_PREEMBED_TOKENIZER", "gpt2")


def _recipe() -> str:
    return os.environ.get("SEMDEX_PREEMBED_RECIPE", "")


def _semantic_model() -> str:
    """The SEMANTIC chunker's breakpoint embedding model ("" = chonkie's own default)."""
    return os.environ.get("SEMDEX_PREEMBED_SEMANTIC_MODEL", "")


def _chunk_config() -> dict[str, Any]:
    """The FULL chunker config for this run - recorded verbatim in every cell's meta.json."""
    return {
        "strategy": _chunker_id(),
        "max_tokens": _max_tokens(),
        "overlap": _overlap(),
        "tokenizer": _tokenizer(),
        "recipe": _recipe(),
        "semantic_model": _semantic_model(),
    }


def _chunk_profile() -> str:
    """A dir-safe id of the full chunk config, so vectors for DIFFERENT chunk settings never
    collide in the cache - each config gets its own ``<corpus>__<profile>__<embedding>`` dir.
    The default config maps to ``recursive-t256-o0-gpt2``. The semantic breakpoint model is
    appended (``...-bp<model>``) ONLY for the semantic strategy, so two semantic runs with
    different breakpoint models land in separate cells while every recursive id is unchanged.
    """
    parts = [_chunker_id(), f"t{_max_tokens()}", f"o{_overlap()}", _dirsafe(_tokenizer())]
    if _recipe():
        parts.append(f"r{_dirsafe(_recipe())}")
    if _chunker_id() == "semantic" and _semantic_model():
        parts.append(f"bp{_dirsafe(_semantic_model())}")
    return "-".join(parts)


def _chunks_dir(corpus: str) -> Path:
    return _cache_root() / "chunks" / f"{corpus}__{_chunk_profile()}"


def _num_batch_from_env() -> int | None:
    """ollama's physical batch from SEMDEX_PREEMBED_NUM_BATCH; None keeps the server default.

    The default (2048 tokens) silently embeds a longer input from its prefix, so a run over long
    chunks must raise it; a value that is not a positive integer stops the run before any spend.
    """
    raw = os.environ.get("SEMDEX_PREEMBED_NUM_BATCH")
    if raw is None:
        return None
    if not raw.isdigit() or int(raw) <= 0:
        raise SystemExit(f"SEMDEX_PREEMBED_NUM_BATCH must be a positive integer, got {raw!r}")
    return int(raw)


def _cell_id(corpus: str, label: str) -> str:
    return f"{corpus}__{_chunk_profile()}__{_dirsafe(label)}"


def _vectors_dir(corpus: str, label: str) -> Path:
    return _cache_root() / "vectors" / _cell_id(corpus, label)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON to ``path.tmp`` then rename - so a half-written marker never looks complete."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.rename(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


# --- corpus loading (ir_datasets, no pytest; mirrors tests/test_e2e_matrix._load_ir_dataset) ---


def _load_msmarco(scale: int) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]]]:
    """Load the on-disk msmarco BEIR dir capped to ``scale`` docs (judged kept), streamed + bounded.

    Mirrors scripts/bench_msmarco_scale.load_corpus: skip the qrels header, keep every judged doc
    plus fill to ``scale`` others, breaking out of the 8.8M-doc corpus.jsonl scan once done.
    """
    import orjson

    base = Path(os.environ.get("SEMDEX_MSMARCO_DIR", "/corpora/bench-msmarco/msmarco"))
    split = os.environ.get("SEMDEX_MSMARCO_SPLIT", "dev")
    max_q = int(os.environ.get("SEMDEX_MSMARCO_MAX_Q", "1000"))
    qrels: dict[str, dict[str, int]] = {}
    with (base / "qrels" / f"{split}.tsv").open(encoding="utf-8") as fh:
        fh.readline()  # header: query-id\tcorpus-id\tscore
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and int(float(parts[2])) > 0:
                qrels.setdefault(parts[0], {})[parts[1]] = int(float(parts[2]))
    if max_q > 0:
        qrels = {q: qrels[q] for q in list(qrels)[:max_q]}
    want = set(qrels)
    queries: dict[str, str] = {}
    with (base / "queries.jsonl").open("rb") as fh:
        for raw in fh:
            obj = orjson.loads(raw)
            if obj["_id"] in want:
                queries[obj["_id"]] = obj.get("text", "")
    qrels = {q: r for q, r in qrels.items() if q in queries}
    judged = {d for r in qrels.values() for d in r}
    docs: dict[str, str] = {}
    fill_budget = max(0, scale - len(judged))
    filled = 0
    pending = set(judged)
    with (base / "corpus.jsonl").open("rb") as fh:
        for raw in fh:
            obj = orjson.loads(raw)
            did = obj["_id"]
            if did in pending:
                pending.discard(did)
            elif filled < fill_budget:
                filled += 1
            elif pending:
                continue  # fill done, still hunting judged stragglers
            else:
                break  # fill done and every judged doc found - stop scanning
            docs[did] = (obj.get("title", "") + " " + obj.get("text", "")).strip()
    qrels = {q: {d: r for d, r in rel.items() if d in docs} for q, rel in qrels.items()}
    qrels = {q: r for q, r in qrels.items() if r}
    queries = {q: queries[q] for q in qrels}
    return docs, queries, qrels


# Every frozen-slice family. Both the loader dispatch and _slice_root read this, so a new corpus
# cannot be half-registered the way gerdalir first was.
_SLICE_FAMILIES = ("miracl_", "mldr_", "gerdalir_")


def _slice_root(name: str) -> tuple[Path, str]:
    """Return (root dir, builder script) for a frozen slice id, by family."""
    if name.startswith("mldr_"):
        return Path(os.environ.get("SEMDEX_MLDR_SLICE_OUT", "/corpora/mldr-slices")), "build_mldr_slices.py"
    if name.startswith("gerdalir_"):
        return (
            Path(os.environ.get("SEMDEX_SLICE_OUT_GERDALIR", "/corpora/gerdalir-slices")),
            "build_gerdalir_slice.py",
        )
    return Path(os.environ.get("SEMDEX_SLICE_OUT", "/corpora/miracl-slices")), "build_miracl_slices.py"


def _load_frozen_slice(name: str) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]]]:
    """Load a pre-built FROZEN slice (MIRACL or MLDR; same on-disk layout).

    A slice is a self-contained, fixed corpus on disk (corpus.jsonl + queries.json +
    qrels.json), so every chunker/embedder run against it indexes the identical document
    set - no ir_datasets, no re-sampling, just read the saved files (title+text joined
    like the msmarco loader).
    """
    root, builder = _slice_root(name)
    base = root / name
    if not (base / "corpus.jsonl").exists():
        raise SystemExit(f"slice {name} not built - run scripts/{builder} first")
    import orjson

    docs: dict[str, str] = {}
    with (base / "corpus.jsonl").open("rb") as fh:
        for raw in fh:
            obj = orjson.loads(raw)
            docs[obj["_id"]] = (obj.get("title", "") + " " + obj.get("text", "")).strip()
    queries = _read_json(base / "queries.json") or {}
    qrels = _read_json(base / "qrels.json") or {}
    return docs, queries, qrels


def _load_corpus(corpus: str) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]]]:
    """Return (docs id->text, queries id->text, qrels qid->{doc_id: rel}) for a corpus unit.

    A ``msmarco_<scale>`` id loads the on-disk msmarco BEIR dir capped to <scale> docs; a
    ``miracl_<lang>_100k_slice``, ``mldr_<lang>_<n>k_slice`` or ``gerdalir_de_<n>k_slice`` id loads
    a pre-built frozen slice; any other id is a BEIR corpus loaded via ir_datasets.
    """
    if corpus.startswith("msmarco_"):
        return _load_msmarco(int(corpus.split("_", 1)[1]))
    if corpus.startswith(_SLICE_FAMILIES) and corpus.endswith("_slice"):
        return _load_frozen_slice(corpus)
    os.environ.setdefault("IR_DATASETS_HOME", "/corpora/ir_datasets")
    import ir_datasets

    dataset = ir_datasets.load(_CORPORA[corpus])
    qrels: dict[str, dict[str, int]] = {}
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:
            qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = int(qrel.relevance)
    queries = {q.query_id: q.text for q in dataset.queries_iter() if q.query_id in qrels}
    docs = {d.doc_id: (getattr(d, "title", "") + " " + d.text).strip() for d in dataset.docs_iter()}

    max_docs = int(os.environ.get("SEMDEX_PREEMBED_MAX_DOCS", "0"))
    if max_docs > 0 and len(docs) > max_docs:
        judged = {doc_id for rels in qrels.values() for doc_id in rels}
        kept = dict(list({**{k: docs[k] for k in judged if k in docs}, **docs}.items())[:max_docs])
        docs = kept
    return docs, queries, qrels


# --- chunk phase (once per corpus; streamed to parquet) -------------------------------


def _chunks_complete(corpus: str) -> int | None:
    """Return the chunk count if the corpus is fully chunked, else None."""
    meta = _read_json(_chunks_dir(corpus) / "meta.json")
    parquet = _chunks_dir(corpus) / "chunks.parquet"
    if not meta or not parquet.exists():
        return None
    if meta.get("count") == pq.read_metadata(parquet).num_rows:
        return int(meta["count"])
    return None


def _build_run_chunker() -> Any:
    """Build the chunker for this run's chunk config.

    For the SEMANTIC strategy with a breakpoint-model LABEL (SEMDEX_PREEMBED_SEMANTIC_MODEL), the
    breakpoint embedder is built from the bench registry - a local static model, or an endpoint
    model reached over SEMDEX_PREEMBED_SEMANTIC_ENDPOINT - and passed as a chonkie embeddings
    object. Every other case uses the package factory with a plain string (or no) breakpoint model.
    """
    strategy = ChunkStrategy(_chunker_id())
    label = _semantic_model()
    if strategy is ChunkStrategy.SEMANTIC and label:
        from _bench_breakpoint_embeddings import build_breakpoint_embeddings

        from semdex.adapters.chunker import ChonkieChunker

        embedder = build_breakpoint_embeddings(label, endpoint=os.environ.get("SEMDEX_PREEMBED_SEMANTIC_ENDPOINT"))
        print(f"[chunks] semantic breakpoint model: {label} ({embedder!r})", flush=True)
        return ChonkieChunker(
            strategy, recipe=_recipe(), overlap=_overlap(), tokenizer=_tokenizer(), semantic_model=embedder
        )
    return build_chunker(strategy, recipe=_recipe(), overlap=_overlap(), tokenizer=_tokenizer())


def _ensure_chunks(corpus: str) -> int:
    """Chunk ``corpus`` once, streaming chunks to parquet + saving queries/qrels. Idempotent."""
    done = _chunks_complete(corpus)
    if done is not None:
        print(f"[chunks] {corpus}: already complete ({done} chunks) - skip", flush=True)
        return done

    out = _chunks_dir(corpus)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[chunks] {corpus}: loading corpus...", flush=True)
    docs, queries, qrels = _load_corpus(corpus)
    _write_json_atomic(out / "queries.json", queries)
    _write_json_atomic(out / "qrels.json", qrels)

    chunker = _build_run_chunker()
    max_tokens = _max_tokens()
    schema = pa.schema(
        [("text", pa.string()), ("source_uri", pa.string()), ("ordinal", pa.int32()), ("token_count", pa.int32())]
    )
    parquet_tmp = out / "chunks.parquet.tmp"
    texts: list[str] = []
    uris: list[str] = []
    ordinals: list[int] = []
    tokens: list[int] = []
    total = 0
    t0 = time.perf_counter()

    def _flush(writer: pq.ParquetWriter) -> None:
        if not texts:
            return
        writer.write_batch(
            pa.record_batch(
                [pa.array(texts), pa.array(uris), pa.array(ordinals, pa.int32()), pa.array(tokens, pa.int32())],
                schema=schema,
            )
        )
        texts.clear()
        uris.clear()
        ordinals.clear()
        tokens.clear()

    with pq.ParquetWriter(parquet_tmp, schema) as writer:
        for doc_id, text in docs.items():
            source = SourceRef(uri=str(Path(doc_id)), label="", content_hash="", mtime=0.0)
            for chunk in chunker(ExtractedDocument(source=source, text=text), max_tokens=max_tokens):
                texts.append(chunk.text)
                uris.append(chunk.source.uri)
                ordinals.append(chunk.ordinal)
                tokens.append(chunk.token_count)
                total += 1
                if len(texts) >= _CHUNK_ROWGROUP:
                    _flush(writer)
        _flush(writer)

    (parquet_tmp).rename(out / "chunks.parquet")
    _write_json_atomic(
        out / "meta.json",
        {
            "corpus": corpus,
            "profile": _chunk_profile(),
            **_chunk_config(),
            "count": total,
            "docs": len(docs),
            "queries": len(queries),
            "created": _now(),
        },
    )
    print(f"[chunks] {corpus}: {total} chunks from {len(docs)} docs in {time.perf_counter() - t0:.1f}s", flush=True)
    return total


# --- embed phase (per corpus x embedding; memmap fill, streamed) ----------------------


def _embed_batch(embedding: Any, texts: list[str], retries: int = 6) -> Any:
    """Embed one batch, retrying on a transient server error with exponential backoff.

    A long GPU pass streams thousands of requests to the ollama/openai server; a single
    transient disconnect or timeout must not kill hours of work. Retries the batch up to
    ``retries`` times (2s, 4s, ... capped at 60s), then re-raises a persistent failure.
    """
    from semdex.domain.errors import EmbeddingError

    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            return embedding.embed_passages(texts)
        except EmbeddingError as exc:
            if attempt == retries:
                raise
            print(f"[embed] transient error (attempt {attempt}/{retries}): {exc}; retry in {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    return []  # unreachable


def _vectors_complete(corpus: str, label: str, count: int, dim: int) -> bool:
    vdir = _vectors_dir(corpus, label)
    meta = _read_json(vdir / "meta.json")
    npy = vdir / "vectors.npy"
    if not meta or not npy.exists() or meta.get("count") != count:
        return False
    try:
        shape = np.load(npy, mmap_mode="r").shape  # reads only the header
    except (ValueError, OSError):
        return False
    return shape == (count, dim)


def _ensure_vectors(corpus: str, label: str, chunk_count: int) -> None:
    """Embed one (corpus, embedding) cell into a memmapped vectors.npy, streamed + idempotent."""
    backend, model_id = _EMBED_MODELS[label]
    # Route the retrieval endpoint by backend: ollama models to the ollama server, openai
    # (sentence-transformers shim) models to the shim /v1; local backends need no endpoint.
    if backend is EmbeddingBackend.OLLAMA:
        endpoint = os.environ.get("SEMDEX_BENCH_OLLAMA_URL")
    elif backend is EmbeddingBackend.OPENAI:
        endpoint = os.environ.get("SEMDEX_BENCH_OPENAI_URL")
    else:
        endpoint = None
    # allow_fallback=False: a benchmark must NEVER silently cache 256-dim placeholder vectors
    # when the real model cannot load (e.g. a reboot-wiped fastembed cache) - fail loudly instead.
    # SEMDEX_PREEMBED_EMBED_THREADS caps the CPU embedder's (fastembed/onnxruntime) intra-op threads.
    # Unbounded, ONNX uses one thread per core; capping BOUNDS CONTENTION with a co-running sweep -
    # it is NOT a per-embedder speedup (fastembed bge-base CPU is only ~a few chunks/s here either
    # way; the real speed lever is GPU embedding, not thread count).
    threads_env = os.environ.get("SEMDEX_PREEMBED_EMBED_THREADS")
    threads = int(threads_env) if threads_env else None
    num_batch = _num_batch_from_env() if backend is EmbeddingBackend.OLLAMA else None
    embedding = build_embedding(
        backend, model=model_id, endpoint=endpoint, threads=threads, num_batch=num_batch, allow_fallback=False
    )
    dim = embedding.dim

    if _vectors_complete(corpus, label, chunk_count, dim):
        print(f"[embed] {_cell_id(corpus, label)}: already complete ({chunk_count}x{dim}) - skip", flush=True)
        return

    vdir = _vectors_dir(corpus, label)
    vdir.mkdir(parents=True, exist_ok=True)
    npy_tmp = vdir / "vectors.npy.tmp"
    batch = int(os.environ.get("SEMDEX_PREEMBED_EMBED_BATCH", "256" if backend is EmbeddingBackend.OLLAMA else "2048"))
    parquet = _chunks_dir(corpus) / "chunks.parquet"
    t0 = time.perf_counter()
    mm = np.lib.format.open_memmap(npy_tmp, mode="w+", dtype="float32", shape=(chunk_count, dim))
    row = 0
    for record_batch in pq.ParquetFile(parquet).iter_batches(batch_size=batch, columns=["text"]):
        chunk_texts = record_batch.column("text").to_pylist()
        vecs = _embed_batch(embedding, chunk_texts)
        mm[row : row + len(chunk_texts)] = np.asarray(vecs, dtype="float32")
        row += len(chunk_texts)
        print(f"[embed] {_cell_id(corpus, label)}: {row}/{chunk_count}", flush=True)
    mm.flush()
    del mm
    npy_tmp.rename(vdir / "vectors.npy")
    _write_json_atomic(
        vdir / "meta.json",
        {
            "cell": _cell_id(corpus, label),
            "corpus": corpus,
            "profile": _chunk_profile(),
            **_chunk_config(),
            "embedding": label,
            "backend": backend.value,
            "model_id": embedding.model_id,
            "dim": dim,
            "count": chunk_count,
            "created": _now(),
        },
    )
    print(f"[embed] {_cell_id(corpus, label)}: done {chunk_count}x{dim} in {time.perf_counter() - t0:.1f}s", flush=True)


# --- catalog (generated from the cache) -----------------------------------------------


def _write_catalog() -> None:
    root = _cache_root()
    rows: list[dict[str, Any]] = []
    for meta_path in sorted((root / "vectors").glob("*/meta.json")):
        meta = _read_json(meta_path)
        if not meta:
            continue
        npy = meta_path.parent / "vectors.npy"
        meta["vectors_bytes"] = npy.stat().st_size if npy.exists() else 0
        rows.append(meta)
    _write_json_atomic(root / "catalog.json", {"generated": _now(), "cells": rows})
    lines = ["# Vector cache catalog", "", f"Generated {_now()}. {len(rows)} cells.", ""]
    lines.append("| cell | corpus | profile | embedding | model_id | dim | count | MB |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        mb = f"{r.get('vectors_bytes', 0) / 1e6:.1f}"
        cols = [r["cell"], r["corpus"], r.get("profile", ""), r["embedding"], r["model_id"], r["dim"], r["count"], mb]
        lines.append("| " + " | ".join(str(c) for c in cols) + " |")
    (root / "catalog.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[catalog] {len(rows)} cells -> {root / 'catalog.json'} + catalog.md", flush=True)


# --- loadstore (populate a fresh store from a cached cell + score nDCG) ----------------


def _ndcg(ranked: list[str], rels: dict[str, int], k: int) -> float:
    import math

    dcg = sum((rels.get(doc_id, 0)) / math.log2(i + 2) for i, doc_id in enumerate(ranked[:k]))
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def _iter_cell_chunks(cell: str, batch: int) -> Any:
    """Yield (chunks, vectors) batches reconstructed from the cell's parquet + vectors.npy.

    The chunks dir is derived from the cell id (``<corpus>__<profile>__<embedding>`` -> its
    ``<corpus>__<profile>`` chunks dir), so loadstore works regardless of the current env.
    """
    parquet = _cache_root() / "chunks" / cell.rsplit("__", 1)[0] / "chunks.parquet"
    vectors = np.load(_cache_root() / "vectors" / cell / "vectors.npy", mmap_mode="r")
    row = 0
    for rb in pq.ParquetFile(parquet).iter_batches(batch_size=batch):
        cols = rb.to_pydict()
        n = len(cols["text"])
        chunks = [
            Chunk(
                text=cols["text"][i],
                source=SourceRef(uri=cols["source_uri"][i], label="", content_hash="", mtime=0.0),
                ordinal=int(cols["ordinal"][i]),
                token_count=int(cols["token_count"][i]),
            )
            for i in range(n)
        ]
        vecs = [tuple(float(x) for x in vectors[row + i]) for i in range(n)]
        row += n
        yield chunks, vecs


def _loadstore() -> None:
    from semdex.application.use_cases.searching import search

    cell = os.environ["SEMDEX_LOADSTORE_CELL"]
    meta = _read_json(_cache_root() / "vectors" / cell / "meta.json")
    if not meta:
        raise SystemExit(f"cell {cell} has no vectors meta.json - embed it first")
    label, model_id, dim = meta["embedding"], meta["model_id"], int(meta["dim"])
    chunks_dir = _cache_root() / "chunks" / cell.rsplit("__", 1)[0]
    backend = _STORE_BACKENDS[os.environ.get("SEMDEX_BENCH_STORE", "sqlite_vec")]
    collection = os.environ.get("SEMDEX_BENCH_COLLECTION", cell)
    k = int(os.environ.get("SEMDEX_LOADSTORE_K", "10"))
    store_dir = Path(os.environ.get("SEMDEX_BENCH_STORE_DIR") or tempfile.mkdtemp(prefix="semdex-loadstore-"))
    dsn = (
        os.environ.get("SEMDEX_BENCH_PGVECTOR_DSN")
        if backend is StoreBackend.PGVECTOR
        else os.environ.get("SEMDEX_BENCH_MARIADB_DSN")
        if backend is StoreBackend.MARIADB
        else None
    )
    store = build_vector_store(backend, store_dir, dsn=dsn)
    store.ensure_collection(Collection(name=collection, model_id=model_id, dim=dim))
    print(f"[loadstore] {cell} -> {backend.value}: populating from cache...", flush=True)
    t0 = time.perf_counter()
    total = 0
    for chunks, vecs in _iter_cell_chunks(cell, batch=2048):
        store.upsert(collection=collection, chunks=chunks, vectors=vecs)
        total += len(chunks)
    store.compact(collection=collection)
    load_s = time.perf_counter() - t0
    print(f"[loadstore] {cell} -> {backend.value}: {total} chunks in {load_s:.1f}s; scoring...", flush=True)

    embedding = build_embedding(
        _EMBED_MODELS[label][0],
        model=model_id,
        endpoint=os.environ.get("SEMDEX_BENCH_OLLAMA_URL"),
        allow_fallback=False,  # scoring must use the real query embedder, never the placeholder
    )
    queries = _read_json(chunks_dir / "queries.json") or {}
    qrels = _read_json(chunks_dir / "qrels.json") or {}
    scores: list[float] = []
    for qid, text in queries.items():
        hits = search(embedding=embedding, store=store, collection=collection, query=text, k=k)
        seen: list[str] = []
        for hit in hits:
            if hit.uri not in seen:
                seen.append(hit.uri)
        scores.append(_ndcg(seen, qrels.get(qid, {}), k))
    store.close()
    ndcg = sum(scores) / len(scores) if scores else 0.0
    print(
        f"[loadstore] {cell} -> {backend.value}: nDCG@{k}={ndcg:.4f} over {len(scores)} queries (load {load_s:.1f}s)",
        flush=True,
    )


def main() -> None:
    mode = os.environ.get("MODE", "embed")
    if mode == "loadstore":
        _loadstore()
        return
    if mode == "catalog":
        _write_catalog()
        return
    corpora = [
        c.strip()
        for c in os.environ.get("SEMDEX_PREEMBED_CORPORA", "nfcorpus,scifact,fiqa,cqadupstack").split(",")
        if c.strip()
    ]
    labels = [
        e.strip() for e in os.environ.get("SEMDEX_PREEMBED_EMBEDDINGS", "fastembed:bge-small").split(",") if e.strip()
    ]
    if mode == "chunk":
        # CPU-side ONLY: produce chunk sets, do NOT embed. Decouples the heavy GPU embedding
        # (MODE=embed_only) so it can run as a separate concentrated batch that frees the GPU.
        for corpus in corpora:
            _ensure_chunks(corpus)
        return
    if mode == "embed_only":
        # GPU-side ONLY: embed corpora whose chunks are ALREADY complete; skip (with notice) the
        # rest so this pass never blocks on chunking - the GPU streams pre-made chunks at full tilt
        # even while the CPU-bound chunk pass (or another CPU sweep) contends for the dev-box cores.
        for corpus in corpora:
            count = _chunks_complete(corpus)
            if count is None:
                print(f"[embed_only] {corpus}: chunks not ready - skip (run MODE=chunk first)", flush=True)
                continue
            for label in labels:
                _ensure_vectors(corpus, label, count)
        _write_catalog()
        return
    for corpus in corpora:
        count = _ensure_chunks(corpus)
        for label in labels:
            _ensure_vectors(corpus, label, count)
    _write_catalog()


# --- shared harness API ---------------------------------------------------------------
# Used by every scoring harness to find the vector cache and read its small json files.
# Public aliases so a strictly-checked sibling can import them without reportPrivateUsage;
# the underscore names stay for the scripts that already import them.
cache_root = _cache_root
read_json = _read_json
__all__ = ["cache_root", "read_json"]


if __name__ == "__main__":
    main()
