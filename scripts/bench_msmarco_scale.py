#!/usr/bin/env python
"""MSMARCO-scale component reference: capable stores x recursive+fastembed.

Purpose: decision support - how retrieval accuracy, index time, search latency,
and on-disk DB size behave as the corpus grows, for the capable components only.

Design: chunk + embed ONCE (the store-independent cost), then upsert the same
(chunks, vectors) into every store and measure the store-dependent costs
(upsert time, search latency, DB size). Vectors are held in RAM as tuples
(~12.4 KB per 384-dim vector), so scales are capped where that fits the box
(250K docs ~ 3.1 GB); beyond that the per-store numbers extrapolate linearly.

Reads a BEIR-format dir (corpus.jsonl + queries.jsonl + qrels/<split>.tsv),
streaming the corpus so a doc cap never materializes the whole 8.84M-line file.
Server DB sizes (pgvector / mariadb) are read with ``du`` inside the container.

Env:
  MSMARCO_DIR   BEIR dir (default ./bench-data/msmarco)
  MAX_DOCS      doc cap, comma list for several scales (default 50000)
  MAX_QUERIES   judged queries to score (default 50)
  STORES        comma list (default sqlite_vec,lancedb,pgvector,mariadb)
  SPLIT         qrels split (default dev)
  K             chunk hits per query (default 100)
  UPSERT_BATCH  chunks per upsert call (default 2000)
  OUT           results json (default ./msmarco-bench-results.json)

Persistent-store knobs (unset -> spin throwaway Docker + throwaway dirs, the default):
  SEMDEX_BENCH_PGVECTOR_DSN  target a running pgvector server instead of a container
  SEMDEX_BENCH_MARIADB_DSN   target a running mariadb server instead of a container
  SEMDEX_BENCH_STORE_DIR     keep embedded (json/sqlite/lance) stores under this root
                             (a kept dir, not wiped) instead of a throwaway workdir
  SEMDEX_BENCH_COLLECTION    collection name to populate/query (default "bench") -
                             point at a pre-embedded kept collection for instant re-eval
"""

# pyright: basic
# One-off benchmark harness on untyped optional deps (orjson, psycopg, pymysql, ir_datasets);
# strict mode would only add reportUnknown* noise for no safety gain (matches the other bench
# scripts and how the tests/fixtures generator is kept out of strict checking).

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path

import orjson

from semdex.application.use_cases import search
from semdex.composition import build_chunker, build_embedding, build_vector_store
from semdex.domain.enums import ChunkStrategy, EmbeddingBackend, StoreBackend
from semdex.domain.models import Collection, ExtractedDocument, SourceRef

NDCG_K = 10


# ----------------------------------------------------------------- metrics
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


def _pct(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


# ----------------------------------------------------------------- BEIR loader (streaming)
def _load_qrels(path: Path, max_q: int):
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as fh:
        fh.readline()  # header: query-id\tcorpus-id\tscore
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], int(float(parts[2]))
            if score > 0:
                qrels.setdefault(qid, {})[did] = score
    kept = list(qrels)[:max_q]
    return {q: qrels[q] for q in kept}


def _load_queries(path: Path, want: set[str]):
    out: dict[str, str] = {}
    with path.open("rb") as fh:
        for raw in fh:
            obj = orjson.loads(raw)
            if obj["_id"] in want:
                out[obj["_id"]] = obj.get("text", "")
    return out


def _stream_docs(path: Path, judged: set[str], max_docs: int):
    """Keep every judged doc + fill to max_docs with others. Streams; bounded to kept."""
    docs: dict[str, str] = {}
    fill_budget = max(0, max_docs - len(judged))
    filled = 0
    pending_judged = set(judged)
    with path.open("rb") as fh:
        for raw in fh:
            obj = orjson.loads(raw)
            did = obj["_id"]
            if did in pending_judged:
                pending_judged.discard(did)
            elif filled < fill_budget:
                filled += 1
            elif pending_judged:
                continue  # fill done, still hunting stragglers
            else:
                break  # fill done and every judged doc found - stop streaming
            docs[did] = (obj.get("title", "") + " " + obj.get("text", "")).strip()
    return docs


def load_corpus(beir_dir: Path, split: str, max_docs: int, max_q: int):
    qrels = _load_qrels(beir_dir / "qrels" / f"{split}.tsv", max_q)
    queries = _load_queries(beir_dir / "queries.jsonl", set(qrels))
    qrels = {q: r for q, r in qrels.items() if q in queries}
    judged = {d for r in qrels.values() for d in r}
    docs = _stream_docs(beir_dir / "corpus.jsonl", judged, max_docs)
    qrels = {q: {d: r for d, r in rel.items() if d in docs} for q, rel in qrels.items()}
    qrels = {q: r for q, r in qrels.items() if r}
    queries = {q: queries[q] for q in qrels}
    return docs, queries, qrels


# ----------------------------------------------------------------- docker DB helpers
def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _run_container(image, container_port, env, ready):
    port = _free_port()
    args = ["docker", "run", "-d", "--rm", "-p", f"{port}:{container_port}"]
    for k, v in env.items():
        args += ["-e", f"{k}={v}"]
    args.append(image)
    cid = subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if ready(port):
            return cid, port
        time.sleep(1)
    subprocess.run(["docker", "rm", "-f", cid], capture_output=True, check=False)
    raise RuntimeError(f"{image} not ready in time")


def _du_bytes(cid, datadir):
    subprocess.run(["docker", "exec", cid, "sync"], capture_output=True, check=False)
    out = subprocess.run(["docker", "exec", cid, "du", "-sb", datadir], capture_output=True, text=True, check=False)
    try:
        return int(out.stdout.split()[0])
    except Exception:
        return 0


def _pg_ready(port):
    import psycopg

    try:
        psycopg.connect(
            f"host=127.0.0.1 port={port} user=postgres password=semdex dbname=postgres", connect_timeout=2
        ).close()
    except Exception:
        return False
    return True


def _maria_ready(port):
    import pymysql

    try:
        pymysql.connect(
            host="127.0.0.1", port=port, user="root", password="semdex", database="semdex", connect_timeout=2
        ).close()
    except Exception:
        return False
    return True


# Persistent-server env knobs: when set, the run targets an already-running server
# (e.g. the px-semdex-test kept DBs) instead of spinning a throwaway Docker container,
# so pre-embedded collections can be re-queried without re-ingesting. Unset -> old behaviour.
_PERSIST_DSN = {
    StoreBackend.PGVECTOR: "SEMDEX_BENCH_PGVECTOR_DSN",
    StoreBackend.MARIADB: "SEMDEX_BENCH_MARIADB_DSN",
}


def _server(backend):
    persist = os.environ.get(_PERSIST_DSN.get(backend, ""))
    if persist:
        # external persistent server: no container to spin or tear down; on-disk size is
        # not measurable via `du` in a container we do not own, so db_bytes is reported 0.
        return {"external": True, "dsn_fixed": persist, "datadir": None}
    if backend is StoreBackend.PGVECTOR:
        return {
            "image": "pgvector/pgvector:pg16",
            "container_port": 5432,
            "env": {"POSTGRES_PASSWORD": "semdex"},
            "ready": _pg_ready,
            "datadir": "/var/lib/postgresql/data",
            "dsn": lambda p: f"host=127.0.0.1 port={p} user=postgres password=semdex dbname=postgres",
        }
    if backend is StoreBackend.MARIADB:
        return {
            "image": "mariadb:11.8",
            "container_port": 3306,
            "env": {"MARIADB_ROOT_PASSWORD": "semdex", "MARIADB_DATABASE": "semdex"},
            "ready": _maria_ready,
            "datadir": "/var/lib/mysql",
            "dsn": lambda p: f"mysql://root:semdex@127.0.0.1:{p}/semdex",
        }
    return None


def _bench_collection() -> str:
    """Target collection name; override to query/populate a named kept collection."""
    return os.environ.get("SEMDEX_BENCH_COLLECTION", "bench")


def _store_dir_for(backend, workdir: Path) -> tuple[Path, bool]:
    """Return (store_dir, persistent). SEMDEX_BENCH_STORE_DIR -> a kept embedded-store
    location that must NOT be wiped between runs; otherwise the throwaway workdir."""
    root = os.environ.get("SEMDEX_BENCH_STORE_DIR")
    if root:
        return Path(root) / backend.value, True
    return workdir / backend.value, False


# ----------------------------------------------------------------- shared prep
def chunk_docs(docs):
    """Chunk once - chunks are store-independent and small enough to hold at 1M."""
    chunker = build_chunker(ChunkStrategy.RECURSIVE, recipe="")
    t0 = time.perf_counter()
    chunks = []
    for did, text in docs.items():
        source = SourceRef(uri=str(Path(did)), label="", content_hash="", mtime=0.0)
        chunks.extend(chunker(ExtractedDocument(source=source, text=text), max_tokens=256))
    return chunks, time.perf_counter() - t0


def chunk_and_embed(docs, batch: int = 2048):
    embedding = build_embedding(EmbeddingBackend.FASTEMBED, allow_fallback=False)
    chunks, chunk_s = chunk_docs(docs)

    t0 = time.perf_counter()
    vectors = []
    for i in range(0, len(chunks), batch):
        vectors.extend(embedding.embed_passages([c.text for c in chunks[i : i + batch]]))
        done = min(i + batch, len(chunks))
        if done % 20480 < batch:
            rate = done / (time.perf_counter() - t0)
            print(f"    embedded {done}/{len(chunks)} ({rate:.0f} chunks/s)", flush=True)
    embed_s = time.perf_counter() - t0
    return chunks, vectors, embedding, chunk_s, embed_s


# ----------------------------------------------------------------- one (scale, store) cell
def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _ranked_docs(hits, limit):
    ranked, seen = [], set()
    for h in hits:
        d = str(h.source_path)
        if d not in seen:
            seen.add(d)
            ranked.append(d)
        if len(ranked) >= limit:
            break
    return ranked


def measure_store(backend, chunks, *, vectors, embedding, queries, qrels, workdir: Path, k: int, upsert_batch: int):
    srv = _server(backend)
    external = bool(srv and srv.get("external"))
    cid = None
    dsn = None
    store_dir, _persistent = _store_dir_for(backend, workdir)
    store_dir.mkdir(parents=True, exist_ok=True)
    collection = _bench_collection()
    try:
        if srv and external:
            dsn = srv["dsn_fixed"]
        elif srv:
            cid, port = _run_container(srv["image"], srv["container_port"], srv["env"], srv["ready"])
            dsn = srv["dsn"](port)
        store = build_vector_store(backend, store_dir, dsn=dsn)

        t0 = time.perf_counter()
        store.ensure_collection(Collection(name=collection, model_id=embedding.model_id, dim=embedding.dim))
        for i in range(0, len(chunks), upsert_batch):
            store.upsert(
                collection=collection, chunks=chunks[i : i + upsert_batch], vectors=vectors[i : i + upsert_batch]
            )
        store.compact(collection=collection)  # upsert only appends; compact builds/folds the ANN index
        upsert_s = time.perf_counter() - t0

        ndcgs, recalls, rrs, lat = [], [], [], []
        for qid, text in queries.items():
            rel = qrels[qid]
            relevant = set(rel)
            t1 = time.perf_counter()
            hits = search(embedding=embedding, store=store, collection=collection, query=text, k=k)
            lat.append(time.perf_counter() - t1)
            ranked = _ranked_docs(hits, NDCG_K)
            ndcgs.append(_ndcg(ranked, rel, NDCG_K))
            recalls.append(_recall(ranked, relevant, NDCG_K))
            rrs.append(_mrr(ranked, relevant))
        lat.sort()

        size = _du_bytes(cid, srv["datadir"]) if (srv and not external) else _dir_size(store_dir)
        return {
            "store": backend.value,
            "ndcg@10": round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else 0.0,
            "recall@10": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
            "mrr@10": round(sum(rrs) / len(rrs), 4) if rrs else 0.0,
            "upsert_s": round(upsert_s, 1),
            "search_p50_ms": round(_pct(lat, 0.50) * 1000, 1),
            "search_p95_ms": round(_pct(lat, 0.95) * 1000, 1),
            "db_bytes": size,
            "db_mb": round(size / 1_000_000, 1),
        }
    finally:
        if cid:
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True, check=False)


def measure_stores_streaming(backends, chunks, *, embedding, queries, qrels, workdir: Path, k: int, batch: int):
    """Streaming variant for scales whose vectors do not fit RAM: each batch is embedded
    ONCE and upserted into every store in the same pass (embedding dominates at ~40 chunks/s
    on this box, so sharing it across stores halves-or-better the wall time). Per-store
    upsert/search/size are still measured separately; embed_s is the shared cost."""
    handles = []  # (backend, store, cid, srv, store_dir)
    collection = _bench_collection()
    try:
        for backend in backends:
            srv = _server(backend)
            cid = None
            dsn = None
            store_dir, _persistent = _store_dir_for(backend, workdir)
            store_dir.mkdir(parents=True, exist_ok=True)
            if srv and srv.get("external"):
                dsn = srv["dsn_fixed"]
            elif srv:
                cid, port = _run_container(srv["image"], srv["container_port"], srv["env"], srv["ready"])
                dsn = srv["dsn"](port)
            store = build_vector_store(backend, store_dir, dsn=dsn)
            store.ensure_collection(Collection(name=collection, model_id=embedding.model_id, dim=embedding.dim))
            handles.append([backend, store, cid, srv, store_dir])

        embed_s = 0.0
        upsert_s = {h[0].value: 0.0 for h in handles}
        t_start = time.perf_counter()
        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            t0 = time.perf_counter()
            vectors = embedding.embed_passages([c.text for c in part])
            embed_s += time.perf_counter() - t0
            for backend, store, _cid, _srv, _sd in handles:
                t0 = time.perf_counter()
                store.upsert(collection=collection, chunks=part, vectors=vectors)
                upsert_s[backend.value] += time.perf_counter() - t0
            done = min(i + batch, len(chunks))
            if done % 51200 < batch:
                rate = done / (time.perf_counter() - t_start)
                print(f"    [stream x{len(handles)}] {done}/{len(chunks)} ({rate:.0f} chunks/s)", flush=True)

        for backend, store, _cid, _srv, _sd in handles:  # upsert only appends; fold the ANN index
            t0 = time.perf_counter()
            store.compact(collection=collection)
            upsert_s[backend.value] += time.perf_counter() - t0

        rows = []
        for backend, store, cid, srv, store_dir in handles:
            ndcgs, recalls, rrs, lat = [], [], [], []
            for qid, text in queries.items():
                rel = qrels[qid]
                relevant = set(rel)
                t1 = time.perf_counter()
                hits = search(embedding=embedding, store=store, collection=collection, query=text, k=k)
                lat.append(time.perf_counter() - t1)
                ranked = _ranked_docs(hits, NDCG_K)
                ndcgs.append(_ndcg(ranked, rel, NDCG_K))
                recalls.append(_recall(ranked, relevant, NDCG_K))
                rrs.append(_mrr(ranked, relevant))
            lat.sort()
            size = _du_bytes(cid, srv["datadir"]) if (srv and not srv.get("external")) else _dir_size(store_dir)
            rows.append(
                {
                    "store": backend.value,
                    "mode": "stream",
                    "ndcg@10": round(sum(ndcgs) / len(ndcgs), 4) if ndcgs else 0.0,
                    "recall@10": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
                    "mrr@10": round(sum(rrs) / len(rrs), 4) if rrs else 0.0,
                    "embed_s": round(embed_s, 1),
                    "upsert_s": round(upsert_s[backend.value], 1),
                    "search_p50_ms": round(_pct(lat, 0.50) * 1000, 1),
                    "search_p95_ms": round(_pct(lat, 0.95) * 1000, 1),
                    "db_bytes": size,
                    "db_mb": round(size / 1_000_000, 1),
                }
            )
        return rows
    finally:
        for _backend, _store, cid, _srv, _sd in handles:
            if cid:
                subprocess.run(["docker", "rm", "-f", cid], capture_output=True, check=False)


def main():
    beir_dir = Path(os.environ.get("MSMARCO_DIR", "./bench-data/msmarco"))
    scales = [int(x) for x in os.environ.get("MAX_DOCS", "50000").split(",")]
    max_q = int(os.environ.get("MAX_QUERIES", "50"))
    split = os.environ.get("SPLIT", "dev")
    k = int(os.environ.get("K", "100"))
    upsert_batch = int(os.environ.get("UPSERT_BATCH", "2000"))
    # Above this scale the vectors will not fit RAM (tuple floats ~12.4 KB/vec on 16 GB),
    # so the run switches to the streaming (re-embed per store) mode.
    embed_once_max = int(os.environ.get("EMBED_ONCE_MAX", "300000"))
    store_names = os.environ.get("STORES", "sqlite_vec,lancedb,pgvector,mariadb").split(",")
    backends = [StoreBackend(s.strip()) for s in store_names if s.strip()]
    out = Path(os.environ.get("OUT", "msmarco-bench-results.json"))
    workroot = Path(os.environ.get("WORKROOT", "./bench-data/stores"))

    import shutil

    results = []
    for scale in scales:
        t = time.perf_counter()
        docs, queries, qrels = load_corpus(beir_dir, split, scale, max_q)
        print(
            f"[scale={scale}] loaded docs={len(docs)} queries={len(queries)} "
            f"judged={sum(len(r) for r in qrels.values())} in {time.perf_counter() - t:.0f}s",
            flush=True,
        )
        streaming = scale > embed_once_max
        if streaming:
            chunks, chunk_s = chunk_docs(docs)
            embedding = build_embedding(EmbeddingBackend.FASTEMBED, allow_fallback=False)
            vectors, embed_s = None, None
            print(f"[scale={scale}] STREAM mode, chunks={len(chunks)} chunk_s={chunk_s:.1f}", flush=True)
        else:
            chunks, vectors, embedding, chunk_s, embed_s = chunk_and_embed(docs)
            print(f"[scale={scale}] chunks={len(chunks)} chunk_s={chunk_s:.1f} embed_s={embed_s:.1f}", flush=True)
        docs = None  # free the raw text; chunks carry what the stores need
        wd = workroot / f"s{scale}"
        if streaming:
            try:
                rows = measure_stores_streaming(
                    backends,
                    chunks,
                    embedding=embedding,
                    queries=queries,
                    qrels=qrels,
                    workdir=wd,
                    k=k,
                    batch=upsert_batch,
                )
                for row in rows:
                    row.update({"scale": scale, "chunks": len(chunks), "chunk_s": round(chunk_s, 1)})
                    results.append(row)
                    print("  " + json.dumps(row), flush=True)
            except Exception as exc:  # record the gap, keep other scales alive
                print(f"  stream scale={scale} FAILED: {type(exc).__name__}: {exc}", flush=True)
                results.append({"scale": scale, "mode": "stream", "error": f"{type(exc).__name__}: {exc}"})
            shutil.rmtree(wd, ignore_errors=True)
            out.write_text(json.dumps(results, indent=2))
            continue
        assert embed_s is not None  # non-streaming path: chunk_and_embed always set embed_s
        for backend in backends:
            try:
                row = measure_store(
                    backend,
                    chunks,
                    vectors=vectors,
                    embedding=embedding,
                    queries=queries,
                    qrels=qrels,
                    workdir=wd,
                    k=k,
                    upsert_batch=upsert_batch,
                )
                row.update(
                    {"scale": scale, "chunks": len(chunks), "chunk_s": round(chunk_s, 1), "embed_s": round(embed_s, 1)}
                )
                results.append(row)
                print("  " + json.dumps(row), flush=True)
            except Exception as exc:  # a store failing is a recorded gap, not a run abort
                print(f"  {backend.value} FAILED: {type(exc).__name__}: {exc}", flush=True)
                results.append({"store": backend.value, "scale": scale, "error": f"{type(exc).__name__}: {exc}"})
            shutil.rmtree(wd / backend.value, ignore_errors=True)  # bound disk between backends
            out.write_text(json.dumps(results, indent=2))  # persist incrementally
    print(f"\nwrote {out} ({len(results)} rows)", flush=True)


if __name__ == "__main__":
    main()
