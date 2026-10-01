#!/usr/bin/env python
"""Freeze a GerDaLIR slice: German long documents with enough queries to resolve a chunking claim.

MLDR is the only German body on this page long enough to carry a chunking comparison, and it ships
200 test queries - the entire language, not a slice cap. Merging its dev split reaches 400, and the
overlap comparisons that matter need roughly 560 to 1,900 to resolve. So German conclusions were
structurally out of reach, and a chunking recommendation was resting on English alone.

GerDaLIR (German Dataset for Legal Information Retrieval, via mteb) fixes both halves at once:
German court decisions average 15,940 characters, about 15.6 chunks per document at 256 tokens
against a fitness threshold of 3, and it ships 12,298 judged queries. A slice of its 10,025 judged
documents plus distractors lands near 150,000 chunks, the same scale as the English MLDR slice, so
the two are directly comparable.

The output is the same frozen layout as the MLDR and MIRACL slices - corpus.jsonl, queries.json,
qrels.json, doc_ids.txt, meta.json - so every chunker and embedder run indexes an identical
document set and the existing pipeline reads it with no change.

Config (env; scripts/ convention, no argparse):
  SEMDEX_GERDALIR_DOCS    document budget (default 12000; judged docs are always kept)
  SEMDEX_GERDALIR_QUERIES cap on queries, 0 = all (default 0)
  SEMDEX_GERDALIR_SEED    reservoir RNG seed (default 20260813)
  SEMDEX_GERDALIR_REPO    HF dataset id (default mteb/GerDaLIR)
  SEMDEX_GERDALIR_HF_CACHE HF cache dir (default /corpora/hf/hub)
  SEMDEX_SLICE_OUT_GERDALIR output root (default /corpora/gerdalir-slices)
"""

from __future__ import annotations

import importlib
import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import orjson

_DEFAULT_OUT = "/corpora/gerdalir-slices"
_DEFAULT_DOCS = 12_000


def slice_name(docs: int) -> str:
    """Dir-safe slice id encoding the doc budget, so two budgets never share a vector cell."""
    return f"gerdalir_de_{round(docs / 1000)}k_slice"


def _snapshot() -> Path:
    """The cached dataset snapshot, downloading it only if absent."""
    repo = os.environ.get("SEMDEX_GERDALIR_REPO", "mteb/GerDaLIR")
    cache = os.environ.get("SEMDEX_GERDALIR_HF_CACHE", "/corpora/hf/hub")
    local = sorted(Path(cache).glob(f"datasets--{repo.replace('/', '--')}/snapshots/*"))
    if local:
        return local[-1]
    return Path(_hub_snapshot_download()(repo, repo_type="dataset", cache_dir=cache))


class _ParquetColumn(Protocol):
    def to_pylist(self) -> list[str]: ...


class _ParquetTable(Protocol):
    def column(self, name: str) -> _ParquetColumn: ...


class _ParquetReader(Protocol):
    def read_table(self, source: Path, *, columns: list[str]) -> _ParquetTable: ...


def _hub_snapshot_download() -> Callable[..., str]:
    """huggingface_hub.snapshot_download, typed to the str path it returns.

    The published package leaves the symbol partially unknown, so a direct import makes every
    value derived from it Unknown under strict mode. Naming the return type here keeps the one
    call site typed.
    """
    hub = importlib.import_module("huggingface_hub")
    return cast(Callable[..., str], hub.snapshot_download)


def _parquet() -> _ParquetReader:
    """pyarrow.parquet behind the two calls this script actually makes.

    pyarrow ships no stubs for the submodule. A local stub package would type it for every script
    here, but it would also SHADOW the real package for the ones that use far more of its API than
    this, so an incomplete stub would break them. Declaring only the surface used here types these
    call sites without taking that risk.
    """
    return cast(_ParquetReader, importlib.import_module("pyarrow.parquet"))


def _read(snapshot: Path, part: str, columns: list[str]) -> list[dict[str, str]]:
    pq = _parquet()
    rows: list[dict[str, str]] = []
    for path in sorted((snapshot / part).glob("*.parquet")):
        table = pq.read_table(path, columns=columns)
        cols = {name: table.column(name).to_pylist() for name in columns}
        rows.extend(dict(zip(columns, values, strict=True)) for values in zip(*cols.values(), strict=True))
    return rows


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def build(docs_budget: int, seed: int, query_cap: int) -> None:
    out = Path(os.environ.get("SEMDEX_SLICE_OUT_GERDALIR", _DEFAULT_OUT)) / slice_name(docs_budget)
    if (out / "meta.json").exists():
        print(f"[slice] {out.name}: already built - skip (delete the dir to rebuild)", flush=True)
        return
    snapshot = _snapshot()
    print(f"[slice] reading GerDaLIR from {snapshot}", flush=True)

    qrels: dict[str, dict[str, int]] = {}
    for row in _read(snapshot, "qrels", ["query-id", "corpus-id", "score"]):
        if int(row["score"]) > 0:
            qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
    queries = {str(r["id"]): str(r["text"]) for r in _read(snapshot, "queries", ["id", "text"])}
    queries = {q: t for q, t in queries.items() if q in qrels}
    if query_cap:
        queries = dict(sorted(queries.items())[:query_cap])
    qrels = {q: r for q, r in qrels.items() if q in queries}
    judged = {doc for rel in qrels.values() for doc in rel}
    print(f"[slice] {len(queries):,} judged queries, {len(judged):,} judged documents", flush=True)

    rng = random.Random(seed)  # noqa: S311 - deterministic reproducible sampling, NOT cryptographic
    judged_docs: dict[str, tuple[str, str]] = {}
    reservoir: list[tuple[str, tuple[str, str]]] = []
    seen = 0
    for row in _read(snapshot, "corpus", ["id", "text", "title"]):
        doc_id = str(row["id"])
        # Title and body stay SEPARATE: the frozen-slice loader joins them itself, exactly as the
        # MLDR and MIRACL slices are written, and pre-joining here would double the separator.
        text = str(row.get("text") or "")
        title = str(row.get("title") or "")
        if doc_id in judged:
            judged_docs[doc_id] = (title, text)
            continue
        # Algorithm R reservoir sample -> a uniform, seed-reproducible distractor set.
        seen += 1
        if len(reservoir) < docs_budget:
            reservoir.append((doc_id, (title, text)))
        else:
            slot = rng.randint(1, seen)
            if slot <= docs_budget:
                reservoir[slot - 1] = (doc_id, (title, text))

    slice_docs = dict(judged_docs)
    slice_docs.update(reservoir[: max(0, docs_budget - len(judged_docs))])
    doc_ids = sorted(slice_docs)  # canonical order = the pin
    present = set(doc_ids)
    # Drop judgements for documents the budget excluded, so recall has a reachable denominator.
    qrels = {q: {d: r for d, r in rel.items() if d in present} for q, rel in qrels.items()}
    qrels = {q: r for q, r in qrels.items() if r}
    queries = {q: t for q, t in queries.items() if q in qrels}

    out.mkdir(parents=True, exist_ok=True)
    _write_atomic(
        out / "corpus.jsonl",
        b"".join(
            orjson.dumps({"_id": d, "title": slice_docs[d][0], "text": slice_docs[d][1]}) + b"\n" for d in doc_ids
        ),
    )
    _write_atomic(out / "queries.json", orjson.dumps(queries))
    _write_atomic(out / "qrels.json", orjson.dumps(qrels))
    _write_atomic(out / "doc_ids.txt", ("\n".join(doc_ids) + "\n").encode())
    chars = sum(len(title) + len(text) for title, text in slice_docs.values())
    _write_atomic(
        out / "meta.json",
        orjson.dumps(
            {
                "slice": out.name,
                "lang": "de",
                "source": "mteb/GerDaLIR",
                "snapshot": snapshot.name,
                "seed": seed,
                "n_docs": len(doc_ids),
                "n_queries": len(queries),
                "n_judged_docs": len(judged_docs),
                "mean_doc_chars": round(chars / len(doc_ids), 1),
                "rels_per_query": round(sum(len(r) for r in qrels.values()) / len(qrels), 2),
            },
            option=orjson.OPT_INDENT_2,
        ),
    )
    print(
        f"[slice] {out.name}: {len(doc_ids):,} docs ({len(judged_docs):,} judged), "
        f"{len(queries):,} queries, mean {chars / len(doc_ids):,.0f} chars/doc -> {out}",
        flush=True,
    )


def main() -> None:
    build(
        docs_budget=int(os.environ.get("SEMDEX_GERDALIR_DOCS", _DEFAULT_DOCS)),
        seed=int(os.environ.get("SEMDEX_GERDALIR_SEED", "20260813")),
        query_cap=int(os.environ.get("SEMDEX_GERDALIR_QUERIES", "0")),
    )


if __name__ == "__main__":
    main()
