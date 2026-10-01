#!/usr/bin/env python
# pyright: basic
"""Build fixed, reproducible MIRACL 100k slices (per language) as reusable corpora.

A slice = ALL qrels-judged docs + a seeded reservoir sample of distractors, to
exactly ``SEMDEX_SLICE_SIZE`` passages. It is saved self-contained (corpus.jsonl
+ queries + qrels + doc_ids + meta) so every later chunk/embed run indexes the
IDENTICAL passage set - the only way a chunker/embedder/store A/B stays
comparable. Deterministic (fixed seed + sorted ids); idempotent (a built slice is
never re-sampled - delete its dir to rebuild).

Config (env; scripts/ convention, no argparse):
  SEMDEX_SLICE_LANGS   comma list of MIRACL langs (default "de,en")
  SEMDEX_SLICE_SIZE    passages per slice (default 100000)
  SEMDEX_SLICE_SEED    reservoir RNG seed (default 20260713)
  IR_DATASETS_HOME     ir_datasets cache (default /corpora/ir_datasets)
  SEMDEX_MIRACL_DOCS   docs shard dir template (default
                       /corpora/ir_datasets/miracl/v1.0/{lang}/docs)
  SEMDEX_SLICE_OUT     output root (default /corpora/miracl-slices)

queries+qrels come from ir_datasets ``miracl/{lang}/dev`` (lazy - no docs
download). Docs are streamed from the local MIRACL ``*.jsonl.gz`` shards
directly, so a lang whose shards are not present yet (e.g. ``en`` before its
corpus download finishes) is skipped with a warning; re-run once they land.
"""

from __future__ import annotations

import gzip
import os
import random
from pathlib import Path
from typing import Any

import orjson

_DEFAULT_DOCS = "/corpora/ir_datasets/miracl/v1.0/{lang}/docs"
_DEFAULT_OUT = "/corpora/miracl-slices"


def _shard_dir(lang: str) -> Path:
    return Path(os.environ.get("SEMDEX_MIRACL_DOCS", _DEFAULT_DOCS).format(lang=lang))


def _out_dir(lang: str) -> Path:
    return Path(os.environ.get("SEMDEX_SLICE_OUT", _DEFAULT_OUT)) / f"miracl_{lang}_100k_slice"


def _write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a temp file + rename (no half-written slice)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _load_queries_qrels(lang: str) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Return (queries id->text, qrels qid->{doc_id: rel}) for ``miracl/{lang}/dev``.

    Only ``queries_iter``/``qrels_iter`` are touched, which ir_datasets serves from
    the tiny topics/qrels files without downloading the big docs corpus.
    """
    os.environ.setdefault("IR_DATASETS_HOME", "/corpora/ir_datasets")
    import ir_datasets

    dataset = ir_datasets.load(f"miracl/{lang}/dev")
    qrels: dict[str, dict[str, int]] = {}
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:
            qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = int(qrel.relevance)
    queries = {q.query_id: q.text for q in dataset.queries_iter() if q.query_id in qrels}
    qrels = {q: r for q, r in qrels.items() if q in queries}
    return queries, qrels


def _iter_shard_rows(shard_dir: Path) -> Any:
    """Yield every doc row from the language's gzipped shards, in shard+line order (fixed)."""
    for shard in sorted(shard_dir.glob("*.jsonl.gz")):
        with gzip.open(shard, "rb") as handle:
            for raw in handle:
                yield orjson.loads(raw)


def _build_lang(lang: str, size: int, seed: int) -> None:
    """Build + save the slice for one language (idempotent)."""
    out = _out_dir(lang)
    if (out / "meta.json").exists():
        print(f"[slice] miracl_{lang}_100k_slice: already built - skip (delete the dir to rebuild)", flush=True)
        return
    shard_dir = _shard_dir(lang)
    if not sorted(shard_dir.glob("*.jsonl.gz")):
        print(f"[slice] {lang}: no docs shards in {shard_dir} - SKIP (download the corpus first)", flush=True)
        return

    print(f"[slice] {lang}: loading queries/qrels...", flush=True)
    queries, qrels = _load_queries_qrels(lang)
    judged = {doc_id for rel in qrels.values() for doc_id in rel}
    # Reservoir target = the full slice size, then trimmed to fill EXACTLY to `size`
    # once the FOUND-judged count is known - some qrels docs are absent from the
    # corpus dump, and those are backfilled with distractors (not left as a shortfall).
    budget = size
    print(f"[slice] {lang}: streaming shards; judged={len(judged)} target={size}", flush=True)

    rng = random.Random(seed)  # noqa: S311 - deterministic reproducible sampling, NOT cryptographic
    judged_docs: dict[str, tuple[str, str]] = {}
    reservoir: list[tuple[str, str, str]] = []
    seen = 0
    for row in _iter_shard_rows(shard_dir):
        doc_id = row["docid"]
        title = row.get("title") or ""
        text = row.get("text") or ""
        if doc_id in judged:
            judged_docs[doc_id] = (title, text)
        else:
            # Algorithm R reservoir sample -> a uniform, seed-reproducible distractor set.
            seen += 1
            if len(reservoir) < budget:
                reservoir.append((doc_id, title, text))
            else:
                slot = rng.randint(1, seen)
                if slot <= budget:
                    reservoir[slot - 1] = (doc_id, title, text)

    slice_docs: dict[str, tuple[str, str]] = dict(judged_docs)
    n_distractors = max(0, size - len(judged_docs))  # fill to EXACTLY size (backfills missing judged docs)
    for doc_id, title, text in reservoir[:n_distractors]:
        slice_docs[doc_id] = (title, text)
    doc_ids = sorted(slice_docs)  # canonical order = the pin
    present = set(doc_ids)
    qrels = {q: {d: r for d, r in rel.items() if d in present} for q, rel in qrels.items()}
    qrels = {q: r for q, r in qrels.items() if r}
    queries = {q: queries[q] for q in qrels}

    out.mkdir(parents=True, exist_ok=True)
    corpus = b"".join(
        orjson.dumps({"_id": d, "title": slice_docs[d][0], "text": slice_docs[d][1]}) + b"\n" for d in doc_ids
    )
    _write_atomic(out / "corpus.jsonl", corpus)
    _write_atomic(out / "queries.json", orjson.dumps(queries, option=orjson.OPT_INDENT_2))
    _write_atomic(out / "qrels.json", orjson.dumps(qrels, option=orjson.OPT_INDENT_2))
    _write_atomic(out / "doc_ids.txt", ("\n".join(doc_ids) + "\n").encode())
    meta = {
        "corpus": f"miracl_{lang}_100k_slice",
        "lang": lang,
        "source": f"miracl/{lang}/dev",
        "seed": seed,
        "total": len(doc_ids),
        "n_judged": len(judged_docs),
        "n_distractors": len(doc_ids) - len(judged_docs),
        "n_queries": len(queries),
        "missing_judged_docs": len(judged) - len(judged_docs),
    }
    _write_atomic(out / "meta.json", orjson.dumps(meta, option=orjson.OPT_INDENT_2))
    print(
        f"[slice] miracl_{lang}_100k_slice: {len(doc_ids)} docs "
        f"(judged {len(judged_docs)} + distractors {len(doc_ids) - len(judged_docs)}), "
        f"{len(queries)} queries -> {out}",
        flush=True,
    )


def main() -> None:
    langs = [x.strip() for x in os.environ.get("SEMDEX_SLICE_LANGS", "de,en").split(",") if x.strip()]
    size = int(os.environ.get("SEMDEX_SLICE_SIZE", "100000"))
    seed = int(os.environ.get("SEMDEX_SLICE_SEED", "20260713"))
    for lang in langs:
        _build_lang(lang, size, seed)


if __name__ == "__main__":
    main()
