#!/usr/bin/env python
# pyright: basic
"""Build fixed, reproducible MLDR slices (per language) as reusable long-document corpora.

MLDR (Multilingual Long-Document Retrieval, from BGE-M3) is the corpus a CHUNK benchmark
needs: its documents are genuinely longer than any chunk size we sweep (en ~13 chunks/doc,
de ~33 chunks/doc at 256 tokens, by MEDIAN as well as mean). Passage corpora like MIRACL and
MS MARCO are ~1.0 chunks/doc, so re-chunking them measures nothing - see
docs/benchmarks/01-method.md ("Corpus fitness").

A slice = ALL qrels-judged docs + a seeded reservoir sample of distractors, to exactly
``docs`` documents. Saved self-contained (corpus.jsonl + queries + qrels + doc_ids + meta) in
the SAME layout as the MIRACL slices, so every later chunk/embed run indexes the IDENTICAL
document set. Deterministic (fixed seed + sorted ids); idempotent (a built slice is never
re-sampled - delete its dir to rebuild).

Size is budgeted in DOCUMENTS but chosen from a CHUNK budget: MLDR inverts MIRACL's shape
(few docs, each huge), so ~100k chunks/profile is ~3k de docs and ~8k en docs. A 100k-DOC
slice is impossible for de - the entire German corpus is 10k docs.

Config (env; scripts/ convention, no argparse):
  SEMDEX_MLDR_LANGS      comma list (default "de,en")
  SEMDEX_MLDR_DOCS       per-lang doc budget "lang:n,..." (default "de:3000,en:8000")
  SEMDEX_MLDR_SEED       reservoir RNG seed (default 20260717)
  SEMDEX_MLDR_REPO       HF dataset id (default Shitao/MLDR)
  SEMDEX_MLDR_HF_CACHE   HF cache dir (default /corpora/hf/hub)
  SEMDEX_MLDR_SLICE_OUT  output root (default /corpora/mldr-slices)
"""

from __future__ import annotations

import gzip
import os
import random
from pathlib import Path
from typing import Any

import orjson

_DEFAULT_DOCS = "de:3000,en:8000"
_DEFAULT_OUT = "/corpora/mldr-slices"


def _slice_name(lang: str, docs: int) -> str:
    """Dir-safe slice id encoding the doc budget, so two budgets never share a vector cell."""
    return f"mldr_{lang}_{round(docs / 1000)}k_slice"


def _out_dir(lang: str, docs: int) -> Path:
    return Path(os.environ.get("SEMDEX_MLDR_SLICE_OUT", _DEFAULT_OUT)) / _slice_name(lang, docs)


def _doc_budgets() -> dict[str, int]:
    raw = os.environ.get("SEMDEX_MLDR_DOCS", _DEFAULT_DOCS)
    return {part.split(":")[0].strip(): int(part.split(":")[1]) for part in raw.split(",") if part.strip()}


def _write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a temp file + rename (no half-written slice)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _splits() -> list[str]:
    """Which MLDR splits to draw queries from.

    German ships only 200 test queries, which is the entire language - not a slice cap - and a
    chunking comparison on 200 queries leaves most differences unresolvable. Its dev split holds a
    further 200 with DISJOINT query ids and qrels of the same construction, so merging them is the
    only way to double German power without leaving MLDR. Default stays "test" so existing slices
    rebuild identically.
    """
    return [s.strip() for s in os.environ.get("SEMDEX_MLDR_SPLITS", "test").split(",") if s.strip()]


def _repo_files(lang: str, split: str) -> tuple[Path, Path, Path]:
    """Return (corpus.jsonl.gz, <split>.jsonl.gz, qrels.tsv) for ``lang``, fetching if absent."""
    from huggingface_hub import hf_hub_download

    repo = os.environ.get("SEMDEX_MLDR_REPO", "Shitao/MLDR")
    cache = os.environ.get("SEMDEX_MLDR_HF_CACHE", "/corpora/hf/hub")
    corpus = hf_hub_download(repo, f"mldr-v1.0-{lang}/corpus.jsonl.gz", repo_type="dataset", cache_dir=cache)
    queries = hf_hub_download(repo, f"mldr-v1.0-{lang}/{split}.jsonl.gz", repo_type="dataset", cache_dir=cache)
    qrels = hf_hub_download(repo, f"qrels/qrels.mldr-v1.0-{lang}-{split}.tsv", repo_type="dataset", cache_dir=cache)
    return Path(corpus), Path(queries), Path(qrels)


def _load_queries_qrels(test_path: Path, qrels_path: Path) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Return (queries id->text, qrels qid->{doc_id: rel}) from the test split + TREC qrels."""
    qrels: dict[str, dict[str, int]] = {}
    with qrels_path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 4 and int(parts[3]) > 0:  # qid Q0 docid rel
                qrels.setdefault(parts[0], {})[parts[2]] = int(parts[3])
    queries: dict[str, str] = {}
    with gzip.open(test_path, "rt", encoding="utf-8") as handle:
        for raw in handle:
            row = orjson.loads(raw)
            if row["query_id"] in qrels:
                queries[row["query_id"]] = row["query"]
    qrels = {q: r for q, r in qrels.items() if q in queries}
    return queries, qrels


def _iter_corpus(corpus_path: Path) -> Any:
    """Yield every doc row from the language's gzipped corpus, in file order (fixed)."""
    with gzip.open(corpus_path, "rb") as handle:
        for raw in handle:
            yield orjson.loads(raw)


def _build_lang(lang: str, docs: int, seed: int) -> None:
    """Build + save the slice for one language (idempotent)."""
    out = _out_dir(lang, docs)
    name = _slice_name(lang, docs)
    if (out / "meta.json").exists():
        print(f"[slice] {name}: already built - skip (delete the dir to rebuild)", flush=True)
        return

    print(f"[slice] {lang}: fetching/locating MLDR files...", flush=True)
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    corpus_path = None
    for split in _splits():
        corpus_path, queries_path, qrels_path = _repo_files(lang, split)
        part_q, part_r = _load_queries_qrels(queries_path, qrels_path)
        # Disjoint by construction in MLDR, but assert it: a silent id collision between splits
        # would let one query's judgements overwrite another's and shrink the set without a word.
        clash = set(part_q) & set(queries)
        if clash:
            raise ValueError(f"{lang}: query ids repeat across splits ({len(clash)}), refusing to merge")
        queries.update(part_q)
        qrels.update(part_r)
        print(f"[slice] {lang}: split {split} -> {len(part_q)} queries (total {len(queries)})", flush=True)
    assert corpus_path is not None
    judged = {doc_id for rel in qrels.values() for doc_id in rel}
    print(f"[slice] {lang}: streaming corpus; judged={len(judged)} target={docs}", flush=True)

    rng = random.Random(seed)  # noqa: S311 - deterministic reproducible sampling, NOT cryptographic
    judged_docs: dict[str, str] = {}
    reservoir: list[tuple[str, str]] = []
    seen = 0
    for row in _iter_corpus(corpus_path):
        doc_id = row["docid"]
        text = row.get("text") or ""
        if doc_id in judged:
            judged_docs[doc_id] = text
        else:
            # Algorithm R reservoir sample -> a uniform, seed-reproducible distractor set.
            seen += 1
            if len(reservoir) < docs:
                reservoir.append((doc_id, text))
            else:
                slot = rng.randint(1, seen)
                if slot <= docs:
                    reservoir[slot - 1] = (doc_id, text)

    slice_docs: dict[str, str] = dict(judged_docs)
    n_distractors = max(0, docs - len(judged_docs))  # fill to EXACTLY docs (backfills missing judged)
    slice_docs.update(reservoir[:n_distractors])  # reservoir items are (doc_id, text) pairs
    doc_ids = sorted(slice_docs)  # canonical order = the pin
    present = set(doc_ids)
    qrels = {q: {d: r for d, r in rel.items() if d in present} for q, rel in qrels.items()}
    qrels = {q: r for q, r in qrels.items() if r}
    queries = {q: queries[q] for q in qrels}

    out.mkdir(parents=True, exist_ok=True)
    # title kept (empty) so the slice matches the MIRACL slice layout the preembed loader reads.
    corpus = b"".join(orjson.dumps({"_id": d, "title": "", "text": slice_docs[d]}) + b"\n" for d in doc_ids)
    _write_atomic(out / "corpus.jsonl", corpus)
    _write_atomic(out / "queries.json", orjson.dumps(queries, option=orjson.OPT_INDENT_2))
    _write_atomic(out / "qrels.json", orjson.dumps(qrels, option=orjson.OPT_INDENT_2))
    _write_atomic(out / "doc_ids.txt", ("\n".join(doc_ids) + "\n").encode())
    chars = sum(len(t) for t in slice_docs.values())
    meta = {
        "corpus": name,
        "lang": lang,
        "source": f"{os.environ.get('SEMDEX_MLDR_REPO', 'Shitao/MLDR')} mldr-v1.0-{lang}/test",
        "seed": seed,
        "total": len(doc_ids),
        "n_judged": len(judged_docs),
        "n_distractors": len(doc_ids) - len(judged_docs),
        "n_queries": len(queries),
        "missing_judged_docs": len(judged) - len(judged_docs),
        "total_chars": chars,
        "mean_chars_per_doc": round(chars / len(doc_ids)) if doc_ids else 0,
    }
    _write_atomic(out / "meta.json", orjson.dumps(meta, option=orjson.OPT_INDENT_2))
    print(
        f"[slice] {name}: {len(doc_ids)} docs "
        f"(judged {len(judged_docs)} + distractors {len(doc_ids) - len(judged_docs)}), "
        f"{len(queries)} queries, {chars / 1e6:.0f}M chars -> {out}",
        flush=True,
    )


def main() -> None:
    langs = [x.strip() for x in os.environ.get("SEMDEX_MLDR_LANGS", "de,en").split(",") if x.strip()]
    budgets = _doc_budgets()
    seed = int(os.environ.get("SEMDEX_MLDR_SEED", "20260717"))
    for lang in langs:
        if lang not in budgets:
            print(f"[slice] {lang}: no doc budget in SEMDEX_MLDR_DOCS - skip", flush=True)
            continue
        _build_lang(lang, budgets[lang], seed)


if __name__ == "__main__":
    main()
