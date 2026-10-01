#!/usr/bin/env python
# pyright: basic
# Benchmark harness on pyarrow/model2vec (no strict stubs); strict mode would only add
# reportUnknown* noise. Same stance as audit_chunk_dimensions.py and score_chunk_sweep.py.
"""Audit whether an embedder silently truncates the chunks a sweep hands it.

A chunk set is capped by the CHUNKER, in the chunker's tokenizer. The embedder then applies
its OWN cap, in its OWN tokenizer, at embed time - and that second cap is silent: no error, no
warning, no log line. The vector is simply computed from a prefix, so a truncated cell is
indistinguishable from an honest one in every downstream table.

The gap between the two caps is normally wide enough that nothing happens. An overlap ladder
closes it: overlap is appended context, so a chunk grows past ``max_tokens`` by design, and far
enough up the ladder it reaches the embedder's limit. At that point the sweep stops varying
overlap alone and starts varying overlap-and-how-much-got-clipped, which is not the measurement
anyone asked for. This names the rung where that begins.

It is measured, never predicted: each chunk goes through the embedder's real tokenize path twice,
once capped and once uncapped, and the two lengths are compared. ``model2vec`` in particular
applies TWO cuts - a raw-CHARACTER cut to ``max_length * median_token_length`` before tokenizing,
then an id slice to ``max_length`` - so counting tokens with the bare tokenizer sees neither, and
which cut binds depends on the text. Driving the real method is the only way to be sure.

Coverage is explicit rather than assumed: ``_PROBES`` lists the embedder families whose cap this
script knows how to drive. An embedder that is not listed is reported as ``uncovered``, never as
untruncated - a silent pass for an embedder nobody measured is the failure this exists to prevent.

Bounded memory: the parquet is streamed one batch at a time and only counters survive a batch,
so peak stays flat regardless of set size.

Exit codes: 0 no truncation found, 1 truncation found (a finding, not an error), 2 error.

Env:
  CACHE_ROOT        cache dir (default /embeddings)
  AUDIT_SETS        optional comma-separated chunk-set dir names (default: every set in the cache)
  AUDIT_EMBEDDINGS  comma-separated embedder labels (default: the model2vec pair the ladder uses)
  AUDIT_SAMPLE      chunks to sample per set, spread across the file; 0 = every chunk (default 0)
  OUT               results json (default tests/benchmarks/raw/embedder-truncation-audit.json).
                    A run MERGES into it by (chunk set, embedder): rows it did not measure are
                    kept, rows it re-measured are replaced, and a full row (sample 0) is never
                    replaced by a sampled one. Each row records the sample it was measured on.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parent.parent
_BATCH = 4096
# onnxruntime warns noisily about thread affinity in a container unless the count is explicit.
_THREADS = 2


class _HasTokenizer(Protocol):
    """The one attribute a fastembed ONNX text model exposes that this audit needs."""

    tokenizer: Any


class Tokenizes(Protocol):
    """The one operation an audit needs from an embedder: tokenize with and without its cap."""

    # Positional-only: the implementation is a third-party model whose parameter is named
    # `sentences`, and the protocol must not impose a name the library does not use.
    def tokenize(self, texts: Sequence[str], /, max_length: int | None) -> list[list[int]]: ...


@dataclass(frozen=True)
class Probe:
    """An embedder's real tokenize path, plus the caps it applies."""

    label: str
    tokenizes: Tokenizes
    token_cap: int
    char_precut: int | None


@dataclass(frozen=True)
class Counts:
    """Truncation counters for one (chunk set, embedder) pair. Merges associatively."""

    chunks: int = 0
    truncated: int = 0
    tokens_full: int = 0
    tokens_kept: int = 0
    max_tokens_full: int = 0
    max_chars: int = 0

    def merge(self, other: Counts) -> Counts:
        return Counts(
            chunks=self.chunks + other.chunks,
            truncated=self.truncated + other.truncated,
            tokens_full=self.tokens_full + other.tokens_full,
            tokens_kept=self.tokens_kept + other.tokens_kept,
            max_tokens_full=max(self.max_tokens_full, other.max_tokens_full),
            max_chars=max(self.max_chars, other.max_chars),
        )


def count_batch(texts: Sequence[str], probe: Probe) -> Counts:
    """Compare capped against uncapped tokenization for one batch of chunk texts.

    Both calls go through the embedder's own method, so every cut it applies is exercised -
    including one that happens before tokenization and is invisible to a token count.
    """
    if not texts:
        return Counts()
    kept = probe.tokenizes.tokenize(texts, max_length=probe.token_cap)
    full = probe.tokenizes.tokenize(texts, max_length=None)
    if len(kept) != len(full) or len(kept) != len(texts):
        raise RuntimeError(f"{probe.label}: tokenize returned {len(kept)}/{len(full)} rows for {len(texts)} texts")
    return Counts(
        chunks=len(texts),
        truncated=sum(1 for k, f in zip(kept, full, strict=True) if len(k) < len(f)),
        tokens_full=sum(len(f) for f in full),
        tokens_kept=sum(len(k) for k in kept),
        max_tokens_full=max(len(f) for f in full),
        max_chars=max(len(t) for t in texts),
    )


def summarise(counts: Counts, probe: Probe) -> dict[str, Any]:
    """Turn counters into the reported record, including which cut is the binding one.

    ``headroom_tokens`` is what the ladder has left before the token cap starts clipping: the
    number to watch when deciding whether one more rung is still measuring overlap alone.
    """
    chunks = counts.chunks
    return {
        "chunks": chunks,
        "truncated": counts.truncated,
        "truncated_pct": round(100.0 * counts.truncated / chunks, 4) if chunks else 0.0,
        "token_mass_lost_pct": (
            round(100.0 * (counts.tokens_full - counts.tokens_kept) / counts.tokens_full, 4)
            if counts.tokens_full
            else 0.0
        ),
        "max_tokens_full": counts.max_tokens_full,
        "max_chars": counts.max_chars,
        "token_cap": probe.token_cap,
        "char_precut": probe.char_precut,
        "headroom_tokens": probe.token_cap - counts.max_tokens_full,
        "headroom_chars": (probe.char_precut - counts.max_chars) if probe.char_precut is not None else None,
    }


class TruncatingTokenizer:
    """Drives a HuggingFace ``tokenizers.Tokenizer`` the way an embedder that owns one does.

    Padding is turned OFF once here, and that is the whole reason this class exists. With padding
    on, ``len(encoding.ids)`` reports the PAD WIDTH rather than the content: every text in a batch
    comes back the same length, so the capped and uncapped counts agree by construction and the
    audit reports "nothing truncated" for any input at all. Measured on bge-base, that produced a
    reading whose median, p95 and max were identical at every rung - a constant, which is an
    instrument failing to discriminate rather than a clean result.
    """

    def __init__(self, tokenizer: Any) -> None:
        tokenizer.no_padding()
        self._tokenizer = tokenizer

    def tokenize(self, texts: Sequence[str], /, max_length: int | None) -> list[list[int]]:
        if max_length is None:
            self._tokenizer.no_truncation()
        else:
            self._tokenizer.enable_truncation(max_length=max_length)
        return [encoding.ids for encoding in self._tokenizer.encode_batch(list(texts))]


def _fastembed_probe(label: str, model_id: str) -> Probe:
    """Load a fastembed model and drive the very tokenizer it embeds with.

    The cap is read from the tokenizer's own truncation config rather than restated, and the cache
    is resolved exactly as the embedding adapter resolves it, so the audit measures the model the
    sweep actually used.
    """
    # Function-local: fastembed and semdex's adapter are only needed for a requested label.
    from fastembed import TextEmbedding

    from semdex.adapters.embedding.fastembed import resolve_cache_dir

    model = TextEmbedding(model_id, cache_dir=resolve_cache_dir(None), threads=_THREADS)
    # `TextEmbeddingBase` declares no `tokenizer`, though every ONNX text model carries one.
    # Naming the shape we rely on types the two attribute reads below instead of silencing them.
    tokenizer = cast("_HasTokenizer", model.model).tokenizer
    truncation = tokenizer.truncation
    if not truncation or not isinstance(truncation.get("max_length"), int):
        raise RuntimeError(f"{label}: tokenizer declares no integer truncation max_length ({truncation!r})")
    return Probe(
        label=label,
        tokenizes=TruncatingTokenizer(tokenizer),
        token_cap=truncation["max_length"],
        char_precut=None,
    )


def _model2vec_probe(label: str, repo_id: str) -> Probe:
    """Load a model2vec model and read its caps from the library rather than restating them.

    The cap is ``StaticModel.encode``'s own ``max_length`` default, because that is what a caller
    who passes no ``max_length`` gets - which is exactly what the embedding adapter does.
    """
    # Function-local: model2vec is an optional bench dependency, and only a requested label loads it.
    from model2vec import StaticModel

    model = StaticModel.from_pretrained(repo_id)
    cap = inspect.signature(StaticModel.encode).parameters["max_length"].default
    if not isinstance(cap, int):
        raise RuntimeError(f"{label}: StaticModel.encode has no integer max_length default (got {cap!r})")
    return Probe(label=label, tokenizes=model, token_cap=cap, char_precut=cap * model.median_token_length)


# Embedder families this script knows how to drive. Anything absent is reported `uncovered`.
_PROBES: dict[str, Callable[[str], Probe]] = {
    "model2vec:potion-base-8M": lambda label: _model2vec_probe(label, "minishlab/potion-base-8M"),
    "model2vec:potion-retrieval-32M": lambda label: _model2vec_probe(label, "minishlab/potion-retrieval-32M"),
    "fastembed:bge-base": lambda label: _fastembed_probe(label, "BAAI/bge-base-en-v1.5"),
    "fastembed:bge-small": lambda label: _fastembed_probe(label, "BAAI/bge-small-en-v1.5"),
}

_DEFAULT_EMBEDDINGS = "model2vec:potion-retrieval-32M,model2vec:potion-base-8M"


def _cache_root() -> Path:
    return Path(os.environ.get("CACHE_ROOT", "/embeddings"))


def _chunk_sets(root: Path) -> list[Path]:
    only = [s for s in os.environ.get("AUDIT_SETS", "").split(",") if s]
    sets = sorted(p for p in (root / "chunks").iterdir() if (p / "chunks.parquet").is_file())
    return [p for p in sets if p.name in only] if only else sets


def _batches(path: Path, sample: int) -> Iterable[list[str]]:
    """Stream chunk texts. A sample takes an even spread across the file, never the head.

    The head of a chunk set is one document's chunks, which is not a sample of the corpus.
    """
    pf = pq.ParquetFile(path)
    if sample <= 0:
        for batch in pf.iter_batches(batch_size=_BATCH, columns=["text"]):
            yield batch.column("text").to_pylist()
        return
    total = pf.metadata.num_rows
    step = max(1, total // sample)
    taken: list[str] = []
    seen = 0
    for batch in pf.iter_batches(batch_size=_BATCH, columns=["text"]):
        for text in batch.column("text").to_pylist():
            if seen % step == 0 and len(taken) < sample:
                taken.append(text)
                if len(taken) == _BATCH:
                    yield taken
                    taken = []
            seen += 1
    if taken:
        yield taken


def cell_dir(root: Path, set_name: str, label: str) -> Path:
    """Where the pre-embed cache keeps this embedder's vectors for this chunk set.

    The cache writes an embedder label with ":" as "-", the same substitution
    ``preembed_vectors._profile_tag`` makes.
    """
    return root / "vectors" / f"{set_name}__{label.replace(':', '-')}"


def _audit_set(root: Path, path: Path, probes: list[Probe], sample: int) -> dict[str, Any]:
    totals = {p.label: Counts() for p in probes}
    for texts in _batches(path / "chunks.parquet", sample):
        for probe in probes:
            totals[probe.label] = totals[probe.label].merge(count_batch(texts, probe))
    out: dict[str, Any] = {}
    for probe in probes:
        record = summarise(totals[probe.label], probe)
        # A chunk set can be measured against an embedder that never embedded it. That row says
        # what the embedder WOULD keep - useful for asking whether a rung is extensible at all -
        # and must not be read as describing a cell that exists.
        record["cell_exists"] = cell_dir(root, path.name, probe.label).is_dir()
        out[probe.label] = record
    return out


def _read_report(path: Path) -> dict[str, Any]:
    """The report a previous run left, or an empty one; a census is built up over several runs."""
    if not path.is_file():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    # The first reports carried ONE top-level sample; a row without its own inherits that value.
    for row in report.get("rows", []):
        row.setdefault("sample", report.get("sample", 0))
    return report


def _merge_rows(previous: list[dict[str, Any]], fresh: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge measured rows into a report by (chunk set, embedder).

    A re-measured row replaces its predecessor, with one exception: a row measured on EVERY chunk
    (sample 0) is never replaced by a sampled estimate, so a screening census cannot downgrade the
    full ladder measurement it runs beside.
    """
    by_key = {(r["chunk_set"], r["embedder"]): r for r in previous}
    for row in fresh:
        key = (row["chunk_set"], row["embedder"])
        held = by_key.get(key)
        if held is not None and held.get("sample", 0) == 0 and row["sample"] != 0:
            continue
        by_key[key] = row
    return [by_key[k] for k in sorted(by_key)]


def main() -> int:
    root = _cache_root()
    sample = int(os.environ.get("AUDIT_SAMPLE", "0"))
    labels = [s.strip() for s in os.environ.get("AUDIT_EMBEDDINGS", _DEFAULT_EMBEDDINGS).split(",") if s.strip()]
    out_path = Path(os.environ.get("OUT", _ROOT / "tests/benchmarks/raw/embedder-truncation-audit.json"))

    covered = [label for label in labels if label in _PROBES]
    uncovered = [label for label in labels if label not in _PROBES]
    for label in uncovered:
        print(f"uncovered: {label} - no probe knows its cap, so it is NOT reported as untruncated", file=sys.stderr)
    if not covered:
        print("no covered embedder requested; nothing measured", file=sys.stderr)
        return 2

    probes = [_PROBES[label](label) for label in covered]
    for probe in probes:
        print(f"{probe.label}: token cap {probe.token_cap}, character pre-cut {probe.char_precut}", file=sys.stderr)

    rows: list[dict[str, Any]] = []
    findings = 0
    for path in _chunk_sets(root):
        per_embedder = _audit_set(root, path, probes, sample)
        # A FLAT row list, the shape every other raw benchmark file uses, so a claim rowset can
        # read it directly; a nested mapping would need a reshaping step nothing else has.
        rows.extend(
            {"chunk_set": path.name, "embedder": label, "sample": sample, **rec} for label, rec in per_embedder.items()
        )
        for label, rec in per_embedder.items():
            if rec["truncated"]:
                findings += 1
                print(
                    f"TRUNCATED {path.name} {label}: {rec['truncated']}/{rec['chunks']} chunks "
                    f"({rec['truncated_pct']}%), {rec['token_mass_lost_pct']}% of token mass lost",
                    file=sys.stderr,
                )
        headroom = ", ".join(f"{k} headroom {v['headroom_tokens']}t" for k, v in per_embedder.items())
        print(f"[audit] {path.name}: {headroom}")

    previous = _read_report(out_path)
    merged = _merge_rows(previous.get("rows", []), rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"uncovered": sorted(set(previous.get("uncovered", [])) | set(uncovered)), "rows": merged}
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[audit] {len(rows)} rows measured, {len(merged)} in report -> {out_path}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
