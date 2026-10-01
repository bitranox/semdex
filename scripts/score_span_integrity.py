#!/usr/bin/env python3
"""Measure whether a chunk boundary cuts an answer in half, and what that costs retrieval.

Every other chunking measurement in this repo is end-to-end nDCG against DOCUMENT-level
relevance judgements. That cannot see the mechanism chunking is supposed to affect. If a boundary
splits the sentence carrying the answer, some other chunk of the same document usually still
ranks, the document is scored as retrieved, and the split is invisible - while a consumer reading
the retrieved chunk finds half an answer. It is also exactly the failure overlap exists to
prevent, so the overlap conclusion rested on a metric blind to overlap's purpose.

SQuAD-style data has character-level answer spans, so the question can be asked directly. Two
measurements, deliberately separate:

**Span integrity** is chunking alone, no embeddings, no retrieval: chunk each context, locate the
answer span, and ask whether ANY chunk wholly contains it. This isolates the boundary effect from
everything downstream.

**Retrieval-aware** composes it with search: index every context's chunks, query with the
question, and compare two verdicts over the same top-k. ``doc_hit`` is what the existing
benchmarks measure - a chunk of the right context was retrieved. ``span_hit`` is whether a
retrieved chunk actually holds the whole answer. The difference between them IS the blind spot,
in the units of the metric that hid it.

Corpus fitness applies here as it does to every chunking claim (see docs/benchmarks/01-method.md):
a context that fits in one chunk has no boundary to cut anything, so its integrity is trivially
1.0 and says nothing about chunking. Every row therefore carries ``chunks_per_context``, and rows
that fail the fitness bar are marked rather than quietly averaged in.

Usage::

    python scripts/score_span_integrity.py                    # full grid, phase A + B
    python scripts/score_span_integrity.py --phase a          # chunking only, no embeddings
    python scripts/score_span_integrity.py --datasets xquad.de,xquad.en
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from bisect import bisect_right
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pyarrow.parquet as pq  # pyright: ignore[reportMissingTypeStubs] - pyarrow ships none; drop when it does

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _score_stats import bootstrap_ci, paired_ci

from semdex.composition import build_chunker, build_embedding
from semdex.domain.enums import ChunkStrategy, EmbeddingBackend
from semdex.domain.models import ExtractedDocument, SourceRef

_HUB = Path(os.environ.get("SEMDEX_SPAN_HUB", "/corpora/hf/hub"))
_OUT = Path(__file__).resolve().parents[1] / "tests" / "benchmarks" / "raw" / "span-integrity.json"

# A context must produce at least this many chunks for its integrity number to mean anything:
# below it there is no boundary that could cut an answer. Same spirit as the chunk sweep's
# corpus-fitness bar, at the lower value that suits a per-context question rather than a per-
# document retrieval one - two chunks is already one boundary.
_MIN_CHUNKS_PER_CONTEXT = 2.0

_STRATEGIES = ("recursive", "markdown", "fast", "whitespace", "semantic", "late")
# chonkie applies OverlapRefinery to RECURSIVE only; semantic-text-splitter (markdown/fast) has
# native overlap. The others ignore the knob, so sweeping it there would report a fake flat line.
_OVERLAP_CAPABLE = frozenset({"recursive", "markdown", "fast"})


@dataclass(frozen=True, slots=True)
class QaItem:
    """One question with its answer located by character offset in its context."""

    context_id: int
    question: str
    answer: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Corpus:
    """A SQuAD-style set reduced to distinct contexts plus offset-validated questions."""

    name: str
    contexts: list[str]
    items: list[QaItem]
    dropped_offsets: int


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _dataset_paths() -> dict[str, Path]:
    """Locate the cached SQuAD-style parquet files, skipping any that are not present."""
    wanted = {
        "xquad.de": "xquad.de/*.parquet",
        "xquad.en": "xquad.en/*.parquet",
        "xquad.es": "xquad.es/*.parquet",
        "germanquad": "datasets--deepset--germanquad/**/test/0000.parquet",
        "mlqa.de": "mlqa.de.de/test/*.parquet",
    }
    found: dict[str, Path] = {}
    for name, pattern in wanted.items():
        match = next(_HUB.rglob(pattern), None)
        if match is not None:
            found[name] = match
    return found


class _ParquetColumn(Protocol):
    """The one method this module uses off a parquet column."""

    def to_pylist(self) -> list[Any]: ...


class _ParquetTable(Protocol):
    """The one method this module uses off a parquet table."""

    def column(self, name: str) -> _ParquetColumn: ...


# pyarrow ships no type stubs, so everything reached through a table is Unknown and leaks that
# through every caller. Declaring the two methods actually used and casting the entry point once
# keeps the rest of this module under strict checking, rather than turning the whole file basic
# and losing the checks that already caught two real defects here.
# The reference to the unstubbed symbol itself is the one thing the facade cannot type.
# Remove this ignore when pyarrow ships stubs (or pyarrow-stubs is added to [dev]).
_read_table = cast(
    "Callable[[Path], _ParquetTable]",
    pq.read_table,  # pyright: ignore[reportUnknownMemberType]
)


def read_squad_columns(path: Path) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """The three SQuAD columns as typed Python lists."""
    table = _read_table(path)
    return (
        cast(list[str], table.column("context").to_pylist()),
        cast(list[str], table.column("question").to_pylist()),
        cast(list[dict[str, Any]], table.column("answers").to_pylist()),
    )


def load_corpus(name: str, path: Path, *, max_contexts: int | None = None) -> Corpus:
    """Read a SQuAD-style parquet into distinct contexts plus offset-verified questions.

    The published ``answer_start`` is trusted only where it actually reproduces the answer text,
    because several sets carry variants with a leading space. A span whose offset does not verify
    would silently be measured against the wrong characters, so it is dropped and counted.
    """
    contexts_raw, questions, answers = read_squad_columns(path)

    index_of: dict[str, int] = {}
    contexts: list[str] = []
    items: list[QaItem] = []
    dropped = 0
    for context, question, answer in zip(contexts_raw, questions, answers, strict=True):
        cid = index_of.get(context)
        if cid is None:
            if max_contexts is not None and len(contexts) >= max_contexts:
                continue
            cid = index_of[context] = len(contexts)
            contexts.append(context)
        span = _first_verifiable_span(context, answer)
        if span is None:
            dropped += 1
            continue
        start, text = span
        items.append(QaItem(cid, question, text, start, start + len(text)))
    return Corpus(name=name, contexts=contexts, items=items, dropped_offsets=dropped)


def _first_verifiable_span(context: str, answer: dict[str, Any]) -> tuple[int, str] | None:
    """The first (offset, text) pair whose offset really holds that text in the context."""
    for text, start in zip(answer["text"], answer["answer_start"], strict=True):
        if context[start : start + len(text)] == text:
            return int(start), text
    return None


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Whitespace-collapsed text plus, for each normalized character, its index in the original.

    Two chunkers do not return byte-identical substrings, so a plain ``str.find`` silently loses
    their chunks and understates coverage for reasons that have nothing to do with chunking. The
    map is what lets the match happen on normalized text while the offsets stay in the ORIGINAL
    coordinate system, which is the one the answer spans live in.
    """
    out: list[str] = []
    index: list[int] = []
    position = 0
    length = len(text)
    while position < length:
        if text[position].isspace():
            run = position
            while run < length and text[run].isspace():
                run += 1
            out.append(" ")
            index.append(position)
            position = run
            continue
        out.append(text[position])
        index.append(position)
        position += 1
    return "".join(out), index


def chunk_offsets(context: str, texts: Sequence[str]) -> list[tuple[int, int]]:
    """Locate each chunk's text in its context as a ``(start, end)`` character range.

    The chunk model carries no offsets, so they are recovered by search. Two things stop that
    being a plain substring lookup, and both were found by the drop counter rather than by
    reading - each biased the result in a direction that flattered a conclusion:

    * the whitespace chunker rejoins on single spaces, so its chunks are not byte-identical to
      the source. Matching is therefore done on whitespace-normalized text through an offset map.
    * chonkie's overlap refinery cuts the prepended context at a TOKEN boundary, which can land
      mid-character and leave a U+FFFD replacement at the end. Those are trimmed before matching.
      Every overlap>0 cell lost chunks this way and only overlap>0 cells did, so the measurement
      was deleting precisely the chunks overlap adds and then scoring overlap.

    The cursor advances one character past each match's START, not past its end. Past the end
    would break overlap, where the next chunk begins before this one finishes: the search would
    miss it and fall back to a whole-string find, quietly returning an earlier occurrence. Staying
    ON the start instead breaks repeated text, where identical consecutive chunks would all
    resolve to the first occurrence. Chunk starts strictly increase, so start+1 satisfies both.
    """
    normalized, index_map = normalize_with_map(context)
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for text in texts:
        needle, _ = normalize_with_map(text.rstrip("\ufffd"))
        if not needle:
            continue
        found = normalized.find(needle, cursor)
        if found < 0:
            found = normalized.find(needle)
        if found < 0:
            continue  # unlocatable: excluded rather than guessed at, and counted by the caller
        offsets.append((index_map[found], index_map[found + len(needle) - 1] + 1))
        cursor = found + 1
    return offsets


def span_is_intact(offsets: Sequence[tuple[int, int]], start: int, end: int) -> bool:
    """Whether some chunk wholly contains ``[start, end)``.

    Chunk starts are non-decreasing, so a binary search finds the split above which no chunk can
    begin early enough. Every chunk below it already starts at or before the span - that is what
    bisect_right establishes - so the only thing left to test is whether one of them also reaches
    far enough to the right. Overlap means several may; one is enough.
    """
    starts = [s for s, _ in offsets]
    upper = bisect_right(starts, start)
    return any(offsets[i][1] >= end for i in range(upper - 1, -1, -1))


def _document(text: str, uri: str) -> ExtractedDocument:
    return ExtractedDocument(source=SourceRef(uri=uri, label="", content_hash="h", mtime=0.0), text=text)


@dataclass(frozen=True, slots=True)
class ChunkedCorpus:
    """Every context chunked once under one profile, with offsets kept for span tests."""

    texts: list[str]
    context_of: list[int]
    offsets: list[tuple[int, int]]
    per_context: list[list[tuple[int, int]]]
    unlocatable: int


def chunk_corpus(corpus: Corpus, chunker: Any, *, max_tokens: int) -> ChunkedCorpus:
    """Chunk every context under one profile, keeping each chunk's character range."""
    texts: list[str] = []
    context_of: list[int] = []
    offsets: list[tuple[int, int]] = []
    per_context: list[list[tuple[int, int]]] = []
    unlocatable = 0
    for cid, context in enumerate(corpus.contexts):
        chunk_texts = [c.text for c in chunker(_document(context, f"mem://{corpus.name}/{cid}"), max_tokens=max_tokens)]
        located = chunk_offsets(context, chunk_texts)
        unlocatable += len(chunk_texts) - len(located)
        per_context.append(located)
        for text, span in zip(chunk_texts[: len(located)], located, strict=False):
            texts.append(text)
            context_of.append(cid)
            offsets.append(span)
    return ChunkedCorpus(texts, context_of, offsets, per_context, unlocatable)


def pack_bits(flags: Sequence[float]) -> str:
    """Per-question outcomes as a hex bitmap, question order defining the bit order.

    Kept on every row so the paired comparisons this repo's method requires can be recomputed
    from the committed data without re-running a 35-minute grid. One bit per question rather than
    a mapping of ids, because within a corpus every cell scores the identical question set by
    construction - which is also precisely what makes the comparison pairable.
    """
    packed = bytearray((len(flags) + 7) // 8)
    for index, flag in enumerate(flags):
        if flag:
            packed[index // 8] |= 1 << (index % 8)
    return packed.hex()


def unpack_bits(hex_bits: str, count: int) -> list[float]:
    """Inverse of :func:`pack_bits`, given how many questions the cell scored."""
    raw = bytes.fromhex(hex_bits)
    return [float(bool(raw[i // 8] & (1 << (i % 8)))) for i in range(count)]


def integrity_row(corpus: Corpus, chunked: ChunkedCorpus, profile: dict[str, Any]) -> dict[str, Any]:
    """Score one (corpus, profile) pair for span integrity, with a fitness verdict."""
    intact = [
        float(span_is_intact(chunked.per_context[item.context_id], item.start, item.end)) for item in corpus.items
    ]
    stats = bootstrap_ci(intact)
    chunks_per_context = len(chunked.texts) / max(1, len(corpus.contexts))
    fit = chunks_per_context >= _MIN_CHUNKS_PER_CONTEXT
    return {
        "corpus": corpus.name,
        **profile,
        "n_questions": len(intact),
        "n_contexts": len(corpus.contexts),
        "chunks_per_context": round(chunks_per_context, 2),
        "fit_for_claim": fit,
        "span_intact": round(stats["mean"], 4),
        "ci_lo": round(stats["ci_lo"], 4),
        "ci_hi": round(stats["ci_hi"], 4),
        "answers_split": sum(1 for value in intact if value == 0.0),
        "mean_answer_chars": round(sum(i.end - i.start for i in corpus.items) / max(1, len(corpus.items)), 1),
        "unlocatable_chunks": chunked.unlocatable,
        "dropped_offsets": corpus.dropped_offsets,
        "intact_bits": pack_bits(intact),
    }


def profiles(strategies: Sequence[str], max_tokens: Sequence[int], overlaps: Sequence[int]) -> Iterator[dict[str, Any]]:
    """The profile grid, skipping combinations that are not real measurements.

    Two exclusions, both so a published row means what it says:

    * strategies that discard the overlap argument (see ``_OVERLAP_CAPABLE``), where an overlap
      row would be a duplicate of the zero-overlap one and read as "overlap does not help here"
      rather than "overlap never happened";
    * an overlap at or above the chunk size, which is not a chunking configuration at all -
      semantic-text-splitter raises on it outright, and chonkie accepts it while producing
      something no deployment would run. Skipping it for EVERY strategy keeps the grid identical
      across them, so a strategy comparison is never a comparison of different grids.
    """
    for strategy in strategies:
        for tokens in max_tokens:
            for overlap in overlaps:
                if overlap and strategy not in _OVERLAP_CAPABLE:
                    continue
                if overlap >= tokens:
                    continue
                yield {"strategy": strategy, "max_tokens": tokens, "overlap_tokens": overlap}


def _chunker_for(profile: dict[str, Any]) -> Any:
    return build_chunker(
        ChunkStrategy(profile["strategy"]),
        recipe="",
        overlap=int(profile["overlap_tokens"]),
    )


def retrieval_row(
    corpus: Corpus,
    chunked: ChunkedCorpus,
    profile: dict[str, Any],
    *,
    embedding: Any,
    k: int,
) -> dict[str, Any]:
    """Compare document-level and span-level success over the same top-k retrieval.

    ``doc_hit`` is the verdict the existing benchmarks record; ``span_hit`` is whether the answer
    survived intact inside something actually retrieved. Their difference is the blind spot.
    """
    from _score_kernel import topk_stream

    passages = np.asarray(embedding.embed_passages(chunked.texts), dtype=np.float32)
    # The port exposes embed_query one at a time (queries arrive singly in real use); batching is
    # the caller's job here because this asks thousands of them at once.
    queries = np.asarray([embedding.embed_query(item.question) for item in corpus.items], dtype=np.float32)
    index, _score = topk_stream(passages, queries, fetch=k)

    doc_hits: list[float] = []
    span_hits: list[float] = []
    blind: list[float] = []
    from_split: list[float] = []
    from_ranking: list[float] = []
    for row, item in enumerate(corpus.items):
        rows = [int(i) for i in index[row]]
        doc = any(chunked.context_of[i] == item.context_id for i in rows)
        span = any(
            chunked.context_of[i] == item.context_id
            and chunked.offsets[i][0] <= item.start
            and chunked.offsets[i][1] >= item.end
            for i in rows
        )
        # A blind spot has two very different causes and reporting only the total would blame
        # chunking for both. Either no chunk anywhere holds the answer whole - the boundary cut
        # it, which is chunking's doing - or one does and the ranking did not retrieve it, which
        # is not.
        intact_somewhere = span_is_intact(chunked.per_context[item.context_id], item.start, item.end)
        doc_hits.append(float(doc))
        span_hits.append(float(span))
        blind.append(float(doc and not span))
        from_split.append(float(doc and not span and not intact_somewhere))
        from_ranking.append(float(doc and not span and intact_somewhere))
    return {
        "corpus": corpus.name,
        **profile,
        "k": k,
        "n_questions": len(doc_hits),
        "n_chunks_indexed": len(chunked.texts),
        "doc_hit": round(float(np.mean(doc_hits)), 4),
        "span_hit": round(float(np.mean(span_hits)), 4),
        "blind_spot": round(float(np.mean(blind)), 4),
        "blind_spot_share_of_doc_hits": round(float(np.sum(blind) / max(1.0, np.sum(doc_hits))), 4),
        "blind_from_split": round(float(np.mean(from_split)), 4),
        "blind_from_ranking": round(float(np.mean(from_ranking)), 4),
    }


_AXES = ("max_tokens", "overlap_tokens", "strategy")


def axis_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Paired one-axis-apart comparisons over the per-question outcomes.

    Comparing two marginal intervals is the weaker test and this repo does not use it (see
    docs/benchmarks/01-method.md). Every cell of a corpus scores the same questions, so the
    difference is taken per question, which removes question difficulty - the dominant source of
    variance, and common to both sides.

    Only cells differing in exactly ONE axis are compared, and only cells that pass the corpus
    fitness bar on BOTH sides: a context that fits in one chunk has no boundary, so pairing an
    unfit cell against a fit one would attribute the difference to the knob rather than to one
    side having nothing to measure.
    """
    fit = [row for row in rows if row.get("fit_for_claim")]
    by_key = {row_key(row): row for row in fit}
    out: list[dict[str, Any]] = []
    for left_key, left in by_key.items():
        for right_key, right in by_key.items():
            axis = _single_axis_between(left, right)
            if axis is None or not _is_forward(left, right, axis) or left_key == right_key:
                continue
            out.append(_effect_row(left, right, axis))
    return sorted(out, key=lambda row: (row["axis"], row["corpus"], str(row["held_fixed"]), str(row["from_level"])))


def _single_axis_between(left: dict[str, Any], right: dict[str, Any]) -> str | None:
    """The one axis two cells differ on, or None if they differ on zero or several."""
    if left["corpus"] != right["corpus"]:
        return None
    differing = [axis for axis in _AXES if left[axis] != right[axis]]
    return differing[0] if len(differing) == 1 else None


def _is_forward(left: dict[str, Any], right: dict[str, Any], axis: str) -> bool:
    """Emit each pair once, in a stable direction (low to high, or alphabetical)."""
    a, b = left[axis], right[axis]
    return str(a) < str(b) if isinstance(a, str) else bool(a < b)


def _effect_row(left: dict[str, Any], right: dict[str, Any], axis: str) -> dict[str, Any]:
    """One paired comparison: moving ``axis`` from the left cell's level to the right cell's."""
    count = int(left["n_questions"])
    # Keys are the question ordinal as a string: paired_ci pairs on query IDS, and within a
    # corpus the ordinal IS the identity, since every cell scores the same questions in order.
    low = {str(i): value for i, value in enumerate(unpack_bits(left["intact_bits"], count))}
    high = {str(i): value for i, value in enumerate(unpack_bits(right["intact_bits"], int(right["n_questions"])))}
    paired = paired_ci(high, low)  # high minus low: positive means the higher level protects more
    held = {axis_name: left[axis_name] for axis_name in _AXES if axis_name != axis}
    return {
        "axis": axis,
        "corpus": left["corpus"],
        "held_fixed": held,
        "from_level": left[axis],
        "to_level": right[axis],
        "mean_delta": round(float(cast(float, paired["mean_delta"])), 4),
        "ci_lo": round(float(cast(float, paired["ci_lo"])), 4),
        "ci_hi": round(float(cast(float, paired["ci_hi"])), 4),
        "wins": int(cast(int, paired["wins"])),
        "losses": int(cast(int, paired["losses"])),
        "ties": int(cast(int, paired["ties"])),
        "n_shared": int(cast(int, paired["n_shared"])),
        "resolved": bool(paired["resolved"]),
    }


_ROW_KEY = ("corpus", "strategy", "max_tokens", "overlap_tokens", "k")


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """What makes a row unique: one measurement per corpus and profile (and k, in phase B)."""
    return tuple(row.get(field) for field in _ROW_KEY)


def merge_payload(existing: dict[str, Any], fresh: dict[str, Any]) -> dict[str, Any]:
    """Fold a run into an earlier one, newest measurement winning per row.

    This grid is dominated by a few slow cells - `late` chunking re-embeds every sentence of
    every context - so extending it with an extra knob level should not mean re-measuring the
    parts that did not change. Rows are replaced by identity rather than appended, so a re-run of
    the same profile updates in place instead of leaving two rows a table would average.

    The environment stamp comes from the FRESH run and therefore describes the latest pass only.
    That is the same compromise the regression baseline makes, and for the same reason: rows are
    what carry meaning here, and a merged file honestly holds several vintages.
    """
    merged = dict(fresh)
    for phase in ("integrity", "retrieval"):
        old_rows = cast(list[dict[str, Any]], existing.get(phase) or [])
        new_rows = cast(list[dict[str, Any]], fresh.get(phase) or [])
        if not old_rows and not new_rows:
            continue
        by_key = {row_key(row): row for row in old_rows}
        by_key.update({row_key(row): row for row in new_rows})
        merged[phase] = sorted(by_key.values(), key=lambda row: [str(v) for v in row_key(row)])
    return merged


def _environment() -> dict[str, Any]:
    return {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "semdex_git_sha": _git_sha(),
        "host": platform.node(),
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "hub": str(_HUB),
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("a", "b", "ab"), default="ab", help="a=integrity, b=retrieval, ab=both")
    parser.add_argument("--datasets", default="", help="comma list; default every cached set")
    parser.add_argument("--strategies", default=",".join(_STRATEGIES))
    parser.add_argument("--max-tokens", default="64,128,256,512")
    parser.add_argument("--overlaps", default="0,32,64")
    parser.add_argument(
        "--retrieval-profiles", default="recursive:256:0,recursive:256:64,recursive:128:0,recursive:128:64"
    )
    parser.add_argument("--retrieval-corpora", default="xquad.de,xquad.en,germanquad")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-contexts", type=int, default=0, help="cap contexts per set (0 = all)")
    parser.add_argument("--out", type=Path, default=_OUT)
    parser.add_argument(
        "--merge",
        action="store_true",
        help="fold into an existing --out instead of replacing it (rows replaced by identity)",
    )
    return parser.parse_args(argv)


def _selected_corpora(args: argparse.Namespace) -> dict[str, Corpus]:
    available = _dataset_paths()
    wanted = [n.strip() for n in args.datasets.split(",") if n.strip()] or list(available)
    missing = [n for n in wanted if n not in available]
    if missing:
        print(f"[span] not cached, skipped: {', '.join(missing)}", file=sys.stderr)
    cap = args.max_contexts or None
    return {name: load_corpus(name, available[name], max_contexts=cap) for name in wanted if name in available}


def _run_phase_a(corpora: dict[str, Corpus], args: argparse.Namespace) -> list[dict[str, Any]]:
    grid = list(
        profiles(
            [s.strip() for s in args.strategies.split(",") if s.strip()],
            [int(t) for t in args.max_tokens.split(",") if t.strip()],
            [int(o) for o in args.overlaps.split(",") if o.strip()],
        )
    )
    rows: list[dict[str, Any]] = []
    for profile in grid:
        chunker = _chunker_for(profile)
        for corpus in corpora.values():
            chunked = chunk_corpus(corpus, chunker, max_tokens=int(profile["max_tokens"]))
            row = integrity_row(corpus, chunked, profile)
            rows.append(row)
            flag = "" if row["fit_for_claim"] else "   [unfit: contexts fit in one chunk]"
            print(
                f"[A] {corpus.name:11s} {profile['strategy']:10s} t{profile['max_tokens']:<4d}"
                f" o{profile['overlap_tokens']:<3d} intact {row['span_intact']:.4f}"
                f"  {row['chunks_per_context']:5.2f} chunks/ctx{flag}",
                flush=True,
            )
    return rows


def _run_phase_b(corpora: dict[str, Corpus], args: argparse.Namespace) -> list[dict[str, Any]]:
    embedding = build_embedding(EmbeddingBackend.FASTEMBED)
    rows: list[dict[str, Any]] = []
    for spec in [s.strip() for s in args.retrieval_profiles.split(",") if s.strip()]:
        strategy, tokens, overlap = spec.split(":")
        profile = {"strategy": strategy, "max_tokens": int(tokens), "overlap_tokens": int(overlap)}
        chunker = _chunker_for(profile)
        for name in [n.strip() for n in args.retrieval_corpora.split(",") if n.strip()]:
            corpus = corpora.get(name)
            if corpus is None:
                continue
            chunked = chunk_corpus(corpus, chunker, max_tokens=int(tokens))
            row = retrieval_row(corpus, chunked, profile, embedding=embedding, k=args.k)
            rows.append(row)
            print(
                f"[B] {name:11s} {spec:20s} doc_hit {row['doc_hit']:.4f}"
                f"  span_hit {row['span_hit']:.4f}  blind {row['blind_spot']:.4f}",
                flush=True,
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    corpora = _selected_corpora(args)
    if not corpora:
        print("[span] no cached datasets found", file=sys.stderr)
        return 2

    payload: dict[str, Any] = {**_environment(), "k": args.k, "min_chunks_per_context": _MIN_CHUNKS_PER_CONTEXT}
    if "a" in args.phase:
        payload["integrity"] = _run_phase_a(corpora, args)
    if "b" in args.phase:
        payload["retrieval"] = _run_phase_b(corpora, args)

    if args.merge and args.out.exists():
        payload = merge_payload(json.loads(args.out.read_text(encoding="utf-8")), payload)
    if payload.get("integrity"):
        # Computed after the merge, so effects always describe the file's FULL grid rather than
        # whichever slice this invocation happened to measure.
        payload["axis_effects"] = axis_effects(payload["integrity"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
