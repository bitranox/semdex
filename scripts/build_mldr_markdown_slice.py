#!/usr/bin/env python
# pyright: basic
"""Render the MLDR English slice with its bare-line section headings marked as markdown.

MLDR English is Wikipedia text with the markup stripped. The section headings survive as bare
lines: a short line with no terminal punctuation, followed by a paragraph of running prose. This
writes a TWIN slice with the same doc ids, queries and qrels and each such line prefixed ``## ``,
so a chunk strategy that cuts on markdown structure can be measured against its own unmarked cell
per query. Stripping every ``## `` prefix gives the source back byte for byte (tested), which is
what makes the pair isolate the markup.

The rule is a heuristic and its precision is measured, not assumed: ``heading-sample.txt`` beside
the slice holds a seeded sample of marked lines with the line that follows each, for review. A
reviewer copies it to ``heading-sample-judged.txt`` and prefixes a wrong line with ``X\t``; the
paired-effect script reports the share.

Heading LEVEL is not recoverable from stripped text; every heading is level two.

Config (env; scripts/ convention, no argparse):
  SEMDEX_MDSLICE_SOURCE  source slice dir (default /corpora/mldr-slices/mldr_en_8k_slice)
  SEMDEX_MDSLICE_OUT     output slice dir (default /corpora/mldr-slices/mldr_en_8k_md_slice)
  SEMDEX_MDSLICE_SAMPLE  marked lines sampled for review (default 200)
  SEMDEX_MDSLICE_SEED    sample RNG seed (default 20260927)
  SEMDEX_MDSLICE_MAX_CHARS / _MAX_SPACES / _MIN_NEXT_CHARS  rule parameters (defaults 79 / 8 / 120)
"""

from __future__ import annotations

import hashlib
import os
import random
import re
from pathlib import Path
from typing import Any

import orjson

MARK = "## "
# The rule's parameters. A heading is short, has few words, and is followed by a paragraph.
MAX_HEADING_CHARS = int(os.environ.get("SEMDEX_MDSLICE_MAX_CHARS", "79"))
MAX_HEADING_SPACES = int(os.environ.get("SEMDEX_MDSLICE_MAX_SPACES", "8"))
MIN_NEXT_LINE_CHARS = int(os.environ.get("SEMDEX_MDSLICE_MIN_NEXT_CHARS", "120"))
_TERMINAL_PUNCTUATION = ".,;:!?)"
_PROSE_ENDINGS = (".", '"', ")")
_FORBIDDEN_CHARS = ("=", "\u2192", "+")  # the arrow as an escape: ASCII source only
_DIGIT_RUN = re.compile(r"\d{3}")
_LIST_MARKERS = ("-", "*", "\u2022")

__all__ = ["MARK", "build", "is_heading", "mark_headings", "strip_marks"]


def is_heading(line: str, following: str | None) -> bool:
    """Is ``line`` a section heading, given the next non-empty line?"""
    if following is None or not line.strip() or len(line) > MAX_HEADING_CHARS:
        return False
    if line.startswith(MARK) or line.startswith(_LIST_MARKERS):
        return False
    if line[-1] in _TERMINAL_PUNCTUATION or line.count(" ") > MAX_HEADING_SPACES:
        return False
    if any(ch in line for ch in _FORBIDDEN_CHARS) or _DIGIT_RUN.search(line):
        return False
    return len(following) >= MIN_NEXT_LINE_CHARS and following.endswith(_PROSE_ENDINGS)


def _next_non_empty(lines: list[str], start: int) -> str | None:
    for candidate in lines[start:]:
        if candidate.strip():
            return candidate
    return None


def mark_headings(text: str) -> tuple[str, int]:
    """Prefix every heading line with ``## ``; return the text and how many lines were marked."""
    lines = text.split("\n")
    count = 0
    for i, line in enumerate(lines):
        if is_heading(line, _next_non_empty(lines, i + 1)):
            lines[i] = MARK + line
            count += 1
    return "\n".join(lines), count


def strip_marks(text: str) -> str:
    """The inverse of ``mark_headings`` for text the source never marked."""
    return "\n".join(line[len(MARK) :] if line.startswith(MARK) else line for line in text.split("\n"))


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _rule() -> dict[str, Any]:
    return {
        "max_chars": MAX_HEADING_CHARS,
        "max_spaces": MAX_HEADING_SPACES,
        "min_next_line_chars": MIN_NEXT_LINE_CHARS,
        "terminal_punctuation": _TERMINAL_PUNCTUATION,
        "prose_endings": "".join(_PROSE_ENDINGS),
        "forbidden_chars": "".join(_FORBIDDEN_CHARS),
        "digit_run": _DIGIT_RUN.pattern,
        "mark": MARK,
    }


def _sample_rows(text: str, doc_id: str) -> list[str]:
    """Every marked line of one document as ``doc_id<TAB>heading<TAB>next line`` for the review file."""
    lines = text.split("\n")
    rows = []
    for i, line in enumerate(lines):
        if line.startswith(MARK):
            following = _next_non_empty(lines, i + 1) or ""
            rows.append(f"{doc_id}\t{line}\t{following[:160]}")
    return rows


def build(source: Path, out: Path, *, sample_size: int, seed: int) -> dict[str, Any]:
    """Write the marked twin of ``source`` into ``out`` (idempotent); return the meta written."""
    if (out / "meta.json").exists():
        print(f"[mdslice] {out.name}: already built - skip (delete the dir to rebuild)", flush=True)
        return orjson.loads((out / "meta.json").read_bytes())
    source_bytes = (source / "corpus.jsonl").read_bytes()
    marked_rows: list[bytes] = []
    review: list[str] = []
    marked_lines = marked_docs = total = 0
    for raw in source_bytes.splitlines():
        row = orjson.loads(raw)
        text, count = mark_headings(row.get("text") or "")
        if strip_marks(text) != (row.get("text") or ""):
            raise ValueError(f"{row['_id']}: stripping the marks does not give the source back")
        total += 1
        marked_lines += count
        marked_docs += count > 0
        review.extend(_sample_rows(text, row["_id"]))
        marked_rows.append(orjson.dumps({"_id": row["_id"], "title": row.get("title", ""), "text": text}) + b"\n")
    out.mkdir(parents=True, exist_ok=True)
    _write_atomic(out / "corpus.jsonl", b"".join(marked_rows))
    for name in ("queries.json", "qrels.json", "doc_ids.txt"):
        _write_atomic(out / name, (source / name).read_bytes())
    sample = random.Random(seed).sample(review, min(sample_size, len(review)))  # noqa: S311 - reproducible review sample, not cryptographic
    _write_atomic(out / "heading-sample.txt", ("\n".join(sample) + "\n").encode() if sample else b"")
    meta = {
        "corpus": out.name,
        "source_corpus": source.name,
        "source_corpus_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "rule": _rule(),
        "total": total,
        "marked_lines": marked_lines,
        "marked_docs": marked_docs,
        "sample_size": len(sample),
        "sample_seed": seed,
    }
    _write_atomic(out / "meta.json", orjson.dumps(meta, option=orjson.OPT_INDENT_2))
    print(f"[mdslice] {out.name}: {total} docs, {marked_lines} headings in {marked_docs} docs -> {out}", flush=True)
    return meta


def main() -> None:
    source = Path(os.environ.get("SEMDEX_MDSLICE_SOURCE", "/corpora/mldr-slices/mldr_en_8k_slice"))
    out = Path(os.environ.get("SEMDEX_MDSLICE_OUT", "/corpora/mldr-slices/mldr_en_8k_md_slice"))
    build(
        source,
        out,
        sample_size=int(os.environ.get("SEMDEX_MDSLICE_SAMPLE", "200")),
        seed=int(os.environ.get("SEMDEX_MDSLICE_SEED", "20260927")),
    )


if __name__ == "__main__":
    main()
