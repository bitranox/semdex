#!/usr/bin/env python
# pyright: basic
# Benchmark harness on numpy (no strict stubs); strict mode would only add reportUnknown* noise.
# Same stance as export_bench_raw.py and _score_stats.py.
"""Does the chunk-overlap effect track QUERY LENGTH? Two measurements on the scored cells.

GerDaLIR's queries are citing passages (median about a hundred words); MLDR's are one-sentence
questions. The two corpora disagree on whether overlap helps, and nothing on the chunking page
described the queries. This asks the query side directly, on cells that already exist:

1. ``by_query_length`` - OBSERVATIONAL. The per-query nDCG delta between two overlap levels,
   binned by the query's own word count (quantile bins over the corpus's queries, plus one bin for
   queries no longer than a short-question ceiling). Every bin gets the same paired bootstrap the
   chunking page uses, so a bin "resolves" on the page's own terms.
2. ``truncated_queries`` - CAUSAL. The same paired delta re-scored with every query cut to its
   first N words (``score_chunk_sweep.py`` with ``SEMDEX_SCORE_QUERY_WORDS``; corpus vectors
   reused, only the query vectors change), beside the full-query delta. If the gain goes with the
   words, the mechanism is query length and not the corpus.

Reads the per-query arrays the scorer wrote (``scores/perquery`` and ``scores/perquery-q<N>w``)
and the query texts of the chunk set; embeds nothing. Provenance is carried from the scorer's
stamps on the cells it reads, never taken from this machine.

Env:
  CACHE_ROOT       cache dir (default /embeddings)
  QLEN_CORPORA     comma list (default gerdalir_de_12k_slice,mldr_en_8k_slice)
  QLEN_PAIRS       comma list of low:high overlap tokens (default 0:51,0:128,0:256)
  QLEN_BINS        quantile bins over query length (default 4)
  QLEN_SHORT_WORDS ceiling for the "short queries" bin, in words (default 38, MLDR's longest)
  QLEN_VARIANTS    comma list of N-word truncations to look for (default 20)
  OUT              results json (default tests/benchmarks/raw/query-length-effect.json)
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _provenance import MEASURED_ON
from _score_stats import paired_ci
from export_bench_raw import measured_on_summary

_ROOT = Path(__file__).resolve().parent.parent
_PROFILE = "recursive-t256-o{overlap}-gpt2"
_METRIC_KEYS = ("ndcg", "ndcg@10")

__all__ = [
    "bin_edges",
    "by_query_length",
    "flatten_bins",
    "query_shape",
    "query_word_counts",
    "truncated_queries",
]


def _cache_root() -> Path:
    return Path(os.environ.get("CACHE_ROOT", "/embeddings"))


def _dirsafe(label: str) -> str:
    return label.replace(":", "-").replace("/", "-")


def _cell(corpus: str, overlap: int, label: str) -> str:
    return f"{corpus}__{_PROFILE.format(overlap=overlap)}__{_dirsafe(label)}"


def _per_query(perquery_dir: Path, cell: str) -> dict[str, float] | None:
    path = perquery_dir / f"{cell}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        key = next((k for k in _METRIC_KEYS if k in data), None)
        if key is None:
            raise KeyError(f"{path} holds no nDCG array: {list(data)}")
        return {str(q): float(v) for q, v in zip(data["qids"], data[key], strict=True)}


def query_word_counts(queries: Mapping[str, str]) -> dict[str, int]:
    """Whitespace word count per query id, the length the truncation variant is defined in."""
    return {qid: len(text.split()) for qid, text in queries.items()}


def query_shape(corpus: str, queries: Mapping[str, str]) -> dict[str, Any]:
    """How long a corpus's queries are, in words: the fact the overlap answer turned on."""
    words = np.asarray(sorted(query_word_counts(queries).values()), dtype=np.int64)
    if words.size == 0:
        return {"corpus": corpus, "n_queries": 0}
    p25, p50, p75, p90 = (float(v) for v in np.percentile(words, [25, 50, 75, 90]))
    return {
        "corpus": corpus,
        "n_queries": int(words.size),
        "median_words": p50,
        "p25_words": p25,
        "p75_words": p75,
        "p90_words": p90,
        "max_words": int(words[-1]),
    }


def flatten_bins(binned: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (corpus, embedder, overlap pair, bin), so a claim can select a single bin."""
    rows: list[dict[str, Any]] = []
    for row in binned:
        keys = {k: row[k] for k in ("corpus", "embedding", "overlap_low", "overlap_high")}
        rows.extend({**keys, **b} for b in row["bins"])
    return rows


def bin_edges(lengths: Iterable[int], bins: int) -> list[int]:
    """Quantile edges over the lengths, ``bins + 1`` values from the minimum to the maximum."""
    values = np.asarray(sorted(lengths), dtype=np.int64)
    if values.size == 0:
        return []
    return [int(v) for v in np.percentile(values, np.linspace(0, 100, bins + 1))]


def _in_bin(length: int, lo: int, hi: int, *, last: bool) -> bool:
    return lo <= length <= hi if last else lo <= length < hi


def _bin_rows(
    deltas_by_query: tuple[dict[str, float], dict[str, float]], words: Mapping[str, int], edges: list[int]
) -> list[dict[str, Any]]:
    """One paired comparison per quantile bin, restricted to the queries whose length falls in it."""
    high, low = deltas_by_query
    rows: list[dict[str, Any]] = []
    for i in range(len(edges) - 1):
        last = i == len(edges) - 2
        member = {q for q, n in words.items() if _in_bin(n, edges[i], edges[i + 1], last=last)}
        paired = paired_ci(
            {q: v for q, v in high.items() if q in member}, {q: v for q, v in low.items() if q in member}
        )
        rows.append({"bin": f"Q{i + 1}", "words_lo": edges[i], "words_hi": edges[i + 1], **_round_paired(paired)})
    return rows


def _short_row(
    deltas_by_query: tuple[dict[str, float], dict[str, float]], words: Mapping[str, int], ceiling: int
) -> dict[str, Any]:
    high, low = deltas_by_query
    member = {q for q, n in words.items() if n <= ceiling}
    paired = paired_ci({q: v for q, v in high.items() if q in member}, {q: v for q, v in low.items() if q in member})
    return {"bin": f"<= {ceiling} words", "words_lo": 0, "words_hi": ceiling, **_round_paired(paired)}


def _round_paired(paired: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(paired)
    for key in ("mean_delta", "ci_lo", "ci_hi"):
        out[key] = round(float(out[key]), 4)
    return out


def by_query_length(
    *,
    corpus: str,
    label: str,
    pair: tuple[int, int],
    queries: Mapping[str, str],
    perquery_dir: Path,
    bins: int,
    short_words: int,
) -> dict[str, Any] | None:
    """The overlap delta for one embedder, by query-length bin; None when either cell is unscored."""
    low_scores = _per_query(perquery_dir, _cell(corpus, pair[0], label))
    high_scores = _per_query(perquery_dir, _cell(corpus, pair[1], label))
    if low_scores is None or high_scores is None:
        return None
    words = query_word_counts(queries)
    edges = bin_edges(words.values(), bins)
    return {
        "corpus": corpus,
        "embedding": label,
        "overlap_low": pair[0],
        "overlap_high": pair[1],
        "all": _round_paired(paired_ci(high_scores, low_scores)),
        "bins": [
            *_bin_rows((high_scores, low_scores), words, edges),
            _short_row((high_scores, low_scores), words, short_words),
        ],
    }


def truncated_queries(
    *,
    corpus: str,
    label: str,
    pair: tuple[int, int],
    perquery_dir: Path,
    variant_dir: Path,
    words: int,
) -> dict[str, Any] | None:
    """The overlap delta at full queries beside the same delta at queries cut to ``words`` words."""
    full_high = _per_query(perquery_dir, _cell(corpus, pair[1], label))
    full_low = _per_query(perquery_dir, _cell(corpus, pair[0], label))
    cut_high = _per_query(variant_dir, _cell(corpus, pair[1], label))
    cut_low = _per_query(variant_dir, _cell(corpus, pair[0], label))
    if full_high is None or full_low is None or cut_high is None or cut_low is None:
        return None
    return {
        "corpus": corpus,
        "embedding": label,
        "overlap_low": pair[0],
        "overlap_high": pair[1],
        "query_words": words,
        "full_queries": _round_paired(paired_ci(full_high, full_low)),
        "truncated_queries": _round_paired(paired_ci(cut_high, cut_low)),
    }


def _labels(perquery_dir: Path, corpus: str) -> list[str]:
    """Every embedder with an overlap-0 cell for the corpus, read off the per-query arrays."""
    prefix = f"{corpus}__{_PROFILE.format(overlap=0)}__"
    found = sorted(p.stem[len(prefix) :] for p in perquery_dir.glob(f"{prefix}*.npz"))
    # The dirsafe label is what the file carries; the scorer's rows carry the real one.
    return [f for f in found if "__" not in f]


def _real_label(dirsafe: str, rows: Iterable[Mapping[str, Any]]) -> str:
    for row in rows:
        if _dirsafe(str(row.get("embedding", ""))) == dirsafe:
            return str(row["embedding"])
    return dirsafe


def _score_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return list(data.values()) if isinstance(data, dict) else []


def _queries(corpus: str) -> dict[str, str]:
    path = _cache_root() / "chunks" / f"{corpus}__{_PROFILE.format(overlap=0)}" / "queries.json"
    return json.loads(path.read_text())


def _pairs() -> list[tuple[int, int]]:
    raw = os.environ.get("QLEN_PAIRS", "0:51,0:128,0:256")
    return [(int(a), int(b)) for a, b in (item.split(":") for item in raw.split(",") if item)]


def main() -> int:
    cache = _cache_root()
    corpora = [c for c in os.environ.get("QLEN_CORPORA", "gerdalir_de_12k_slice,mldr_en_8k_slice").split(",") if c]
    bins = int(os.environ.get("QLEN_BINS", "4"))
    short_words = int(os.environ.get("QLEN_SHORT_WORDS", "38"))
    variants = [int(v) for v in os.environ.get("QLEN_VARIANTS", "20").split(",") if v]
    out = Path(os.environ.get("OUT", _ROOT / "tests/benchmarks/raw/query-length-effect.json"))
    perquery_dir = cache / "scores" / "perquery"

    observational: list[dict[str, Any]] = []
    causal: list[dict[str, Any]] = []
    stamped_rows: list[dict[str, Any]] = []
    shapes: list[dict[str, Any]] = []
    for corpus in corpora:
        score_rows = [
            r for p in (cache / "scores").glob("*_scores.json") for r in _score_rows(p) if r.get("corpus") == corpus
        ]
        stamped_rows.extend(score_rows)
        queries = _queries(corpus)
        shapes.append(query_shape(corpus, queries))
        for dirsafe in _labels(perquery_dir, corpus):
            label = _real_label(dirsafe, score_rows)
            for pair in _pairs():
                row = by_query_length(
                    corpus=corpus,
                    label=label,
                    pair=pair,
                    queries=queries,
                    perquery_dir=perquery_dir,
                    bins=bins,
                    short_words=short_words,
                )
                if row:
                    observational.append(row)
                for words in variants:
                    variant_dir = cache / "scores" / f"perquery-q{words}w"
                    causal_row = truncated_queries(
                        corpus=corpus,
                        label=label,
                        pair=pair,
                        perquery_dir=perquery_dir,
                        variant_dir=variant_dir,
                        words=words,
                    )
                    if causal_row:
                        causal.append(causal_row)
    payload = {
        MEASURED_ON: measured_on_summary(stamped_rows),
        "note": (
            "Paired per-query nDCG@10 delta (high overlap minus low) on recursive cap256 cells. "
            "by_query_length bins the SAME cells by each query's word count; truncated_queries "
            "re-scores the cells with every query cut to its first N words. resolved=false means "
            "the 95 percent interval spans zero."
        ),
        "query_shape": shapes,
        "by_query_length": observational,
        "by_query_length_bins": flatten_bins(observational),
        "truncated_queries": causal,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[qlen] {len(observational)} binned comparisons, {len(causal)} truncated-query comparisons -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
