#!/usr/bin/env python
# pyright: basic
# Benchmark harness on numpy (no strict stubs); strict mode would only add reportUnknown* noise.
"""Promote sweep results from the out-of-repo cache into committed, gated raw data.

The sweep writes its scores to ``/embeddings/scores``, which is outside the repo and covered by
no test, no comparator and no review. Roughly 130 KB of published benchmark prose was then
hand-transcribed from it, and only one table in the whole doc set was ever checked against its
source. This is the bridge: it copies the scores into ``tests/benchmarks/raw`` in a stable,
diff-readable shape, so the docs can be GENERATED from data that CI can see.

What it deliberately does:

* rounds metrics to 4 decimals, because the digits beyond that are noise against a confidence
  interval two orders of magnitude wider, and they would churn the diff on every re-run;
* records ``has_interval`` per cell rather than inventing one. Cells scored before the scorer
  retained per-query values have no bootstrap interval, and a table must be able to tell the
  difference between "the interval is wide" and "there is no interval";
* keeps the per-query arrays OUT of the repo and references them by sha256, so an interval can be
  re-derived without git carrying 400 float arrays;
* counts the vector cells NOBODY scored, by diffing the vector cache against every score file
  that publishes the cell's corpus, and fails naming them unless ``withheld-cells.json`` beside
  the raw files records each with a reason. Every stage before this reports the effort it made;
  this is the one count taken from the corpus it was handed;
* attaches the parsed chunking axes to every row, so a chart or the explorer can facet by overlap
  or strategy without re-parsing profile strings in three places;
* carries each cell's provenance through untouched - the host, git sha, numpy and BLAS build and
  thread count recorded by the scorer AT THE MOMENT it measured that cell - because a latency or
  throughput number without the box it ran on is not a measurement. It collects none of this
  itself: the exporting machine measured nothing, and stamping it here described the wrong box on
  data weeks older, on the very files carrying the latency numbers. A cell measured before the
  scorers recorded any of it is reported as ``not_recorded`` rather than given today's machine.

Nothing in an output file therefore depends on the environment running the export, so re-exporting
unchanged sources reproduces it byte for byte instead of leaving a provenance-only diff to revert
by hand. When the file landed is git's to record, not the file's.

Env:
  CACHE_ROOT   cache dir (default /embeddings)
  OUT_DIR      destination (default tests/benchmarks/raw)
  ONLY         comma list of output names to refresh (default: all that have sources present)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _provenance import MEASURED_ON
from _score_stats import DEFAULT_ALPHA, MATERIAL_FLOOR, MATERIAL_FLOOR_METRIC, is_material, paired_ci
from audit_chunk_dimensions import parse_profile
from score_chunk_sweep import NPZ_KEYS

_ROOT = Path(__file__).resolve().parent.parent
_METRICS = ("ndcg@10", "recall@10", "mrr", "p@1")
_PLACES = 4
# Fields a specific sweep contributes beyond the shared metric set.
_PASSTHROUGH = (
    "store",
    "exact_scan",
    "rows",
    "upsert_s",
    "search_p50_ms",
    "search_p95_ms",
    "store_mb",
    "doc_recall_at_k",
    "exact_ndcg@10",
    "ndcg_delta_vs_exact",
    # the ANN setting a store row was measured at: without it the number cannot be compared
    # against a later run of the same cell, and the published rows once silently described a
    # different configuration from the one that produced them
    "ann_params",
    "control_failed",
    "reranker",
    "rerank_depth",
)

# Which cache score files become which committed file, and what the reader must be told about the
# body it was measured on. The corpus-fitness note is not editorial: a chunking comparison on a
# corpus that yields about one chunk per document is measuring nothing, and these files are the
# only place that warning can travel WITH the numbers.
_EXPORTS: dict[str, dict[str, Any]] = {
    "chunk-sweep-mldr.json": {
        # The marked twin is a THIRD corpus id on purpose: the scorer refuses to merge a foreign
        # corpus into mldr_chunk_scores.json, so its cells live in their own source file, and the
        # export publishes it beside the unmarked slice so the strategy tables pair within it.
        "sources": ["mldr_chunk_scores.json", "mldr_md_chunk_scores.json"],
        "corpora": ["mldr_de_3k_slice", "mldr_en_8k_md_slice", "mldr_en_8k_slice"],
        "note": (
            "Chunk-parameter sweep on long documents (MLDR: 18.5 chunks/doc en, 67.3 de at "
            "recursive-t256-ov0). This is the body the chunk axes can actually be judged on. "
            "mldr_en_8k_md_slice is the English slice with its section headings marked '## ' "
            "(same documents, queries and qrels); the cross-corpus pairing is in "
            "markdown-structure-effect.json."
        ),
        "fit_for_chunk_claims": True,
    },
    "chunk-sweep-gerdalir.json": {
        "sources": ["gerdalir_chunk_scores.json"],
        "corpora": ["gerdalir_de_12k_slice"],
        "note": (
            "German legal retrieval (GerDaLIR: 12000 documents, 35.98 chunks/doc at "
            "recursive-t256-ov0, 12298 queries). A second long-document body beside MLDR, in one "
            "language and one domain, so a chunking result that holds on both is not an artefact "
            "of MLDR's web-scraped text."
        ),
        "fit_for_chunk_claims": True,
    },
    "chunk-sweep-beir.json": {
        "sources": ["nf_sci_chunk_scores.json", "beir_chunk_scores.json"],
        # Four corpora, and the note now says so. The two extra ones were deliberately measured,
        # not stray: they are in the pre-embed matrix's own default corpus list, cqadupstack is
        # in the profile sweep's, and both were scored into a file named for BEIR, which needs
        # SEMDEX_SCORE_OUT and SEMDEX_SCORE_CORPORA set away from their MIRACL defaults. What
        # drifted was the note: this export began on nf_sci_chunk_scores.json alone and the note
        # was written to match, then the four-corpus source was added beside it without the note
        # being revisited. The list and the note are corrected together, which is the only way
        # this allowlist may ever be widened.
        "corpora": ["cqadupstack", "fiqa", "nfcorpus", "scifact"],
        "note": (
            "Short-document BEIR controls: nfcorpus 1.91 chunks/doc, scifact 1.81, "
            "cqadupstack 1.65, fiqa 1.26. Useful as a contrast to MLDR, NOT as evidence about "
            "chunk size: at under 2 chunks per document most profiles produce nearly the same "
            "chunks. cqadupstack and fiqa are swept over a partial profile grid and far fewer "
            "cells than the other two, so they are coarser again; the per-corpus counts are in "
            "the summary."
        ),
        "fit_for_chunk_claims": False,
    },
    "chunk-sweep-miracl.json": {
        "sources": ["chunk_sweep_scores.json", "chunk_sweep_scores_qwen3.json"],
        "corpora": ["miracl_de_100k_slice", "miracl_en_100k_slice"],
        "note": (
            "MIRACL slices at about 1.0 chunks/doc: VOID for chunk-parameter claims, since every "
            "profile yields essentially one chunk per passage. Retained because the same cells "
            "are a sound EMBEDDER comparison and are the real-vector fodder for the store and "
            "dimension benchmarks."
        ),
        "fit_for_chunk_claims": False,
    },
    "hybrid-dense-bm25.json": {
        "sources": ["hybrid_scores.json"],
        "corpora": ["mldr_de_3k_slice", "mldr_en_8k_slice"],
        "note": (
            "Dense, BM25 and their Reciprocal Rank Fusion on identical chunks, queries and "
            "judgements. The BM25 analyzer is stemmed and stopworded in the corpus language. The "
            "bm25 rows repeat across embedders by construction, because the lexical index does not "
            "depend on the embedding model."
        ),
        "fit_for_chunk_claims": True,
    },
    "store-quality.json": {
        "sources": ["store_quality.json"],
        "corpora": ["mldr_de_3k_slice", "mldr_en_8k_slice"],
        "note": (
            "Retrieval quality measured THROUGH a real vector store rather than through an exact "
            "numpy scan, with latency and disk from the same run. The exhaustive stores are the "
            "control: their nDCG must match the kernel, so an ANN number beside them is only "
            "trustworthy when it does."
        ),
        "fit_for_chunk_claims": True,
    },
    "rerank-cross-encoder.json": {
        "sources": ["rerank_scores.json"],
        "corpora": ["mldr_de_3k_slice", "mldr_en_8k_slice"],
        "note": (
            "Cross-encoder reranking of each retrieval method's shortlist. Baseline and reranked "
            "rows come from the SAME candidates at the same depth, marked @20, so a pair differs "
            "only by the reranking step and not by the size of the list."
        ),
        "fit_for_chunk_claims": True,
    },
    "embedding-multilingual-mldr.json": {
        "sources": ["bp_sweep_scores.json"],
        "corpora": ["mldr_de_3k_slice", "mldr_en_8k_slice"],
        "note": (
            "Semantic breakpoint-model sweep on MLDR: the only body where bge-m3 and e5-large "
            "are measured, and the fair German comparison, because the plain semantic profiles "
            "use chonkie's English-only default breakpoint model."
        ),
        "fit_for_chunk_claims": True,
    },
}

# The product-k measurement scores the DELIVERED chunk list beside the deduplicated document list
# (scripts/score_product_k.py). Its rows carry per-rung views, not the chunk-sweep metric set, so
# it is written by its own writer below rather than through export_row, and FLAT - one row per
# cell and rung - because the claim checker's dotted getter walks dicts only, so a figure inside a
# nested list could never carry a claim. It is deliberately NOT in _EXPORTS: unscored_cells unions
# those sources, and this file scores a subset of cells by design.
_PRODUCT_K_FILE = "product-k.json"
_PRODUCT_K_SOURCE = "product_k_scores.json"
_PRODUCT_K_CORPORA = ("gerdalir_de_12k_slice", "mldr_de_3k_slice", "mldr_en_8k_slice")
_PRODUCT_K_NOTE = (
    "The product's own unit: semdex search returns default_k chunks with NO dedup, while every "
    "other chunking table scores ten deduplicated documents. One row per cell and rung: nDCG@k "
    "over the delivered chunk list (a repeated document earns zero gain), nDCG@k over k distinct "
    "documents from the SAME top list, the mean distinct documents in the k slots, and the paired "
    "documents-minus-delivered gap. Budget rungs compare chunk sizes at a fixed number of tokens "
    "read (k = budget // cap); the product rung is the shipped default_k."
)


def _cache_root() -> Path:
    return Path(os.environ.get("CACHE_ROOT", "/embeddings"))


def measured_on_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Which measurement runs the rows in this file actually came from.

    Derived entirely from the rows, so re-exporting unchanged sources reproduces the file byte for
    byte. Nothing here describes the machine running the export: that one measured none of this.

    ``runs`` collapses identical stamps, so a 400-cell sweep on one box prints one entry, while
    cells from different boxes stay visible as the separate runs they are. ``not_recorded`` counts
    cells measured before any of this was stamped - it is never filled in from the current
    machine, because that is exactly the misattribution this replaced.
    """
    stamps = [row[MEASURED_ON] for row in rows if isinstance(row.get(MEASURED_ON), dict)]
    unique = {json.dumps(stamp, sort_keys=True): stamp for stamp in stamps}
    # Chronological, so the runs read as the timeline they are. The canonical JSON breaks a tie
    # between two runs stamped in the same second, which keeps the order total and the bytes stable.
    ordered = sorted(unique.items(), key=lambda item: (str(item[1].get("measured_utc", "")), item[0]))
    return {
        "recorded": len(stamps),
        "not_recorded": len(rows) - len(stamps),
        "runs": [stamp for _, stamp in ordered],
    }


def _round(value: Any) -> Any:
    return round(value, _PLACES) if isinstance(value, float) else value


def export_row(cell: str, row: dict[str, Any]) -> dict[str, Any]:
    """One committed row: rounded metrics, parsed axes, and an honest interval flag."""
    has_interval = f"{_METRICS[0]}_ci_lo" in row
    out: dict[str, Any] = {
        "cell": cell,
        "corpus": row.get("corpus"),
        "profile": row.get("profile"),
        "embedding": row.get("embedding"),
        "dim": row.get("dim"),
        "n_queries": row.get("n_queries"),
        "has_interval": has_interval,
        "axes": parse_profile(str(row.get("profile", ""))),
    }
    if row.get("method"):
        out["method"] = row["method"]
    # Measurement-specific fields a particular sweep produced. Carried through verbatim rather
    # than re-listed per sweep, so a new measurement does not silently lose its own columns the
    # way the store run lost exact_scan and its latency.
    for extra in _PASSTHROUGH:
        if extra in row:
            out[extra] = _round(row[extra])
    for metric in _METRICS:
        out[metric] = _round(row.get(metric))
        if has_interval:
            for suffix in ("ci_lo", "ci_hi", "sd"):
                out[f"{metric}_{suffix}"] = _round(row.get(f"{metric}_{suffix}"))
    if row.get("perquery_sha256"):
        out["perquery_sha256"] = row["perquery_sha256"]
    # Carried through verbatim and never rounded or regenerated: it describes the run that
    # produced this row, so anything computed here would describe the export instead. Absent on a
    # cell measured before scorers stamped it, and left absent rather than filled in.
    if isinstance(row.get(MEASURED_ON), dict):
        out[MEASURED_ON] = row[MEASURED_ON]
    return out


def _load_sources(names: list[str], *, quiet: bool = False) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for name in names:
        path = _cache_root() / "scores" / name
        if not path.exists():
            if not quiet:
                print(f"[export] source missing, skipped: {path}", flush=True)
            continue
        merged.update(json.loads(path.read_text()))
    return merged


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Enough shape for a reader to see what the file covers without loading every row."""
    return {
        "cells": len(rows),
        "with_interval": sum(1 for r in rows if r["has_interval"]),
        "corpora": sorted({str(r["corpus"]) for r in rows}),
        "profiles": sorted({str(r["profile"]) for r in rows}),
        "embeddings": sorted({str(r["embedding"]) for r in rows}),
        "dims": sorted({int(r["dim"]) for r in rows if r.get("dim")}),
        "query_counts": sorted({int(r["n_queries"]) for r in rows if r.get("n_queries")}),
    }


def _reject_foreign_corpora(name: str, allowed: list[str], rows: list[dict[str, Any]]) -> None:
    """Refuse to publish a cell from a corpus this export does not claim to cover.

    Each export's ``note`` tells the reader which body the numbers came from and whether that body
    can carry a chunking claim at all - the MIRACL note says outright that it is VOID for one. The
    rows, though, come from whatever the source file holds, and ``score_chunk_sweep.py`` defaults
    ``SEMDEX_SCORE_OUT`` to the MIRACL source file. Scoring any other corpus without setting that
    variable therefore lands foreign cells there and publishes them under the wrong note; 44 MLDR
    cells nearly went out as MIRACL that way. Prose nobody re-reads cannot catch it, so fail here
    and name the cells.
    """
    permitted = set(allowed)
    strays = sorted({str(r["corpus"]) for r in rows if r.get("corpus") not in permitted})
    if not strays:
        return
    cells = sorted(str(r["cell"]) for r in rows if str(r.get("corpus")) in set(strays))
    raise ValueError(
        f"{name} declares corpora {sorted(permitted)} but its sources hold {strays}: "
        f"{len(cells)} foreign cell(s), first {cells[:3]}. Either the scorer wrote into the wrong "
        "source file (check SEMDEX_SCORE_OUT) or this export's corpora list and its note have to "
        "be updated together."
    )


# Share of an embedder's token mass a cell may lose to its silent input cap and still be
# published. The ladder audit's worst row lost 0.0007 percent; the whitespace cells on German lost
# 17 to 24. Five percent sits between them with a wide margin on both sides.
_CLIPPED_MASS_PCT = 5.0


class Voids:
    """The audits' judgements of which cells must not be published, keyed the way rows are."""

    def __init__(self, *, moved: dict[tuple[str, str], dict[str, Any]], clipped: dict[tuple[str, str], float]) -> None:
        self.moved = moved  # (corpus, profile) -> the audit's row: a re-cut recursive overlap set
        self.clipped = clipped  # (chunk_set, embedder) -> token mass lost, percent


def load_voids(raw_dir: Path) -> Voids:
    """Read both audits from the raw dir. A missing audit voids nothing and is reported."""
    moved: dict[tuple[str, str], dict[str, Any]] = {}
    clipped: dict[tuple[str, str], float] = {}
    dimension = raw_dir / "chunk-dimension-audit.json"
    truncation = raw_dir / "embedder-truncation-audit.json"
    for path in (dimension, truncation):
        if not path.exists():
            print(f"[export] WARNING: {path.name} missing, so no cell can be voided on it", file=sys.stderr, flush=True)
    if dimension.exists():
        judged = json.loads(dimension.read_text()).get("integrity", {}).get("moved_boundaries_under_overlap", [])
        moved = {(str(m["corpus"]), str(m["profile"])): m for m in judged}
    if truncation.exists():
        for row in json.loads(truncation.read_text()).get("rows", []):
            clipped[(str(row["chunk_set"]), str(row["embedder"]))] = float(row.get("token_mass_lost_pct", 0.0))
    return Voids(moved=moved, clipped=clipped)


def _void_reason(row: dict[str, Any], voids: Voids) -> str | None:
    corpus, profile, embedding = str(row.get("corpus")), str(row.get("profile")), str(row.get("embedding"))
    moved = voids.moved.get((corpus, profile))
    if moved is not None:
        return (
            f"chunk set re-cut under overlap: {moved['rows']} chunks against {moved['rows_at_overlap_0']} at "
            "overlap 0 (chunk-dimension-audit.json moved_boundaries_under_overlap)"
        )
    lost = voids.clipped.get((f"{corpus}__{profile}", embedding))
    if lost is not None and lost > _CLIPPED_MASS_PCT:
        return f"embedder clipped {lost} percent of the token mass (embedder-truncation-audit.json)"
    return None


def void_cells(rows: list[dict[str, Any]], voids: Voids) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into the publishable ones and the voided ones, each of the latter with its reason."""
    kept: list[dict[str, Any]] = []
    voided: list[dict[str, Any]] = []
    for row in rows:
        reason = _void_reason(row, voids)
        if reason is None:
            kept.append(row)
        else:
            voided.append({"cell": row["cell"], "reason": reason})
    return kept, voided


def build_export(name: str, spec: dict[str, Any], *, voids: Voids | None = None) -> dict[str, Any] | None:
    merged = _load_sources(spec["sources"])
    if not merged:
        return None
    rows = sorted((export_row(cell, row) for cell, row in merged.items()), key=lambda r: str(r["cell"]))
    _reject_foreign_corpora(name, spec["corpora"], rows)
    rows, voided = void_cells(rows, voids) if voids is not None else (rows, [])
    return {
        MEASURED_ON: measured_on_summary(rows),
        "note": spec["note"],
        "fit_for_chunk_claims": spec["fit_for_chunk_claims"],
        "sources": spec["sources"],
        "summary": _summary(rows),
        "voided": voided,
        "cells": rows,
    }


def _per_query(cell: str, metric: str) -> dict[str, float] | None:
    """Per-query scores for one cell, from the arrays the scorer wrote beside the cache."""
    path = Path(os.environ.get("SEMDEX_SCORE_PERQUERY", str(_cache_root() / "scores" / "perquery"))) / f"{cell}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        # Arrays written before the npz keys were made identifier-safe are stored under the raw
        # metric name ("ndcg@10"); newer ones use the alias ("ndcg"). Accept either rather than
        # silently dropping every cell scored before the rename - which would quietly shrink the
        # paired comparisons instead of failing.
        key = next((k for k in (NPZ_KEYS[metric], metric) if k in data), None)
        if key is None:
            raise KeyError(f"{path} holds neither {NPZ_KEYS[metric]!r} nor {metric!r}: {list(data)}")
        return {str(q): float(v) for q, v in zip(data["qids"], data[key], strict=True)}


_CHUNK_AXES = ("strategy", "max_tokens", "overlap_tokens", "breakpoint_model")
_DENSE_SUFFIX = "__dense"
_STAND_IN = "_dense_stand_in"


def mark_dense_stand_ins(rows: list[dict[str, Any]]) -> None:
    """Flag each declared ``dense`` cell whose plain twin was never scored.

    A method sweep scores every embedder it covers as bm25, dense and hybrid, and its dense cell is
    the same measurement as the plain cell of the dense-only sweep. Where that plain cell exists
    the dense one is a duplicate and must stay off the chunking and embedding axes; where it does
    not (an embedder the dense-only sweep never ran), the dense cell is the only dense measurement
    there is, and dropping it would silently remove that embedder from those axes.

    Args:
        rows: Exported cells. Each flagged cell gains a transient ``_dense_stand_in`` key.
    """
    plain = {row["cell"] for row in rows if row.get("method") is None}
    for row in rows:
        if row.get("method") == "dense" and row["cell"].removesuffix(_DENSE_SUFFIX) not in plain:
            row[_STAND_IN] = True


def _one_axis_apart(left: dict[str, Any], right: dict[str, Any]) -> str | None:
    """The single axis two cells differ in, or None if they differ in zero or several.

    Comparing cells that differ in two axes at once is how a sweep attributes an overlap effect
    to a max_tokens change. Requiring exactly one difference is what makes the delta attributable.

    The embedding model counts as an axis too: two cells over the same corpus and the same chunk
    profile differ only in the model, so the same paired machinery answers "which embedder wins"
    with the same rigour, instead of that question being settled by eyeballing two rankings.

    A cell that declares a retrieval method pairs on the method axis only. The chunking and
    embedding axes are measured on the plain (dense-only) cells, or on a declared ``dense`` cell
    that ``mark_dense_stand_ins`` found no plain twin for.
    """
    if left["corpus"] != right["corpus"]:
        return None
    # Every parsed axis counts toward "how far apart", not only the knobs reported: a recipe or
    # tokenizer difference riding along with an overlap change would otherwise be credited to it.
    differing = sorted(
        axis for axis in left["axes"].keys() | right["axes"].keys() if left["axes"].get(axis) != right["axes"].get(axis)
    )
    # The retrieval method (dense / bm25 / hybrid) is an axis like any other, so "does adding
    # lexical search help" is answered by the same paired bootstrap as "does overlap help",
    # instead of by comparing two means.
    left_method, right_method = left.get("method"), right.get("method")
    if left_method != right_method:
        if left_method is not None and right_method is not None:
            return "method" if not differing and left["embedding"] == right["embedding"] else None
        # One side is a plain cell, the other declares a method. A plain cell IS the dense
        # measurement, so pairing it with its own declared-dense twin would compare a cell with
        # itself and report a method effect; the only declared cell that may cross this line is a
        # dense cell standing in for a plain one that was never scored.
        if not (left.get(_STAND_IN) or right.get(_STAND_IN)):
            return None
    elif left_method is not None and not (left.get(_STAND_IN) and right.get(_STAND_IN)):
        # The same declared method on both sides. These cells were scored to measure the method
        # axis and pair on nothing else: two bm25 cells embed nothing, so an "embedding" effect
        # between them compares one lexical ranking with itself, and the dense and hybrid variants
        # would restate every dense-only effect under a second label the effect row cannot show.
        return None
    if left["embedding"] != right["embedding"]:
        return "embedding" if not differing else None
    return differing[0] if len(differing) == 1 and differing[0] in _CHUNK_AXES else None


def knob_effects(rows: list[dict[str, Any]], metric: str = MATERIAL_FLOOR_METRIC) -> list[dict[str, Any]]:
    """Paired per-query deltas for every pair of cells that differ in exactly one axis.

    The paired comparison is the instrument the ranking tables need. Two cells answer the SAME
    queries, and most of the spread in a retrieval metric is query difficulty common to both, so
    comparing their marginal intervals by eye is a far weaker test than bootstrapping the
    per-query difference. Cells scored before per-query arrays were retained are skipped rather
    than compared with a cruder method, so no row here is quietly less rigorous than its
    neighbours.
    """
    if metric != MATERIAL_FLOOR_METRIC:
        raise ValueError(f"the material floor is defined for {MATERIAL_FLOOR_METRIC}, not {metric}")
    mark_dense_stand_ins(rows)
    effects: list[dict[str, Any]] = []
    same_list_twice = 0
    for i, left in enumerate(rows):
        for right in rows[i + 1 :]:
            axis = _one_axis_apart(left, right)
            if axis is None:
                continue
            left_scores, right_scores = _per_query(left["cell"], metric), _per_query(right["cell"], metric)
            if left_scores is None or right_scores is None:
                continue
            if _same_measurement(left_scores, right_scores):
                same_list_twice += 1
                continue
            effects.append(_effect_row(axis, left, right, metric, paired_ci(left_scores, right_scores)))
    if same_list_twice:
        print(
            f"[export] chunk-knob-effects.json: {same_list_twice} pair(s) scoring identically on every "
            "shared query dropped as one measurement under two labels",
            flush=True,
        )
    return sorted(effects, key=lambda e: (e["axis"], e["corpus"], e["embedding"], str(e["from_level"])))


def _same_measurement(left: dict[str, float], right: dict[str, float]) -> bool:
    """Whether two cells score identically on every query they share.

    Such a pair is one ranked list under two labels, not a comparison: a single ranker read at
    two shortlist depths (``bm25`` against ``bm25@20``, ``dense`` against ``dense@20``) is the
    same top-10 at nDCG@10, so pairing them reports delta zero, no wins, no losses and a tie on
    every query, inflating the axis's denominator and its tie count. Decided on the scores rather
    than on the level names so that any future same-list-twice pair is caught the same way.
    """
    shared = left.keys() & right.keys()
    return bool(shared) and all(left[q] == right[q] for q in shared)


def _level_of(cell: dict[str, Any], axis: str) -> Any:
    """The value of the axis under test for one cell."""
    if axis == "embedding":
        return cell["embedding"]
    if axis == "method":
        return cell.get("method")
    return cell["axes"].get(axis)


def _effect_row(
    axis: str, left: dict[str, Any], right: dict[str, Any], metric: str, paired: dict[str, Any]
) -> dict[str, Any]:
    mean_delta = _round(float(paired["mean_delta"]))
    return {
        "axis": axis,
        "metric": metric,
        "corpus": left["corpus"],
        "embedding": left["embedding"],
        "dim": left.get("dim"),
        "held_fixed": {k: v for k, v in left["axes"].items() if k != axis and k not in ("recipe", "tokenizer")},
        "from_level": _level_of(right, axis),
        "to_level": _level_of(left, axis),
        "from_cell": right["cell"],
        "to_cell": left["cell"],
        "mean_delta": mean_delta,
        "ci_lo": _round(float(paired["ci_lo"])),
        "ci_hi": _round(float(paired["ci_hi"])),
        "wins": paired["wins"],
        "ties": paired["ties"],
        "losses": paired["losses"],
        "n_shared": paired["n_shared"],
        # False means the query set cannot tell these two configurations apart. A table that
        # prints a winner here is inventing one.
        "resolved": paired["resolved"],
        # Resolved AND at least MATERIAL_FLOOR apart, judged on the ROUNDED mean_delta this row
        # publishes (not the unrounded delta paired_ci computed), so the flag is derivable from
        # the row itself: a reader must never see "+0.0050 resolved" stamped immaterial against
        # a 0.005 floor. A 12,298-query corpus resolves a step of 0.0009; a table that counts it
        # as a finding is counting the query set, not the knob.
        "material": is_material({"resolved": paired["resolved"], "mean_delta": mean_delta}),
    }


_WITHHELD_FILE = "withheld-cells.json"


def unscored_cells() -> dict[str, list[str]]:
    """Vector cells that no score file publishing their corpus knows about, by corpus.

    A stage reports the effort it made, never the corpus it was handed, which is how 124 MLDR
    cells sat embedded and unscored while the pages read as if the sweep were complete. This is
    the count from the other side: every cell in the vector cache whose corpus some export
    publishes, minus every cell any of that export's sources scored. A hybrid sweep's dense arm
    (``<cell>__dense``) counts as a score, because the export stands it in for the plain cell. A
    corpus no export publishes (the msmarco scale bench) is not the exporter's business.
    """
    scored_by_corpus: dict[str, set[str]] = {}
    for spec in _EXPORTS.values():
        keys = set(_load_sources(spec["sources"], quiet=True))
        for corpus in spec["corpora"]:
            scored_by_corpus.setdefault(corpus, set()).update(keys)
    vectors = _cache_root() / "vectors"
    census: dict[str, list[str]] = {}
    if not vectors.is_dir():
        return census
    for cell_dir in sorted(vectors.iterdir()):
        cell = cell_dir.name
        corpus = cell.split("__", 1)[0]
        scored = scored_by_corpus.get(corpus)
        if scored is None or cell in scored or f"{cell}{_DENSE_SUFFIX}" in scored:
            continue
        census.setdefault(corpus, []).append(cell)
    return census


def load_withheld(raw_dir: Path) -> dict[str, str]:
    """The cells deliberately left unscored, each with the reason, committed beside the data.

    An omission recorded here travels with the numbers it qualifies; one recorded nowhere is the
    operator's memory, which is what this file replaces. An entry without a reason is refused for
    the same reason.
    """
    path = raw_dir / _WITHHELD_FILE
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"[export] {_WITHHELD_FILE}: expected an object of cell -> reason")
    unreasoned = [cell for cell, reason in data.items() if not str(reason).strip()]
    if unreasoned:
        raise SystemExit(f"[export] {_WITHHELD_FILE}: no reason given for {', '.join(unreasoned)}")
    return {cell: str(reason) for cell, reason in data.items()}


def report_unscored(raw_dir: Path) -> list[str]:
    """Print every unscored cell by name and return the ones nothing withholds."""
    withheld = load_withheld(raw_dir)
    census = unscored_cells()
    failing: list[str] = []
    for corpus, cells in census.items():
        for cell in cells:
            if cell in withheld:
                print(f"[export]   withheld {cell}: {withheld[cell]}", flush=True)
            else:
                print(f"[export]   UNSCORED {cell}: vectors present, no score in any published file", flush=True)
                failing.append(cell)
        print(f"[export] {corpus}: {len(cells)} vector cell(s) without a score", flush=True)
    stale = sorted(set(withheld) - {cell for cells in census.values() for cell in cells})
    for cell in stale:
        print(f"[export]   withheld entry no longer unscored, drop it: {cell}", flush=True)
    return failing


def _write_knob_effects(out_dir: Path) -> None:
    """Effects across every exported file that has per-query arrays behind it."""
    rows: list[dict[str, Any]] = []
    for name in _EXPORTS:
        path = out_dir / name
        if path.exists():
            rows.extend(json.loads(path.read_text())["cells"])
    effects = knob_effects(rows)
    if not effects:
        print("[export] chunk-knob-effects.json: no per-query arrays available, not written", flush=True)
        return
    payload = {
        MEASURED_ON: measured_on_summary(rows),
        "alpha": DEFAULT_ALPHA,
        "material_floor": {"metric": MATERIAL_FLOOR_METRIC, "value": MATERIAL_FLOOR},
        "note": (
            "Paired per-query deltas for cell pairs differing in exactly ONE chunking axis. "
            "resolved=false means the interval at alpha spans zero, so the query set cannot "
            "separate the two configurations. material=true means resolved AND an absolute "
            "mean_delta at or above material_floor: a difference worth acting on, not only one "
            "the query set can see."
        ),
        "effects": effects,
    }
    (out_dir / "chunk-knob-effects.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    resolved = sum(1 for e in effects if e["resolved"])
    material = sum(1 for e in effects if e["material"])
    print(
        f"[export] chunk-knob-effects.json: {len(effects)} paired comparisons, "
        f"{resolved} resolved, {material} material",
        flush=True,
    )


_PRODUCT_K_CELL_FIELDS = ("corpus", "profile", "embedding", "dim", "n_queries", "product_k")


def product_k_rows(cell: str, row: dict[str, Any]) -> list[dict[str, Any]]:
    """One committed row per rung of the cell: the cell's fields, the rounded rung, the stamp verbatim."""
    base: dict[str, Any] = {"cell": cell}
    for field in _PRODUCT_K_CELL_FIELDS:
        base[field] = row.get(field)
    base["axes"] = parse_profile(str(row.get("profile", "")))
    if row.get("perquery_sha256"):
        base["perquery_sha256"] = row["perquery_sha256"]
    if isinstance(row.get(MEASURED_ON), dict):
        base[MEASURED_ON] = row[MEASURED_ON]
    rungs = sorted(row.get("rungs", []), key=lambda r: int(r["k"]))
    return [{**base, **{key: _round(value) for key, value in rung.items()}} for rung in rungs]


def product_k_payload(cells: dict[str, Any]) -> dict[str, Any]:
    """The committed product-k file from the scorer's results, guarded by the corpus allowlist."""
    rows = [flat for cell in sorted(cells) for flat in product_k_rows(cell, cells[cell])]
    _reject_foreign_corpora(_PRODUCT_K_FILE, list(_PRODUCT_K_CORPORA), rows)
    budgets = sorted({int(b) for row in rows for b in row.get("budgets", [])})
    product_ks = sorted({int(row["product_k"]) for row in rows if row.get("product_k") is not None})
    return {
        MEASURED_ON: measured_on_summary(rows),
        "note": _PRODUCT_K_NOTE,
        "sources": [_PRODUCT_K_SOURCE],
        "corpora": list(_PRODUCT_K_CORPORA),
        "product_k": product_ks[0] if len(product_ks) == 1 else product_ks,
        "budgets": budgets,
        "summary": {
            "cells": len({str(r["cell"]) for r in rows}),
            "rows": len(rows),
            "corpora": sorted({str(r["corpus"]) for r in rows}),
            "profiles": sorted({str(r["profile"]) for r in rows}),
            "embeddings": sorted({str(r["embedding"]) for r in rows}),
        },
        "rows": rows,
    }


def _write_product_k(out_dir: Path) -> None:
    """Write product-k.json from the scorer's results file, or say plainly that there is none."""
    source = _cache_root() / "scores" / _PRODUCT_K_SOURCE
    if not source.exists():
        print(f"[export] {_PRODUCT_K_FILE}: source missing, not written: {source}", flush=True)
        return
    payload = product_k_payload(json.loads(source.read_text()))
    (out_dir / _PRODUCT_K_FILE).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    summary = payload["summary"]
    print(
        f"[export] {_PRODUCT_K_FILE}: {summary['cells']} cells, {summary['rows']} rows, "
        f"corpora={len(summary['corpora'])} profiles={len(summary['profiles'])} "
        f"embedders={len(summary['embeddings'])}",
        flush=True,
    )


def main() -> None:
    out_dir = Path(os.environ.get("OUT_DIR", str(_ROOT / "tests" / "benchmarks" / "raw")))
    out_dir.mkdir(parents=True, exist_ok=True)
    only = [n for n in os.environ.get("ONLY", "").split(",") if n]

    rejected: list[str] = []
    voids = load_voids(out_dir)
    for name, spec in _EXPORTS.items():
        if only and name not in only:
            continue
        try:
            payload = build_export(name, spec, voids=voids)
        except ValueError as exc:
            # Report and carry on. Aborting the whole run would mean one drifted export blocks
            # every unrelated one, and the pressure then is to widen its allowlist just to get the
            # others out - which defeats the guard through the person it protects. The non-zero
            # exit below is what stops a partial set being read as a success.
            print(f"[export] {name}: REJECTED, not written: {exc}", flush=True)
            rejected.append(name)
            continue
        if payload is None:
            print(f"[export] {name}: no sources present, not written", flush=True)
            continue
        (out_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        summary = payload["summary"]
        print(
            f"[export] {name}: {summary['cells']} cells "
            f"({summary['with_interval']} with intervals) "
            f"corpora={len(summary['corpora'])} profiles={len(summary['profiles'])} "
            f"embedders={len(summary['embeddings'])} voided={len(payload['voided'])}",
            flush=True,
        )
        for void in payload["voided"]:
            print(f"[export]   void {void['cell']}: {void['reason']}", flush=True)
    _write_knob_effects(out_dir)
    try:
        _write_product_k(out_dir)
    except ValueError as exc:
        # Same stance as a rejected export: report, keep going, fail the run at the end.
        print(f"[export] {_PRODUCT_K_FILE}: REJECTED, not written: {exc}", flush=True)
        rejected.append(_PRODUCT_K_FILE)
    unscored = report_unscored(out_dir)
    problems: list[str] = []
    if rejected:
        problems.append(f"{len(rejected)} export(s) rejected: {', '.join(rejected)}")
    if unscored:
        problems.append(f"{len(unscored)} vector cell(s) unscored and not withheld: {', '.join(unscored)}")
    if problems:
        raise SystemExit("[export] " + "; ".join(problems))


if __name__ == "__main__":
    main()
