#!/usr/bin/env python
# pyright: basic
# Benchmark harness reading committed JSON; strict mode adds nothing over the shapes asserted here.
"""Generate every published benchmark table from committed data.

About sixty tables of benchmark prose were hand-transcribed from a results file that lives outside
the repo, and exactly one of them was ever checked against its source. The predictable happened:
docs/BENCHMARKS.md states the store quality lever as 0.71/0.81/0.83 where the raw data says
0.6534/0.7933/0.8257, and claims lancedb "held 100% recall" where the same repo's own numbers say
0.783 at 250K.

Rather than add another parser that re-implements a table's semantics in a regex, this makes the
doc a RENDERING of the data. Each table is registered once, rendered into the doc between
``<!-- BEGIN GENERATED <id> -->`` markers, and hashed into a manifest. Drift stops being something
a test has to catch, because there is no longer a second copy of the number to drift.

``collect_table_data`` is pure and reads ONLY ``tests/benchmarks/raw/*.json``. It never touches
/embeddings, so the test that re-renders and compares runs in CI on a machine with no cache.

Usage:
  python scripts/gen_bench_tables.py            # rewrite the generated blocks in docs/
  python scripts/gen_bench_tables.py --check    # exit 1 if any block is stale (what CI asks)
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
_RAW = _ROOT / "tests" / "benchmarks" / "raw"
_DOCS = _ROOT / "docs"
_MANIFEST = _DOCS / "benchmarks" / "generated-tables.manifest.json"

# A corpus can only distinguish chunk configurations if its documents are long enough to chunk.
# Judged on ONE reference profile, so the verdict describes the corpus rather than whichever
# profile happened to cut it finest.
_FITNESS_REFERENCE_PROFILE = "recursive-t256-o0-gpt2"
_FIT_CHUNKS_PER_DOC = 3

# The strategies that act on [chunker].chunk_overlap. recursive appends context to chunks it has
# already cut; markdown and fast re-split the text into overlapping windows. chonkie ignores the
# setting for semantic and late, so a profile label showing an overlap on those would be false.
_OVERLAP_AWARE_STRATEGIES = frozenset({"recursive", "markdown", "fast"})

# The axes the fitness filter governs. An axis outside this set (retrieval method, embedding
# model) is measurable on a corpus of one-chunk documents and is never dropped by it.
_CHUNK_AXES = frozenset({"overlap_tokens", "max_tokens", "strategy", "breakpoint_model"})

# The chunker-throughput raw file has two vintages: a bare list of rows, and a dict carrying the
# rows plus the run's metadata. Both are committed history, so the union is real rather than
# defensive, and naming it lets _throughput_shape narrow it once.
ThroughputFile = list[dict[str, Any]] | dict[str, Any]

_BEGIN = "<!-- BEGIN GENERATED {id} (scripts/gen_bench_tables.py) -->"
_END = "<!-- END GENERATED {id} -->"
_BLOCK_RE = re.compile(
    r"<!-- BEGIN GENERATED (?P<id>[a-z0-9_]+) \(scripts/gen_bench_tables\.py\) -->\n"
    r"(?P<body>.*?)"
    r"<!-- END GENERATED (?P=id) -->",
    re.DOTALL,
)


# ---------------------------------------------------------------- data loading


def _load(name: str) -> dict[str, Any] | None:
    path = _RAW / name
    return json.loads(path.read_text()) if path.exists() else None


def _fmt(value: Any, places: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return str(value)


def _interval(row: dict[str, Any], metric: str = "ndcg@10") -> str:
    """A metric with its interval, or a marker saying the interval does not exist.

    Printing a bare mean where no interval was computed would make an old cell look as
    well-characterised as a new one.
    """
    if not row.get("has_interval"):
        return f"{_fmt(row.get(metric))} (no interval)"
    return f"{_fmt(row.get(metric))} [{_fmt(row.get(f'{metric}_ci_lo'), 3)}, {_fmt(row.get(f'{metric}_ci_hi'), 3)}]"


def _overlap_percent(overlap_tokens: int, cap: int) -> int:
    """Overlap as the percent of the chunk cap, the unit industry guidance quotes (10-25 percent).

    Rounded half up, so the ladder's token rungs (26, 38, 51 at cap256; 51, 77, 102, 128 at cap512)
    print as the round percentages they were chosen to be.
    """
    return int(overlap_tokens * 100 / cap + 0.5)


def _overlap_text(overlap_tokens: Any, cap: Any) -> str:
    """An overlap level in tokens, with its percent of the cap when the cap is known."""
    if _is_rung(overlap_tokens) and _is_rung(cap) and cap > 0:
        return f"{overlap_tokens} ({_overlap_percent(overlap_tokens, cap)}%)"
    return level_label(overlap_tokens)


def _display_profile(axes: dict[str, Any]) -> str:
    """A profile named so it needs no legend.

    The cache directory names stay as they are - 405 cells are keyed by them - but nothing
    user-facing should print o10 for an overlap of ten TOKENS, or t256 for a number that is a hard
    cap under recursive and an ignored hint under semantic.
    """
    strategy = axes.get("strategy") or "?"
    size = axes.get("max_tokens")
    overlap = axes.get("overlap_tokens") or 0
    label = f"{strategy} {'cap' if strategy == 'recursive' else 'hint'}{size}"
    # Every strategy that HONOURS overlap has to show it, or its levels print as one repeated
    # label carrying different numbers. markdown and fast honour it as a re-split; only the
    # chonkie strategies ignore the setting, and printing ov0tok on those would be a lie.
    if strategy in _OVERLAP_AWARE_STRATEGIES:
        percent = f" ({_overlap_percent(overlap, size)}%)" if isinstance(size, int) and size > 0 else ""
        label += f" ov{overlap}tok{percent}"
    breakpoint_model = axes.get("breakpoint_model")
    if breakpoint_model:
        label += f" breakpoint-{breakpoint_model}"
    elif strategy == "semantic":
        label += " breakpoint-default(potion-base-32M)"
    return label


# ---------------------------------------------------------------- table builders


def _table(title: str, columns: list[str], rows: list[list[str]], note: str = "") -> dict[str, Any]:
    return {"title": title, "columns": columns, "rows": rows, "note": note}


def _axis_status_table(audit: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for _key, value in sorted(audit["axis_status"].items()):
        counts = ", ".join(f"{k} {v}" for k, v in sorted(value["status_counts"].items()))
        rows.append([f"`{value['axis']}`", value["strategy"], value["verdict"], str(value["groups"]), counts])
    return _table(
        "What each sweep axis actually did",
        ["Axis", "Strategy", "Verdict", "Groups compared", "Per-group outcome"],
        rows,
        "Measured from the chunk sets, not read off the profile names. `inert` means the chunk "
        "boundaries were byte-identical across the axis levels, `content-only` means the "
        "boundaries did not move but the chunk text did, `weak` means under 5 percent of chunks "
        "moved, `conditional` means the axis bites under some held-fixed values and not others.",
    )


def _corpus_fitness_table(audit: dict[str, Any]) -> dict[str, Any]:
    seen: dict[str, dict[str, Any]] = {}
    for row in audit["chunk_sets"]:
        if row["profile"] == _FITNESS_REFERENCE_PROFILE:
            seen[row["corpus"]] = row
    rows = []
    for corpus, row in sorted(seen.items(), key=lambda kv: -(kv[1]["chunks_per_doc"] or 0)):
        fit = "yes" if (row["chunks_per_doc"] or 0) >= _FIT_CHUNKS_PER_DOC else "NO"
        rows.append(
            [
                corpus,
                f"{row['docs']:,}",
                f"{row['rows']:,}",
                _fmt(row["chunks_per_doc"], 2),
                f"{row['token_p50']}",
                fit,
            ]
        )
    return _table(
        "Corpus fitness for chunking claims",
        ["Corpus", "Documents", "Chunks", "Chunks/doc", "Median chunk tokens", "Can carry a chunk claim"],
        rows,
        "Measured at `recursive cap256 ov0tok`. A corpus that yields about one chunk per document "
        "cannot distinguish chunk configurations at all, because every profile produces the same "
        "single chunk. The threshold of 3 chunks/doc is the point below which the profiles in this "
        "sweep stop differing meaningfully.",
    )


def _profile_shape_table(audit: dict[str, Any], corpus: str) -> dict[str, Any]:
    rows = []
    for row in sorted(audit["chunk_sets"], key=lambda r: str(r["profile"])):
        if row["corpus"] != corpus or row["axes"].get("breakpoint_model"):
            continue
        rows.append(
            [
                _display_profile(row["axes"]),
                f"{row['rows']:,}",
                _fmt(row["chunks_per_doc"], 2),
                f"{row['token_p50']}",
                f"{row['token_max']}",
                str(row["true_cap"]),
                "yes" if row["cap_enforced"] else "NO",
                f"{row['chars_per_token']:.2f}" if row["chars_per_token"] else "n/a",
            ]
        )
    return _table(
        f"What each chunk profile produces ({corpus})",
        ["Profile", "Chunks", "Chunks/doc", "Median tokens", "Max tokens", "True cap", "Cap held", "Chars/token"],
        rows,
        "The true cap is `max_tokens + overlap` for recursive, whose overlap is appended context "
        "that deliberately overshoots, and `max_tokens` for every other strategy. Chars/token "
        "differs across strategies because chonkie's "
        "SemanticChunker takes no tokenizer argument and counts in its embedding model's "
        "tokenizer, so token counts are NOT comparable between recursive and semantic rows.",
    )


def _held_fixed_label(held: dict[str, Any], axis: str) -> str:
    """Describe what was held constant, naming only the axes that are not the one under test."""
    parts: list[str] = []
    if axis != "strategy" and held.get("strategy"):
        parts.append(str(held["strategy"]))
    if axis != "max_tokens" and held.get("max_tokens"):
        cap = "cap" if held.get("strategy") == "recursive" else "hint"
        parts.append(f"{cap}{held['max_tokens']}")
    if axis != "overlap_tokens" and held.get("overlap_tokens") is not None:
        parts.append(f"ov{held['overlap_tokens']}tok")
    if axis != "breakpoint_model" and held.get("breakpoint_model"):
        parts.append(f"breakpoint-{held['breakpoint_model']}")
    return " ".join(parts) or "-"


def _is_rung(level: Any) -> bool:
    """Whether a knob level is a numeric rung on a ladder, rather than a category label.

    Shared by ``_oriented`` (numeric ordering) and ``_is_ladder_axis`` (ladder detection), so the
    two can never disagree on what counts as a rung. ``bool`` is excluded because it is an ``int``
    to ``isinstance`` and never a rung.
    """
    return isinstance(level, (int, float)) and not isinstance(level, bool)


def _oriented(effect: dict[str, Any]) -> dict[str, Any]:
    """Orient a comparison low level to high level, so the sign means the same thing every row.

    The pair order is arbitrary, so half the rows would otherwise read "512 to 256" and half
    "256 to 512", and a reader scanning the delta column would have to check the direction on
    every line.
    """
    left, right = effect["from_level"], effect["to_level"]
    ordered = (left is None or right is None) or (
        left <= right if _is_rung(left) and _is_rung(right) else str(left) <= str(right)
    )
    if ordered:
        return effect
    return {
        **effect,
        "from_level": right,
        "to_level": left,
        "mean_delta": -effect["mean_delta"],
        "ci_lo": -effect["ci_hi"],
        "ci_hi": -effect["ci_lo"],
        "wins": effect["losses"],
        "losses": effect["wins"],
    }


def level_label(level: Any) -> str:
    """Print one level of a knob.

    A null level is the knob left at its default: the plain semantic profile sets no breakpoint
    model, and the exporter records that side of the comparison as null. Printing Python's None
    reads as a missing value, when it names the very configuration a deployer starts from.
    """
    return "default" if level is None else str(level)


def _levels_text(effect: dict[str, Any], axis: str) -> str:
    """The "from to" cell of an effect row; an overlap level also shows its percent of the held cap."""
    if axis == "overlap_tokens":
        cap = effect["held_fixed"].get("max_tokens")
        return f"{_overlap_text(effect['from_level'], cap)} to {_overlap_text(effect['to_level'], cap)}"
    return f"{level_label(effect['from_level'])} to {level_label(effect['to_level'])}"


def _knob_rows(
    effects: list[dict[str, Any]],
    axis: str,
    fit: set[str] | None,
    profile: str | None = None,
    levels: set[str] | None = None,
) -> list[list[str]]:
    rows = []
    for raw in effects:
        if raw["axis"] != axis:
            continue
        if fit is not None and raw["corpus"] not in fit:
            continue
        if profile is not None and _held_fixed_label(raw["held_fixed"], axis) != profile:
            continue
        # A levels filter keeps two families of one axis apart. The retrieval methods exist at two
        # shortlist depths, and a table mixing them would confound the reranker with the size of
        # the list it reranked.
        if levels is not None and not {str(raw["from_level"]), str(raw["to_level"])} <= levels:
            continue
        effect = _oriented(raw)
        rows.append(
            [
                effect["corpus"],
                # For the embedding axis the model is the axis, so repeating it here would just
                # echo the Change column; the dimension is what a reader wants beside it.
                str(effect.get("dim") or "-") if axis == "embedding" else effect["embedding"],
                _held_fixed_label(effect["held_fixed"], axis),
                _levels_text(effect, axis),
                f"{effect['mean_delta']:+.4f}",
                f"[{effect['ci_lo']:+.4f}, {effect['ci_hi']:+.4f}]",
                f"{effect['wins']}/{effect['losses']}",
                "resolved" if effect["resolved"] else "not resolved",
            ]
        )
    return sorted(rows, key=lambda r: (r[0], r[2], r[1]))


def _knob_table(
    effects_doc: dict[str, Any],
    axis: str,
    title: str,
    note: str,
    *,
    fit: set[str] | None = None,
    profile: str | None = None,
    levels: set[str] | None = None,
) -> dict[str, Any]:
    rows = _knob_rows(effects_doc["effects"], axis, fit, profile, levels)
    fit_note = (
        " Only corpora that can carry a chunking claim are shown; the rest yield about one chunk "
        "per document, where every profile produces the same chunk. The full set is in "
        "`tests/benchmarks/raw/chunk-knob-effects.json`."
        if fit is not None
        else ""
    )
    return _table(
        title,
        [
            "Corpus",
            "Dim" if axis == "embedding" else "Embedder",
            "Held fixed",
            "Change (low to high)",
            "Delta nDCG@10",
            "95% CI",
            "Win/loss",
            "Verdict",
        ],
        rows,
        note + fit_note,
    )


def _favoured_level(effect: dict[str, Any]) -> Any:
    """Which of the two levels a resolved comparison came out in favour of."""
    return effect["to_level"] if effect["mean_delta"] > 0 else effect["from_level"]


def _is_ladder_axis(effects: list[dict[str, Any]], axis: str) -> bool:
    """Whether an axis's levels are all numeric, so its comparisons form ordered ladders.

    Detected from the values rather than kept in a list: a new numeric knob (a minimum chunk
    size, say) would otherwise fall back silently to the per-level tally this rule replaces.
    Uses ``_is_rung`` so this and ``_oriented`` never disagree on what counts as a rung.
    """
    levels = {
        level for effect in effects if effect["axis"] == axis for level in (effect["from_level"], effect["to_level"])
    }
    return bool(levels) and all(_is_rung(level) for level in levels)


def _ladders(
    effects_doc: dict[str, Any], axis: str, fit: set[str] | None
) -> dict[tuple[str, str, str], dict[tuple[Any, Any], dict[str, Any]]]:
    """One ladder per corpus, held-fixed set and embedder: every oriented pair on the axis, keyed low to high."""
    ladders: dict[tuple[str, str, str], dict[tuple[Any, Any], dict[str, Any]]] = {}
    for raw in effects_doc["effects"]:
        if raw["axis"] != axis or (fit is not None and raw["corpus"] not in fit):
            continue
        effect = _oriented(raw)
        key = (effect["corpus"], _held_fixed_label(effect["held_fixed"], axis), effect["embedding"])
        ladders.setdefault(key, {})[(effect["from_level"], effect["to_level"])] = effect
    return ladders


def _ladder_verdicts(ladders: dict[tuple[str, str, str], dict[tuple[Any, Any], dict[str, Any]]]) -> dict[str, int]:
    """Judge each ladder ONCE, by the pair of its highest and lowest level.

    A per-level tally counts partners: on a 20-rung ladder a middle rung has 19 and collects the
    most favours by beating everything below it, which says nothing about direction. The
    end-to-end pair is one comparison per ladder, so the counts here are ladders.
    ``undecided`` is a ladder whose end-to-end pair is unresolved or immaterial; ``unspanned`` is
    one whose end-to-end pair was never measured, which the committed data does not contain.
    """
    verdicts = {"more": 0, "less": 0, "undecided": 0, "unspanned": 0}
    for pairs in ladders.values():
        levels = sorted({level for pair in pairs for level in pair})
        span = pairs.get((levels[0], levels[-1]))
        if span is None:
            verdicts["unspanned"] += 1
        elif not span["material"]:
            verdicts["undecided"] += 1
        elif span["mean_delta"] > 0:
            verdicts["more"] += 1
        else:
            verdicts["less"] += 1
    return verdicts


def _ladder_direction(ladders: dict[tuple[str, str, str], dict[tuple[Any, Any], dict[str, Any]]]) -> str:
    verdicts = _ladder_verdicts(ladders)
    text = (
        f"{verdicts['more']} favour more, {verdicts['less']} favour less, "
        f"{verdicts['undecided']} unresolved or immaterial"
    )
    if verdicts["unspanned"]:
        text += f", {verdicts['unspanned']} without an end-to-end pair"
    return f"{text} (of {len(ladders)} ladders)"


def _knob_summary_table(
    effects_doc: dict[str, Any], fit: set[str] | None = None, axes: frozenset[str] | None = None
) -> dict[str, Any]:
    """One row per axis: comparisons, resolved, the chance ceiling, material, and which way.

    ``axes`` restricts the rows. The chunking page shows only the chunk axes: the embedding and
    retrieval-method rows belong to their own pages, and on the chunking page they would outweigh
    every chunking axis with nothing in the prose to explain them. An ``effects_doc`` exported
    before ``alpha``/``material_floor`` were recorded raises ``KeyError``; regenerate it with the
    current exporter rather than patching a default in here.
    """
    alpha = float(effects_doc["alpha"])
    floor = float(effects_doc["material_floor"]["value"])
    metric = str(effects_doc["material_floor"]["metric"])
    buckets: dict[str, dict[str, Any]] = {}
    for effect in effects_doc["effects"]:
        if fit is not None and effect["corpus"] not in fit:
            continue
        if axes is not None and effect["axis"] not in axes:
            continue
        # winners is populated for every axis, ladder ones included, even though the direction
        # column below never reads it for a ladder axis (it uses _ladder_direction instead):
        # whether an axis IS a ladder is not known until every effect for it has been grouped, so
        # deferring the tally to a second pass would mean re-walking effects_doc["effects"] a
        # second time rather than a single one-line dict increment here.
        bucket = buckets.setdefault(effect["axis"], {"n": 0, "resolved": 0, "material": 0, "winners": {}})
        bucket["n"] += 1
        if effect["resolved"]:
            bucket["resolved"] += 1
        if effect["material"]:
            bucket["material"] += 1
            level = level_label(_favoured_level(effect))
            bucket["winners"][level] = bucket["winners"].get(level, 0) + 1
    rows = []
    for axis, bucket in sorted(buckets.items()):
        share = f"{bucket['resolved'] / bucket['n'] * 100:.0f}%" if bucket["n"] else "n/a"
        # Ladder-ness is checked against EVERY effect for this axis, not effects_doc scoped to
        # fit: whether an axis's levels are numeric is a property of the axis itself, not of
        # which corpora are in scope, so a fit filter must never change what counts as a ladder.
        if _is_ladder_axis(effects_doc["effects"], axis):
            direction = _ladder_direction(_ladders(effects_doc, axis, fit))
        else:
            direction = (
                ", ".join(
                    f"{count} favour {level}"
                    for level, count in sorted(bucket["winners"].items(), key=lambda kv: -kv[1])
                )
                or "none"
            )
        rows.append(
            [
                f"`{axis}`",
                str(bucket["n"]),
                f"{bucket['resolved']} ({share})",
                str(math.ceil(bucket["n"] * alpha)),
                str(bucket["material"]),
                direction,
            ]
        )
    return _table(
        "Do the chunking knobs move retrieval at all",
        [
            "Axis",
            "Paired comparisons",
            f"Resolved at {1 - alpha:.0%}",
            "Up to N by chance",
            f"Resolved and material (>= {floor:g} {metric})",
            "Which way",
        ],
        rows,
        "Each comparison pairs two cells differing in exactly one axis and bootstraps the "
        "per-query difference over the queries they share. Unresolved means the interval spans "
        f"zero: the query set cannot separate those two configurations. Up to N by chance is comparisons "
        f"times alpha {alpha:.2f}, rounded up: the resolutions that many true nulls would produce on average, "
        f"a ceiling on how much of the resolved count is noise. Material means resolved AND an absolute "
        f"difference of at least {floor:g} {metric}; a 12,298-query corpus resolves steps far below that. "
        "For a numeric axis the direction is judged once per ladder (corpus, embedder, held-fixed set) by its "
        "top against bottom pair, so the counts are ladders; where the optimum sits within a ladder is the "
        "ceiling table's question. For a categorical axis the material comparisons are tallied per level. "
        "Restricted to corpora long enough to carry a chunking claim.",
    )


def _ranking_table(doc: dict[str, Any], corpus: str, limit: int = 12) -> dict[str, Any]:
    cells = [c for c in doc["cells"] if c["corpus"] == corpus]
    cells.sort(key=lambda c: -(c.get("ndcg@10") or 0))
    rows = []
    for rank, cell in enumerate(cells[:limit], start=1):
        rows.append(
            [
                str(rank),
                _display_profile(cell["axes"]),
                cell["embedding"],
                str(cell["dim"]),
                _interval(cell),
                _fmt(cell.get("recall@10")),
                str(cell.get("n_queries")),
            ]
        )
    return _table(
        f"Top configurations on {corpus}",
        ["#", "Profile", "Embedder", "Dim", "nDCG@10 [95% CI]", "Recall@10", "Queries"],
        rows,
        "Ranked by mean nDCG@10. Neighbouring rows whose intervals overlap are not separated by "
        "this query set; see the paired comparisons for which differences actually resolve.",
    )


def _embedder_ranking_table(doc: dict[str, Any], corpora: list[str]) -> dict[str, Any]:
    """Best configuration per embedder, so the model is compared at its own best setting.

    Ranking every cell would let a model with more cells in the grid dominate the top of the
    table by sheer count. One row per model, at whichever profile that model did best on, is the
    comparison a reader is actually making.
    """
    rows = []
    for corpus in corpora:
        best: dict[str, dict[str, Any]] = {}
        for cell in doc["cells"]:
            if cell["corpus"] != corpus:
                continue
            current = best.get(str(cell["embedding"]))
            if current is None or (cell.get("ndcg@10") or 0) > (current.get("ndcg@10") or 0):
                best[str(cell["embedding"])] = cell
        rows.extend(
            [
                corpus.replace("_slice", ""),
                str(cell["embedding"]),
                str(cell["dim"]),
                _display_profile(cell["axes"]),
                _interval(cell),
                _fmt(cell.get("recall@10")),
            ]
            for cell in sorted(best.values(), key=lambda c: -(c.get("ndcg@10") or 0))
        )
    return _table(
        "Embedding models at their best chunk configuration",
        ["Corpus", "Model", "Dim", "Best profile for this model", "nDCG@10 [95% CI]", "Recall@10"],
        rows,
        "One row per model, at whichever profile that model scored highest on, so a model is not "
        "penalised for a setting that suits another. Dimension rides along with the model and "
        "cannot be separated from it here.",
    )


def _precision_table(docs: list[dict[str, Any]]) -> dict[str, Any]:
    by_corpus: dict[str, dict[str, Any]] = {}
    for doc in docs:
        for cell in doc["cells"]:
            if not cell.get("has_interval"):
                continue
            half = (cell["ndcg@10_ci_hi"] - cell["ndcg@10_ci_lo"]) / 2
            entry = by_corpus.setdefault(cell["corpus"], {"n": cell["n_queries"], "halves": []})
            entry["halves"].append(half)
    rows = []
    for corpus, entry in sorted(by_corpus.items(), key=lambda kv: -kv[1]["n"]):
        mean_half = math.fsum(entry["halves"]) / len(entry["halves"])
        rows.append([corpus, str(entry["n"]), f"+/-{mean_half:.3f}", f"{mean_half * 2:.3f}"])
    return _table(
        "How large a difference has to be before it is a result",
        ["Corpus", "Queries", "Mean 95% half-width", "Smallest separable difference"],
        rows,
        "A difference smaller than the half-width is not a finding, it is noise. This is why "
        "several previously published verdicts, decided on gaps of 0.01 or less, are reported "
        "here as unresolved.",
    )


def _extractor_tables(baseline: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Coverage and latency for the document extractors, from the gated slice."""
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for key, value in baseline.get("results", {}).items():
        if not key.startswith("extract/"):
            continue
        name, _, fmt = key[len("extract/") :].partition("@")
        cells[(name, fmt)] = value
    extractors = sorted({name for name, _ in cells})
    formats = sorted({fmt for _, fmt in cells})

    coverage_rows = []
    latency_rows = []
    for name in extractors:
        cover, latency = [name], [name]
        for fmt in formats:
            cell = cells.get((name, fmt))
            if cell is None:
                cover.append("-")
                latency.append("-")
                continue
            status = cell.get("status")
            metrics = cell.get("metrics", {})
            recall = metrics.get("phrase_recall")
            cover.append(f"{recall * 100:.0f}%" if status == "ok" and recall is not None else str(status))
            ms = metrics.get("latency_ms")
            latency.append(f"{ms:.1f}" if status == "ok" and ms is not None else str(status))
        coverage_rows.append(cover)
        latency_rows.append(latency)

    return {
        "extractor_coverage": _table(
            "Which extractor reads which format",
            ["Extractor", *formats],
            coverage_rows,
            "Share of known phrases recovered from a fixture of that format. `n/a` means the "
            "extractor does not claim the format; `skip` means it was not exercised in this run. "
            "These are self-authored fixtures, so this measures agreement with our expectations "
            "rather than fidelity to documents in the wild - see the gaps page.",
        ),
        "extractor_latency": _table(
            "Extraction latency per document (ms)",
            ["Extractor", *formats],
            latency_rows,
            "One document per cell on the reference machine, single run. Read the orders of "
            "magnitude, not the digits: these are not repeated measurements and the box was "
            "shared.",
        ),
    }


def _ocr_table(doc: dict[str, Any]) -> dict[str, Any]:
    rows = [
        [
            row["engine"],
            row.get("model", "-"),
            "vision LLM" if row["kind"] == "vlm" else "classic",
            f"{row['content_pass'] * 100:.1f}%",
            f"{row['present'] * 100:.1f}%",
            f"{row['order'] * 100:.1f}%",
        ]
        for row in sorted(doc["results"], key=lambda r: -r["content_pass"])
    ]
    return _table(
        "OCR engines on degraded scans",
        ["Engine", "Model", "Kind", "Content pass", "Text present", "Reading order"],
        rows,
        f"olmOCR-bench old-scans split: {doc.get('n_pdfs')} degraded scans, "
        f"{doc.get('n_content_tests')} unit tests. Content pass is the headline; the other two "
        "columns separate 'found the text' from 'put it in the right order'.",
    )


def _store_dim_synthetic_table(doc: dict[str, Any]) -> dict[str, Any]:
    """The synthetic dimension sweep: curve SHAPE only, never an absolute or a recall claim."""
    rows = []
    for row in doc["results"]:
        exact, ann = row.get("sqlite_vec") or {}, row.get("lancedb") or {}
        if "error" in exact or "error" in ann:
            continue
        rows.append(
            [
                str(row["dim"]),
                f"{row['scale']:,}",
                _fmt(exact.get("search_p50_ms"), 2),
                _fmt(ann.get("search_p50_ms"), 2),
                "ANN" if row.get("ann_faster") else "exact",
            ]
        )
    return _table(
        "Synthetic dimension sweep (shape only)",
        ["Dim", "Rows", "sqlite_vec exact p50 ms", "lancedb ANN p50 ms", "Faster"],
        rows,
        "Random unit vectors. Latency depends only on row count and dimension, so these reproduce "
        "the SHAPE of the curve with no model or corpus needed. They do NOT reproduce recall: "
        "random vectors are near-orthogonal and evenly spread, the easiest possible case for a "
        "partitioning index, whereas real embeddings cluster. Read the shape here and the "
        "absolute numbers from the real-vector table. "
        f"lancedb index threshold lowered to {doc.get('lance_index_threshold')} so ANN is active "
        "at every scale.",
    )


def _method_ranking_table(doc: dict[str, Any]) -> dict[str, Any]:
    """Dense, BM25 and hybrid side by side for each corpus and embedder."""
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for cell in doc["cells"]:
        by_key.setdefault((str(cell["corpus"]), str(cell["embedding"])), {})[str(cell.get("method"))] = cell
    rows = []
    for (corpus, embedding), methods in sorted(by_key.items()):
        dense, bm25, hybrid = methods.get("dense"), methods.get("bm25"), methods.get("hybrid")
        if not (dense and bm25 and hybrid):
            continue
        best = max((dense, bm25, hybrid), key=lambda c: c.get("ndcg@10") or 0)
        rows.append(
            [
                corpus.replace("_slice", ""),
                embedding,
                str(dense["dim"]),
                _interval(dense),
                _interval(bm25),
                _interval(hybrid),
                str(best.get("method")),
            ]
        )
    return _table(
        "Dense, BM25 and hybrid on the same queries",
        ["Corpus", "Embedder", "Dim", "Dense nDCG@10", "BM25 nDCG@10", "Hybrid nDCG@10", "Best"],
        rows,
        "One chunk set, one query set, one set of judgements, three retrieval methods. The BM25 "
        "column repeats down each corpus because the lexical index does not depend on the "
        "embedding model, which is a free check that the three runs really did share an index. "
        "Hybrid is Reciprocal Rank Fusion of the other two.",
    )


def declared_renderings(raw_dir: Path = _RAW) -> set[str]:
    """Corpora that a committed raw file declares to be a re-rendering of another corpus.

    Any raw JSON whose top-level ``slice`` object carries both ``corpus`` and ``source_corpus``
    declares ``slice["corpus"]`` a rendering of ``slice["source_corpus"]``: same documents, same
    queries and qrels, re-rendered to answer one paired question. Scanning for the shape (rather
    than naming a file) means a future rendering declared the same way is excluded with no code
    change. ``raw_dir`` is a parameter so a test can point it at a throwaway fixture.
    """
    renderings: set[str] = set()
    for path in sorted(raw_dir.glob("*.json")):
        # No skip on a broken file: if the broken one is the file that declares a rendering, the
        # rendering would silently re-enter every cross-corpus aggregate and double-count its source.
        doc = json.loads(path.read_text())
        if not isinstance(doc, dict):
            continue
        slice_info = doc.get("slice")
        if isinstance(slice_info, dict) and "corpus" in slice_info and "source_corpus" in slice_info:
            renderings.add(slice_info["corpus"])
    return renderings


def fit_corpora(audit: dict[str, Any] | None, renderings: set[str]) -> set[str] | None:
    """Corpora long enough to carry a chunking claim, judged on ONE reference profile.

    Judging it per profile let nfcorpus in at 3.44 chunks/doc through its semantic profile while
    its reference profile yields 1.91, which is exactly the corpus-fitness error this prevents. A
    corpus named in ``renderings`` is excluded even when its own chunks/doc clears the threshold:
    it is the same documents as another corpus, re-rendered, and answers only its own paired
    question, never a fourth corpus in a cross-corpus aggregate.
    """
    fit = {
        r["corpus"]
        for r in (audit or {}).get("chunk_sets", [])
        if r["profile"] == _FITNESS_REFERENCE_PROFILE and (r.get("chunks_per_doc") or 0) >= _FIT_CHUNKS_PER_DOC
    }
    return (fit - renderings) or None


def _fit_corpora(audit: dict[str, Any] | None) -> set[str] | None:
    """The fit set for the committed raw directory: fit_corpora with its renderings read in."""
    return fit_corpora(audit, declared_renderings())


def _register_chunk_tables(tables: dict[str, Any], audit: dict[str, Any]) -> None:
    tables["chunk_axis_status"] = _axis_status_table(audit)
    tables["corpus_fitness"] = _corpus_fitness_table(audit)
    tables["chunk_profile_shape_mldr_en"] = _profile_shape_table(audit, "mldr_en_8k_slice")


def unjudged_corpora(effects: dict[str, Any] | None, audit: dict[str, Any] | None) -> list[str]:
    """Corpora with chunk-axis measurements that the fitness audit never judged.

    A corpus the audit judged UNFIT is dropped on purpose and says so in the fitness table. A
    corpus the audit has never SEEN is dropped by the same filter for a completely different
    reason - the audit is stale - and that drop is silent. A three-week-old audit once removed
    every one of GerDaLIR's 162 comparisons from every table on the chunking page while the prose
    went on asserting an overlap conclusion GerDaLIR reverses.
    """
    judged = {
        row["corpus"] for row in (audit or {}).get("chunk_sets", []) if row["profile"] == _FITNESS_REFERENCE_PROFILE
    }
    measured = {effect["corpus"] for effect in (effects or {}).get("effects", []) if effect["axis"] in _CHUNK_AXES}
    return sorted(measured - judged)


def _warn_unjudged(corpora: list[str]) -> None:
    if not corpora:
        return
    print(
        f"[tables] WARNING: {', '.join(corpora)} carry chunk-axis measurements that the fitness "
        f"audit has never judged, so every table drops them silently. Re-run "
        f"scripts/audit_chunk_dimensions.py over ALL corpora.",
        file=sys.stderr,
        flush=True,
    )


def _rung_table(effects: dict[str, Any], fit: set[str] | None) -> dict[str, Any]:
    """What each STEP UP the overlap ladder buys, rather than each level against zero.

    Against zero every rung of a rising ladder resolves, which says a ladder rises and not where
    it stops. Consecutive steps answer that: a step whose gain has fallen to nothing is the top,
    and a ladder still gaining at its last rung was cut short by the sweep rather than by the
    data.
    """
    ladders: dict[tuple[str, str, int], dict[tuple[int, int], list[dict[str, Any]]]] = {}
    for raw in effects["effects"]:
        if raw["axis"] != "overlap_tokens" or (fit is not None and raw["corpus"] not in fit):
            continue
        effect = _oriented(raw)
        cap = effect["held_fixed"].get("max_tokens")
        if cap is None:
            continue
        key = (effect["corpus"], _held_fixed_label(effect["held_fixed"], "overlap_tokens"), int(cap))
        # A list, not an assignment: one level pair is measured once per embedder, and keying by
        # the pair alone would keep whichever embedder happened to come last.
        ladders.setdefault(key, {}).setdefault((effect["from_level"], effect["to_level"]), []).append(effect)

    rows = []
    for (corpus, held, cap), pairs in sorted(ladders.items()):
        levels = sorted({level for pair in pairs for level in pair})
        # Two levels are one comparison, which the effect table already shows; a ladder needs three.
        if len(levels) < 3:
            continue
        for low, high in itertools.pairwise(levels):
            step = pairs.get((low, high))
            if not step:
                continue
            deltas = [effect["mean_delta"] for effect in step]
            rows.append(
                [
                    corpus,
                    held,
                    f"{low} to {high}",
                    f"{low / cap:.1%} to {high / cap:.1%}",
                    f"{sum(1 for effect in step if effect['resolved'])}/{len(step)}",
                    f"{math.fsum(deltas) / len(deltas):+.4f}",
                    f"[{min(deltas):+.4f}, {max(deltas):+.4f}]",
                ]
            )
    return _table(
        "What each step up the overlap ladder buys",
        ["Corpus", "Held fixed", "Rung step", "Percent of cap", "Resolved", "Mean delta", "Delta range"],
        rows,
        "One row per CONSECUTIVE pair of overlap levels, aggregated over the embedders measured at "
        "that pair, so the column reads as the marginal gain of that step rather than of the whole "
        "ladder. Resolved counts how many of those embedders separated the two levels at 95 "
        "percent. A ladder whose last step is still positive has not been measured to its top.",
    )


def _ladders_by_embedder(
    effects: dict[str, Any], fit: set[str] | None
) -> dict[tuple[str, str, int, str], dict[tuple[int, int], dict[str, Any]]]:
    """Overlap ladders keyed with their cap, one per EMBEDDER, for the ceiling table.

    ``_rung_table`` deliberately pools the embedders at a step; this keys them apart, because the
    question "has the ladder stopped paying" is asked of one embedder at a time. The
    grouping is ``_ladders``; this adds the cap the ceiling table prints and drops a ladder with
    no cap.
    """
    ladders: dict[tuple[str, str, int, str], dict[tuple[int, int], dict[str, Any]]] = {}
    for (corpus, held, embedding), pairs in _ladders(effects, "overlap_tokens", fit).items():
        cap = next(iter(pairs.values()))["held_fixed"].get("max_tokens")
        if cap is None:
            continue
        ladders[(corpus, held, int(cap), embedding)] = pairs
    return ladders


def _step_verdict(step: dict[str, Any]) -> str:
    """A step's delta with what it means: material, resolved but under the floor, or unresolved."""
    if step["material"]:
        state = "material"
    elif step["resolved"]:
        state = "resolved, immaterial"
    else:
        state = "unresolved"
    return f"{step['mean_delta']:+.4f} {state}"


def _rung_ceiling_table(effects: dict[str, Any], fit: set[str] | None) -> dict[str, Any]:
    """Where each embedder's overlap ladder stops paying - the split a mean over embedders hides.

    A mean marginal step summarises AGREEMENT. When some embedders have found their ceiling and
    others are still climbing, it reports a small gain describing neither, and a resolved count
    beside it does not rescue the reading: "2 of 6 separate" is equally compatible with
    weak-everywhere and with strong-in-two, which are opposite conclusions. This is one row per
    embedder, so the two cases look different.

    The decisive column is the LAST step: a ladder whose final step is still material has been cut
    short by the sweep rather than by the data, and its optimum is outside the measured range.
    """
    rows = []
    for (corpus, held, cap, embedding), pairs in sorted(_ladders_by_embedder(effects, fit).items()):
        levels = sorted({level for pair in pairs for level in pair})
        # Two levels are one comparison; a ladder needs three to have a consecutive step at all.
        if len(levels) < 3:
            continue
        steps = [(low, high, pairs[(low, high)]) for low, high in itertools.pairwise(levels) if (low, high) in pairs]
        if not steps:
            continue
        resolved = [step for step in steps if step[2]["resolved"]]
        material = [step for step in steps if step[2]["material"]]
        top_low, top_high, top = steps[-1]
        rows.append(
            [
                corpus,
                held,
                f"`{embedding}`",
                f"{len(resolved)}/{len(steps)} ({len(material)} material)",
                f"{material[-1][0]} to {material[-1][1]}" if material else "none",
                f"{top_low} to {top_high} ({top_high / cap:.0%} of cap)",
                _step_verdict(top),
            ]
        )
    return _table(
        "Where each embedder's overlap ladder stops paying",
        [
            "Corpus",
            "Held fixed",
            "Embedder",
            "Steps resolved (material)",
            "Highest material step",
            "Last step",
            "Last step delta",
        ],
        rows,
        "One row per EMBEDDER rather than per step, because a mean over embedders cannot "
        "distinguish a plateau from a split - embedders that have found their ceiling and "
        "embedders still climbing average to a small gain that describes neither. A step is "
        "material when it resolves AND clears the file's material floor; a resolved step under it "
        "is a difference the query set can see and no deployer would act on. A last step that is "
        "still material means the sweep stopped before the data did, so that embedder's optimum "
        "lies above the range measured here.",
    )


def _register_effect_tables(tables: dict[str, Any], effects: dict[str, Any], fit: set[str] | None) -> None:
    """The paired-comparison tables, one per axis. An axis with no rows is not registered.

    An empty table would read as "no effect" where the truth is "not measured", so an axis whose
    cells carry no per-query arrays is left out entirely rather than rendered blank.
    """
    tables["chunk_knob_summary"] = _knob_summary_table(effects, fit)
    tables["chunk_knob_summary_chunking"] = _knob_summary_table(effects, fit, axes=_CHUNK_AXES)
    rungs = _rung_table(effects, fit)
    if rungs["rows"]:
        tables["chunk_overlap_rungs"] = rungs
    ceilings = _rung_ceiling_table(effects, fit)
    if ceilings["rows"]:
        tables["chunk_overlap_ceiling"] = ceilings
    specs = [
        (
            "chunk_knob_overlap",
            "overlap_tokens",
            "Effect of chunk overlap",
            "Overlap is counted in gpt2 tokens and shown with its percent of the chunk cap, the unit "
            "industry guidance quotes. Three strategies honour it - recursive, markdown and fast - "
            "and they implement it differently; chonkie ignores it for semantic and late. Every "
            "verdict is document-level nDCG@10 (a document scores by its best chunk), while "
            "`semdex search` returns chunks.",
            None,
        ),
        (
            "chunk_knob_max_tokens",
            "max_tokens",
            "Effect of chunk size",
            "Larger chunks mean fewer, longer chunks per document.",
            None,
        ),
        (
            "chunk_knob_strategy",
            "strategy",
            "Effect of chunking strategy",
            'Recursive here runs chonkie\'s generic rules (recipe ""), not the markdown recipe semdex '
            "ships; semantic splits on sentence-similarity boundaries found with a breakpoint embedding "
            "model.",
            None,
        ),
        (
            "chunk_knob_breakpoint",
            "breakpoint_model",
            "Effect of the semantic breakpoint model",
            "The model that decides where semantic boundaries fall. chonkie's default is "
            "English-distilled, which matters for non-English corpora.",
            None,
        ),
        (
            "method_effects",
            "method",
            "Effect of adding lexical retrieval",
            "Each row bootstraps the per-query difference between two retrieval methods over the "
            "queries they share, on identical chunks and judgements.",
            None,
        ),
        (
            "rerank_effects",
            "method",
            "Effect of cross-encoder reranking",
            "Baseline and reranked run come from the SAME shortlist at the same depth, so each "
            "pair differs by exactly one thing: the reranking. The @20 suffix is that depth.",
            None,
        ),
        (
            "embedder_effects",
            "embedding",
            "Effect of the embedding model",
            "Two cells over the same corpus and the same chunk profile, differing only in the model. "
            "Compare the resolve rate here against the chunking axes above: the model choice is "
            "settled far more often than any chunking knob. Shown at the recommended chunk profile; "
            "the other profiles agree and are in the raw file.",
            "recursive cap256 ov0tok",
        ),
    ]
    base_methods = {"dense", "bm25", "hybrid"}
    rerank_levels = {f"{m}@20" for m in base_methods} | {f"{m}@20+rerank" for m in base_methods}
    levels_by_table = {"method_effects": base_methods, "rerank_effects": rerank_levels}
    for table_id, axis, title, note, profile in specs:
        table = _knob_table(effects, axis, title, note, fit=fit, profile=profile, levels=levels_by_table.get(table_id))
        if table["rows"]:
            tables[table_id] = table


def _register_ranking_tables(tables: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-corpus and per-model rankings; returns the docs that carry confidence intervals."""
    mldr = _load("chunk-sweep-mldr.json")
    if mldr:
        tables["embedder_ranking_mldr"] = _embedder_ranking_table(mldr, ["mldr_en_8k_slice", "mldr_de_3k_slice"])
        tables["ranking_mldr_en"] = _ranking_table(mldr, "mldr_en_8k_slice")
        tables["ranking_mldr_de"] = _ranking_table(mldr, "mldr_de_3k_slice")
    gerdalir = _load("chunk-sweep-gerdalir.json")
    if gerdalir:
        tables["ranking_gerdalir_de"] = _ranking_table(gerdalir, "gerdalir_de_12k_slice")
    multilingual = _load("embedding-multilingual-mldr.json")
    if multilingual:
        tables["embedder_ranking_multilingual"] = _embedder_ranking_table(
            multilingual, ["mldr_en_8k_slice", "mldr_de_3k_slice"]
        )
    hybrid = _load("hybrid-dense-bm25.json")
    if hybrid:
        tables["method_ranking"] = _method_ranking_table(hybrid)
    return [d for d in (mldr, _load("chunk-sweep-beir.json"), multilingual, hybrid) if d]


def _register_component_tables(tables: dict[str, Any]) -> None:
    baseline_path = _ROOT / "tests" / "benchmarks" / "baseline.json"
    if baseline_path.exists():
        tables.update(_extractor_tables(json.loads(baseline_path.read_text())))
    ocr = _load("ocr-badscans.json")
    if ocr:
        tables["ocr_engines"] = _ocr_table(ocr)
    synthetic = _load("dim-crossover.json")
    if synthetic:
        tables["store_dim_synthetic"] = _store_dim_synthetic_table(synthetic)
    store = _load("dim-crossover-real.json")
    if store:
        tables["store_dim_real"] = _store_table(store)
    quality = _load("store-quality.json")
    if quality:
        tables["store_quality"] = _store_quality_table(quality)


def _store_quality_table(doc: dict[str, Any]) -> dict[str, Any]:
    """What an ANN index costs in real retrieval quality, beside what it buys in latency."""
    rows = []
    for cell in sorted(doc["cells"], key=lambda c: (str(c["corpus"]), -(c.get("dim") or 0), str(c.get("store")))):
        exact = cell.get("exact_scan")
        rows.append(
            [
                str(cell["corpus"]).replace("_slice", ""),
                str(cell["embedding"]).split(":")[-1],
                str(cell.get("dim")),
                str(cell.get("store")),
                "exhaustive" if exact else str(cell.get("ann_params") or "ANN"),
                _interval(cell),
                "0 (control)" if exact else f"{cell.get('ndcg_delta_vs_exact'):+.4f}",
                _fmt(cell.get("doc_recall_at_k"), 3),
                _fmt(cell.get("search_p50_ms"), 1),
                _fmt(cell.get("store_mb"), 0),
            ]
        )
    return _table(
        "Retrieval quality measured through the store, with its latency",
        [
            "Corpus",
            "Model",
            "Dim",
            "Store",
            "Search / ANN setting",
            "nDCG@10 [95% CI]",
            "vs exact",
            "Doc recall@10",
            "p50 ms",
            "MB",
        ],
        rows,
        "One run per row: the same cached vectors and the same queries, loaded into a real store. "
        "The exhaustive stores are the control - their ranking is exact by construction, so their "
        "nDCG must match the kernel's, and an ANN row beside them is only meaningful when it "
        "does. `vs exact` is the quality actually lost to approximation, which is the number the "
        "recall column cannot give you: dropping the rank-1 document costs far more than dropping "
        "rank 9. The ANN rows name the setting they were measured at, which is what a stock "
        "install resolves - not the driver's own default, which is a different and worse thing.",
    )


def _span_axis_rows(
    effects: list[dict[str, Any]],
    axis: str,
    *,
    held: str | None = None,
    from_level: Any = None,
) -> list[list[str]]:
    """Paired span-integrity effects for one axis.

    ``from_level`` narrows an axis with many levels to the comparison a deployment actually
    faces. For overlap that is "off versus on": whether 10 beats 15 is in the raw file, and
    publishing all 222 pairwise combinations would bury the answer rather than support it.
    """
    rows = []
    for effect in effects:
        if effect["axis"] != axis:
            continue
        if from_level is not None and effect["from_level"] != from_level:
            continue
        fixed = ", ".join(f"{k} {v}" for k, v in sorted(effect["held_fixed"].items()))
        if held is not None and held not in fixed:
            continue
        rows.append(
            [
                effect["corpus"],
                fixed,
                _levels_text(effect, axis),
                f"{effect['mean_delta']:+.4f}",
                f"[{effect['ci_lo']:+.4f}, {effect['ci_hi']:+.4f}]",
                f"{effect['wins']}/{effect['losses']}",
                "resolved" if effect["resolved"] else "not resolved",
            ]
        )
    return rows


_SPAN_EFFECT_COLUMNS = [
    "Corpus",
    "Held fixed",
    "Change (low to high)",
    "Delta intact",
    "95% CI",
    "Win/loss",
    "Verdict",
]


def _register_span_tables(tables: dict[str, Any], doc: dict[str, Any]) -> None:
    """Span integrity: the direct measure of whether a boundary cuts an answer."""
    fit = [row for row in doc["integrity"] if row["fit_for_claim"]]
    effects = doc.get("axis_effects", [])

    tables["span_integrity_levels"] = _table(
        "Answers surviving inside one chunk, by chunk size and overlap",
        ["Corpus", "Strategy", "Max tokens", "Overlap", "Chunks/context", "Intact", "95% CI", "Split"],
        [
            [
                row["corpus"],
                row["strategy"],
                str(row["max_tokens"]),
                str(row["overlap_tokens"]),
                f"{row['chunks_per_context']:.2f}",
                f"{row['span_intact']:.4f}",
                f"[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]",
                f"{row['answers_split']}/{row['n_questions']}",
            ]
            for row in sorted(
                (r for r in fit if r["strategy"] == "recursive"),
                key=lambda r: (r["corpus"], r["max_tokens"], r["overlap_tokens"]),
            )
        ],
        "Fraction of answers lying wholly inside at least one chunk, measured on SQuAD-style "
        "character spans - no embedding and no retrieval, so this is the chunk boundary alone. "
        "Only corpora whose contexts produce at least two chunks are shown: a context that fits "
        "in one chunk has no boundary and scores 1.0 for a reason that has nothing to do with "
        "chunking. `recursive` only here; every strategy is in the raw file.",
    )

    tables["span_knob_max_tokens"] = _table(
        "Chunk size against answer integrity (paired)",
        _SPAN_EFFECT_COLUMNS,
        _span_axis_rows(effects, "max_tokens", held="overlap_tokens 0, strategy recursive"),
        "Paired per question, so question difficulty cancels. Positive means the LARGER chunk "
        "kept more answers intact. Zero overlap and `recursive` throughout, isolating size from "
        "the remedy and from the strategy. This axis is close to tautological at the limit - one "
        "chunk per document scores 1.0 - so it is a constraint to keep acceptable, never a "
        "quantity to maximise; the retrieval sweep wants the opposite direction.",
    )

    tables["span_knob_overlap"] = _table(
        "Overlap against answer integrity (paired)",
        _SPAN_EFFECT_COLUMNS,
        _span_axis_rows(effects, "overlap_tokens", held="strategy recursive", from_level=0),
        "Overlap off versus on, `recursive`, paired per question. Positive means overlap kept "
        "more answers intact. This is the measurement the end-to-end sweep could not make: "
        "overlap exists to stop a boundary cutting an answer, and document-level relevance "
        "cannot see that happen. The full level-by-level grid, including 10 against 15, is in "
        "`tests/benchmarks/raw/span-integrity.json`.",
    )

    tables["span_knob_strategy"] = _table(
        "Chunking strategy against answer integrity (paired)",
        _SPAN_EFFECT_COLUMNS,
        _span_axis_rows(effects, "strategy", held="max_tokens 256, overlap_tokens 0"),
        "At 256 tokens with no overlap. Positive favours the alphabetically later strategy named in the change column.",
    )

    retrieval = doc.get("retrieval") or []
    if retrieval:
        tables["span_blind_spot"] = _table(
            "What document-level scoring counts as a hit but does not deliver",
            [
                "Corpus",
                "Profile",
                "k",
                "Chunks indexed",
                "Doc hit",
                "Span hit",
                "Blind spot",
                "of which split",
                "of which ranking",
            ],
            [
                [
                    row["corpus"],
                    f"{row['strategy']} t{row['max_tokens']} o{row['overlap_tokens']}",
                    str(row["k"]),
                    f"{row['n_chunks_indexed']:,}",
                    f"{row['doc_hit']:.4f}",
                    f"{row['span_hit']:.4f}",
                    f"{row['blind_spot']:.4f}",
                    f"{row.get('blind_from_split', 0.0):.4f}",
                    f"{row.get('blind_from_ranking', 0.0):.4f}",
                ]
                for row in sorted(retrieval, key=lambda r: (r["corpus"], r["max_tokens"], r["overlap_tokens"]))
            ],
            "`Doc hit` is the verdict every other page here records: a chunk of the right document "
            "was retrieved. `Span hit` asks whether a retrieved chunk actually holds the whole "
            "answer. `Blind spot` is the difference - queries counted as successes that do not "
            "deliver a usable answer - split by cause: `split` means no chunk anywhere holds the "
            "answer whole, which is the boundary's doing; `ranking` means one does and it was not "
            "retrieved, which is not. Only the first is a chunking failure.",
        )


def _register_ann_frontier_tables(tables: dict[str, Any], doc: dict[str, Any]) -> None:
    """The recall-latency frontier per store, and what the shipped presets actually buy."""
    points = doc.get("points") or []
    if not points:
        return
    query_points = [p for p in points if p["build_level"] == "driver default"]
    build_points = [p for p in points if p["query_level"] == "driver default"]

    tables["ann_query_frontier"] = _table(
        "Recall-latency frontier: query-time knobs (no reindex)",
        ["Store", "Setting", "Recall vs exact", "nDCG@10", "p50 ms", "p95 ms"],
        [
            [
                point["store"],
                point["query_level"],
                f"{point['recall_vs_exact']:.3f}",
                f"{point['ndcg@10']:.4f}",
                f"{point['search_p50_ms']:.1f}",
                f"{point['search_p95_ms']:.1f}",
            ]
            for point in sorted(query_points, key=lambda p: (p["store"], p["search_p50_ms"]))
        ],
        "`Recall vs exact` is the share of the exact top-10 DOCUMENTS the store returned, scored "
        "against the same streamed kernel the rest of these pages use. These knobs change per "
        "search, so a deployment can move along this frontier without rebuilding anything. "
        "`driver default` is the setting every previously published store number was taken at.",
    )

    tables["ann_build_frontier"] = _table(
        "Recall-latency frontier: build-time knobs (each needs a reindex)",
        ["Store", "Index setting", "Recall vs exact", "nDCG@10", "Build s", "Store MB", "p50 ms"],
        [
            [
                point["store"],
                point["build_level"],
                f"{point['recall_vs_exact']:.3f}",
                f"{point['ndcg@10']:.4f}",
                f"{point['build_s']:.0f}",
                f"{point['store_mb']:.0f}",
                f"{point['search_p50_ms']:.1f}",
            ]
            for point in sorted(build_points, key=lambda p: (p["store"], p["build_level"]))
        ],
        "Measured at each driver's own query-time default, so the two axes never move together. "
        "These are baked into the index, so changing one costs a full rebuild - which is why the "
        "grid is short and why build time and size are reported here rather than on the query "
        "frontier.",
    )


def _extraction_real_rows(rows: list[dict[str, Any]], key: str) -> list[list[str]]:
    return [
        [
            row["extractor"],
            str(row[key]),
            str(row["pages"]),
            f"{row['word_recall']:.3f}",
            f"{row['edit_similarity']:.3f}",
            f"{row['latency_p50_ms']:.0f}",
            str(row["failed"]),
        ]
        for row in rows
    ]


_EXTRACTION_REAL_COLUMNS = ["Extractor", "", "Pages", "Word recall", "Edit similarity", "p50 ms", "Failed"]


def _register_extraction_real_tables(tables: dict[str, Any], doc: dict[str, Any]) -> None:
    """Extraction fidelity on real documents, not on fixtures written here."""
    control = doc.get("text_layer_control_chars") or {}
    control_note = ", ".join(f"{name} {chars}" for name, chars in sorted(control.items()))

    tables["extraction_real_overall"] = _table(
        "Extraction fidelity on OmniDocBench",
        ["Extractor", "Pages", "Word recall", "Edit similarity", "p50 ms", "Failed"],
        [
            [
                row["extractor"],
                str(row["pages"]),
                f"{row['word_recall']:.3f}",
                f"{row['edit_similarity']:.3f}",
                f"{row['latency_p50_ms']:.0f}",
                str(row["failed"]),
            ]
            for row in doc.get("overall", [])
        ],
        "Real document pages with human annotations, sampled across ten document types and three "
        "languages, each wrapped identically as a single-page PDF. `Word recall` is the share of "
        "annotated text the extraction contains, counted as a token multiset and so indifferent "
        "to bullets, punctuation and reading order; `Edit similarity` compares whole pages and "
        "therefore does punish dropped and reordered content. A zero is a converter with no OCR, "
        "not a broken harness: the same code path on a text-layer PDF returns "
        f"{control_note} characters respectively.",
    )

    tables["extraction_real_by_source"] = _table(
        "Extraction fidelity by document type",
        [c or "Document type" for c in _EXTRACTION_REAL_COLUMNS],
        _extraction_real_rows(doc.get("by_source", []), "source"),
        "The axis the self-authored fixture grid cannot have: how each converter behaves on a "
        "newspaper against a slide deck against a handwritten note.",
    )

    tables["extraction_real_by_language"] = _table(
        "Extraction fidelity by language",
        [c or "Language" for c in _EXTRACTION_REAL_COLUMNS],
        _extraction_real_rows(doc.get("by_language", []), "language"),
        "OCR ran with the tesseract pack matching each page's annotated language, so a low score "
        "here is the converter rather than a missing language pack.",
    )


def _rate(value: float) -> str:
    """Thousands separators where the digits matter, a decimal where rounding would erase it.

    `late` runs at 2.9 docs/s, and printing that as "3" both loses the measurement and hides that
    it sits below one document per third of a second.
    """
    return f"{value:,.0f}" if value >= 100 else f"{value:.1f}"


def _humanised_duration(seconds: float) -> str:
    """A wall-clock figure a reader can weigh, because docs/s over five orders of magnitude cannot.

    The gap between 100,000 docs/s and 3 docs/s is a column of digits until it is expressed as
    ten seconds against four days, which is the form the deployment decision is actually made in.
    """
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def _throughput_shape(doc: ThroughputFile) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Split the raw file into its rows and its run metadata, whichever vintage it is.

    The file was a bare list of rows before the benchmark gained repeats, load capture and a
    worker sweep. Declaring that union here and narrowing it once keeps every caller working with
    real types instead of propagating the ambiguity: a `dict[str, Any]` narrowed by
    `isinstance(x, list)` gives an element type of `str`, which is how one honest union produced
    nineteen type errors downstream.
    """
    if isinstance(doc, list):
        return doc, {}
    return doc["rows"], doc


def _register_throughput_tables(tables: dict[str, Any], doc: ThroughputFile) -> None:
    """Chunking rate per strategy, and whether a second worker buys anything."""
    rows, meta = _throughput_shape(doc)
    single = [r for r in rows if r.get("threads", 1) == 1]
    corpora = sorted({r["corpus"] for r in single})
    rate_of = {(r["corpus"], r["strategy"]): r for r in single}
    strategies = sorted(
        {r["strategy"] for r in single},
        key=lambda st: -min(rate_of[(c, st)]["docs_per_s"] for c in corpora),
    )
    load: dict[str, Any] = meta.get("load_before") or {}
    load_note = (
        f" Taken at load {load['load_1min']} on {load['cores']} cores, with the native thread pools "
        f"capped at {meta.get('native_threads', 'default')}."
        if load
        else ""
    )

    tables["chunker_throughput"] = _table(
        "Chunking throughput, one worker",
        ["Strategy", *[f"{c.split(' (')[0]} docs/s" for c in corpora], "Spread", "1M docs"],
        [
            [
                strategy,
                *[_rate(rate_of[(c, strategy)]["docs_per_s"]) for c in corpora],
                f"+/-{max(rate_of[(c, strategy)].get('spread_pct', 0.0) for c in corpora):.0f}%",
                # The SLOWER of the two corpora, so the projection is never the optimistic one.
                _humanised_duration(1_000_000 / min(rate_of[(c, strategy)]["docs_per_s"] for c in corpora)),
            ]
            for strategy in strategies
        ],
        "Chunking only: no embedding provider, no store, no search. `semantic` and `late` do embed "
        "internally because that IS their algorithm, so their rate honestly includes it. Each figure "
        "is the median of repeated samples, batched to a floor duration so the fast strategies are "
        "not timed over the clock's own resolution; `Spread` is the worst gap between the extremes "
        "across the two corpora and is what says whether the box was quiet. `1M docs` projects the "
        "SLOWER of the two corpora, so it is never the flattering figure." + load_note,
    )

    swept = sorted({r["threads"] for r in rows if r.get("threads", 1) > 1})
    if not swept:
        return
    counts = [1, *swept]
    by = {(r["strategy"], r["threads"]): r["docs_per_s"] for r in rows if r["corpus"] == corpora[0]}
    tables["chunker_workers"] = _table(
        "Throughput against worker threads, relative to one worker",
        ["Strategy", *[f"{n} worker{'s' if n > 1 else ''}" for n in counts]],
        [
            [strategy, *[f"{by[(strategy, n)] / by[(strategy, 1)]:.2f}x" for n in counts]]
            for strategy in strategies
            if all((strategy, n) in by for n in counts)
        ],
        "Each worker holds its own chunker, which is both what a server does and what a "
        "HuggingFace tokenizer requires. `semantic` and `late` are not swept: their cost is inside "
        "an embedding model whose own thread pool is capped here, so the sweep would measure that "
        f"pool rather than the chunker. Measured on {corpora[0]}.",
    )


_ARM_LABEL = {
    "plain": "no prefix (as published)",
    "query": "instruction on the query",
    "both": "instruction on query + passage",
}


def _register_qwen3_instruction_tables(tables: dict[str, Any], docs: list[dict[str, Any]]) -> None:
    """What the Qwen3 instruction prefix is worth, per model and as a paired verdict."""
    arm_rows: list[list[str]] = []
    paired_rows: list[list[str]] = []
    for doc in docs:
        model = doc["model"]
        for arm in ("plain", "query", "both"):
            cell = doc["arms"][arm]
            arm_rows.append(
                [
                    model,
                    _ARM_LABEL[arm],
                    f"{cell['ndcg@10']:.4f} [{cell['ndcg@10_ci_lo']:.3f}, {cell['ndcg@10_ci_hi']:.3f}]",
                    f"{cell['recall@10']:.4f}",
                    f"{cell['mrr']:.4f}",
                ]
            )
        for key, verdict in doc["paired"].items():
            comparison, metric = key.split("/")
            if metric != "ndcg@10":
                continue
            left, right = comparison.split("_vs_")
            paired_rows.append(
                [
                    model,
                    f"{_ARM_LABEL[left]} vs {_ARM_LABEL[right]}",
                    f"{verdict['mean_delta']:+.4f}",
                    f"[{verdict['ci_lo']:+.4f}, {verdict['ci_hi']:+.4f}]",
                    f"{verdict['wins']}/{verdict['losses']}",
                    "resolved" if verdict["resolved"] else "not resolved",
                ]
            )
    first = docs[0]
    tables["qwen3_instruction"] = _table(
        "Qwen3-Embedding with and without its instruction prefix",
        ["Model", "Configuration", "nDCG@10 [95% CI]", "Recall@10", "MRR"],
        arm_rows,
        f"`{first['corpus']}`, {first['docs']:,} documents, {first['arms']['plain']['queries']} queries. "
        f"The prefix is the model card's own: `Instruct: <task>\\nQuery:` with the task "
        f'"{first["task_description"]}". Passages are identical between the first two rows of '
        "each model, because a query prefix does not touch them.",
    )
    tables["qwen3_instruction_paired"] = _table(
        "The same queries, differenced pairwise",
        ["Model", "Comparison", "Delta nDCG@10", "95% CI", "Better/worse", "Verdict"],
        paired_rows,
        "Differences per query rather than between two means, because the arms differ by far less "
        "than the query-to-query spread. `Better/worse` counts queries; the remainder tied.",
    )


_CONTAINER_GB = 8


# Growth smaller than this is measurement noise, not a trend, and dividing a container by it
# manufactures a precise-looking capacity from nothing: an early version of this turned 0.7 MB of
# drift across 99,000 rows into "about 1,152,205,428 chunks".
_NOISE_FLOOR_MB = 5.0
_NOISE_FLOOR_SHARE = 0.10


def _rows_per_gb(rows: list[dict[str, Any]]) -> dict[str, str]:
    """How many chunks each store can serve inside a fixed container.

    Extrapolated LINEARLY from the measured span, which is honest only for a store whose footprint
    actually grows with the corpus. Two cases must not get a number instead of a caveat:

    * a FLAT store, where the growth is under the noise floor - its limit is disk, not memory;
    * a NON-MONOTONIC one, where a threshold inside the backend changes the regime partway up the
      ladder. lancedb builds its ANN index at 100,000 rows and gets both lighter and faster
      crossing it, so a line drawn through that discontinuity describes neither side of it.
    """
    verdicts: dict[str, str] = {}
    for backend in sorted({r["backend"] for r in rows}):
        cells = sorted((r for r in rows if r["backend"] == backend), key=lambda r: r["rows"])
        if len(cells) < 2:
            continue
        footprints = [c["resident_mb"] for c in cells]
        first, last = cells[0], cells[-1]
        grew = last["resident_mb"] - first["resident_mb"]
        floor = max(_NOISE_FLOOR_MB, first["resident_mb"] * _NOISE_FLOOR_SHARE)
        if any(b < a - floor for a, b in itertools.pairwise(footprints)):
            verdicts[backend] = f"not extrapolated: footprint falls between {first['rows']:,} and {last['rows']:,}"
            continue
        if grew < floor:
            verdicts[backend] = f"flat to {last['rows']:,} chunks; bounded by disk, not memory"
            continue
        per_row_kb = (grew / (last["rows"] - first["rows"])) * 1024
        headroom_mb = _CONTAINER_GB * 1024 - last["resident_mb"]
        verdicts[backend] = f"about {last['rows'] + int(headroom_mb * 1024 / per_row_kb):,} chunks"
    return verdicts


def _register_memory_tables(tables: dict[str, Any], doc: dict[str, Any]) -> None:
    """Resident memory per store and per embedding provider."""
    stores = doc.get("stores") or []
    capacity = _rows_per_gb(stores)
    tables["store_memory"] = _table(
        "Vector store memory, net of the interpreter",
        ["Backend", "Rows", "Serving RSS", "Ingest peak", "Query p50"],
        [
            [
                row["backend"],
                f"{row['rows']:,}",
                f"{row['resident_mb']:,.0f} MB",
                f"{row['ingest_peak_mb']:,.0f} MB",
                f"{row['query_ms']:,.2f} ms",
            ]
            for row in sorted(stores, key=lambda r: (r["backend"], r["rows"]))
        ],
        f"`Serving RSS` is what a process holds while answering queries, measured in a FRESH "
        f"process that opened the store from disk and never built it - the fixture needed to load "
        f"a store is larger than the store, so measuring both in one process measures mostly the "
        f"harness. `Ingest peak` is the high-water mark of the process that built it, harness "
        f"included, and is the number that decides whether a machine can create an index at all. "
        f"Both are net of a {doc.get('interpreter_baseline_mb', 0)} MB interpreter baseline, "
        f"measured the same way. Vectors are {doc.get('dim')}-dimensional.",
    )
    tables["store_memory_capacity"] = _table(
        f"What fits in a {_CONTAINER_GB} GB container",
        ["Backend", "Chunks servable"],
        [[backend, verdict] for backend, verdict in sorted(capacity.items())],
        "Extrapolated linearly from the measured span above, so it is an order-of-magnitude "
        "answer rather than a guarantee. A store whose footprint does not grow with the corpus "
        "gets no number: its limit is disk, which is measured elsewhere.",
    )
    embedders = doc.get("embedders") or []
    tables["embedder_memory"] = _table(
        "Embedding provider memory",
        ["Provider", "Model", "Dim", "Resident", "Peak while loading"],
        [
            [
                row["provider"],
                f"`{row['model_id']}`",
                str(row["dim"]),
                f"{row['resident_mb']:,.0f} MB",
                f"{row['peak_mb']:,.0f} MB",
            ]
            for row in sorted(embedders, key=lambda r: r["resident_mb"])
        ],
        "One process per provider, holding the model and one batch of 64 passages. `ollama`, "
        "`openai`, `gemini` and `cohere` are absent by nature rather than by omission: they hold "
        "no model in this process at all, which is the entire reason to choose one of them on a "
        "small machine. Their memory is their server's.",
    )


def _register_energy_tables(tables: dict[str, Any], cpu: dict[str, Any], gpu: dict[str, Any] | None) -> None:
    """Energy per document, and what it costs against the hosted APIs."""
    rows = list(cpu.get("rows") or [])
    where = {r["model_id"]: f"CPU ({cpu.get('cpu', 'unknown')})" for r in rows}
    if gpu:
        rows += list(gpu.get("rows") or [])
        where.update({r["model_id"]: f"GPU ({gpu.get('gpu', 'unknown')})" for r in (gpu.get("rows") or [])})
    tariff = cpu.get("tariff_eur_per_kwh", 0.0)

    tables["energy_per_document"] = _table(
        "Energy to embed one document",
        ["Provider", "Model", "Runs on", "J/doc", "Docs/s", "EUR per 1M docs", "Signal/noise"],
        [
            [
                row["provider"],
                f"`{row['model_id']}`",
                where.get(row["model_id"], "?").split(" (")[0],
                f"{row['marginal_j_per_doc']:.4f}" if row["resolved"] else "below noise floor",
                f"{row['docs_per_s']:,.1f}",
                f"{row['eur_per_million_docs_marginal']:.4f}" if row["resolved"] else "-",
                f"{row['signal_to_noise']:.1f}" if row["resolved"] else f"{row['signal_to_noise']:.1f} (unresolved)",
            ]
            for row in sorted(rows, key=lambda r: r["marginal_j_per_doc"])
        ],
        "MARGINAL energy: what the work adds over the same machine sitting idle, with idle windows "
        "INTERLEAVED between the work windows rather than measured once beforehand. CPU figures "
        "come from the package RAPL counter and GPU figures from integrating card power, so "
        "neither includes the rest of the machine. `Signal/noise` is that marginal draw against "
        f"the idle swing it had to be pulled out of; below 2.0 the row is arithmetic rather than a "
        f"measurement and reports no figure. Electricity at {tariff} EUR/kWh. Denominated in "
        "DOCUMENTS, never tokens - see the note below the next table.",
    )


def _register_cloud_cost_tables(tables: dict[str, Any], prices: dict[str, Any], cpu: dict[str, Any]) -> None:
    """Hosted API list price for the same million documents."""
    per_tokenizer = prices["tokens_per_document"]["by_tokenizer"]
    low_name = min(per_tokenizer, key=lambda k: per_tokenizer[k])
    high_name = max(per_tokenizer, key=lambda k: per_tokenizer[k])
    low, high = per_tokenizer[low_name], per_tokenizer[high_name]
    rows: list[list[str]] = []
    for entry in prices["providers"]:
        price = entry["usd_per_million_tokens"]
        if price is None:
            rows.append([entry["provider"], f"`{entry['model']}`", "not published", "-"])
            continue
        rows.append(
            [
                entry["provider"],
                f"`{entry['model']}`",
                f"{price:.2f}",
                f"{price * low:,.2f} to {price * high:,.2f}",
            ]
        )
    tables["cloud_cost_per_million"] = _table(
        "Hosted API list price for the same million documents",
        ["Provider", "Model", "USD per 1M tokens", "USD per 1M documents"],
        rows,
        f"List prices checked {prices['checked_utc']}, each recorded with its source in the raw "
        f"file. The document column is a RANGE because it needs a tokenizer and the answer depends "
        f"on which: `{low_name}` counts {low:.0f} tokens for the mean document on this corpus and "
        f"`{high_name}` counts {high:.0f}, a {(high - low) / low * 100:.0f} percent spread, and a "
        "vendor bills on its own tokenizer which is neither of these. That spread is why the "
        "energy table above is denominated in documents rather than tokens. Local figures are in "
        "EUR and these in USD; the gap between them is large enough that no exchange rate flips it.",
    )


# Below this many swept cells a rank correlation is unstable, so the corpus is shown but its
# view count is flagged rather than read as a result.
_MIN_CELLS_FOR_CORRELATION = 20


def _register_metric_redundancy_tables(tables: dict[str, Any], doc: dict[str, Any]) -> None:
    """How many independent views the four reported metrics actually carry, per corpus."""
    corpora = sorted(doc.get("corpora") or [], key=lambda c: c["effective_views"])
    tables["metric_redundancy"] = _table(
        "Four metrics, how many views",
        ["Corpus", "Cells", "Relevant docs/query", "Effective views", "Near-duplicate pairs"],
        [
            [
                row["corpus"],
                str(row["cells"]),
                f"{row['judged_per_query']:.2f}" if row.get("judged_per_query") else "n/a",
                f"{row['effective_views']:.2f}" + ("" if row["cells"] >= _MIN_CELLS_FOR_CORRELATION else " (thin)"),
                f"{row['near_duplicate_pairs']}/6",
            ]
            for row in corpora
        ],
        "`Effective views` is the participation ratio of the four metrics' rank-correlation matrix "
        "across the swept configurations: 4.00 would mean four independent perspectives and 1.00 "
        "means one. A pair counts as near-duplicate at |rho| >= "
        f"{doc.get('near_duplicate_rho')}. Marked `(thin)` below {_MIN_CELLS_FOR_CORRELATION} "
        "cells, where a rank correlation is not stable enough to read.",
    )

    # Range across corpora rather than a row per corpus per pair: 48 rows would bury the one thing
    # worth seeing, which is that two pairs never separate and the other four sometimes do.
    spans: dict[tuple[str, str], list[tuple[float, str]]] = {}
    for row in corpora:
        if row["cells"] < _MIN_CELLS_FOR_CORRELATION:
            continue
        for pair in row["pairs"]:
            spans.setdefault((pair["left"], pair["right"]), []).append((pair["disagreement_rate"], row["corpus"]))
    tables["metric_disagreement"] = _table(
        "How often two metrics pick a different winner",
        ["Metric pair", "Lowest", "Highest", "Where it disagrees most"],
        [
            [
                f"{left} vs {right}",
                f"{min(values)[0] * 100:.1f}%",
                f"{max(values)[0] * 100:.1f}%",
                max(values)[1],
            ]
            for (left, right), values in sorted(spans.items(), key=lambda kv: max(kv[1])[0])
        ],
        "Over every pair of swept configurations within a corpus, the share where the two metrics "
        "rank them oppositely. Pairs tied under either metric are excluded rather than counted as "
        "agreement, since a tie is the metric declining to choose. Thin corpora are left out.",
    )


def _register_query_power_tables(tables: dict[str, Any], doc: dict[str, Any]) -> None:
    """What it would take to settle the comparisons that came out as ties."""
    rows = sorted(doc.get("corpora") or [], key=lambda r: r["median_required_queries"] or 0)
    ladder = doc.get("ladder") or []
    tables["query_power"] = _table(
        "What the ties would cost to settle",
        ["Corpus", "Queries", "Ties", "Median effect", "Median half-width", "Queries needed", "Never"],
        [
            [
                row["corpus"],
                f"{row['queries']:,}",
                f"{row['unresolved']}/{row['comparisons']}",
                f"{row['median_effect']:.4f}",
                f"{row['median_half_width']:.4f}",
                f"{row['median_required_queries']:,}" if row["median_required_queries"] else "-",
                str(row["never_resolvable"]),
            ]
            for row in rows
        ],
        "`Queries needed` is the median over that corpus's unresolved comparisons of "
        "`n x (half-width / effect)^2`, the count at which the interval would clear the observed "
        "effect. `Never` counts comparisons whose effect is indistinguishable from zero: those are "
        "not underpowered, they are answered, and the answer is that the two configurations are "
        "the same. Both assume the OBSERVED effect is the true one, which for a comparison that "
        "came out a tie is optimistic, so read the counts as a floor.",
    )
    if not ladder:
        return
    tables["query_power_ladder"] = _table(
        "How many ties each query count would settle",
        ["Corpus", "Ties", *[f"{target:,}" for target in ladder]],
        [
            [
                row["corpus"],
                str(row["unresolved"]),
                *[str(row["resolved_at"][str(target)]) for target in ladder],
            ]
            for row in rows
        ],
        "Cumulative: each column counts the corpus's currently unresolved comparisons that would "
        "separate at that many queries. A row that barely moves across the whole ladder is a "
        "corpus whose ties are near-zero effects rather than a corpus starved of data.",
    )


# Each entry is (raw file, registrar). The registrar is called with the loaded document only when
# the file exists, so a missing raw file skips its tables rather than failing the whole render.
# A table is one line here; the alternative was an if-block per file, which grew past the branch
# limit and made every addition edit the same function body.
def _register_query_power_validation(tables: dict[str, Any], doc: dict[str, Any]) -> None:
    """Whether the queries-needed prediction survives a holdout, and by how much it is off."""
    rounds = doc.get("rounds") or []
    if not rounds:
        return
    ratios = sorted(r["half_width_ratio"] for r in rounds)
    predicted = [r["predicted_resolve"] for r in rounds]
    actual = [r["actually_resolved"] for r in rounds]
    precision = [r["precision"] for r in rounds if r["precision"] is not None]
    recall = [r["recall"] for r in rounds if r["recall"] is not None]
    overshoot = (math.fsum(predicted) / len(predicted)) / (math.fsum(actual) / len(actual)) if sum(actual) else 0.0
    tables["query_power_validation"] = _table(
        "Does the prediction hold up on a holdout",
        ["Check", "Expected", "Measured", "Reading"],
        [
            [
                "Half-width scaling",
                f"{doc['expected_half_width_ratio']:.2f}x",
                f"{_median_of(ratios):.2f}x ({ratios[0]:.2f} to {ratios[-1]:.2f})",
                "the 1/sqrt(n) mechanism holds",
            ],
            [
                "Ties predicted to resolve",
                "matches actual",
                f"{math.fsum(predicted) / len(predicted):.0f} predicted, {math.fsum(actual) / len(actual):.0f} actual",
                f"optimistic by {overshoot:.2f}x",
            ],
            [
                "Which ties resolve",
                "1.00 precision",
                f"{math.fsum(precision) / len(precision):.2f} precision, {math.fsum(recall) / len(recall):.2f} recall",
                "barely better than chance per comparison",
            ],
        ],
        f"The English slice runs the same comparisons at {doc['full_queries']} queries. Subsampling "
        f"it to {doc['subsample_queries']} - the size of the German set - predicting from that, and "
        f"scoring against the full set the prediction never saw, over {len(rounds)} subsamples. The "
        "count is inflated because the effect estimated from a small sample is noisy, and the "
        "comparisons that look nearly resolvable are the ones most likely to be overstated.",
    )


def _median_of(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def _verdict_cell(paired: dict[str, Any]) -> str:
    return f"{paired['mean_delta']:+.4f} {'resolved' if paired['resolved'] else 'unresolved'}"


def _register_query_length_tables(tables: dict[str, Any], doc: dict[str, Any]) -> None:
    """The overlap effect read against QUERY LENGTH: binned on the scored cells, then re-scored on cut queries."""
    binned = doc.get("by_query_length") or []
    if binned:
        bin_names = [b["bin"] for b in binned[0]["bins"]]
        ranges = sorted(
            {
                f"{row['corpus']}: "
                + ", ".join(
                    f"{b['bin']} {b['words_lo']}-{b['words_hi']} words" for b in row["bins"] if b["bin"].startswith("Q")
                )
                for row in binned
            }
        )
        tables["chunk_overlap_by_query_length"] = _table(
            "Overlap against query length (paired, per bin)",
            ["Corpus", "Embedder", "Overlap (low to high)", "All queries", *bin_names],
            [
                [
                    row["corpus"],
                    f"`{row['embedding']}`",
                    f"{row['overlap_low']} to {row['overlap_high']}",
                    _verdict_cell(row["all"]),
                    *(_verdict_cell(b) for b in row["bins"]),
                ]
                for row in binned
            ],
            "The SAME two cells as the overlap table, their per-query nDCG@10 deltas split by the query's own "
            "word count into quantile bins (Q1 shortest), plus one bin holding only the queries no longer than "
            "the ceiling named in its header. Each bin gets the page's paired bootstrap, so `resolved` means "
            "what it means everywhere else. Bins in words: " + "; ".join(ranges) + ".",
        )
    causal = doc.get("truncated_queries") or []
    if causal:
        words = causal[0]["query_words"]
        tables["chunk_overlap_truncated_queries"] = _table(
            "Overlap with every query cut to its first words (paired)",
            ["Corpus", "Embedder", "Overlap (low to high)", "Full queries", f"First {words} words"],
            [
                [
                    row["corpus"],
                    f"`{row['embedding']}`",
                    f"{row['overlap_low']} to {row['overlap_high']}",
                    _verdict_cell(row["full_queries"]),
                    _verdict_cell(row["truncated_queries"]),
                ]
                for row in causal
            ],
            "The same cells re-scored with each query truncated to its first words: the corpus vectors are "
            "reused and only the query vectors change, so nothing but query length differs between the two "
            "columns. A gain that goes with the words was the words' doing.",
        )


def _pair_cell(marked: int | None, unmarked: int | None, *, thousands: bool) -> str:
    """Two counts side by side, or n/a when the audit had neither side."""
    if marked is None or unmarked is None:
        return "n/a"
    return f"{marked:,} / {unmarked:,}" if thousands else f"{marked} / {unmarked}"


def _share_cell(share: float | None) -> str:
    """A fraction as a percentage with one decimal, or n/a when it was not measured."""
    return "n/a" if share is None else f"{100 * share:.1f}%"


def _register_markdown_structure_tables(tables: dict[str, Any], doc: dict[str, Any]) -> None:
    """Marked against unmarked MLDR English: what heading markup buys each strategy, per embedder."""
    effects = doc.get("effects") or []
    if not effects:
        return
    missing = doc.get("missing") or []
    absent = "; ".join(f"{m['strategy']} / {m['embedding']}" for m in missing)
    tables["markdown_structure_effect"] = _table(
        "Heading markup on MLDR English: marked against unmarked, paired per query",
        [
            "Strategy",
            "Embedder",
            "Marked minus unmarked",
            "95% CI",
            "Wins/losses",
            "Chunks (marked / unmarked)",
            "Median tokens (marked / unmarked)",
            "Chunks starting at a heading",
        ],
        [
            [
                f"`{row['strategy']}`",
                f"`{row['embedding']}`",
                _verdict_cell(row),
                f"[{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]",
                f"{row['wins']}/{row['losses']}",
                _pair_cell(row["marked_rows"], row["unmarked_rows"], thousands=True),
                _pair_cell(row["marked_token_p50"], row["unmarked_token_p50"], thousands=False),
                _share_cell(row["marked_heading_start_share"]),
            ]
            for row in effects
        ],
        "The marked corpus is the MLDR English slice with every bare-line section heading prefixed "
        "`## `; the unmarked corpus is the slice as published: identical documents, queries and qrels, "
        "so the per-query difference isolates the markup. Positive favours the marked corpus. "
        "`recursive`, `fast` and `semantic` are controls: `recursive` ran chonkie's generic rules "
        '(recipe ""), which never cut at a heading, `fast` shares `markdown`\'s splitter family without '
        "reading structure, and `semantic` ignores structure entirely."
        + (f" Pairs with an unscored twin: {absent}." if absent else ""),
    )


_PRODUCT_K_LADDER_CAPS = (64, 128, 256, 512)


def _product_k_corpus(corpus: str) -> str:
    """Body and language, the chart script's convention, so a table row and a chart label agree."""
    parts = corpus.split("_")
    return "-".join(parts[:2]) if len(parts) > 1 else corpus


def _product_k_strategy(axes: dict[str, Any]) -> str:
    strategy = f"`{axes.get('strategy')}`"
    breakpoint_model = axes.get("breakpoint_model")
    return f"{strategy} bp {breakpoint_model}" if breakpoint_model else strategy


def _register_product_k_verdict(tables: dict[str, Any], doc: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    k = int(doc["product_k"])
    out: list[list[str]] = []
    for row in rows:
        axes = row["axes"]
        if not row.get("product") or axes.get("max_tokens") != 256 or axes.get("overlap_tokens") != 0:
            continue
        out.append(
            [
                _product_k_corpus(row["corpus"]),
                _product_k_strategy(axes),
                f"`{row['embedding']}`",
                f"{row['ndcg_documents']:.4f}",
                f"{row['ndcg_delivered']:.4f}",
                _verdict_cell({"mean_delta": row["gap_mean_delta"], "resolved": row["gap_resolved"]}),
                f"[{row['gap_ci_lo']:+.4f}, {row['gap_ci_hi']:+.4f}]",
                f"{row['distinct_docs']:.2f}",
            ]
        )
    if not out:
        return
    out.sort(key=lambda r: (r[0], r[1], r[2]))
    tables["product_k_verdict"] = _table(
        f"The product's {k} slots: delivered chunks against deduplicated documents, paired per query",
        [
            "Corpus",
            "Strategy",
            "Embedder",
            f"Documents nDCG@{k}",
            f"Delivered nDCG@{k}",
            "Cost of no dedup",
            "95% CI",
            f"Distinct docs in {k}",
        ],
        out,
        f"Both views come from the same top list per query at cap 256, zero overlap. `Documents` "
        f"deduplicates to {k} distinct documents, as every other table here does; `Delivered` scores "
        f"the {k} chunks `semdex search` returns, where a repeated document earns zero gain. The cost "
        "is documents minus delivered; positive means the product's no-dedup loses that much nDCG. "
        "`Distinct docs` is the mean number of different documents in the delivered slots.",
    )


def _register_product_k_budget(tables: dict[str, Any], doc: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    budgets = [int(b) for b in doc["budgets"]]
    ladder = [r for r in rows if r["axes"].get("strategy") == "recursive" and r["axes"].get("overlap_tokens") == 0]
    ladder = [r for r in ladder if int(r["axes"]["max_tokens"]) in _PRODUCT_K_LADDER_CAPS]
    if not ladder:
        return
    by_cell: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in ladder:
        key = (_product_k_corpus(row["corpus"]), str(row["embedding"]), int(row["axes"]["max_tokens"]))
        by_cell.setdefault(key, []).append(row)
    out: list[list[str]] = []
    present: dict[tuple[str, str], set[int]] = {}
    for (label, embedding, cap), rungs in sorted(by_cell.items()):
        present.setdefault((label, embedding), set()).add(cap)
        line = [label, f"`{embedding}`", str(cap)]
        for budget in budgets:
            rung = next((r for r in rungs if budget in r.get("budgets", [])), None)
            line.extend([str(rung["k"]), f"{rung['ndcg_delivered']:.4f}"] if rung else ["n/a", "n/a"])
        out.append(line)
    missing = sorted(
        {
            f"{label}: {', '.join(str(c) for c in _PRODUCT_K_LADDER_CAPS if c not in caps)}"
            for (label, _), caps in present.items()
            if any(c not in caps for c in _PRODUCT_K_LADDER_CAPS)
        }
    )
    columns = ["Corpus", "Embedder", "Cap"]
    for budget in budgets:
        columns.extend([f"k at {budget}", f"Delivered nDCG@k, {budget}"])
    tables["product_k_budget"] = _table(
        "Chunk size at a fixed number of tokens read: delivered nDCG@k with k = budget / cap",
        columns,
        out,
        "`recursive`, zero overlap. A consumer reading k chunks of cap tokens reads k times cap "
        "tokens whatever the cap, so each column holds the cap's k at that budget and the delivered "
        "score there. " + (f"Not measured (no cell at that cap): {'; '.join(missing)}." if missing else ""),
    )


def _register_product_k_tables(tables: dict[str, Any], doc: dict[str, Any]) -> None:
    """The product's delivered chunk list against deduplicated documents, and chunk size at a fixed budget."""
    rows = doc.get("rows") or []
    if not rows:
        return
    _register_product_k_verdict(tables, doc, rows)
    _register_product_k_budget(tables, doc, rows)


def _register_voided_cells(tables: dict[str, Any]) -> None:
    """Every cell an export dropped on an audit's judgement, with the reason, so a reader can see what is absent."""
    rows: list[list[str]] = []
    for name in (
        "chunk-sweep-mldr.json",
        "chunk-sweep-gerdalir.json",
        "chunk-sweep-beir.json",
        "chunk-sweep-miracl.json",
    ):
        doc = _load(name)
        rows.extend([f"`{void['cell']}`", void["reason"]] for void in (doc or {}).get("voided", []))
    if rows:
        tables["voided_cells"] = _table(
            "Cells withheld from every table above",
            ["Cell", "Why"],
            rows,
            "Dropped by the export on an audit's judgement, never by hand: a recursive overlap set whose chunk "
            "count moved against its overlap-0 sibling was re-cut by a size-guard and carries crumb chunks; a "
            "cell whose embedder clipped more than five percent of the token mass was scored on text the "
            "embedder never saw. The judgements are in `chunk-dimension-audit.json` and "
            "`embedder-truncation-audit.json`.",
        )


_Registrar = Callable[[dict[str, Any], Any], None]

_SIMPLE_SOURCES: tuple[tuple[str, _Registrar], ...] = (
    ("span-integrity.json", _register_span_tables),
    ("query-length-effect.json", _register_query_length_tables),
    ("markdown-structure-effect.json", _register_markdown_structure_tables),
    ("product-k.json", _register_product_k_tables),
    ("ann-frontier.json", _register_ann_frontier_tables),
    ("extraction-omnidocbench.json", _register_extraction_real_tables),
    ("chunker-throughput.json", _register_throughput_tables),
    ("memory.json", _register_memory_tables),
    ("metric-redundancy.json", _register_metric_redundancy_tables),
    ("query-power.json", _register_query_power_tables),
    ("query-power-validation.json", _register_query_power_validation),
)


def collect_table_data() -> dict[str, dict[str, Any]]:
    """Every registered table, built purely from committed raw data."""
    tables: dict[str, dict[str, Any]] = {}
    audit = _load("chunk-dimension-audit.json")
    if audit:
        _register_chunk_tables(tables, audit)
    effects = _load("chunk-knob-effects.json")
    if effects:
        _warn_unjudged(unjudged_corpora(effects, audit))
        _register_effect_tables(tables, effects, _fit_corpora(audit))
    with_intervals = _register_ranking_tables(tables)
    if with_intervals:
        tables["ci_precision"] = _precision_table(with_intervals)
    _register_component_tables(tables)
    for name, register in _SIMPLE_SOURCES:
        document = _load(name)
        if document:
            register(tables, document)
    _register_multi_source_tables(tables)
    return tables


def _register_multi_source_tables(tables: dict[str, Any]) -> None:
    """The tables that need more than one raw file, so they cannot sit in _SIMPLE_SOURCES."""
    _register_voided_cells(tables)
    instruction = [d for d in (_load("qwen3-instruction.json"), _load("qwen3-instruction-8b.json")) if d]
    if instruction:
        _register_qwen3_instruction_tables(tables, instruction)
    energy = _load("energy.json")
    if energy:
        _register_energy_tables(tables, energy, _load("energy-gpu.json"))
    prices = _load("cloud-prices.json")
    if prices and energy:
        _register_cloud_cost_tables(tables, prices, energy)


def _store_table(doc: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for row in doc["results"]:
        exact = row.get("sqlite_vec") or {}
        ann = row.get("lancedb") or {}
        if "error" in exact or "error" in ann:
            rows.append(
                [
                    str(row["dim"]),
                    f"{row['scale']:,}",
                    exact.get("error", "n/a"),
                    ann.get("error", "n/a"),
                    "n/a",
                    "n/a",
                    "n/a",
                ]
            )
            continue
        note = f" (from {row['truncated_from']})" if row.get("truncated_from") else ""
        rows.append(
            [
                f"{row['dim']}{note}",
                f"{row['scale']:,}",
                _fmt(exact.get("search_p50_ms"), 2),
                _fmt(ann.get("search_p50_ms"), 2),
                _fmt(ann.get("recall_at_k"), 3),
                _fmt(exact.get("store_mb"), 1),
                _fmt(ann.get("store_mb"), 1),
            ]
        )
    return _table(
        "Exact against ANN on real embeddings, by dimension",
        [
            "Dim",
            "Rows",
            "sqlite_vec exact p50 ms",
            "lancedb ANN p50 ms",
            "lancedb recall@10",
            "exact MB",
            "ANN MB",
        ],
        rows,
        f"Real embeddings of one fixed corpus and chunk profile ({doc.get('corpus')}, "
        f"{doc.get('profile')}), so only the embedding model changes down the ladder. "
        f"{doc.get('n_queries')} held-out passage vectors as queries, {doc.get('repeats')} repeats. "
        "Recall is against exact top-10 on the same rows, at the adapter's DEFAULT ANN "
        "parameters, which are not tuned here. sqlite_vec is exact, so its recall is 1.0 by "
        "construction and serves as the control on the measurement.",
    )


# ---------------------------------------------------------------- rendering


def render(spec: dict[str, Any]) -> str:
    """A padded markdown table, matching the style the rest of the docs already use."""
    columns, rows = spec["columns"], spec["rows"]
    if not rows:
        return f"_No data for: {spec['title']}._\n"
    widths = [max(len(str(columns[i])), *(len(str(r[i])) for r in rows)) for i in range(len(columns))]
    head = "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(columns)) + " |"
    rule = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = "\n".join("| " + " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(columns))) + " |" for r in rows)
    note = f"\n\n{spec['note']}\n" if spec["note"] else "\n"
    return f"{head}\n{rule}\n{body}\n{note}"


def _splice(text: str, tables: dict[str, dict[str, Any]]) -> tuple[str, list[str], list[str]]:
    written: list[str] = []
    unknown: list[str] = []

    def replace(match: re.Match[str]) -> str:
        table_id = match.group("id")
        if table_id not in tables:
            unknown.append(table_id)
            return match.group(0)
        written.append(table_id)
        body = render(tables[table_id])
        return _BEGIN.format(id=table_id) + "\n" + body + _END.format(id=table_id)

    return _BLOCK_RE.sub(replace, text), written, unknown


def manifest(tables: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {table_id: hashlib.sha256(render(spec).encode()).hexdigest() for table_id, spec in sorted(tables.items())}


def main() -> None:
    check = "--check" in sys.argv
    tables = collect_table_data()
    stale: list[str] = []
    unknown_all: list[str] = []

    for path in sorted(_DOCS.rglob("*.md")):
        text = path.read_text()
        if "BEGIN GENERATED" not in text:
            continue
        new_text, written, unknown = _splice(text, tables)
        unknown_all.extend(f"{path.name}:{u}" for u in unknown)
        if new_text != text:
            stale.extend(f"{path.relative_to(_ROOT)}:{w}" for w in written)
            if not check:
                path.write_text(new_text)
        print(f"[tables] {path.relative_to(_ROOT)}: {len(written)} blocks", flush=True)

    if not check:
        _MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        _MANIFEST.write_text(json.dumps(manifest(tables), indent=2, sort_keys=True) + "\n")

    if unknown_all:
        print(f"[tables] UNKNOWN table ids referenced by docs: {unknown_all}", flush=True)
    if check and (stale or unknown_all):
        sys.exit(f"stale generated blocks: {stale or unknown_all}. Run: python scripts/gen_bench_tables.py")
    print(f"[tables] {len(tables)} tables registered: {', '.join(sorted(tables))}", flush=True)


if __name__ == "__main__":
    main()
