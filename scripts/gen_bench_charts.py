#!/usr/bin/env python
"""Generate the benchmark ranking charts (PNG) from the raw measurement JSON.

The charts are DERIVED from the data, so they cannot drift: re-run this after any
benchmark change and commit the refreshed PNGs. ``collect_chart_data`` is pure (no
matplotlib) and returns exactly the values each chart plots; ``main`` writes those
values' hashes to ``docs/benchmarks/img/charts.manifest.json`` and renders the PNGs. A separate
test re-derives the hashes from the raw JSON and fails if a committed chart is stale -
so a data change without a chart regen is caught in CI, with no matplotlib needed there.

Run: python scripts/gen_bench_charts.py   (needs semdex[charts]: matplotlib)
"""

# pyright: basic
# One-off chart generator on matplotlib (no stubs); strict mode would only add noise.

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_RAW = _ROOT / "tests" / "benchmarks" / "raw"

# The tables generator owns the low-to-high orientation of a paired comparison and the printing of
# a level; a second copy of either here is how a chart and a table come to disagree on one row.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from gen_bench_tables import _oriented, declared_renderings, fit_corpora, level_label  # noqa: E402

_BASELINE = _ROOT / "tests" / "benchmarks" / "baseline.json"
_IMG = _ROOT / "docs" / "benchmarks" / "img"

# CVD-safe categorical palette (bitranox dataviz default, light mode; worst adjacent CVD dE 24.2).
_BLUE, _AQUA, _YELLOW, _GREEN, _VIOLET, _RED, _MAGENTA, _ORANGE = (
    "#2a78d6",
    "#1baf7a",
    "#eda100",
    "#008300",
    "#4a3aa7",
    "#e34948",
    "#e87ba4",
    "#eb6834",
)
_SURFACE, _INK, _MUTED, _GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dcdbd6"
# Dark mode is a SELECTED set of steps, not an inversion: the light hues lose contrast against a
# dark surface, so each is lifted to a lighter, slightly less saturated step that still separates
# under colour-vision deficiency. GitHub picks between the pair with prefers-color-scheme.
_DARK = {
    "surface": "#0d1117",
    "ink": "#e6edf3",
    "muted": "#9198a1",
    "grid": "#30363d",
    _BLUE: "#6cb6ff",
    _AQUA: "#4ad2a0",
    _YELLOW: "#f0c674",
    _GREEN: "#57c76a",
    _VIOLET: "#a78bfa",
    _RED: "#ff7b72",
    _MAGENTA: "#f0a6c8",
    _ORANGE: "#ff9f6e",
}
# The active theme is module state for the same reason matplotlib's rcParams are: every renderer
# is parameterised by one global style, and _style() already mutates rcParams globally. Threading
# a theme argument through eleven render functions would duplicate that mechanism rather than
# replace it. _render_all sets it once per pass and restores it.
_ACTIVE = {"theme": "light"}


def _is_dark() -> bool:
    return _ACTIVE["theme"] == "dark"


def _c(colour: str) -> str:
    """Map a palette colour to the current theme's step."""
    return _DARK.get(colour, colour) if _is_dark() else colour


def _surface() -> str:
    """The chart's own background, used as the spacer colour between stacked segments."""
    return _DARK["surface"] if _is_dark() else _SURFACE


def _ink() -> str:
    return _DARK["ink"] if _is_dark() else _INK


def _muted() -> str:
    return _DARK["muted"] if _is_dark() else _MUTED


# provider / store -> colour, fixed order so a chart never repaints when rows are added.
_PROVIDER_COLOR = {"fastembed": _BLUE, "model2vec": _AQUA, "st": _VIOLET, "ollama": _ORANGE}
_STORE_COLOR = {"json": _BLUE, "sqlite_vec": _AQUA, "lancedb": _GREEN, "pgvector": _VIOLET, "mariadb": _ORANGE}


##### chart data (pure; re-derived by the drift test) #####
def collect_chart_data() -> dict[str, Any]:
    """Extract exactly the values each chart plots. Pure - the drift test re-runs this."""
    return {
        "store_dim_crossover": _dim_crossover(),
        "store_scale_trends": _store_scale_trends(),
        "embedding_model_ranking": _embed_ranking(),
        "embedding_german_ranking": _german_ranking(),
        "chunker_throughput": _chunker_throughput(),
        "store_memory": _store_memory(),
        "chunker_quality": _chunker_quality(),
        "extractor_latency": _extractor_latency(),
        "end_to_end_quality": _end_to_end_quality(),
        "ocr_badscans": _ocr_badscans(),
        "chunk_knob_effects": _knob_effects(),
        "markdown_structure": _markdown_structure(),
        "product_k_budget": _product_k_budget(),
        "store_dim_real": _store_dim_real(),
        "span_integrity": _span_integrity(),
        "ann_frontier": _ann_frontier(),
        "extraction_real": _extraction_real(),
    }


def _span_integrity() -> dict[str, Any]:
    """Two views: the boundary mechanism, and what it costs a consumer.

    The left panel needs a corpus with a real range of fit chunk sizes. Only germanquad has one -
    the XQuAD contexts fit in a single chunk above 128 tokens, so faceting by corpus would print
    three panels holding one or two points each and invite the reader to compare empty space.
    The corpus axis moves to the right panel, where every corpus does carry a value.
    """
    path = _RAW / "span-integrity.json"
    if not path.exists():
        return {}
    doc = json.loads(path.read_text())
    rows = [r for r in doc["integrity"] if r["fit_for_claim"] and r["strategy"] == "recursive"]
    sizes_by_corpus = {c: {r["max_tokens"] for r in rows if r["corpus"] == c} for c in {r["corpus"] for r in rows}}
    corpus = max(sizes_by_corpus, key=lambda c: len(sizes_by_corpus[c]))
    mine = [r for r in rows if r["corpus"] == corpus]
    sizes = sorted({r["max_tokens"] for r in mine})
    overlaps = sorted({r["overlap_tokens"] for r in mine})
    by_cell = {(r["max_tokens"], r["overlap_tokens"]): r["span_intact"] for r in mine}

    blind = sorted(doc.get("retrieval") or [], key=lambda r: (r["corpus"], -r["max_tokens"], r["overlap_tokens"]))
    return {
        "corpus": corpus,
        "sizes": sizes,
        "overlaps": overlaps,
        "intact": {str(o): [by_cell.get((size, o)) for size in sizes] for o in overlaps},
        "blind_labels": [f"{r['corpus']}  {r['max_tokens']}/{r['overlap_tokens']}" for r in blind],
        "blind_split": [r.get("blind_from_split", 0.0) for r in blind],
        "blind_ranking": [r.get("blind_from_ranking", 0.0) for r in blind],
    }


def _ann_frontier() -> dict[str, Any]:
    """Query-time frontier points per store: latency against recall, plus the default marked.

    The default is carried separately rather than as just another point, because it is the
    setting every previously published store number was taken at and the chart's whole job is to
    show how far from the frontier it sits.
    """
    path = _RAW / "ann-frontier.json"
    if not path.exists():
        return {}
    points = [p for p in json.loads(path.read_text())["points"] if p["build_level"] == "driver default"]
    if not points:
        return {}
    stores = sorted({p["store"] for p in points})
    return {
        "stores": stores,
        "series": {
            store: sorted(
                (
                    {
                        "label": p["query_level"],
                        "recall": p["recall_vs_exact"],
                        "p50": p["search_p50_ms"],
                        "ndcg": p["ndcg@10"],
                        "default": p["query_level"] == "driver default",
                    }
                    for p in points
                    if p["store"] == store
                ),
                key=lambda point: point["p50"],
            )
            for store in stores
        },
    }


def _extraction_real() -> dict[str, Any]:
    """Word recall per extractor per document type, the axis fixtures cannot supply."""
    path = _RAW / "extraction-omnidocbench.json"
    if not path.exists():
        return {}
    doc = json.loads(path.read_text())
    rows = doc.get("by_source") or []
    if not rows:
        return {}
    extractors = sorted({r["extractor"] for r in rows})
    # ordered by how hard the page type proves to be, so the chart reads left to right
    by_source: dict[str, list[float]] = {}
    for source in {r["source"] for r in rows}:
        by_source[source] = [
            next((r["word_recall"] for r in rows if r["extractor"] == e and r["source"] == source), 0.0)
            for e in extractors
        ]
    order = sorted(by_source, key=lambda s: -max(by_source[s]))
    return {
        "extractors": extractors,
        "sources": order,
        "recall": {s: by_source[s] for s in order},
        "overall": {r["extractor"]: r["word_recall"] for r in doc.get("overall", [])},
    }


def _ocr_badscans() -> dict[str, Any]:
    doc = json.loads((_RAW / "ocr-badscans.json").read_text())
    rows = sorted(doc["results"], key=lambda r: r["content_pass"])  # ascending: best ends up on top in a barh
    return {
        "engines": [r["engine"] for r in rows],
        "kinds": [r["kind"] for r in rows],
        "content": [round(r["content_pass"], 3) for r in rows],
        "present": [round(r["present"], 3) for r in rows],
        "order": [round(r["order"], 3) for r in rows],
    }


def _dim_crossover() -> dict[str, Any]:
    rows = json.loads((_RAW / "dim-crossover.json").read_text())["results"]
    dims = sorted({r["dim"] for r in rows})
    scales = sorted({r["scale"] for r in rows})
    by = {(r["dim"], r["scale"]): r for r in rows}
    return {
        "dims": dims,
        "scales": scales,
        "sqlite": {d: [round(by[(d, s)]["sqlite_vec"]["search_p50_ms"], 2) for s in scales] for d in dims},
        "lance": {d: [round(by[(d, s)]["lancedb"]["search_p50_ms"], 2) for s in scales] for d in dims},
    }


def _store_scale_trends() -> dict[str, Any]:
    rows = json.loads((_RAW / "msmarco-scale.json").read_text())
    scales = sorted({r["scale"] for r in rows})
    stores = [s for s in _STORE_COLOR if any(r["store"] == s for r in rows)]
    by = {(r["store"], r["scale"]): r for r in rows}

    def series(store: str, key: str) -> list[float | None]:
        return [round(by[(store, sc)][key], 3) if (store, sc) in by else None for sc in scales]

    return {
        "scales": scales,
        "stores": stores,
        "p50": {s: series(s, "search_p50_ms") for s in stores},
        "ndcg": {s: series(s, "ndcg@10") for s in stores},
        "disk_mb": {s: series(s, "db_mb") for s in stores},
    }


def _embed_ranking() -> dict[str, Any]:
    cells = json.loads(_BASELINE.read_text())["results"]
    items = [
        (k.split("/", 1)[1].split("@", 1)[0], round(v["metrics"]["ndcg@10"], 4))
        for k, v in cells.items()
        if k.startswith("embed_model/") and v.get("status") == "ok"
    ]
    items.sort(key=lambda t: t[1])
    return {"labels": [i[0] for i in items], "ndcg": [i[1] for i in items]}


def _german_ranking() -> dict[str, Any]:
    rows = json.loads((_RAW / "embedding-german-miracl.json").read_text())["results"]
    items = sorted(((r["label"], round(r["ndcg@10"], 4)) for r in rows if "ndcg@10" in r), key=lambda t: t[1])
    return {"labels": [i[0] for i in items], "ndcg": [i[1] for i in items]}


def _chunker_throughput() -> dict[str, Any]:
    doc = json.loads((_RAW / "chunker-throughput.json").read_text())
    # The file was a bare list before the bench gained repeats, load capture and a worker sweep;
    # accept both so the chart survives whichever vintage is committed.
    rows = doc if isinstance(doc, list) else doc["rows"]
    single = [r for r in rows if r.get("threads", 1) == 1]  # one worker is the comparable rate
    corpora = sorted({r["corpus"] for r in single})
    # Slowest first, so the log axis reads as a ranking rather than as whatever order the rows
    # happened to be written in. 36,000x separates the ends; the order is the whole message.
    rate_of = {(r["corpus"], r["strategy"]): r["docs_per_s"] for r in single}
    strategies = sorted({r["strategy"] for r in single}, key=lambda st: rate_of[(corpora[0], st)])
    spread_of = {(r["corpus"], r["strategy"]): r.get("spread_pct", 0.0) for r in single}
    return {
        "corpora": corpora,
        "strategies": strategies,
        "docs_per_s": {c: [rate_of[(c, st)] for st in strategies] for c in corpora},
        "spread_pct": {c: [spread_of[(c, st)] for st in strategies] for c in corpora},
        "workers": _worker_scaling(rows, corpora[0]),
    }


def _worker_scaling(rows: list[dict[str, Any]], corpus: str) -> dict[str, Any]:
    """Throughput at N workers relative to one, for the strategies the sweep covers.

    Relative rather than absolute because the question is not how fast each one is - the other
    panel answers that - but whether adding a worker buys anything. Plotted against a line at
    1.0, a value below it is a strategy that got slower for being given more.
    """
    swept = sorted({r["threads"] for r in rows if r["corpus"] == corpus and r.get("threads", 1) > 1})
    if not swept:
        return {}
    counts = [1, *swept]
    by = {(r["strategy"], r["threads"]): r["docs_per_s"] for r in rows if r["corpus"] == corpus}
    series = {
        strategy: [by[(strategy, n)] / by[(strategy, 1)] for n in counts]
        for strategy in sorted({s for s, _ in by})
        if all((strategy, n) in by for n in counts)
    }
    return {"counts": counts, "series": series, "corpus": corpus}


def _store_memory() -> dict[str, Any]:
    doc = json.loads((_RAW / "memory.json").read_text())
    rows = doc.get("stores") or []
    backends = sorted({r["backend"] for r in rows})
    series = {
        backend: sorted(
            ((r["rows"], r["resident_mb"], r["ingest_peak_mb"]) for r in rows if r["backend"] == backend),
            key=lambda t: t[0],
        )
        for backend in backends
    }
    embedders = sorted(((r["provider"], r["resident_mb"]) for r in (doc.get("embedders") or [])), key=lambda t: t[1])
    return {"series": series, "embedders": embedders, "baseline_mb": doc.get("interpreter_baseline_mb", 0)}


def render_store_memory(d: dict[str, Any]) -> None:
    plt = _style()
    fig, (mem_ax, emb_ax) = plt.subplots(1, 2, figsize=(11.6, 4.2))
    palette = (_BLUE, _AQUA, _VIOLET, _ORANGE, _GREEN, _MAGENTA)
    for i, (backend, points) in enumerate(sorted(d["series"].items())):
        colour = palette[i % len(palette)]
        mem_ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker="o",
            markersize=5,
            linewidth=2,
            color=colour,
            label=f"{backend} (serving)",
        )
        mem_ax.plot(
            [p[0] for p in points], [p[2] for p in points], linestyle=":", linewidth=1.5, color=colour, alpha=0.7
        )
    mem_ax.axhline(8 * 1024, color=_muted(), linewidth=1, linestyle="--", zorder=1)
    mem_ax.text(mem_ax.get_xlim()[1], 8 * 1024 * 1.05, "8 GB container", ha="right", fontsize=7, color=_muted())
    mem_ax.set_xscale("log")
    mem_ax.set_yscale("log")
    mem_ax.set_xlabel("chunks stored (log scale)", fontsize=9)
    mem_ax.set_ylabel("resident MB (log scale)", fontsize=9)
    mem_ax.tick_params(labelsize=8)
    mem_ax.legend(fontsize=8, frameon=False, loc="upper left")
    mem_ax.set_title("Store memory: solid serving, dotted ingest peak", fontsize=11, color=_ink(), loc="left")

    import numpy as np

    names = [e[0] for e in d["embedders"]]
    y = np.arange(len(names))
    emb_ax.barh(y, [e[1] for e in d["embedders"]], height=0.55, color=_BLUE, zorder=3)
    for i, (_name, mb) in enumerate(d["embedders"]):
        emb_ax.text(mb * 1.05, y[i], f"{mb:,.0f} MB", va="center", fontsize=7, color=_muted())
    emb_ax.set_yticks(list(y))
    emb_ax.set_yticklabels(names, fontsize=9)
    emb_ax.set_xlim(right=max((e[1] for e in d["embedders"]), default=1) * 1.45)
    emb_ax.set_xlabel("resident MB, model plus one batch", fontsize=9)
    emb_ax.tick_params(labelsize=8)
    emb_ax.grid(axis="y", visible=False)
    emb_ax.set_title("Embedding provider memory", fontsize=11, color=_ink(), loc="left")
    fig.tight_layout()
    _save(fig, "store_memory")
    plt.close(fig)


def _chunker_quality() -> dict[str, Any]:
    cells = json.loads(_BASELINE.read_text())["results"]
    quality = {k: v for k, v in cells.items() if k.startswith("quality/") and v.get("status") == "ok"}
    embeddings = ["fastembed", "model2vec"]  # both cover every chunker; placeholder is only partial
    corpora = sorted({k.split("@", 1)[1] for k in quality})
    chunkers = sorted({k.split("/", 1)[1].split("+", 1)[0] for k in quality})

    def ndcg(chunker: str, embedding: str, corpus: str) -> float | None:
        cell = quality.get(f"quality/{chunker}+{embedding}@{corpus}")
        return round(cell["metrics"]["ndcg@10"], 4) if cell else None

    grid = {c: {e: [ndcg(ch, e, c) for ch in chunkers] for e in embeddings} for c in corpora}
    return {"corpora": corpora, "chunkers": chunkers, "embeddings": embeddings, "ndcg": grid}


def _extractor_latency() -> dict[str, Any]:
    cells = json.loads(_BASELINE.read_text())["results"]
    ex = {k: v for k, v in cells.items() if k.startswith("extract/")}
    extractors = sorted({k.split("/")[1].split("@")[0] for k in ex})
    formats = sorted({k.split("@")[1] for k in ex})
    lat: dict[str, float | None] = {}
    for e in extractors:
        for f in formats:
            cell = ex.get(f"extract/{e}@{f}")
            metrics = cell.get("metrics", {}) if cell else {}
            value = metrics.get("latency_ms")
            lat[f"{e}|{f}"] = round(value) if value is not None else None
    return {"extractors": extractors, "formats": formats, "latency_ms": lat}


def _end_to_end_quality() -> dict[str, Any]:
    cells = json.loads(_BASELINE.read_text())["results"]
    # A few USEFUL full-pipeline stacks spanning the quality/cost space; corpora easy -> hard.
    stacks = ["fast+fastembed", "semantic+fastembed", "whitespace+model2vec", "fast+placeholder"]
    corpora = ["scifact", "fiqa", "cqadupstack", "nfcorpus"]

    def ndcg(stack: str, corpus: str) -> float | None:
        cell = cells.get(f"quality/{stack}@{corpus}")
        return round(cell["metrics"]["ndcg@10"], 3) if cell and cell.get("status") == "ok" else None

    return {"stacks": stacks, "corpora": corpora, "ndcg": {s: [ndcg(s, c) for c in corpora] for s in stacks}}


def _manifest(data: dict[str, Any]) -> dict[str, str]:
    return {k: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest() for k, v in data.items()}


##### rendering (matplotlib; not needed by the drift test) #####
def _style() -> Any:
    import matplotlib

    matplotlib.use("svg")
    import matplotlib.pyplot as plt

    dark = _is_dark()
    matplotlib.rcParams.update(
        {
            "svg.hashsalt": "semdex-bench",
            "font.family": "DejaVu Sans",
            "figure.facecolor": _DARK["surface"] if dark else _SURFACE,
            "axes.facecolor": _DARK["surface"] if dark else _SURFACE,
            "axes.edgecolor": _muted(),
            "axes.labelcolor": _ink(),
            "text.color": _ink(),
            "xtick.color": _muted(),
            "ytick.color": _muted(),
            "axes.grid": True,
            "grid.color": _DARK["grid"] if dark else _GRID,
            "grid.linewidth": 0.6,
        }
    )
    return plt


def _save(fig: Any, name: str) -> None:
    # PNG (not SVG): GitHub Markdown renders committed PNGs reliably; relative-path SVG is flaky.
    # The dark twin is a separate file so the doc can offer both to <picture> and let the reader's
    # colour scheme choose; a single image would be unreadable in one of the two.
    suffix = "-dark" if _is_dark() else ""
    fig.savefig(_IMG / f"{name}{suffix}.png", format="png", dpi=120, bbox_inches="tight", metadata={"Software": None})


def _label(ax: Any, x: float, y: float, text: str) -> None:
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(4, 0), va="center", fontsize=7, color=_muted())


def _fmt_n(n: int) -> str:
    return f"{n // 1000}K" if n < 1_000_000 else f"{n // 1_000_000}M"


def render_store_dim_crossover(d: dict[str, Any]) -> None:
    plt = _style()
    dims, scales = d["dims"], d["scales"]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.8), sharex=True)
    x = [_fmt_n(s) for s in scales]
    for ax, dim in zip(axes.flat, dims, strict=True):
        ax.plot(x, d["sqlite"][dim], "-o", color=_BLUE, lw=1.8, ms=4.5, label="sqlite_vec (exact)")
        ax.plot(x, d["lance"][dim], "-o", color=_AQUA, lw=1.8, ms=4.5, label="lancedb (ANN)")
        ax.set_yscale("log")
        ax.set_title(f"{dim}-dim", color=_INK, fontsize=10)
        ax.set_ylabel("p50 ms", fontsize=8)
        ax.tick_params(labelsize=8)
    axes.flat[0].legend(loc="upper left", fontsize=7.5, frameon=False)
    fig.suptitle("Search latency vs corpus size, by embedding dimension", fontsize=11, color=_INK, y=1.0)
    _save(fig, "store_dim_crossover")
    plt.close(fig)


def render_store_scale_trends(d: dict[str, Any]) -> None:
    plt = _style()
    scales, stores = d["scales"], d["stores"]
    x = [_fmt_n(s) for s in scales]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1))
    panels = (
        ("p50", "search p50 latency (ms)", True),
        ("ndcg", "recall - nDCG@10", False),
        ("disk_mb", "on-disk size (MB)", True),
    )
    for ax, (key, title, logy) in zip(axes, panels, strict=True):
        for s in stores:
            pts = [(x[i], v) for i, v in enumerate(d[key][s]) if v is not None]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=_STORE_COLOR[s], lw=1.8, ms=5, label=s)
        if logy:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=10, color=_INK)
        ax.set_xlabel("corpus size", fontsize=9)
        ax.tick_params(labelsize=8)
        # lancedb flat-scans below its default 100K index threshold, so it is SLOWER at 50K than at
        # 250K (where the ANN index has engaged). Flag that point so the dip is not read as an error.
        first = d["p50"].get("lancedb", [None])[0]
        if key == "p50" and first is not None:
            ax.annotate(
                "lancedb: flat-scan\nbelow index threshold",
                xy=(x[0], first),
                xytext=(0.16, 0.9),
                textcoords="axes fraction",
                fontsize=6,
                color=_GREEN,
                ha="left",
                va="top",
                arrowprops={"arrowstyle": "->", "color": _GREEN, "lw": 0.8},
            )
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.suptitle(
        "Vector stores vs corpus size (MSMARCO, 384-dim): latency, recall, disk", fontsize=11, color=_INK, y=1.02
    )
    _save(fig, "store_scale_trends")
    plt.close(fig)


def _provider_of(label: str) -> str:
    return label.split(":", 1)[0]


def _embed_hbar(name: str, labels: list[str], values: list[float], title: str, xmax: float) -> None:
    plt = _style()
    from matplotlib.patches import Patch

    colors = [_PROVIDER_COLOR.get(_provider_of(lbl), _MUTED) for lbl in labels]
    fig, ax = plt.subplots(figsize=(6.6, 0.34 * len(labels) + 1.3))
    y = list(range(len(labels)))
    ax.barh(y, values, color=colors, height=0.68, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("nDCG@10", fontsize=9)
    ax.set_xlim(0, xmax)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, values, strict=True):
        _label(ax, v, yi, f"{v:g}")
    # Legend maps colour -> provider (identity is never colour-alone: the y-label names each model).
    present = [p for p in _PROVIDER_COLOR if p in {_provider_of(lbl) for lbl in labels}]
    ax.legend(
        handles=[Patch(facecolor=_PROVIDER_COLOR[p], label=p) for p in present],
        title="provider",
        fontsize=7.5,
        title_fontsize=7.5,
        frameon=False,
        loc="lower right",
    )
    ax.set_title(title, fontsize=11, color=_INK, loc="left")
    _save(fig, name)
    plt.close(fig)


def render_embedding_model_ranking(d: dict[str, Any]) -> None:
    _embed_hbar(
        "embedding_model_ranking", d["labels"], d["ndcg"], "Embedding quality - English (NFCorpus, nDCG@10)", xmax=0.75
    )


def render_embedding_german_ranking(d: dict[str, Any]) -> None:
    _embed_hbar(
        "embedding_german_ranking",
        d["labels"],
        d["ndcg"],
        "Embedding quality - native German (MIRACL-de, nDCG@10)",
        xmax=1.05,
    )


def render_chunker_quality(d: dict[str, Any]) -> None:
    plt = _style()
    import numpy as np

    corpora, chunkers, embeddings = d["corpora"], d["chunkers"], d["embeddings"]
    ecolor = {"fastembed": _BLUE, "model2vec": _AQUA}
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0), sharex=True)
    yb = np.arange(len(chunkers))
    h = 0.38
    for ax, corpus in zip(axes.flat, corpora, strict=True):
        for i, e in enumerate(embeddings):
            offs = (i - (len(embeddings) - 1) / 2) * h
            vals = [v if v is not None else 0.0 for v in d["ndcg"][corpus][e]]
            ax.barh(yb + offs, vals, height=h, color=ecolor[e], zorder=3, label=e)
        ax.set_yticks(list(yb))
        ax.set_yticklabels(chunkers, fontsize=8)
        ax.set_xlim(0, 1.0)
        ax.set_title(corpus, fontsize=10, color=_INK)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", visible=False)
    axes.flat[0].legend(title="embedding", fontsize=7.5, title_fontsize=7.5, frameon=False, loc="lower right")
    fig.suptitle(
        "Chunker quality by corpus and embedding (nDCG@10) - strategy barely moves it", fontsize=11, color=_INK, y=1.0
    )
    _save(fig, "chunker_quality")
    plt.close(fig)


def render_chunker_throughput(d: dict[str, Any]) -> None:
    plt = _style()
    workers = d.get("workers") or {}
    fig, axes = plt.subplots(
        1, 2 if workers else 1, figsize=(11.6 if workers else 6.6, 0.42 * len(d["strategies"]) + 1.6)
    )
    rate_ax, worker_ax = (axes[0], axes[1]) if workers else (axes, None)
    _render_throughput_rates(rate_ax, d)
    if worker_ax is not None:
        _render_worker_scaling(worker_ax, workers)
    fig.tight_layout()
    _save(fig, "chunker_throughput")
    plt.close(fig)


def _render_throughput_rates(ax: Any, d: dict[str, Any]) -> None:
    import numpy as np

    strategies, corpora = d["strategies"], d["corpora"]
    y = np.arange(len(strategies))
    h = 0.38
    for i, corpus in enumerate(corpora):
        offs = (i - (len(corpora) - 1) / 2) * h
        ax.barh(y + offs, d["docs_per_s"][corpus], height=h, color=(_BLUE, _AQUA)[i % 2], zorder=3, label=corpus)
    # The spread belongs on the chart, not only in the raw file: it is what says whether the bar
    # is a measurement or a lucky sample. Worst of the two corpora, so it cannot read better than
    # the noisiest thing behind it.
    for i in range(len(strategies)):
        worst = max(d["spread_pct"][c][i] for c in corpora)
        ax.text(
            max(d["docs_per_s"][c][i] for c in corpora) * 1.35,
            y[i],
            f"+/-{worst:.0f}%",
            va="center",
            fontsize=7,
            color=_muted(),
        )
    ax.set_yticks(list(y))
    ax.set_yticklabels(strategies, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(right=max(max(v) for v in d["docs_per_s"].values()) * 6)
    ax.set_xlabel("docs / second, one worker (log scale)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("Throughput, median of repeats (higher is better)", fontsize=11, color=_ink(), loc="left")


def _render_worker_scaling(ax: Any, workers: dict[str, Any]) -> None:
    counts = workers["counts"]
    palette = (_BLUE, _AQUA, _VIOLET, _ORANGE, _GREEN, _MAGENTA)
    for i, (strategy, rel) in enumerate(sorted(workers["series"].items())):
        ax.plot(counts, rel, marker="o", markersize=5, linewidth=2, color=palette[i % len(palette)], label=strategy)
    ax.axhline(1.0, color=_muted(), linewidth=1, linestyle="--", zorder=1)
    ax.text(counts[-1], 1.02, "no gain", ha="right", va="bottom", fontsize=7, color=_muted())
    ax.set_xscale("log", base=2)
    ax.set_xticks(counts)
    ax.set_xticklabels([str(c) for c in counts], fontsize=8)
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("worker threads", fontsize=9)
    ax.set_ylabel("throughput relative to one worker", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.set_title("Every worker added made it slower", fontsize=11, color=_ink(), loc="left")


def render_extractor_latency(d: dict[str, Any]) -> None:
    plt = _style()
    import numpy as np
    from matplotlib.colors import LogNorm

    extractors, formats = d["extractors"], d["formats"]
    grid = np.full((len(extractors), len(formats)), np.nan)
    for i, e in enumerate(extractors):
        for j, f in enumerate(formats):
            value = d["latency_ms"][f"{e}|{f}"]
            if value is not None:
                grid[i, j] = value
    fig, ax = plt.subplots(figsize=(6.8, 2.9))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#eeede8")  # unsupported/failed cell -> light gray, distinct from the ramp
    finite = grid[np.isfinite(grid)]
    im = ax.imshow(
        np.ma.masked_invalid(grid),
        aspect="auto",
        cmap=cmap,
        norm=LogNorm(vmin=max(1.0, finite.min()), vmax=finite.max()),
    )
    ax.set_xticks(range(len(formats)))
    ax.set_xticklabels(formats, fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(len(extractors)))
    ax.set_yticklabels(extractors, fontsize=9)
    ax.grid(visible=False)
    for i in range(len(extractors)):
        for j in range(len(formats)):
            if np.isfinite(grid[i, j]):
                shade = _SURFACE if grid[i, j] > finite.mean() else _INK
                ax.text(j, i, f"{int(grid[i, j])}", ha="center", va="center", fontsize=6.5, color=shade)
    fig.colorbar(im, ax=ax, label="latency ms (log)", fraction=0.046, pad=0.03)
    ax.set_title("Extractor latency per format (ms; blank = unsupported)", fontsize=11, color=_INK, loc="left")
    _save(fig, "extractor_latency")
    plt.close(fig)


def render_end_to_end_quality(d: dict[str, Any]) -> None:
    plt = _style()
    stacks, corpora = d["stacks"], d["corpora"]
    colors = (_BLUE, _AQUA, _VIOLET, _ORANGE)
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    for stack, color in zip(stacks, colors, strict=True):
        ys = d["ndcg"][stack]
        pts = [(corpora[i], v) for i, v in enumerate(ys) if v is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=color, lw=1.8, ms=5.5, label=stack)
    ax.set_ylabel("nDCG@10 (whole chain)", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("corpus (easier -> harder)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(title="chunker + embedding", fontsize=7.5, title_fontsize=7.5, frameon=False, loc="lower left")
    ax.set_title(
        "End-to-end retrieval quality: pipeline stack x corpus (exact store)", fontsize=11, color=_INK, loc="left"
    )
    _save(fig, "end_to_end_quality")
    plt.close(fig)


def render_ocr_badscans(d: dict[str, Any]) -> None:
    plt = _style()
    import numpy as np

    engines = d["engines"]
    metrics = [
        ("content", d["content"], _BLUE),
        ("text present", d["present"], _AQUA),
        ("reading order", d["order"], _VIOLET),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    yb = np.arange(len(engines))
    h = 0.26
    for i, (name, vals, color) in enumerate(metrics):
        offs = (i - (len(metrics) - 1) / 2) * h
        ax.barh(yb + offs, vals, height=h, color=color, zorder=3, label=name)
        for y, v in zip(yb + offs, vals, strict=True):
            ax.annotate(
                f"{v * 100:.0f}",
                (v, y),
                textcoords="offset points",
                xytext=(3, 0),
                va="center",
                fontsize=6.5,
                color=_MUTED,
            )
    ax.set_yticks(yb)
    labels = [f"{e} (VLM)" if k == "vlm" else e for e, k in zip(engines, d["kinds"], strict=True)]
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("olmOCR-bench Old-Scans pass rate", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.set_title("OCR on degraded scans: VLM vs incumbents (Old-Scans, 526 tests)", fontsize=11, color=_INK, loc="left")
    _save(fig, "ocr_badscans")
    plt.close(fig)


def _read(path: Path) -> dict[str, Any] | None:
    """Read a raw file, or None when a measurement has not been taken yet.

    A chart whose data does not exist is skipped rather than rendered empty, so a missing run is
    visible as an absent figure instead of an authoritative-looking blank one.
    """
    return json.loads(path.read_text()) if path.exists() else None


##### charts from the swept data (chunk knobs, real-vector store ladder) #####
def _fit_corpora() -> set[str]:
    """Corpora long enough to carry a chunking claim - delegates to the one shared filter.

    gen_bench_tables.py owns the profile and threshold literals and the rendering exclusion; a
    second copy here is how a chart and a table came to disagree on which corpora count.
    """
    audit = _read(_RAW / "chunk-dimension-audit.json")
    return fit_corpora(audit, declared_renderings(_RAW)) or set()


def _corpus_label(corpus: str) -> str:
    """Body and language, not language alone.

    The overlap axis splits by CORPUS and not by language - GerDaLIR German goes one way and MLDR
    German the other - so a label taking only the language collapses the two bodies that disagree
    into one name, and the chart stops being able to show the split it is evidence for.
    """
    parts = corpus.split("_")
    return "-".join(parts[:2]) if len(parts) > 1 else corpus


def _knob_effects() -> dict[str, Any]:
    """Paired per-query deltas with intervals, per chunking axis."""
    doc = _read(_RAW / "chunk-knob-effects.json")
    fit = _fit_corpora()
    if not doc or not fit:
        return {"axes": [], "rows": {}}
    rows: dict[str, list[dict[str, Any]]] = {}
    for raw in doc["effects"]:
        if raw["corpus"] not in fit:
            continue
        effect = _oriented(raw)
        held = effect["held_fixed"]
        rows.setdefault(effect["axis"], []).append(
            {
                "label": f"{_corpus_label(effect['corpus'])} {effect['embedding'].split(':')[1]}",
                "held": f"{held.get('strategy', '')} {held.get('max_tokens', '')}".strip(),
                "change": f"{level_label(effect['from_level'])} to {level_label(effect['to_level'])}",
                "delta": effect["mean_delta"],
                "lo": effect["ci_lo"],
                "hi": effect["ci_hi"],
                "resolved": effect["resolved"],
            }
        )
    rows = {axis: sorted(entries, key=lambda r: r["delta"]) for axis, entries in rows.items()}
    return {"axes": sorted(rows), "rows": rows}


def _markdown_structure() -> dict[str, Any]:
    """Marked against unmarked MLDR English, one paired delta per strategy and embedder."""
    doc = _read(_RAW / "markdown-structure-effect.json")
    if not doc:
        return {"rows": []}
    rows = [
        {
            "label": f"{e['strategy']} {e['embedding'].split(':')[1]}",
            "delta": e["mean_delta"],
            "lo": e["ci_lo"],
            "hi": e["ci_hi"],
            "resolved": e["resolved"],
        }
        for e in doc["effects"]
    ]
    return {"rows": sorted(rows, key=lambda r: r["delta"])}


_PRODUCT_K_CAPS = [64, 128, 256, 512]
_PRODUCT_K_CORPUS = "mldr_en_8k_slice"


def _product_k_budget() -> dict[str, Any]:
    """Delivered nDCG@k against cap at fixed token budgets, MLDR English, recursive, zero overlap."""
    doc = _read(_RAW / "product-k.json")
    if not doc:
        return {"caps": [], "budgets": [], "series": {}}
    budgets = [int(b) for b in doc["budgets"]]
    series: dict[str, dict[str, list[float | None]]] = {str(b): {} for b in budgets}
    for row in doc["rows"]:
        axes = row["axes"]
        if row["corpus"] != _PRODUCT_K_CORPUS or axes.get("strategy") != "recursive" or axes.get("overlap_tokens") != 0:
            continue
        cap = int(axes["max_tokens"])
        if cap not in _PRODUCT_K_CAPS:
            continue
        for budget in budgets:
            if budget not in row.get("budgets", []):
                continue
            line = series[str(budget)].setdefault(str(row["embedding"]), [None] * len(_PRODUCT_K_CAPS))
            line[_PRODUCT_K_CAPS.index(cap)] = float(row["ndcg_delivered"])
    return {"caps": list(_PRODUCT_K_CAPS), "budgets": budgets, "series": series}


def _store_dim_real() -> dict[str, Any]:
    """Exact against ANN on real embeddings: latency, recall and size by dimension."""
    doc = _read(_RAW / "dim-crossover-real.json")
    if not doc:
        return {"dims": [], "scales": [], "points": []}
    points = []
    for row in doc["results"]:
        exact, ann = row.get("sqlite_vec") or {}, row.get("lancedb") or {}
        if "error" in exact or "error" in ann or not exact or not ann:
            continue
        points.append(
            {
                "dim": row["dim"],
                "scale": row["scale"],
                "truncated": bool(row.get("truncated_from")),
                "exact_ms": exact["search_p50_ms"],
                "ann_ms": ann["search_p50_ms"],
                "ann_recall": ann.get("recall_at_k"),
                "exact_mb": exact["store_mb"],
                "ann_mb": ann["store_mb"],
            }
        )
    return {
        "dims": sorted({p["dim"] for p in points}),
        "scales": sorted({p["scale"] for p in points}),
        "points": points,
    }


def render_chunk_knob_effects(d: dict[str, Any]) -> None:
    """One forest plot per axis: the paired delta with its interval, against a zero line.

    This is the chart the chunking decision actually needs. A bar chart of two means invites the
    reader to compare bar heights, which is the comparison the query counts cannot support.
    Plotting the DIFFERENCE with its interval makes "this does not resolve" visible instead of
    hidden, and the zero line is the only reference that matters.

    One figure per axis, not three stacked panels: stacked, the tallest axis forced a 2400px
    image that no page can show at a readable size, and each axis belongs beside its own section.
    """
    for axis in d["axes"]:
        _render_one_knob_axis(axis, d["rows"][axis])


def _render_one_knob_axis(axis: str, rows: list[dict[str, Any]]) -> None:
    plt = _style()
    height = max(2.2, 0.185 * len(rows) + 1.0)
    fig, ax = plt.subplots(figsize=(7.6, height))
    ys = list(range(len(rows)))
    for y, row in zip(ys, rows, strict=True):
        resolved = row["resolved"]
        colour = _c(_BLUE if row["delta"] > 0 else _ORANGE) if resolved else _muted()
        ax.plot(
            [row["lo"], row["hi"]],
            [y, y],
            color=colour,
            lw=2.1 if resolved else 1.1,
            alpha=1.0 if resolved else 0.5,
            solid_capstyle="round",
        )
        ax.plot([row["delta"]], [y], "o", color=colour, ms=5.0 if resolved else 3.2, alpha=1.0 if resolved else 0.5)
    ax.axvline(0, color=_ink(), lw=1.1, alpha=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['label']}  {r['held']}  {r['change']}" for r in rows], fontsize=6.8)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.tick_params(labelsize=7.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("paired nDCG@10 difference, 95% CI (positive favours the higher level)", fontsize=8)
    resolved_count = sum(1 for r in rows if r["resolved"])
    ax.set_title(
        f"{axis}: {resolved_count} of {len(rows)} comparisons resolve; faded bars cross zero",
        color=_ink(),
        fontsize=10,
        loc="left",
        pad=8,
    )
    fig.tight_layout()
    _save(fig, f"chunk_knob_{axis}")
    plt.close(fig)


def render_markdown_structure(d: dict[str, Any]) -> None:
    """Forest plot: the paired delta of heading markup per strategy and embedder against zero.

    Same form as the knob plots and for the same reason: the question is whether the difference
    resolves, and a bar chart of two means hides exactly that.
    """
    rows = d["rows"]
    if not rows:
        return
    plt = _style()
    fig, ax = plt.subplots(figsize=(7.6, max(2.2, 0.26 * len(rows) + 1.0)))
    ys = list(range(len(rows)))
    for y, row in zip(ys, rows, strict=True):
        resolved = row["resolved"]
        colour = _c(_BLUE if row["delta"] > 0 else _ORANGE) if resolved else _muted()
        ax.plot(
            [row["lo"], row["hi"]],
            [y, y],
            color=colour,
            lw=2.1 if resolved else 1.1,
            alpha=1.0 if resolved else 0.5,
            solid_capstyle="round",
        )
        ax.plot([row["delta"]], [y], "o", color=colour, ms=5.0 if resolved else 3.2, alpha=1.0 if resolved else 0.5)
    ax.axvline(0, color=_ink(), lw=1.1, alpha=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=7.5)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.tick_params(labelsize=7.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("paired nDCG@10 difference, 95% CI (positive favours the marked corpus)", fontsize=8)
    resolved_count = sum(1 for r in rows if r["resolved"])
    ax.set_title(
        f"heading markup on MLDR English: {resolved_count} of {len(rows)} pairs resolve; faded bars cross zero",
        color=_ink(),
        fontsize=10,
        loc="left",
        pad=8,
    )
    fig.tight_layout()
    _save(fig, "chunk_markdown_structure")
    plt.close(fig)


def render_product_k_budget(d: dict[str, Any]) -> None:
    """One panel per budget: delivered nDCG@k against cap, a line per embedder.

    Read left to right as "the same number of tokens handed to the reader, cut finer or coarser":
    at a fixed budget a smaller cap means more, shorter chunks. The x axis is log2 so the four
    caps sit evenly.
    """
    if not d["series"]:
        return
    plt = _style()
    palette = [_BLUE, _AQUA, _ORANGE, _VIOLET, _GREEN, _RED]
    fig, axes = plt.subplots(1, len(d["budgets"]), figsize=(4.0 * len(d["budgets"]), 3.6), sharey=True)
    panels = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, budget in zip(panels, d["budgets"], strict=True):
        lines = d["series"][str(budget)]
        for i, (embedding, values) in enumerate(sorted(lines.items())):
            colour = palette[i % len(palette)]
            xs = [cap for cap, v in zip(d["caps"], values, strict=True) if v is not None]
            ys = [v for v in values if v is not None]
            ax.plot(xs, ys, "o-", color=_c(colour), lw=1.6, ms=4.5, label=embedding.split(":")[1])
        ax.set_xscale("log", base=2)
        ax.set_xticks(d["caps"])
        ax.set_xticklabels([str(c) for c in d["caps"]], fontsize=8)
        ax.set_xlabel("cap (tokens per chunk)", fontsize=8, color=_muted())
        ax.set_title(f"{budget} tokens read (k = {budget} / cap)", fontsize=9, color=_ink())
        ax.tick_params(labelsize=8)
    panels[0].set_ylabel("delivered nDCG@k", fontsize=8, color=_muted())
    panels[0].legend(fontsize=7, frameon=False, loc="lower right", title="embedder", title_fontsize=7)
    fig.suptitle("Chunk size at a fixed number of tokens read, MLDR English", fontsize=11, color=_ink(), y=1.0)
    _save(fig, "chunk_product_k_budget")
    plt.close(fig)


def render_store_dim_real(d: dict[str, Any]) -> None:
    """Three panels sharing a dimension axis: latency, ANN recall, and store size.

    Deliberately three panels rather than one chart with two y-axes. Latency and recall are the
    two halves of the same decision, and a dual axis would let their curves cross wherever the
    scaling happened to put them, which is the most reliable way to mislead with a chart.
    """
    if not d["points"]:
        return
    plt = _style()
    scales = d["scales"]
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.5))
    shades = [0.45, 0.7, 1.0]
    for i, scale in enumerate(scales):
        pts = sorted((p for p in d["points"] if p["scale"] == scale), key=lambda p: p["dim"])
        if not pts:
            continue
        dims = [p["dim"] for p in pts]
        alpha = shades[min(i, len(shades) - 1)]
        axs[0].plot(
            [p["dim"] for p in pts],
            [p["exact_ms"] for p in pts],
            "-o",
            color=_c(_BLUE),
            lw=1.7,
            ms=4,
            alpha=alpha,
            label=f"exact, {_fmt_n(scale)}",
        )
        axs[0].plot(
            dims,
            [p["ann_ms"] for p in pts],
            "-s",
            color=_c(_AQUA),
            lw=1.7,
            ms=4,
            alpha=alpha,
            label=f"ANN, {_fmt_n(scale)}",
        )
        recalls = [p["ann_recall"] for p in pts if p["ann_recall"] is not None]
        if recalls:
            axs[1].plot(
                [p["dim"] for p in pts if p["ann_recall"] is not None],
                recalls,
                "-o",
                color=_c(_ORANGE),
                lw=1.7,
                ms=4,
                alpha=alpha,
                label=f"{_fmt_n(scale)} rows",
            )
        axs[2].plot(dims, [p["exact_mb"] for p in pts], "-o", color=_c(_BLUE), lw=1.7, ms=4, alpha=alpha)
        axs[2].plot(dims, [p["ann_mb"] for p in pts], "-s", color=_c(_AQUA), lw=1.7, ms=4, alpha=alpha)

    axs[0].set_yscale("log")
    axs[0].set_title("Search latency", color=_ink(), fontsize=9.5, loc="left")
    axs[0].set_ylabel("p50 ms (log)", fontsize=8)
    axs[1].set_title("What ANN speed costs", color=_ink(), fontsize=9.5, loc="left")
    axs[1].set_ylabel("lancedb recall@10 vs exact", fontsize=8)
    axs[1].set_ylim(0, 1.02)
    axs[1].axhline(1.0, color=_c(_BLUE), lw=1.2, ls="--")
    axs[1].annotate(
        "exact = 1.0", (dims[0], 1.0), textcoords="offset points", xytext=(2, -10), fontsize=7, color=_c(_BLUE)
    )
    axs[2].set_title("On-disk size", color=_ink(), fontsize=9.5, loc="left")
    axs[2].set_ylabel("MB", fontsize=8)
    dims_all = sorted({p["dim"] for p in d["points"]})
    for ax in axs:
        ax.set_xscale("log", base=2)
        # Tick at the dimensions actually measured, labelled with those numbers. A log2 axis
        # otherwise reads 2^8 to 2^12, and nobody chooses a model by exponent.
        ax.set_xticks(dims_all)
        ax.set_xticklabels([str(v) for v in dims_all], rotation=45, ha="right")
        ax.minorticks_off()
        ax.set_xlabel("embedding dimension", fontsize=8)
        ax.tick_params(labelsize=7)
    axs[0].legend(fontsize=6.5, frameon=False, ncols=2)
    axs[1].legend(fontsize=6.5, frameon=False)
    fig.suptitle("Exact against ANN on real embeddings, by dimension", fontsize=10.5, color=_ink())
    fig.tight_layout()
    _save(fig, "store_dim_real")
    plt.close(fig)


def render_span_integrity(d: dict[str, Any]) -> None:
    """Left: the boundary mechanism. Right: what it costs, split by who is responsible."""
    if not d:
        return
    plt = _style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.6, 4.2), gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.30})
    _span_mechanism_panel(left, d)
    _span_blind_panel(right, d)
    fig.suptitle(
        "A boundary cuts the answer; overlap prevents it - but most undelivered answers are a ranking miss",
        fontsize=11,
        color=_ink(),
        y=1.0,
    )
    _save(fig, "span_integrity")
    plt.close(fig)


def _span_mechanism_panel(ax: Any, d: dict[str, Any]) -> None:
    palette = [_BLUE, _AQUA, _YELLOW, _ORANGE, _VIOLET]
    for i, overlap in enumerate(d["overlaps"]):
        series = d["intact"][str(overlap)]
        points = [(size, value) for size, value in zip(d["sizes"], series, strict=True) if value is not None]
        if not points:
            continue
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker="o",
            markersize=5,
            linewidth=2,
            color=_c(palette[i % len(palette)]),
            zorder=3,
            label=f"overlap {overlap}",
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(d["sizes"])
    ax.set_xticklabels([str(s) for s in d["sizes"]], fontsize=8)
    ax.set_xlabel("max tokens", fontsize=8, color=_muted())
    ax.set_ylabel("answers intact in one chunk", fontsize=8, color=_muted())
    ax.set_title(f"{d['corpus']}: chunking alone", fontsize=9.5, color=_ink())
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")


def _span_blind_panel(ax: Any, d: dict[str, Any]) -> None:
    """Stacked by CAUSE, because the total alone would read as chunking's failure."""
    import numpy as np

    labels = d["blind_labels"]
    if not labels:
        return
    y = np.arange(len(labels))
    split = np.asarray(d["blind_split"])
    ranking = np.asarray(d["blind_ranking"])
    ax.barh(y, split, height=0.62, color=_c(_RED), zorder=3, label="answer split by a boundary")
    # A 2px surface gap keeps the two segments legible where one is thin.
    ax.barh(
        y,
        ranking,
        height=0.62,
        left=split,
        color=_c(_BLUE),
        zorder=3,
        label="intact but not retrieved",
        edgecolor=_surface(),
        linewidth=2,
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("share of queries scored as a document hit that deliver no whole answer", fontsize=8, color=_muted())
    ax.set_title("composed with retrieval, by cause  (corpus  size/overlap)", fontsize=9.5, color=_ink())
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")


def render_ann_frontier(d: dict[str, Any]) -> None:
    """Latency against recall, one line per store, with each driver default called out."""
    if not d:
        return
    plt = _style()
    palette = {"lancedb": _BLUE, "pgvector": _AQUA, "mariadb": _ORANGE}
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for store in d["stores"]:
        series = d["series"][store]
        colour = _c(palette.get(store, _VIOLET))
        ax.plot(
            [p["p50"] for p in series],
            [p["recall"] for p in series],
            marker="o",
            markersize=5,
            linewidth=2,
            color=colour,
            zorder=3,
            label=store,
        )
        for point in series:
            if not point["default"]:
                continue
            # A 2px surface ring keeps the marker legible where it overlaps the line.
            ax.plot(
                [point["p50"]],
                [point["recall"]],
                marker="D",
                markersize=10,
                color=colour,
                markeredgecolor=_surface(),
                markeredgewidth=2,
                zorder=4,
            )
            # Offset upward rather than to the right: mariadb's points cluster inside a few
            # milliseconds, so a sideways label lands on top of its own series.
            ax.annotate(
                "driver default",
                (point["p50"], point["recall"]),
                textcoords="offset points",
                xytext=(0, 11),
                ha="center",
                fontsize=7.5,
                color=_muted(),
            )
    ax.set_xlabel("search p50 (ms)", fontsize=8, color=_muted())
    ax.set_ylabel("recall of the exact top-10 documents", fontsize=8, color=_muted())
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="lower right", title="store", title_fontsize=8)
    fig.suptitle(
        "Every published ANN number was taken at the diamond, not on the frontier",
        fontsize=11,
        color=_ink(),
        y=0.99,
    )
    _save(fig, "ann_frontier")
    plt.close(fig)


def render_extraction_real(d: dict[str, Any]) -> None:
    """Grouped bars: document type along y, one bar per extractor, word recall on x."""
    if not d:
        return
    import numpy as np

    plt = _style()
    extractors, sources = d["extractors"], d["sources"]
    palette = {"docling": _BLUE, "xberg": _AQUA, "markitdown": _ORANGE, "mineru": _VIOLET}
    # Height scales with the number of GROUPS, not groups x bars: the latter gave a 22-inch
    # figure nobody can read on a page.
    fig, ax = plt.subplots(figsize=(8.6, 0.62 * len(sources) + 1.9))
    y = np.arange(len(sources))
    height = 0.8 / max(1, len(extractors))
    for i, name in enumerate(extractors):
        offset = (i - (len(extractors) - 1) / 2) * height
        values = [d["recall"][s][i] for s in sources]
        ax.barh(
            y + offset,
            values,
            height=height,
            color=_c(palette.get(name, _YELLOW)),
            zorder=3,
            label=f"{name} ({d['overall'].get(name, 0.0):.2f} overall)",
        )
    ax.set_yticks(list(y))
    ax.set_yticklabels(sources, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("share of annotated text recovered", fontsize=8, color=_muted())
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", visible=False)
    # Legend above the axes: inside it lands on the longest bars, which are the point.
    ax.legend(fontsize=7.5, frameon=False, ncol=len(extractors), loc="lower center", bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Extraction fidelity on real documents, by document type", fontsize=11, color=_ink(), y=1.07)
    _save(fig, "extraction_real")
    plt.close(fig)


def main() -> None:
    _IMG.mkdir(parents=True, exist_ok=True)
    data = collect_chart_data()
    (_IMG / "charts.manifest.json").write_text(json.dumps(_manifest(data), indent=2, sort_keys=True) + "\n")
    for theme in ("light", "dark"):
        _ACTIVE["theme"] = theme
        _render_all(data)
    _ACTIVE["theme"] = "light"
    print(f"wrote {len(data)} charts x 2 themes + manifest to {_IMG}")


def _render_all(data: dict[str, Any]) -> None:
    render_ocr_badscans(data["ocr_badscans"])
    render_store_dim_crossover(data["store_dim_crossover"])
    render_store_scale_trends(data["store_scale_trends"])
    render_embedding_model_ranking(data["embedding_model_ranking"])
    render_embedding_german_ranking(data["embedding_german_ranking"])
    render_chunker_throughput(data["chunker_throughput"])
    render_store_memory(data["store_memory"])
    render_chunker_quality(data["chunker_quality"])
    render_extractor_latency(data["extractor_latency"])
    render_end_to_end_quality(data["end_to_end_quality"])
    render_chunk_knob_effects(data["chunk_knob_effects"])
    render_markdown_structure(data["markdown_structure"])
    render_product_k_budget(data["product_k_budget"])
    render_store_dim_real(data["store_dim_real"])
    render_span_integrity(data["span_integrity"])
    render_ann_frontier(data["ann_frontier"])
    render_extraction_real(data["extraction_real"])


if __name__ == "__main__":
    main()
