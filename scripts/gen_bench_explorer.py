#!/usr/bin/env python
# pyright: basic
# Generator for a self-contained HTML page; strict typing adds nothing over the JSON shapes here.
"""Build the interactive benchmark explorer as one self-contained HTML file.

The static pages answer the questions we thought to ask. This answers the ones a reader brings:
which profile wins for German long documents at 768 dimensions on a CPU budget, and is that
difference real. There are several hundred measured cells across six axes, and no fixed set of
figures covers every slice of that without either exploding the chart count or quietly dropping
comparisons.

Self-contained on purpose: the data is inlined, there is no script tag pointing anywhere, and no
font or stylesheet is fetched. It therefore works from a file:// path with no server, and inside a
strict content-security policy.

It does NOT replace the static pages. GitHub will not run this file inline, so the generated
tables remain the canonical, reviewable, drift-gated form; this is the exploration surface.

Run: python scripts/gen_bench_explorer.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
_RAW = _ROOT / "tests" / "benchmarks" / "raw"
_OUT = _ROOT / "docs" / "benchmarks" / "explorer.html"

# The tables generator owns the fitness-filter literals and the rendering exclusion; a second
# copy here is how the explorer and a table came to disagree on which corpora count.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from gen_bench_tables import declared_renderings, fit_corpora  # noqa: E402 - needs the sys.path entry above

_CELL_FILES = (
    "chunk-sweep-mldr.json",
    "chunk-sweep-beir.json",
    "chunk-sweep-miracl.json",
    "embedding-multilingual-mldr.json",
    "hybrid-dense-bm25.json",
    "rerank-cross-encoder.json",
    "store-quality.json",
)


def _load(name: str) -> dict[str, Any] | None:
    path = _RAW / name
    return json.loads(path.read_text()) if path.exists() else None


def _fit_corpora(audit: dict[str, Any] | None) -> list[str]:
    """The fit set as a sorted list (the explorer's JSON has no use for a set)."""
    return sorted(fit_corpora(audit, declared_renderings(_RAW)) or [])


def _chunk_shapes(audit: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Per (corpus, profile), how many chunks and tokens it produced - the cost side of a cell."""
    if not audit:
        return {}
    return {
        f"{r['corpus']}__{r['profile']}": {
            "chunks": r["rows"],
            "chunks_per_doc": r["chunks_per_doc"],
            "tokens": r["tokens_total"],
        }
        for r in audit["chunk_sets"]
    }


def collect() -> dict[str, Any]:
    audit = _load("chunk-dimension-audit.json")
    shapes = _chunk_shapes(audit)
    cells: list[dict[str, Any]] = []
    for name in _CELL_FILES:
        doc = _load(name)
        if not doc:
            continue
        for cell in doc["cells"]:
            axes = cell.get("axes") or {}
            shape = shapes.get(f"{cell['corpus']}__{cell['profile']}", {})
            cells.append(
                {
                    "corpus": cell["corpus"],
                    "profile": cell["profile"],
                    "strategy": axes.get("strategy"),
                    "max_tokens": axes.get("max_tokens"),
                    "overlap": axes.get("overlap_tokens"),
                    "breakpoint": axes.get("breakpoint_model") or "",
                    "embedding": cell["embedding"],
                    # Dense-only cells carry no method; naming it here keeps the filter honest,
                    # because a blank chip would read as "no method" rather than "dense".
                    "method": cell.get("method") or "dense",
                    "dim": cell["dim"],
                    "n": cell["n_queries"],
                    "ndcg": cell.get("ndcg@10"),
                    "lo": cell.get("ndcg@10_ci_lo"),
                    "hi": cell.get("ndcg@10_ci_hi"),
                    "recall": cell.get("recall@10"),
                    "mrr": cell.get("mrr"),
                    "p1": cell.get("p@1"),
                    "chunks": shape.get("chunks"),
                    "tokens": shape.get("tokens"),
                    "source": name,
                }
            )
    return {
        "cells": cells,
        "fit_corpora": _fit_corpora(audit),
    }


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>semdex benchmark explorer</title>
<style>
:root {
  --surface: #fcfcfb; --ink: #0b0b0b; --muted: #52514e; --grid: #dcdbd6; --panel: #ffffff;
  --blue: #2a78d6; --aqua: #1baf7a; --yellow: #eda100; --green: #008300;
  --violet: #4a3aa7; --red: #e34948; --magenta: #e87ba4; --orange: #eb6834;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface: #0d1117; --ink: #e6edf3; --muted: #9198a1; --grid: #30363d; --panel: #161b22;
    --blue: #6cb6ff; --aqua: #4ad2a0; --yellow: #f0c674; --green: #57c76a;
    --violet: #a78bfa; --red: #ff7b72; --magenta: #f0a6c8; --orange: #ff9f6e;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--ink);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
header { padding: 18px 20px 10px; border-bottom: 1px solid var(--grid); }
h1 { margin: 0 0 4px; font-size: 18px; font-weight: 650; }
header p { margin: 0; color: var(--muted); font-size: 13px; max-width: 70ch; }
main { display: grid; grid-template-columns: 260px 1fr; gap: 18px; padding: 16px 20px 40px; }
@media (max-width: 860px) { main { grid-template-columns: 1fr; } }
.panel { background: var(--panel); border: 1px solid var(--grid); border-radius: 8px; padding: 12px 14px; }
fieldset { border: 0; margin: 0 0 14px; padding: 0; }
legend { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted);
  padding: 0 0 6px; font-weight: 600; }
.chips { display: flex; flex-wrap: wrap; gap: 5px; }
.chip { border: 1px solid var(--grid); background: transparent; color: var(--ink); cursor: pointer;
  border-radius: 999px; padding: 3px 9px; font-size: 11.5px; font-family: inherit; }
.chip[aria-pressed="true"] { background: var(--blue); border-color: var(--blue); color: #fff; }
@media (prefers-color-scheme: dark) { .chip[aria-pressed="true"] { color: #0d1117; } }
.count { color: var(--muted); font-size: 12px; margin: 0 0 10px; }
svg { width: 100%; height: auto; display: block; overflow: visible; }
.axis line, .axis path { stroke: var(--grid); }
.axis text { fill: var(--muted); font-size: 10px; }
.mark:hover { stroke: var(--ink); stroke-width: 2; }
table { border-collapse: collapse; width: 100%; font-size: 12px; margin-top: 6px; }
th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th { cursor: pointer; color: var(--muted); font-weight: 600; position: sticky; top: 0; background: var(--panel); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.scroll { max-height: 460px; overflow: auto; }
.legend { display: flex; flex-wrap: wrap; gap: 10px; font-size: 11.5px; color: var(--muted); margin-top: 8px; }
.swatch { width: 9px; height: 9px; border-radius: 2px; display: inline-block; margin-right: 4px; }
.note { color: var(--muted); font-size: 12px; margin: 10px 0 0; max-width: 78ch; }
#tip { position: fixed; pointer-events: none; background: var(--panel); color: var(--ink);
  border: 1px solid var(--grid); border-radius: 6px; padding: 6px 8px; font-size: 11.5px;
  opacity: 0; transition: opacity .1s; box-shadow: 0 2px 10px rgba(0,0,0,.15); z-index: 10; }
select { font: inherit; font-size: 12px; background: var(--panel); color: var(--ink);
  border: 1px solid var(--grid); border-radius: 5px; padding: 3px 6px; width: 100%; }
</style>
</head>
<body>
<header>
  <h1>semdex benchmark explorer</h1>
  <p>Every measured cell. Bars are 95% confidence intervals: where two cells' intervals overlap,
  this query set cannot tell them apart. Cells scored before per-query values were retained have
  no interval and are drawn without bars.</p>
</header>
<main>
  <div class="panel" id="filters"></div>
  <div>
    <div class="panel">
      <p class="count" id="count"></p>
      <svg id="chart" viewBox="0 0 760 420" role="img" aria-label="Retrieval quality by configuration"></svg>
      <div class="legend" id="legend"></div>
      <p class="note" id="axisnote"></p>
    </div>
    <div class="panel" style="margin-top:14px">
      <div class="scroll"><table id="table"></table></div>
    </div>
  </div>
</main>
<div id="tip" role="status"></div>
<script>
const DATA = __DATA__;
const PALETTE = ["--blue","--aqua","--orange","--violet","--green","--magenta","--yellow","--red"];
const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

const AXES = [
  {key:"corpus", label:"Corpus"},
  {key:"strategy", label:"Strategy"},
  {key:"max_tokens", label:"Chunk size"},
  {key:"overlap", label:"Overlap (tokens)"},
  {key:"embedding", label:"Embedding model"},
  {key:"method", label:"Retrieval method"},
  {key:"dim", label:"Dimension"},
];
const X_CHOICES = [
  {key:"chunks", label:"chunks produced (cost)"},
  {key:"tokens", label:"tokens embedded (cost)"},
  {key:"dim", label:"embedding dimension"},
  {key:"max_tokens", label:"chunk size"},
  {key:"recall", label:"Recall@10"},
];
const state = {selected:{}, x:"chunks", colour:"embedding"};

const values = key => [...new Set(DATA.cells.map(c => c[key]).filter(v => v !== null && v !== ""))]
  .sort((a,b) => (typeof a === "number" && typeof b === "number") ? a-b : String(a).localeCompare(String(b)));

function colourOf(cell) {
  const keys = values(state.colour);
  return cssVar(PALETTE[keys.indexOf(cell[state.colour]) % PALETTE.length]);
}

function visible() {
  return DATA.cells.filter(c => AXES.every(a => {
    const sel = state.selected[a.key];
    return !sel || sel.size === 0 || sel.has(String(c[a.key]));
  }));
}

function buildFilters() {
  const host = document.getElementById("filters");
  host.innerHTML = "";
  const xf = document.createElement("fieldset");
  xf.innerHTML = '<legend>Horizontal axis</legend>';
  const sel = document.createElement("select");
  X_CHOICES.forEach(o => { const opt = document.createElement("option");
    opt.value = o.key; opt.textContent = o.label; sel.appendChild(opt); });
  sel.value = state.x;
  sel.addEventListener("change", () => { state.x = sel.value; render(); });
  xf.appendChild(sel); host.appendChild(xf);

  const cf = document.createElement("fieldset");
  cf.innerHTML = '<legend>Colour by</legend>';
  const csel = document.createElement("select");
  AXES.forEach(a => { const opt = document.createElement("option");
    opt.value = a.key; opt.textContent = a.label; csel.appendChild(opt); });
  csel.value = state.colour;
  csel.addEventListener("change", () => { state.colour = csel.value; render(); });
  cf.appendChild(csel); host.appendChild(cf);

  AXES.forEach(a => {
    const fs = document.createElement("fieldset");
    const lg = document.createElement("legend"); lg.textContent = a.label; fs.appendChild(lg);
    const wrap = document.createElement("div"); wrap.className = "chips";
    values(a.key).forEach(v => {
      const b = document.createElement("button");
      b.className = "chip"; b.type = "button"; b.textContent = String(v);
      b.setAttribute("aria-pressed", "false");
      b.addEventListener("click", () => {
        state.selected[a.key] = state.selected[a.key] || new Set();
        const set = state.selected[a.key];
        if (set.has(String(v))) { set.delete(String(v)); b.setAttribute("aria-pressed","false"); }
        else { set.add(String(v)); b.setAttribute("aria-pressed","true"); }
        render();
      });
      wrap.appendChild(b);
    });
    fs.appendChild(wrap); host.appendChild(fs);
  });
}

function render() {
  const rows = visible();
  document.getElementById("count").textContent =
    rows.length + " of " + DATA.cells.length + " cells shown" +
    " (" + rows.filter(r => r.lo != null).length + " with intervals)";
  drawChart(rows);
  drawLegend(rows);
  drawTable(rows);
  document.getElementById("axisnote").textContent =
    "Vertical axis is nDCG@10. Horizontal axis is " +
    (X_CHOICES.find(o => o.key === state.x) || {}).label +
    ". A cost axis makes the trade visible: up and to the left is better.";
}

function drawChart(rows) {
  const svg = document.getElementById("chart");
  const W = 760, H = 420, m = {t:12, r:14, b:38, l:46};
  const pts = rows.filter(r => r.ndcg != null && r[state.x] != null);
  svg.innerHTML = "";
  if (!pts.length) { svg.innerHTML = '<text x="24" y="40" fill="' + cssVar("--muted") +
    '" font-size="13">No cells match these filters.</text>'; return; }
  const xs = pts.map(p => p[state.x]);
  const logx = state.x === "chunks" || state.x === "tokens";
  const fx = v => logx ? Math.log10(Math.max(v, 1)) : v;
  const x0 = Math.min(...xs.map(fx)), x1 = Math.max(...xs.map(fx));
  const ys = pts.flatMap(p => [p.lo ?? p.ndcg, p.hi ?? p.ndcg]);
  const y0 = Math.max(0, Math.min(...ys) - 0.03), y1 = Math.min(1, Math.max(...ys) + 0.03);
  const px = v => m.l + (x1 === x0 ? 0.5 : (fx(v) - x0) / (x1 - x0)) * (W - m.l - m.r);
  const py = v => H - m.b - (y1 === y0 ? 0.5 : (v - y0) / (y1 - y0)) * (H - m.t - m.b);
  const ns = "http://www.w3.org/2000/svg";
  const add = (tag, attrs, parent) => { const e = document.createElementNS(ns, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]); (parent || svg).appendChild(e); return e; };

  for (let i = 0; i <= 4; i++) {
    const v = y0 + (y1 - y0) * i / 4;
    add("line", {x1:m.l, x2:W-m.r, y1:py(v), y2:py(v), stroke:cssVar("--grid")});
    add("text", {x:m.l-7, y:py(v)+3, "text-anchor":"end", fill:cssVar("--muted"), "font-size":10})
      .textContent = v.toFixed(2);
  }
  for (let i = 0; i <= 4; i++) {
    const t = x0 + (x1 - x0) * i / 4;
    const raw = logx ? Math.pow(10, t) : t;
    add("text", {x:m.l + i/4*(W-m.l-m.r), y:H-m.b+16, "text-anchor":"middle",
      fill:cssVar("--muted"), "font-size":10})
      .textContent = raw >= 1000 ? Math.round(raw/1000) + "k" : (Math.round(raw*100)/100);
  }
  add("text", {x:14, y:H/2, fill:cssVar("--muted"), "font-size":11,
    transform:"rotate(-90 14 " + (H/2) + ")", "text-anchor":"middle"}).textContent = "nDCG@10";

  const tip = document.getElementById("tip");
  pts.forEach(p => {
    const c = colourOf(p);
    if (p.lo != null) add("line", {x1:px(p[state.x]), x2:px(p[state.x]), y1:py(p.lo), y2:py(p.hi),
      stroke:c, "stroke-width":1.6, opacity:.55});
    const dot = add("circle", {cx:px(p[state.x]), cy:py(p.ndcg), r:4.2, fill:c,
      class:"mark", "stroke-width":0});
    dot.addEventListener("mousemove", ev => {
      tip.style.opacity = 1;
      tip.style.left = (ev.clientX + 14) + "px";
      tip.style.top = (ev.clientY + 14) + "px";
      tip.innerHTML = "<b>" + p.embedding + "</b> " + p.dim + "d<br>" + p.profile +
        "<br>" + p.corpus + "<br>nDCG@10 " + p.ndcg.toFixed(4) +
        (p.lo != null ? " [" + p.lo.toFixed(3) + ", " + p.hi.toFixed(3) + "]" : " (no interval)") +
        "<br>" + p.n + " queries";
    });
    dot.addEventListener("mouseleave", () => { tip.style.opacity = 0; });
  });
}

function drawLegend(rows) {
  const host = document.getElementById("legend");
  const keys = values(state.colour).filter(v => rows.some(r => r[state.colour] === v));
  host.innerHTML = keys.map(v => '<span><span class="swatch" style="background:' +
    cssVar(PALETTE[values(state.colour).indexOf(v) % PALETTE.length]) + '"></span>' + v + "</span>").join("");
}

let sortKey = "ndcg", sortDir = -1;
function drawTable(rows) {
  const cols = [
    {k:"corpus", l:"Corpus"}, {k:"profile", l:"Profile"}, {k:"embedding", l:"Model"},
    {k:"dim", l:"Dim", n:1}, {k:"ndcg", l:"nDCG@10", n:1}, {k:"recall", l:"Recall@10", n:1},
    {k:"mrr", l:"MRR", n:1}, {k:"chunks", l:"Chunks", n:1}, {k:"n", l:"Queries", n:1},
  ];
  const sorted = [...rows].sort((a,b) => {
    const x = a[sortKey], y = b[sortKey];
    if (x == null) return 1;
    if (y == null) return -1;
    return (typeof x === "number" ? x - y : String(x).localeCompare(String(y))) * sortDir;
  });
  const t = document.getElementById("table");
  t.innerHTML = "<thead><tr>" + cols.map(c =>
    '<th class="' + (c.n ? "num" : "") + '" data-k="' + c.k + '">' + c.l +
    (sortKey === c.k ? (sortDir < 0 ? " v" : " ^") : "") + "</th>").join("") + "</tr></thead><tbody>" +
    sorted.slice(0, 400).map(r => "<tr>" + cols.map(c => {
      let v = r[c.k];
      if (c.k === "ndcg" && v != null) {
        v = v.toFixed(4) + (r.lo != null ? " [" + r.lo.toFixed(3) + ", " + r.hi.toFixed(3) + "]" : "");
      } else if (typeof v === "number" && !Number.isInteger(v)) { v = v.toFixed(4); }
      return '<td class="' + (c.n ? "num" : "") + '">' + (v == null ? "-" : v) + "</td>";
    }).join("") + "</tr>").join("") + "</tbody>";
  t.querySelectorAll("th").forEach(th => th.addEventListener("click", () => {
    const k = th.dataset.k;
    if (sortKey === k) { sortDir = -sortDir; } else { sortKey = k; sortDir = -1; }
    drawTable(visible());
  }));
}

buildFilters();
render();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
</script>
</body>
</html>
"""


def main() -> None:
    data = collect()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    # json.dumps with no spaces keeps the page small; </ is escaped so a string in the data can
    # never close the script tag early.
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    _OUT.write_text(_PAGE.replace("__DATA__", payload))
    size_kb = _OUT.stat().st_size / 1024
    print(f"wrote {_OUT} ({size_kb:.0f} KB, {len(data['cells'])} cells)")


if __name__ == "__main__":
    main()
