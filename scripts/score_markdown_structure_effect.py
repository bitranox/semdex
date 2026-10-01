#!/usr/bin/env python
# pyright: basic
# Benchmark harness on numpy (no strict stubs); strict mode would only add reportUnknown* noise.
# Same stance as score_query_length_effect.py.
"""Does marking the headings change what each chunk strategy retrieves? Marked against unmarked.

``mldr_en_8k_md_slice`` is ``mldr_en_8k_slice`` with its bare-line section headings prefixed
``## `` (scripts/build_mldr_markdown_slice.py): the same documents, the same queries and qrels.
The exporter pairs cells within ONE corpus, so this script pairs each marked cell with its
unmarked twin across the two corpus ids, per strategy and embedder, and runs the shared paired
bootstrap over the per-query nDCG@10 arrays the scorer wrote. Positive means the marked corpus won.

Beside each pair it carries what the markers did to the chunks (count and median size of both
cells from the chunk-dimension audit, and the share of marked chunks that begin at a heading) and
the heading rule's reviewed precision (``heading-sample-judged.txt`` beside the slice, rows the
reviewer prefixed ``X<TAB>`` are wrong). A missing twin is reported by name, never skipped in
silence; a pair whose query id sets differ is refused. Provenance is carried from the scorer's
stamps, never taken from this machine.

Env:
  CACHE_ROOT           cache dir (default /embeddings)
  MDSTRUCT_MARKED      marked corpus id (default mldr_en_8k_md_slice)
  MDSTRUCT_UNMARKED    unmarked twin (default mldr_en_8k_slice)
  MDSTRUCT_STRATEGIES  comma list (default markdown,recursive,fast,semantic)
  MDSTRUCT_EMBEDDINGS  comma list of embedder labels (default the six the strategy tables carry)
  MDSTRUCT_SLICE_DIR   marked slice dir (default /corpora/mldr-slices/mldr_en_8k_md_slice)
  MDSTRUCT_AUDIT       chunk-dimension audit (default tests/benchmarks/raw/chunk-dimension-audit.json)
  OUT                  results json (default tests/benchmarks/raw/markdown-structure-effect.json)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _provenance import MEASURED_ON
from _score_stats import paired_ci
from export_bench_raw import measured_on_summary

_ROOT = Path(__file__).resolve().parent.parent
_PROFILE = "{strategy}-t256-o0-gpt2"
_METRIC_KEYS = ("ndcg", "ndcg@10")
_MARK = "## "
_DEFAULT_EMBEDDINGS = (
    "model2vec:potion-retrieval-32M",
    "model2vec:potion-base-8M",
    "fastembed:bge-base",
    "ollama:bge-m3",
    "ollama:qwen3-embedding-4b",
    "ollama:qwen3-embedding-8b",
)

__all__ = ["heading_start_share", "judged_wrong", "main", "pair_cells"]


def _cache_root() -> Path:
    return Path(os.environ.get("CACHE_ROOT", "/embeddings"))


def _dirsafe(label: str) -> str:
    return label.replace(":", "-").replace("/", "-")


def _cell(corpus: str, strategy: str, label: str) -> str:
    return f"{corpus}__{_PROFILE.format(strategy=strategy)}__{_dirsafe(label)}"


def _per_query(perquery_dir: Path, cell: str) -> dict[str, float] | None:
    path = perquery_dir / f"{cell}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        key = next((k for k in _METRIC_KEYS if k in data), None)
        if key is None:
            raise KeyError(f"{path} holds no nDCG array: {list(data)}")
        return {str(q): float(v) for q, v in zip(data["qids"], data[key], strict=True)}


def _round_paired(paired: dict[str, Any]) -> dict[str, Any]:
    out = dict(paired)
    for key in ("mean_delta", "ci_lo", "ci_hi"):
        out[key] = round(float(out[key]), 4)
    return out


def pair_cells(
    perquery_dir: Path,
    *,
    strategy: str,
    label: str,
    marked: str,
    unmarked: str,
) -> dict[str, Any] | None:
    """The paired delta (marked minus unmarked) for one strategy and embedder; None when a twin is unscored."""
    marked_cell, unmarked_cell = _cell(marked, strategy, label), _cell(unmarked, strategy, label)
    left, right = _per_query(perquery_dir, marked_cell), _per_query(perquery_dir, unmarked_cell)
    if left is None or right is None:
        return None
    if set(left) != set(right):
        raise ValueError(f"{marked_cell} against {unmarked_cell}: query ids differ, the twins are not twins")
    return {
        "strategy": strategy,
        "embedding": label,
        "marked_cell": marked_cell,
        "unmarked_cell": unmarked_cell,
        **_round_paired(paired_ci(left, right)),
    }


def heading_start_share(parquet: Path) -> float:
    """The share of chunks whose text begins at a heading mark, read column-wise in row groups."""
    reader = pq.ParquetFile(parquet)
    starts = total = 0
    for batch in reader.iter_batches(columns=["text"], batch_size=65536):
        texts = batch.column("text").to_pylist()
        total += len(texts)
        starts += sum(1 for t in texts if t.startswith(_MARK))
    return round(starts / total, 4) if total else 0.0


def judged_wrong(path: Path) -> tuple[int, int]:
    """(rows the reviewer marked wrong, rows judged) from the review file; (0, 0) when there is none."""
    if not path.exists():
        return 0, 0
    rows = [line for line in path.read_text().splitlines() if line.strip()]
    return sum(1 for line in rows if line.startswith("X\t")), len(rows)


def _missing_cells(perquery_dir: Path, cells: tuple[str, str]) -> list[str]:
    return [cell for cell in cells if not (perquery_dir / f"{cell}.npz").exists()]


def _audit_index(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    doc = json.loads(path.read_text()) if path.exists() else {"chunk_sets": []}
    return {(r["corpus"], r["profile"]): r for r in doc["chunk_sets"]}


def _dimensions(
    row: dict[str, Any],
    audit: dict[tuple[str, str], dict[str, Any]],
    cache: Path,
    marked: str,
    unmarked: str,
) -> dict[str, Any]:
    profile = _PROFILE.format(strategy=row["strategy"])
    m, u = audit.get((marked, profile), {}), audit.get((unmarked, profile), {})
    parquet = cache / "chunks" / f"{marked}__{profile}" / "chunks.parquet"
    return {
        "marked_rows": m.get("rows"),
        "unmarked_rows": u.get("rows"),
        "marked_token_p50": m.get("token_p50"),
        "unmarked_token_p50": u.get("token_p50"),
        "marked_heading_start_share": (heading_start_share(parquet) if parquet.exists() else None),
    }


def _score_rows(cache: Path, corpora: tuple[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((cache / "scores").glob("*_scores.json")):
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            rows.extend(r for r in data.values() if isinstance(r, dict) and r.get("corpus") in corpora)
    return rows


def _slice_facts(slice_dir: Path) -> dict[str, Any]:
    meta_path = slice_dir / "meta.json"
    facts = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    wrong, judged = judged_wrong(slice_dir / "heading-sample-judged.txt")
    return {**facts, "sample_judged_wrong": wrong, "sample_judged": judged}


def main() -> int:
    cache = _cache_root()
    marked = os.environ.get("MDSTRUCT_MARKED", "mldr_en_8k_md_slice")
    unmarked = os.environ.get("MDSTRUCT_UNMARKED", "mldr_en_8k_slice")
    strategies = [s for s in os.environ.get("MDSTRUCT_STRATEGIES", "markdown,recursive,fast,semantic").split(",") if s]
    labels = [e for e in os.environ.get("MDSTRUCT_EMBEDDINGS", ",".join(_DEFAULT_EMBEDDINGS)).split(",") if e]
    slice_dir = Path(os.environ.get("MDSTRUCT_SLICE_DIR", f"/corpora/mldr-slices/{marked}"))
    audit = _audit_index(
        Path(
            os.environ.get(
                "MDSTRUCT_AUDIT",
                _ROOT / "tests/benchmarks/raw/chunk-dimension-audit.json",
            )
        )
    )
    out = Path(os.environ.get("OUT", _ROOT / "tests/benchmarks/raw/markdown-structure-effect.json"))
    perquery_dir = cache / "scores" / "perquery"

    effects: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for strategy in strategies:
        for label in labels:
            row = pair_cells(
                perquery_dir,
                strategy=strategy,
                label=label,
                marked=marked,
                unmarked=unmarked,
            )
            if row is None:
                cells = (
                    _cell(marked, strategy, label),
                    _cell(unmarked, strategy, label),
                )
                missing.append(
                    {
                        "strategy": strategy,
                        "embedding": label,
                        "missing": _missing_cells(perquery_dir, cells),
                    }
                )
                continue
            effects.append({**row, **_dimensions(row, audit, cache, marked, unmarked)})
    payload = {
        MEASURED_ON: measured_on_summary(_score_rows(cache, (marked, unmarked))),
        "note": (
            "Paired per-query nDCG@10 delta (marked minus unmarked) between a cell on the MLDR English "
            "slice with its section headings marked '## ' and the same strategy and embedder on the "
            "unmarked slice: identical documents, queries and qrels, only the markup differs. "
            "resolved=false means the 95 percent interval spans zero. missing names every pair a "
            "twin is unscored for."
        ),
        "slice": _slice_facts(slice_dir),
        "effects": effects,
        "missing": missing,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[mdstruct] {len(effects)} pairs, {len(missing)} missing -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
