#!/usr/bin/env python
"""Do the four reported metrics carry four perspectives, or fewer?

Every ranking table here prints nDCG@10, Recall@10, MRR and P@1 side by side, which implies four
views on a result. The gaps page argues they are closer to one and a half on the long-document
corpora, because those ship about one relevant document per query: Recall@10 can then only be 0
or 1, and nDCG@10 collapses toward MRR since a single relevant document makes the gain vector one
non-zero entry whose position is what MRR already reports.

This measures that instead of arguing it, over the cells already swept, per corpus - because the
whole point is that the answer DEPENDS on the corpus and a single number would hide it.

Three readings, in increasing order of how directly they bear on a decision:
  correlation   do the metrics order the configurations alike?
  disagreement  how often do two metrics pick a different winner from the same pair?
  views         the correlation matrix collapsed to an effective count.

Judgment density (relevant documents per query) is read from the corpora themselves, so the
premise the argument rests on is checked rather than assumed.

Env: OUT (default metric-redundancy.json), SKIP_QRELS=1 to skip the ir_datasets density pass.
"""

# pyright: basic
# One-off benchmark harness over committed JSON plus untyped ir_datasets.

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _metric_agreement import NEAR_DUPLICATE_RHO, correlation_matrix, effective_views, pairwise_disagreement

_RAW = Path(__file__).resolve().parents[1] / "tests" / "benchmarks" / "raw"
_SWEEPS = ("chunk-sweep-mldr.json", "chunk-sweep-beir.json", "chunk-sweep-miracl.json")
METRICS = ("ndcg@10", "recall@10", "mrr", "p@1")

# Which ir_datasets corpus each swept slice was drawn from, so judgment density is read from the
# source rather than inferred from the slice name.
_CORPUS_SOURCE = {
    "nfcorpus": "beir/nfcorpus/test",
    "fiqa": "beir/fiqa/test",
    "cqadupstack": "beir/cqadupstack/programmers",
    "scifact": "beir/scifact/test",
}

# MLDR and the MIRACL slices are built from HuggingFace rather than ir_datasets, and ship their own
# qrels beside the slice. Read those directly: the density of the MLDR slices is the exact premise
# the entry rests on, so inferring it from a sibling corpus would test the wrong thing.
_SLICE_ROOTS = (Path("/corpora/mldr-slices"), Path("/corpora/miracl-slices"))


def load_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for name in _SWEEPS:
        path = _RAW / name
        if not path.exists():
            continue
        cells.extend(json.loads(path.read_text()).get("cells") or [])
    return [c for c in cells if all(c.get(m) is not None for m in METRICS)]


def judged_per_query(corpus: str) -> float | None:
    """Mean relevant documents per query, straight from the corpus.

    The premise the whole entry rests on. If a corpus really does ship about one relevant document
    per query, Recall@10 has almost no room to vary and nDCG@10 has almost nothing to discount,
    which is the mechanism that would make the four columns collapse into fewer.
    """
    local = _slice_qrels(corpus)
    if local is not None:
        return local
    dataset_id = _CORPUS_SOURCE.get(corpus)
    if not dataset_id:
        return None
    try:
        import ir_datasets

        counts: dict[str, int] = {}
        for qrel in ir_datasets.load(dataset_id).qrels_iter():
            if qrel.relevance > 0:
                counts[qrel.query_id] = counts.get(qrel.query_id, 0) + 1
        return round(sum(counts.values()) / len(counts), 2) if counts else None
    except Exception as exc:  # a corpus that is not cached must not end the run
        print(f"  qrels unavailable for {dataset_id}: {type(exc).__name__}", flush=True)
        return None


def _slice_qrels(corpus: str) -> float | None:
    """Density from a locally built slice's own ``qrels.json``, if one exists."""
    for root in _SLICE_ROOTS:
        path = root / corpus / "qrels.json"
        if not path.exists():
            continue
        qrels = json.loads(path.read_text())
        relevant = [sum(1 for score in docs.values() if score > 0) for docs in qrels.values()]
        return round(sum(relevant) / len(relevant), 2) if relevant else None
    return None


def analyse_corpus(corpus: str, cells: list[dict[str, Any]]) -> dict[str, Any]:
    # float() at the boundary: these come out of untyped JSON, where a metric that happens to be
    # 0 or 1 arrives as an int and the column stops satisfying the Sequence[float] the correlation
    # helpers declare.
    columns: dict[str, Sequence[float]] = {metric: [float(c[metric]) for c in cells] for metric in METRICS}
    names, matrix = correlation_matrix(columns)
    pairs: list[dict[str, Any]] = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            verdict = pairwise_disagreement(columns[left], columns[right])
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "spearman": round(float(matrix[names.index(left), names.index(right)]), 4),
                    "comparable_pairs": verdict["pairs"],
                    "disagreements": verdict["disagreements"],
                    "disagreement_rate": round(verdict["rate"], 4),
                }
            )
    recalls = columns["recall@10"]
    return {
        "corpus": corpus,
        "cells": len(cells),
        "n_queries": cells[0].get("n_queries"),
        "effective_views": round(effective_views(matrix), 2),
        "recall_mean": round(sum(recalls) / len(recalls), 4),
        "recall_span": round(max(recalls) - min(recalls), 4),
        "near_duplicate_pairs": sum(1 for p in pairs if abs(p["spearman"]) >= NEAR_DUPLICATE_RHO),
        "pairs": pairs,
    }


def main() -> None:
    cells = load_cells()
    corpora = sorted({c["corpus"] for c in cells})
    print(f"{len(cells)} cells over {len(corpora)} corpora", flush=True)

    rows: list[dict[str, Any]] = []
    for corpus in corpora:
        subset = [c for c in cells if c["corpus"] == corpus]
        if len(subset) < 3:
            print(f"  SKIP {corpus}: {len(subset)} cells, too few to correlate", flush=True)
            continue
        row = analyse_corpus(corpus, subset)
        if not os.environ.get("SKIP_QRELS"):
            row["judged_per_query"] = judged_per_query(corpus)
        rows.append(row)
        print(
            f"  {corpus:24s} {row['cells']:>3} cells  {row['effective_views']:.2f} views  "
            f"recall {row['recall_mean']:.3f} (span {row['recall_span']:.3f})  "
            f"{row['near_duplicate_pairs']}/6 near-duplicate pairs  "
            f"rels/query {row.get('judged_per_query')}",
            flush=True,
        )

    payload = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "metrics": list(METRICS),
        "near_duplicate_rho": NEAR_DUPLICATE_RHO,
        "note": "Measured across the swept configurations PER CORPUS. A metric pair is called "
        "near-duplicate at |rho| >= the threshold above; effective_views is the participation "
        "ratio of the correlation matrix eigenvalues, so 4.0 means four independent views and "
        "1.0 means one.",
        "corpora": rows,
    }
    out_path = Path(os.environ.get("OUT", "metric-redundancy.json"))
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
