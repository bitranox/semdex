#!/usr/bin/env python
"""Does the "queries needed" prediction actually work? Tested on a holdout, not asserted.

`scripts/score_query_power.py` predicts how many queries an unresolved comparison needs, from
``n x (half_width / effect)^2``. Gap #12 then uses it to say a German corpus of about 1,500 would
settle a bit over half the current ties. Building that corpus on the strength of an untested
formula would be the expensive way to find out it is wrong.

It can be tested for free. The English slice of the same corpus family runs the SAME 373
comparisons at 800 queries. Subsample it to 200 - the size of the German set - predict from that
subsample which comparisons would resolve by 800, then check against the real 800-query answer.
That is a genuine holdout: the prediction never sees the data it is judged against.

Two things come out of it:

  scaling     does the paired half-width really fall as 1/sqrt(n)? The whole formula rests on it,
              and a bootstrap over a skewed per-query distribution need not obey it.
  prediction  of the comparisons unresolved at 200, how many of those predicted to resolve by 800
              actually did, and how many resolved that were not predicted to.

Env: SUBSAMPLE (default 200), SEEDS (default 5), CORPUS (default mldr_en_8k_slice), OUT.
"""

# pyright: basic
# One-off validation harness over committed JSON plus the cached per-query arrays.

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _query_power import queries_to_resolve
from _score_stats import paired_ci

_RAW = Path(__file__).resolve().parents[1] / "tests" / "benchmarks" / "raw"
_PERQUERY = Path(os.environ.get("SEMDEX_SCORE_PERQUERY", "/embeddings/scores/perquery"))
METRIC = "ndcg@10"

# The cached arrays come in two vintages: older cells store the metric as "ndcg" and newer ones as
# "ndcg@10". They are the same measurement - every cached array's mean reproduces its committed
# cell score to four decimals under either name - so the older key is read rather than skipped,
# which would have dropped 83 of the 99 cells this validation needs.
_METRIC_ALIASES = ("ndcg@10", "ndcg")


def load_cell(cell: str) -> dict[str, float] | None:
    """Per-query metric for one swept cell, keyed by query id."""
    path = _PERQUERY / f"{cell}.npz"
    if not path.exists():
        return None
    # No allow_pickle: these hold only string and float arrays, so the unpickling code path is
    # not needed and is not opened.
    with np.load(path) as data:
        key = next((name for name in _METRIC_ALIASES if name in data), None)
        if key is None:
            return None
        return dict(zip([str(q) for q in data["qids"]], [float(v) for v in data[key]], strict=True))


def _number(value: object) -> float:
    """Narrow one field of paired_ci's return to a float.

    It returns ``dict[str, object]`` because it mixes interval bounds, win/loss counts and a bool,
    so every field arrives untyped at a call site. Checked at runtime rather than cast: a cast
    would keep type-checking silently if that helper's shape ever changed underneath this one.
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"expected a number from paired_ci, got {type(value).__name__}")
    return float(value)


def resolved_at(left: dict[str, float], right: dict[str, float], qids: list[str]) -> dict[str, Any]:
    """Paired verdict over exactly ``qids``, so a subsample is scored the same way as the full set."""
    verdict = paired_ci({q: left[q] for q in qids}, {q: right[q] for q in qids})
    return {
        "mean_delta": _number(verdict["mean_delta"]),
        "half_width": (_number(verdict["ci_hi"]) - _number(verdict["ci_lo"])) / 2,
        "resolved": bool(verdict["resolved"]),
    }


def main() -> None:
    corpus = os.environ.get("CORPUS", "mldr_en_8k_slice")
    size = int(os.environ.get("SUBSAMPLE", "200"))
    seeds = int(os.environ.get("SEEDS", "5"))
    effects = [
        e
        for e in json.loads((_RAW / "chunk-knob-effects.json").read_text())["effects"]
        if e["corpus"] == corpus and e["metric"] == METRIC
    ]
    print(f"{len(effects)} comparisons on {corpus}; subsampling {size} of its queries, {seeds} seeds", flush=True)

    cells: dict[str, dict[str, float]] = {}
    pairs: list[tuple[dict[str, float], dict[str, float]]] = []
    for effect in effects:
        for name in (effect["from_cell"], effect["to_cell"]):
            if name not in cells:
                loaded = load_cell(name)
                if loaded is not None:
                    cells[name] = loaded
        if effect["from_cell"] in cells and effect["to_cell"] in cells:
            pairs.append((cells[effect["to_cell"]], cells[effect["from_cell"]]))
    print(f"  {len(pairs)}/{len(effects)} comparisons have cached per-query scores", flush=True)
    if not pairs:
        print("  nothing to validate", flush=True)
        return

    all_qids = sorted(set(pairs[0][0]) & set(pairs[0][1]))
    full = [resolved_at(left, right, all_qids) for left, right in pairs]
    print(f"  full set: {len(all_qids)} queries, {sum(1 for f in full if f['resolved'])} resolved", flush=True)

    rounds = [_one_seed(pairs, full, all_qids, size, seed) for seed in range(seeds)]
    for seed, row in enumerate(rounds):
        print(
            f"  seed {seed}: half-width ratio {row['half_width_ratio']:.2f} (expect "
            f"{(len(all_qids) / size) ** 0.5:.2f})  predicted {row['predicted_resolve']} of "
            f"{row['unresolved_at_subsample']} ties resolve, actual {row['actually_resolved']}",
            flush=True,
        )

    payload = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "corpus": corpus,
        "metric": METRIC,
        "full_queries": len(all_qids),
        "subsample_queries": size,
        "expected_half_width_ratio": round((len(all_qids) / size) ** 0.5, 3),
        "comparisons": len(pairs),
        "note": "A holdout test of the queries-needed formula: predict from a subsample of the "
        "size of the German set, check against the full set the prediction never saw.",
        "rounds": rounds,
    }
    out_path = Path(os.environ.get("OUT", "query-power-validation.json"))
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}", flush=True)


def _one_seed(
    pairs: list[tuple[dict[str, float], dict[str, float]]],
    full: list[dict[str, Any]],
    all_qids: list[str],
    size: int,
    seed: int,
) -> dict[str, Any]:
    """One subsample: predict from ``size`` queries, score against the full set."""
    rng = np.random.default_rng(seed)
    qids = [all_qids[i] for i in sorted(rng.choice(len(all_qids), size=size, replace=False))]
    small = [resolved_at(left, right, qids) for left, right in pairs]

    ratios = [s["half_width"] / f["half_width"] for s, f in zip(small, full, strict=True) if f["half_width"] > 0]
    predicted_hits = 0
    predicted_total = 0
    actual_total = 0
    missed = 0
    for s, f in zip(small, full, strict=True):
        if s["resolved"]:
            continue  # only the ties are being predicted about
        needed = queries_to_resolve(half_width=s["half_width"], mean_delta=s["mean_delta"], queries=size)
        predicted = needed <= len(all_qids)
        predicted_total += int(predicted)
        actual_total += int(f["resolved"])
        predicted_hits += int(predicted and f["resolved"])
        missed += int(f["resolved"] and not predicted)
    return {
        "seed": seed,
        "half_width_ratio": round(float(np.median(ratios)), 3),
        "unresolved_at_subsample": sum(1 for s in small if not s["resolved"]),
        "predicted_resolve": predicted_total,
        "actually_resolved": actual_total,
        "predicted_and_did": predicted_hits,
        "resolved_but_not_predicted": missed,
        "precision": round(predicted_hits / predicted_total, 3) if predicted_total else None,
        "recall": round(predicted_hits / actual_total, 3) if actual_total else None,
    }


if __name__ == "__main__":
    main()
