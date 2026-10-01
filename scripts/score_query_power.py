#!/usr/bin/env python
"""How many queries would it take to resolve the comparisons that are currently ties?

The gaps page says the German results rest on small query sets, that several comparisons are
unresolved for that reason alone, and that no further computation changes it. The last part is
right about the INTERVAL and wrong about the question: what a reader needs to know is whether more
queries would buy anything, and that is computable from the comparisons already made.

A paired bootstrap half-width shrinks as 1/sqrt(n), so a comparison with observed effect ``d`` and
half-width ``h`` at ``n`` queries needs about ``n * (h/d)^2`` to separate. Applied to every
unresolved comparison this turns "underpowered" into a number, and separates two cases that the
word hides:

* a real effect too small to see at this n, which more queries WOULD resolve;
* an effect of essentially zero, which no query set resolves because there is nothing to find -
  those comparisons are answered already, and the answer is "these are the same".

The English slice of the same corpus family is the control. It runs the SAME comparisons at four
times the queries, so if query count were the whole story its unresolved rate would be far lower.

Env: OUT (default query-power.json), METRIC (default ndcg@10).
"""

# pyright: basic
# One-off benchmark harness over committed JSON.

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _query_power import queries_to_resolve, summarise_corpus

_RAW = Path(__file__).resolve().parents[1] / "tests" / "benchmarks" / "raw"

# Query counts a reader could plausibly reach: roughly the current sets, a doubling, and the
# largest German IR sets that exist. Beyond the last one the answer is "no dataset will do it".
LADDER = (500, 1_000, 2_000, 5_000, 20_000, 100_000)


def main() -> None:
    metric = os.environ.get("METRIC", "ndcg@10")
    effects = [
        e for e in json.loads((_RAW / "chunk-knob-effects.json").read_text())["effects"] if e["metric"] == metric
    ]
    corpora = sorted({e["corpus"] for e in effects})
    print(f"{len(effects)} paired comparisons on {metric} over {len(corpora)} corpora", flush=True)

    rows: list[dict[str, Any]] = []
    for corpus in corpora:
        subset = [e for e in effects if e["corpus"] == corpus]
        row = summarise_corpus(corpus, subset, ladder=LADDER)
        rows.append(row)
        print(
            f"  {corpus:22s} n={row['queries']:<5} {row['unresolved']:>3}/{row['comparisons']:<3} unresolved "
            f"({row['unresolved_share'] * 100:4.1f}%)  median needed "
            f"{row['median_required_queries']:,}  never {row['never_resolvable']}",
            flush=True,
        )

    payload = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "metric": metric,
        "ladder": list(LADDER),
        "note": "required_queries = n * (half_width / |mean_delta|)^2, from the paired bootstrap "
        "half-width shrinking as 1/sqrt(n). It assumes the OBSERVED effect is the true one, which "
        "for a comparison that came out unresolved is optimistic - a noisy estimate of a small "
        "effect lands above its true size as often as below, and the ones that look nearly "
        "resolvable are the ones most likely to be overstated. Read these as a floor.",
        "corpora": rows,
    }
    out_path = Path(os.environ.get("OUT", "query-power.json"))
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()


__all__ = ["LADDER", "queries_to_resolve"]
