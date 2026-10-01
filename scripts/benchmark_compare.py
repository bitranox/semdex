#!/usr/bin/env python3
"""Compare a benchmark run against a committed baseline and flag regressions.

The E2E matrices (``tests/test_e2e_matrix.py``, ``tests/test_e2e_extractor_matrix.py``)
write ``benchmark-results.json`` via ``tests/_benchmark_report.py``. This script diffs
that file against ``tests/benchmarks/baseline.json`` (the known-good reference) and exits
non-zero when a cell regresses, so CI catches a drop introduced by a new semdex version or
a new dependency/container-image version.

A cell regresses when either:

- its status flips ``ok`` -> ``fail`` / ``skip`` (a combination that used to work stopped), or
- a higher-is-better quality metric regresses. Where both runs recorded per-query scores the
  comparison is PAIRED over the shared queries, which is what makes a 20-query slice able to
  detect anything: unpaired, its mean carries a 95 percent half-width around 0.13, so the 0.03
  tolerance could neither see a real 0.10 drop nor be trusted when it fired. Paired, a consistent
  drop past ``--paired-floor`` (default 0.005) is a regression. Cells with no per-query data on
  either side fall back to the unpaired ``--tolerance`` comparison.

Timing/size metrics (suffix ``_ms`` / ``_s`` / ``_mb``) are machine-dependent and noisy, so
they are shown as deltas but never gate. A key present in the baseline but missing from the
current run is reported as "not run", not a regression (a job may cover only some cells). A
key present in the current run but not the baseline is "new" - fold it in with ``--update``.

Usage::

    python scripts/benchmark_compare.py CURRENT.json [--baseline PATH] [--tolerance F]
    python scripts/benchmark_compare.py CURRENT.json --update    # fold current into the baseline

Exit code 0 = no regression, 1 = at least one regression, 2 = usage/IO error.
"""

from __future__ import annotations

import argparse
import sys
from enum import StrEnum
from pathlib import Path
from typing import NoReturn

from pydantic import BaseModel, Field, ValidationError

SCHEMA_VERSION = 1
DEFAULT_BASELINE = Path("tests/benchmarks/baseline.json")
# A metric regresses only past this absolute drop, absorbing run-to-run jitter in the
# stochastic pieces (ANN recall, model non-determinism) without hiding a real fall.
DEFAULT_TOLERANCE = 0.03
# The drop a PAIRED comparison must exceed. Far smaller than the unpaired tolerance because
# pairing removes query difficulty, which is what that tolerance existed to absorb: on a 20-query
# slice the mean carries a 95 percent half-width near 0.13, while the per-query difference between
# two runs of a deterministic slice is zero unless behaviour changed. This is a "did anything
# move" threshold, not a noise floor.
DEFAULT_PAIRED_FLOOR = 0.005
# Suffixes that mark a metric as a reported-only timing/size measurement, never gated.
_REPORTED_SUFFIXES = ("_ms", "_s", "_mb")


class ResultStatus(StrEnum):
    OK = "ok"
    FAIL = "fail"
    SKIP = "skip"
    NA = "n/a"


class BenchmarkResult(BaseModel):
    """One cell, as read back from a results file written by ``tests/_benchmark_report.py``.

    ``per_query`` maps a metric to ``{query id: score}`` and is what lets this script compare a
    run to the baseline query by query. It is optional so that a baseline captured before it
    existed still loads and still gates, through the unpaired path.
    """

    status: ResultStatus
    metrics: dict[str, float] = Field(default_factory=dict)
    per_query: dict[str, dict[str, float]] = Field(default_factory=dict)
    measured_utc: str | None = None


class BenchmarkResults(BaseModel):
    schema_version: int = SCHEMA_VERSION
    generated_utc: str | None = None
    versions: dict[str, object] = Field(default_factory=dict)
    results: dict[str, BenchmarkResult] = Field(default_factory=dict)


def _load(path: Path) -> BenchmarkResults:
    try:
        doc = BenchmarkResults.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _fail(f"no such results file: {path}")
    except ValidationError as exc:
        _fail(f"{path} is not a valid benchmark-results file:\n{exc}")
    if doc.schema_version != SCHEMA_VERSION:
        _fail(f"{path} has schema_version {doc.schema_version}, this tool understands {SCHEMA_VERSION}")
    return doc


def _fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def _is_gated(metric: str) -> bool:
    """A metric gates the comparison unless it is a reported-only timing/size measurement."""
    return not metric.endswith(_REPORTED_SUFFIXES)


def paired_deltas(base: BenchmarkResult, cur: BenchmarkResult, metric: str) -> list[float]:
    """Per-query ``current - baseline`` over the queries BOTH runs scored, or [] if unavailable."""
    base_scores = base.per_query.get(metric) or {}
    cur_scores = cur.per_query.get(metric) or {}
    shared = sorted(set(base_scores) & set(cur_scores))
    return [cur_scores[qid] - base_scores[qid] for qid in shared]


def paired_verdict(deltas: list[float], floor: float) -> tuple[bool, str]:
    """Decide from per-query differences whether a metric regressed, and describe the evidence.

    Two runs of a fixed slice answer the SAME queries, so the comparison can be made query by
    query. That removes query difficulty, which is nearly all of the variance in a retrieval
    metric and the reason a 20-query mean carries a 95 percent half-width around 0.13 - four
    times the tolerance this gate fires on. Paired, the same slice resolves a consistent drop of
    a few thousandths.

    A regression needs BOTH: every-query evidence that the drop is real (no query improved by
    more than the mean fell, expressed as the sign of the whole set) and a mean drop past
    ``floor``. The floor is small precisely because the pairing removed the noise it used to
    have to absorb.
    """
    if not deltas:
        return False, "no paired data"
    mean = sum(deltas) / len(deltas)
    worse = sum(1 for d in deltas if d < 0)
    better = sum(1 for d in deltas if d > 0)
    if mean >= -floor:
        return False, f"paired mean {mean:+.4f} over {len(deltas)} queries ({worse} worse, {better} better)"
    # A drop carried by a single query on a small slice is a query that changed, not a trend.
    if worse <= better:
        return False, f"paired mean {mean:+.4f} but {better} queries improved against {worse} worse"
    return True, f"paired mean {mean:+.4f} over {len(deltas)} queries ({worse} worse, {better} better)"


def _metric_regressions(combo: str, base: BenchmarkResult, cur: BenchmarkResult, tol: float, floor: float) -> list[str]:
    """Gated metrics that regressed, paired per query where both runs recorded it."""
    problems: list[str] = []
    for metric, base_value in base.metrics.items():
        if not _is_gated(metric):
            continue
        cur_value = cur.metrics.get(metric)
        if cur_value is None:
            continue  # metric no longer emitted - not itself a quality regression
        deltas = paired_deltas(base, cur, metric)
        if deltas:
            regressed, evidence = paired_verdict(deltas, floor)
            if regressed:
                problems.append(f"{combo}: {metric} {base_value:.4f} -> {cur_value:.4f} paired, {evidence}")
            continue
        # No per-query data on one side: fall back to the far blunter mean comparison, which is
        # all the older baselines support.
        if cur_value < base_value - tol:
            problems.append(f"{combo}: {metric} {base_value:.4f} -> {cur_value:.4f} (unpaired drop > {tol})")
    return problems


class Comparison(BaseModel):
    """The outcome of diffing a current run against the baseline."""

    regressions: list[str] = Field(default_factory=list)
    not_run: list[str] = Field(default_factory=list)
    new: list[str] = Field(default_factory=list)
    ok: list[str] = Field(default_factory=list)


def compare(
    baseline: BenchmarkResults, current: BenchmarkResults, tolerance: float, floor: float = DEFAULT_PAIRED_FLOOR
) -> Comparison:
    """Diff current vs baseline into regressions / not-run / new / ok buckets."""
    out = Comparison()
    for combo, base in baseline.results.items():
        cur = current.results.get(combo)
        if cur is None:
            out.not_run.append(combo)
            continue
        if base.status is ResultStatus.OK and cur.status is not ResultStatus.OK:
            out.regressions.append(f"{combo}: status {base.status.value} -> {cur.status.value}")
            continue
        drops = _metric_regressions(combo, base, cur, tolerance, floor)
        (out.regressions if drops else out.ok).extend(drops or [combo])
    out.new = [combo for combo in current.results if combo not in baseline.results]
    return out


def merge_baseline(baseline: BenchmarkResults, current: BenchmarkResults) -> tuple[BenchmarkResults, str]:
    """Fold a run into the baseline per cell, keeping cells this run did not cover.

    The baseline is deliberately a superset of any one run: the comparator reports a baseline
    cell missing from the current run as "not run" precisely because a job may cover only part
    of the matrix. CI, for instance, measures one corpus and two extractors out of four and
    eight. Replacing the file wholesale would therefore let an update from a partial run delete
    the coverage it did not measure, and the next comparison would pass because it had stopped
    looking - a green that means less checking rather than no regression.

    A cell that genuinely goes away lingers here and keeps reporting "not run", which is visible
    and harmless. Losing coverage silently is neither, so the accounting is printed.
    """
    # Stamp each cell with the run that produced it. A merged baseline holds cells of several
    # vintages while the file's own `versions` block describes only the latest run, so without
    # this a regression in a kept cell would be attributed to dependency versions it was never
    # measured under.
    results = {
        combo: cell if cell.measured_utc else cell.model_copy(update={"measured_utc": baseline.generated_utc})
        for combo, cell in baseline.results.items()
    }
    updated = sum(1 for combo in current.results if combo in results)
    results.update(
        {
            combo: cell.model_copy(update={"measured_utc": current.generated_utc})
            for combo, cell in current.results.items()
        }
    )
    kept = len(results) - len(current.results)
    merged = current.model_copy(update={"results": results})
    summary = (
        f"cells: {len(current.results) - updated} new, {updated} updated, "
        f"{kept} kept from earlier runs, {len(results)} total"
    )
    return merged, summary


def _load_or_empty(path: Path) -> BenchmarkResults:
    """The baseline, or an empty one when it does not exist yet (first capture)."""
    return _load(path) if path.exists() else BenchmarkResults()


def _render(cmp: Comparison) -> str:
    lines = ["# Benchmark regression check", ""]
    lines.append(f"- ok:          {len(cmp.ok)}")
    lines.append(f"- regressions: {len(cmp.regressions)}")
    lines.append(f"- not run:     {len(cmp.not_run)}")
    lines.append(f"- new:         {len(cmp.new)}")
    for title, items in (("Regressions", cmp.regressions), ("New cells", cmp.new), ("Not run", cmp.not_run)):
        if items:
            lines += ["", f"## {title}", *(f"- {item}" for item in items)]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare a benchmark run against the committed baseline.")
    parser.add_argument("current", type=Path, help="benchmark-results.json from this run")
    parser.add_argument(
        "--baseline", type=Path, default=DEFAULT_BASELINE, help=f"baseline (default {DEFAULT_BASELINE})"
    )
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="max quality drop before regression")
    parser.add_argument(
        "--paired-floor",
        type=float,
        default=DEFAULT_PAIRED_FLOOR,
        help="max paired per-query mean drop before regression (used when per-query data exists)",
    )
    parser.add_argument("--update", action="store_true", help="adopt current as the new baseline and exit 0")
    args = parser.parse_args(argv)

    current = _load(args.current)
    if args.update:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        merged, summary = merge_baseline(_load_or_empty(args.baseline), current)
        args.baseline.write_text(merged.model_dump_json(indent=2), encoding="utf-8")
        print(f"baseline updated from {args.current} -> {args.baseline}")
        print(summary)
        return 0

    baseline = _load(args.baseline)
    cmp = compare(baseline, current, args.tolerance, args.paired_floor)
    print(_render(cmp))
    return 1 if cmp.regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
