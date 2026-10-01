#!/usr/bin/env python
# pyright: basic
# Benchmark harness on fastembed/pyarrow (no strict stubs); strict mode would only add
# reportUnknown* noise. Same stance as audit_chunk_dimensions.py and score_chunk_sweep.py.
"""Decide whether a candidate onnxruntime is safe to raise the cap to.

`pyproject.toml` pins ``onnxruntime>=1.28.0,<1.29.0`` because 1.29.0 embeds 8.5x slower on CPU
(886 vs 104 ms/row on real GerDaLIR chunks with fastembed bge-base). A cap like that rots: the day
upstream fixes it, the cap keeps the project on 1.28.0 anyway, because nobody re-runs an evening of
manual A/B. This makes the re-test one command, so the cap can be re-opened on evidence.

    uv run scripts/bench_embed_stack_ab.py --candidate 1.30.0

It builds one throwaway venv per arm, embeds the SAME real chunks in each, and INTERLEAVES the arms
(A,B,A,B) so the wall clock cannot be confounded with the arm - a sequential all-A-then-all-B run
measures separation, not causation. Real chunks, not synthetic text: a synthetic probe of
fixed-length passages overstates throughput because real chunks vary in length.

Exit codes are format-independent: 0 the candidate is at parity (safe to raise the cap), 1 it is
slower (leave the cap), 2 the arms overlap and the run decided nothing.

Config (env; scripts/ convention, plus the flags below):
  SEMDEX_BENCH_CHUNKS   parquet to sample from (default the GerDaLIR fast-t256-o26 chunk set)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

_BASELINE = "1.28.0"
_MODEL = "BAAI/bge-base-en-v1.5"
_DEFAULT_CHUNKS = "/embeddings/chunks/gerdalir_de_12k_slice__fast-t256-o26-gpt2/chunks.parquet"
_WARMUP = 8

# The worker is written to a file rather than passed with -c: it must run under a DIFFERENT
# interpreter than this script, and keeping it readable is what lets a reader check what was timed.
_WORKER = """
import json, os, sys, time
os.environ.setdefault("FASTEMBED_CACHE_PATH", "/embeddings/.fastembed_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from importlib.metadata import version
from fastembed import TextEmbedding
texts = json.load(open(sys.argv[1], encoding="utf-8"))
warmup = int(sys.argv[2])
emb = TextEmbedding(model_name=sys.argv[3])
list(emb.embed(texts[:warmup]))          # load the ONNX session; one-off, not measured
measured = texts[warmup:]
t0 = time.perf_counter()
rows = len(list(emb.embed(measured)))
elapsed = time.perf_counter() - t0
print(json.dumps({"ms_per_row": elapsed / rows * 1000, "rows": rows,
                  "onnxruntime": version("onnxruntime")}))
"""


def interleaved_plan(arms: Sequence[str], rounds: int) -> list[str]:
    """The run order: A,B,A,B rather than all of A then all of B.

    A sequential block per arm confounds the arm with the wall clock - anything that drifts during
    the run (thermal state, a co-running job) is attributed to whichever arm ran during it.
    """
    return [arm for _ in range(rounds) for arm in arms]


def summarise(
    measurements: Iterable[tuple[str, float]], expected_arms: Sequence[str] | None = None
) -> dict[str, dict[str, float]]:
    """Per-arm mean and spread. An arm with no measurements is an error, never an average."""
    grouped: dict[str, list[float]] = {}
    for arm, ms in measurements:
        grouped.setdefault(arm, []).append(ms)
    for arm in expected_arms or ():
        if not grouped.get(arm):
            raise ValueError(f"arm {arm!r} has no measurements - every run of it failed")
    return {
        arm: {
            "mean_ms": statistics.fmean(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "rounds": len(values),
        }
        for arm, values in grouped.items()
    }


def verdict(*, baseline: dict[str, float], candidate: dict[str, float], tolerance: float) -> str:
    """PARITY when the candidate is not meaningfully worse, SLOWER when the arms are disjoint.

    The middle case is decided by whether the arms actually OVERLAP, not by a second fixed ratio.
    That matters because the honest response to an ambiguous result is to run more rounds, and more
    rounds only ever tightens the spread - against a fixed ratio band it could never change the
    answer, so the advice would have been unfollowable. A candidate whose slowest round still beats
    the baseline's worst is a real separation; anything that interleaves is not yet a result.

    One round per arm carries no spread at all, so disjointness there is meaningless. In that case
    a gross gap is still reported (it is what makes the cheap smoke test useful), but a modest one
    is INCONCLUSIVE rather than a confident claim built on a single reading.
    """
    if candidate["mean_ms"] <= baseline["mean_ms"] * (1 + tolerance):
        return "PARITY"
    single_round = min(baseline["rounds"], candidate["rounds"]) < 2
    if single_round:
        gross = candidate["mean_ms"] >= baseline["mean_ms"] * (1 + 4 * tolerance)
        return "SLOWER" if gross else "INCONCLUSIVE"
    return "SLOWER" if candidate["min_ms"] > baseline["max_ms"] else "INCONCLUSIVE"


def exit_code(result: str) -> int:
    """0 safe to raise the cap, 1 leave it, 2 the run decided nothing."""
    return {"PARITY": 0, "SLOWER": 1, "INCONCLUSIVE": 2}[result]


def _sample(parquet: Path, rows: int) -> list[str]:
    """Real chunk texts, because a synthetic fixed-length probe overstates throughput."""
    import pyarrow.parquet as pq

    batch = next(pq.ParquetFile(parquet).iter_batches(batch_size=rows + _WARMUP, columns=["text"]))
    return batch.column("text").to_pylist()[: rows + _WARMUP]


def _build_venv(work: Path, version: str) -> Path:
    subprocess.run(["uv", "venv", "--python", "3.14", "-q", str(work)], check=True)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "-q",
            "--python",
            str(work / "bin" / "python"),
            "fastembed",
            f"onnxruntime=={version}",
        ],
        check=True,
    )
    return work / "bin" / "python"


def _run_arm(python: Path, worker: Path, sample: Path) -> float:
    out = subprocess.run(
        [str(python), str(worker), str(sample), str(_WARMUP), _MODEL],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return float(json.loads(out.stdout.strip().splitlines()[-1])["ms_per_row"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="onnxruntime version to test")
    parser.add_argument("--baseline", default=_BASELINE, help=f"known-good version (default {_BASELINE})")
    parser.add_argument("--chunks", default=os.environ.get("SEMDEX_BENCH_CHUNKS", _DEFAULT_CHUNKS))
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=0.15)
    args = parser.parse_args(argv)

    parquet = Path(args.chunks)
    if not parquet.exists():
        print(f"chunk corpus not found: {parquet}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="embed-stack-ab-") as tmp:
        work = Path(tmp)
        sample_path = work / "sample.json"
        sample_path.write_text(json.dumps(_sample(parquet, args.rows)), encoding="utf-8")
        worker = work / "worker.py"
        worker.write_text(_WORKER, encoding="utf-8")

        pythons = {
            "baseline": _build_venv(work / "venv_baseline", args.baseline),
            "candidate": _build_venv(work / "venv_candidate", args.candidate),
        }
        measurements: list[tuple[str, float]] = []
        for arm in interleaved_plan(["baseline", "candidate"], args.rounds):
            measurements.append((arm, _run_arm(pythons[arm], worker, sample_path)))
            print(f"[ab] {arm}: {measurements[-1][1]:.1f} ms/row", flush=True)

    stats = summarise(measurements, expected_arms=("baseline", "candidate"))
    result = verdict(baseline=stats["baseline"], candidate=stats["candidate"], tolerance=args.tolerance)
    payload: dict[str, Any] = {
        "verdict": result,
        "baseline_version": args.baseline,
        "candidate_version": args.candidate,
        "rows_per_round": args.rows,
        "stats": stats,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
