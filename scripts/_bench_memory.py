"""Peak resident memory of one workload, measured in a process that ran nothing else.

Disk size is published for every store; resident memory for none, so "does this fit in an 8 GB
container" has no answer here. This is the measuring end of that.

The whole module exists because of one property: **peak RSS is monotonic within a process**. The
kernel's high-water mark never falls, so measuring five stores in one process reports, for each,
the maximum of itself and everything measured before it. The numbers come out ordered, plausible,
and wrong - and wrong in the direction that makes whichever ran first look best. Every cell
therefore runs in its own interpreter, and the parent only collects.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# A bare interpreter already holds several MB before it does anything. Reporting it as part of a
# model's footprint would add a constant to every row and flatter the heavy ones by comparison.
_BASELINE_PROBE = "import sys; sys.exit(0)"


# Injectable so a test can hand in a status file with known values. Which FIELD each function
# reads is the whole contract here - VmHWM is the high-water mark and VmRSS the present - and
# testing that through real allocations would rest on the allocator returning pages to the
# kernel, which is not a guarantee and made the test flake once under memory pressure.
STATUS_PATH = Path("/proc/self/status")


def _proc_status_kb(field: str, *, source: Path = STATUS_PATH) -> int | None:
    """One /proc status field in kB, or None where the file does not exist.

    Read from /proc rather than resource.getrusage because ru_maxrss is kB on Linux and BYTES on
    macOS, a units trap that silently reports a 1024x difference.
    """
    try:
        text = source.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith(f"{field}:"):
            return int(line.split()[1])
    return None


def _getrusage_peak_kb() -> int | None:
    """The high-water mark from getrusage, in kB, or None where there is no ``resource`` module.

    ru_maxrss is kB on Linux and BYTES on macOS. Windows has no ``resource`` module at all.
    """
    try:
        import resource
    except ImportError:
        return None
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw // 1024 if sys.platform == "darwin" else raw


def can_measure(*, source: Path = STATUS_PATH) -> bool:
    """Whether this platform yields a real peak at all (/proc or getrusage; Windows has neither)."""
    return _proc_status_kb("VmHWM", source=source) is not None or _getrusage_peak_kb() is not None


def peak_rss_mb(*, source: Path = STATUS_PATH) -> float:
    """High-water resident memory of THIS process, in MB. 0.0 where it cannot be read."""
    kb = _proc_status_kb("VmHWM", source=source)
    if kb is None:
        kb = _getrusage_peak_kb()
    return round(kb / 1024, 1) if kb is not None else 0.0


def current_rss_mb(*, source: Path = STATUS_PATH) -> float:
    """Resident memory right NOW, in MB. Falls back to the peak where /proc is unavailable.

    Distinct from :func:`peak_rss_mb` on purpose: the serving figure has to be what the process
    holds while answering queries, not the high-water mark it touched while loading.
    """
    kb = _proc_status_kb("VmRSS", source=source)
    return round(kb / 1024, 1) if kb is not None else peak_rss_mb(source=source)


@dataclass(frozen=True, slots=True)
class Measurement:
    """One cell's memory, already net of the interpreter it ran in."""

    peak_mb: float
    resident_mb: float
    baseline_mb: float
    extra: dict[str, Any]

    def as_row(self) -> dict[str, Any]:
        return {
            "peak_mb": round(max(0.0, self.peak_mb - self.baseline_mb), 1),
            "resident_mb": round(max(0.0, self.resident_mb - self.baseline_mb), 1),
            "peak_mb_gross": self.peak_mb,
            "baseline_mb": self.baseline_mb,
            **self.extra,
        }


def interpreter_baseline_mb() -> float:
    """Peak RSS of an interpreter that did nothing, measured the same way as every cell.

    Measured rather than assumed: it moves with the Python version and the build, and it is a
    large share of the smallest rows here.
    """
    probe = "import sys, pathlib\n" + _READ_PEAK + "\nprint(peak())\n"
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, encoding="utf-8", errors="replace", check=True
    )
    return float(out.stdout.strip())


# The same order as peak_rss_mb (/proc, then getrusage), written inline so the probe interpreter
# imports nothing beyond sys and pathlib: importing this module would add its own imports to the
# baseline it exists to measure.
_READ_PEAK = """
def peak():
    try:
        text = pathlib.Path("/proc/self/status").read_text()
    except OSError:
        text = ""
    for line in text.splitlines():
        if line.startswith("VmHWM:"):
            return round(int(line.split()[1]) / 1024, 1)
    try:
        import resource
    except ImportError:
        return 0.0
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round((raw // 1024 if sys.platform == "darwin" else raw) / 1024, 1)
"""


def run_cell(script: Path, env: dict[str, str], *, timeout: float = 900.0) -> dict[str, Any] | None:
    """Run one measurement in a fresh interpreter and return its JSON line.

    Returns None when the cell fails, so an unavailable backend skips instead of killing a sweep
    that has already paid for the cells before it. The reason is printed, never swallowed.
    """
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, **env},
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        print(f"  SKIP {env}: exit {result.returncode}: {result.stderr.strip().splitlines()[-1:]}", flush=True)
        return None
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    print(f"  SKIP {env}: no JSON on stdout", flush=True)
    return None


__all__ = [
    "STATUS_PATH",
    "Measurement",
    "can_measure",
    "current_rss_mb",
    "interpreter_baseline_mb",
    "peak_rss_mb",
    "run_cell",
]
