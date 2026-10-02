#!/usr/bin/env python
# pyright: basic
# Benchmark harness on numpy (no strict stubs); strict mode would only add reportUnknown* noise.
"""The box and the code a measurement ran on, recorded at the moment it is measured.

A latency or throughput number without the machine behind it is not a measurement, so every
scorer stamps this onto each cell as it writes it. Collecting the same facts later - at export,
at commit, at read - answers a different question: it describes whoever ran that step, on data
that may be weeks old and may never have touched their machine.

Two properties this file exists to keep:

* **Per cell, not per file.** A sweep resumes across days and hosts, and cached cells are skipped
  rather than re-measured, so one source file legitimately holds rows from several machines. A
  store-quality file mixes lancedb rows, which need AVX2, with rows a node without AVX2 can
  produce - one stamp for the file has to misdescribe some of them.
* **Never backfilled.** A cell measured before this existed carries no stamp, and there is no
  honest way to recover the box it ran on. Readers report it as not recorded; substituting the
  current machine is the defect this replaced.
"""

from __future__ import annotations

import os
import platform
import re
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["MEASURED_ON", "measurement_provenance", "source_git_sha", "stamped"]

# The key a stamp is written under, on the cell it describes.
MEASURED_ON = "measured_on"

_ROOT = Path(__file__).resolve().parent.parent

# A sweep run from a copy of scripts/ frozen outside the repo cannot ask git which commit it is; its
# run script exports the frozen sha here instead.
SOURCE_SHA_ENV = "SEMDEX_SOURCE_SHA"
_SHA_RE = re.compile(r"[0-9a-f]{7,40}")


def _run(args: list[str]) -> str:
    try:
        out = subprocess.run(args, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def _blas() -> str:
    """Which BLAS numpy resolved. A gemm-bound timing means nothing without it."""
    try:
        config = np.show_config(mode="dicts") or {}
        build = config.get("Build Dependencies", {}).get("blas", {})
        return f"{build.get('name', '?')} {build.get('version', '?')}"
    except Exception:
        return "unknown"


def source_git_sha() -> str:
    """The commit of the code doing the measuring: SEMDEX_SOURCE_SHA, else git, else "unknown".

    The variable wins because a frozen copy has no repository to ask, and a git lookup from there
    could reach an unrelated enclosing repository. A value that is not a hex sha is ignored rather
    than stamped, so a mistyped export cannot put arbitrary text into every row.
    """
    declared = os.environ.get(SOURCE_SHA_ENV, "").strip().lower()
    if _SHA_RE.fullmatch(declared):
        return declared
    return _run(["git", "-C", str(_ROOT), "rev-parse", "--short", "HEAD"])


def measurement_provenance() -> dict[str, Any]:
    """Stamp for a cell being measured RIGHT NOW.

    Call it at the moment the cell is produced and store the result on that cell. Calling it
    anywhere else records the caller's machine against somebody else's numbers.
    """
    return {
        "measured_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "semdex_git_sha": source_git_sha(),
        "host": socket.gethostname(),
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "blas": _blas(),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
    }


def stamped(row: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``row`` carrying the provenance of the run measuring it now.

    A row that already carries a stamp keeps it. Scorers persist incrementally into one results
    dict and rewrite the whole file on every cell, so without that guard a sweep resumed on
    another machine would re-date every cell it merely copied - the original defect, moved from
    the exporter into the scorers where the file would still look freshly measured.

    Returns a copy rather than stamping in place, because the caller holds these rows in the dict
    it is about to serialise and an in-place write would alias into it.
    """
    if isinstance(row.get(MEASURED_ON), dict):
        return dict(row)
    return {**row, MEASURED_ON: measurement_provenance()}
