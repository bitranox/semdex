"""Structured benchmark-result recorder shared by the local_only E2E matrices.

A benchmark run emits two artefacts from one recorder so they never drift:

- ``benchmark-report.md``   - human-readable tables (see each matrix's ``_append_report``).
- ``benchmark-results.json`` - machine-readable ``{versions, results}`` for regression
  detection. ``scripts/benchmark_compare.py`` diffs it against a committed baseline and
  fails CI when a combination stops working or a quality metric drops past a tolerance.

A result is keyed ``"<component>/<combo>"`` (optionally ``"@<corpus>"``) and carries a
:class:`ResultStatus` plus a flat ``metric name -> value`` map. Higher-is-better quality
metrics (``ndcg@10``, ``recall@10``, ``mrr``, ``p@1``, ``phrase_recall``, ...) gate the
comparator; timing/size metrics (suffix ``_ms`` / ``_s`` / ``_mb``) are recorded and
reported but never gated, because they are machine-dependent and noisy.

The recorder keeps one process-wide registry (module global), so running both matrix
files in a single pytest session accumulates into one ``benchmark-results.json`` instead
of the last file clobbering the first. The file is rewritten on every ``record`` call, so
a run that aborts mid-way still leaves the results gathered so far.

The Pydantic models below define ``SCHEMA_VERSION`` 1; ``scripts/benchmark_compare.py``
implements the same schema and rejects a file whose version it does not understand.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1

# Packages whose version is worth pinning to a benchmark result: a bump here is the most
# likely off-project cause of a retrieval/extraction regression. Absent ones are skipped.
_TRACKED_PACKAGES = (
    "semdex",
    "fastembed",
    "model2vec",
    "sentence-transformers",
    "chonkie",
    "semantic-text-splitter",
    "sqlite-vec",
    "lancedb",
    "ir_datasets",
    "huggingface-hub",
    "httpx",
    "mcp",
)


class ResultStatus(StrEnum):
    """Outcome of one benchmark cell.

    ``str, Enum`` (not ``IntEnum``) to round-trip through JSON as a readable label, matching
    the domain-enum convention.
    """

    OK = "ok"  # ran and produced metrics
    FAIL = "fail"  # ran but failed its assertion / errored
    SKIP = "skip"  # backend/model absent, cell not run
    NA = "n/a"  # format unsupported by this extractor - not a failure


class BenchmarkResult(BaseModel):
    """One cell: its status, a flat metric map, and optionally its per-query scores.

    ``per_query`` maps a metric to ``{query id: score}``. It exists so the comparator can pair a
    run against the baseline query by query instead of comparing two means. On a 20-query slice
    the mean carries a 95 percent half-width around 0.13, which is four times the tolerance the
    gate fires on; pairing removes the query-difficulty variance that half-width is made of, and
    the slice becomes able to detect the regressions it was always supposed to guard.
    """

    status: ResultStatus
    metrics: dict[str, float] = Field(default_factory=dict)
    per_query: dict[str, dict[str, float]] = Field(default_factory=dict)
    # Which run produced this cell. Left unset here (the file's own generated_utc covers every
    # cell it holds) and filled by the comparator when folding runs into a multi-vintage baseline.
    measured_utc: str | None = None


class BenchmarkVersions(BaseModel):
    """Everything a regression could be attributed to: semdex, the interpreter, deps, images."""

    semdex: str
    python: str
    packages: dict[str, str] = Field(default_factory=dict)
    images: dict[str, str] = Field(default_factory=dict)


class BenchmarkResults(BaseModel):
    """The whole results file: schema tag, capture time, versions, and keyed results."""

    schema_version: int = SCHEMA_VERSION
    generated_utc: str
    versions: BenchmarkVersions
    results: dict[str, BenchmarkResult] = Field(default_factory=dict)


def _maybe_version(name: str) -> str | None:
    """The installed version of *name*, or None if the (optional) package is absent."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _capture_versions() -> BenchmarkVersions:
    """Snapshot semdex/interpreter/dep versions (container image digests are added later)."""
    packages = {name: version for name in _TRACKED_PACKAGES if (version := _maybe_version(name)) is not None}
    return BenchmarkVersions(
        semdex=packages.get("semdex", "unknown"),
        python=platform.python_version(),
        packages=packages,
    )


_RESULTS_PATH = Path(os.environ.get("SEMDEX_BENCH_RESULTS", "benchmark-results.json"))
_results: dict[str, BenchmarkResult] = {}
# Captured once at import; record() only mutates its .images (per container), never rebinds it.
_versions: BenchmarkVersions = _capture_versions()


def image_digest(image: str) -> str:
    """Return ``image@sha256:...`` for a pulled image, or the bare tag if inspect fails.

    Records the exact image content a container cell ran against, so an extraction
    regression can be pinned to an upstream image rebuild, not just a floating ``:latest``.
    """
    try:
        proc = subprocess.run(
            ["docker", "inspect", "--format", "{{index .RepoDigests 0}}", image],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return image
    digest = proc.stdout.strip()
    return digest or image


def record(
    component: str,
    combo: str,
    status: ResultStatus,
    metrics: dict[str, float] | None = None,
    *,
    corpus: str | None = None,
    images: dict[str, str] | None = None,
    per_query: dict[str, dict[str, float]] | None = None,
) -> None:
    """Record one benchmark cell and rewrite ``benchmark-results.json``.

    ``component`` is the axis (``quality``/``store``/``extract``), ``combo`` the cell within
    it (``recursive+fastembed``, ``pgvector``, ``markitdown@docx``). ``images`` maps a name to
    the digest from :func:`image_digest` when the cell ran a container. ``per_query`` carries the
    unaveraged scores so the comparator can pair against the baseline rather than diff two means.
    """
    key = f"{component}/{combo}" + (f"@{corpus}" if corpus else "")
    _results[key] = BenchmarkResult(status=status, metrics=metrics or {}, per_query=per_query or {})
    if images:
        _versions.images.update(images)
    doc = BenchmarkResults(
        generated_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        versions=_versions,
        results=dict(_results),
    )
    _RESULTS_PATH.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
