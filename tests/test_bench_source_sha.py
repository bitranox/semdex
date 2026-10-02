"""Which code a benchmark row was measured with, also when the script runs from a frozen copy.

Long sweeps run from a copy of scripts/ frozen at one commit, outside the repo, so a later checkout
in the shared working tree cannot change the code under a run that lasts days. Asking git from the
script's own location then finds no repository and every row is stamped "unknown". The run scripts
know the sha they froze, so they export it as SEMDEX_SOURCE_SHA and the stamp takes it from there.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"

pytestmark = pytest.mark.os_agnostic


def _stamp_from_copy(tmp_path: Path, env_sha: str | None) -> str:
    """Copy _provenance.py outside the repo and return the sha it stamps, as the frozen run sees it."""
    frozen = tmp_path / "frozen-scripts"
    frozen.mkdir()
    shutil.copy2(_SCRIPTS / "_provenance.py", frozen / "_provenance.py")
    probe = subprocess.run(["git", "-C", str(frozen), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    assert probe.returncode != 0, f"precondition: {frozen} must not sit inside a git repository"
    env = {k: v for k, v in os.environ.items() if k != "SEMDEX_SOURCE_SHA"}
    if env_sha is not None:
        env["SEMDEX_SOURCE_SHA"] = env_sha
    code = "import _provenance; print(_provenance.measurement_provenance()['semdex_git_sha'])"
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=frozen,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return out.stdout.strip()


def test_a_frozen_copy_stamps_the_sha_its_run_script_exports(tmp_path: Path) -> None:
    assert _stamp_from_copy(tmp_path, "26d3c958") == "26d3c958"


def test_a_frozen_copy_without_the_variable_still_says_unknown(tmp_path: Path) -> None:
    assert _stamp_from_copy(tmp_path, None) == "unknown"


def test_a_value_that_is_not_a_sha_is_not_stamped(tmp_path: Path) -> None:
    assert _stamp_from_copy(tmp_path, "main; rm -rf /") == "unknown"


def test_no_script_asks_git_for_the_sha_on_its_own() -> None:
    """Every stamp goes through _provenance.source_git_sha, so the variable reaches all of them."""
    offenders = [
        path.name
        for path in sorted(_SCRIPTS.glob("*.py"))
        if path.name != "_provenance.py" and re.search(r"rev-parse", path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
