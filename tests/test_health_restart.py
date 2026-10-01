"""Unit tests for the command restart hook.

No process is spawned: ``run`` is injected so a zero exit, a non-zero exit, a
timeout, and a spawn OSError are all exercised offline. The hook must NEVER raise;
every failure maps to ``False``.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

from semdex.adapters.health import CommandRestartHook

if TYPE_CHECKING:
    from collections.abc import Mapping

pytestmark = pytest.mark.os_agnostic


def _completed(command: str, returncode: int) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(args=command, returncode=returncode, stdout=b"", stderr=b"")


class _RecordingRun:
    """An injectable ``subprocess.run`` stand-in that records its one call."""

    def __init__(self, *, returncode: int = 0) -> None:
        self.returncode = returncode
        self.command: str | None = None
        self.shell: bool | None = None
        self.timeout: float | None = None
        self.env: dict[str, str] = {}
        self.capture_output: bool | None = None

    def __call__(
        self,
        command: str,
        *,
        shell: bool,
        timeout: float,
        env: Mapping[str, str],
        capture_output: bool,
    ) -> subprocess.CompletedProcess[bytes]:
        self.command = command
        self.shell = shell
        self.timeout = timeout
        self.env = dict(env)
        self.capture_output = capture_output
        return _completed(command, self.returncode)


def test_zero_exit_is_success_and_passes_shell_and_env() -> None:
    """A zero exit returns True; the command runs via shell with the overlaid env."""
    run = _RecordingRun(returncode=0)
    hook = CommandRestartHook("systemctl restart ollama", timeout=30.0, env={"SEMDEX_X": "1"}, run=run)

    assert hook.restart() is True
    assert run.command == "systemctl restart ollama"
    assert run.shell is True
    assert run.timeout == 30.0
    assert run.capture_output is True
    assert run.env["SEMDEX_X"] == "1"


def test_nonzero_exit_is_failure() -> None:
    """A non-zero exit code maps to False (restart did not succeed)."""
    assert CommandRestartHook("false", run=_RecordingRun(returncode=1)).restart() is False


def test_timeout_is_failure_not_raised() -> None:
    """A TimeoutExpired is caught and reported as False, never raised."""

    def run(
        command: str, *, shell: bool, timeout: float, env: Mapping[str, str], capture_output: bool
    ) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(cmd=command, timeout=timeout)

    assert CommandRestartHook("sleep 999", timeout=0.1, run=run).restart() is False


def test_spawn_error_is_failure_not_raised() -> None:
    """An OSError from the spawn (e.g. command not found) maps to False."""

    def run(
        command: str, *, shell: bool, timeout: float, env: Mapping[str, str], capture_output: bool
    ) -> subprocess.CompletedProcess[bytes]:
        raise OSError("no such file")

    assert CommandRestartHook("/nope", run=run).restart() is False
