"""Restart hook that runs the operator's configured recovery command.

``CommandRestartHook`` runs ``[health].restart_command`` when a backend is judged
down. The command is an OPERATOR-set config value (a trusted string an admin put
in the config), NOT untrusted user input, so running it through the shell is
safe here; never pass user-supplied text to this hook. ``run`` is injectable so
tests drive it without spawning a process.
"""

from __future__ import annotations

import logging
import os

# subprocess is used only for the operator-configured restart hook below.
import subprocess  # nosec B404
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 120.0


class CommandRunner(Protocol):
    """The ``subprocess.run`` slice this hook depends on (injected in tests)."""

    def __call__(
        self,
        command: str,
        *,
        shell: bool,
        timeout: float,
        env: Mapping[str, str],
        capture_output: bool,
    ) -> subprocess.CompletedProcess[bytes]: ...


def _default_run(
    command: str, *, shell: bool, timeout: float, env: Mapping[str, str], capture_output: bool
) -> subprocess.CompletedProcess[bytes]:
    """The production runner: a thin, typed wrapper over ``subprocess.run``.

    Kept as a named function (rather than passing ``subprocess.run`` directly) so it
    presents the exact :class:`CommandRunner` shape - ``subprocess.run`` is heavily
    overloaded and does not match the Protocol on its own.
    """
    # shell is the caller's operator-set choice; see CommandRestartHook.restart for the rationale.
    return subprocess.run(  # noqa: S603  # nosec B602
        command, shell=shell, timeout=timeout, env=env, capture_output=capture_output, check=False
    )


class CommandRestartHook:
    """RestartHook that runs a shell command and reports its exit status.

    ``env`` entries are overlaid on the current environment (so the hook can pass
    which endpoint failed to the restart script). Returns ``True`` iff the command
    exits zero; a non-zero exit, a timeout, or a spawn failure logs a warning and
    returns ``False`` - the hook never raises.
    """

    def __init__(
        self,
        command: str,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        env: Mapping[str, str] | None = None,
        run: CommandRunner = _default_run,
    ) -> None:
        self._command = command
        self._timeout = timeout
        self._env = dict(env) if env else {}
        self._run = run

    def restart(self) -> bool:
        """Run the restart command; ``True`` on a zero exit, ``False`` otherwise."""
        merged_env = {**os.environ, **self._env}
        try:
            # shell=True is safe here: command is an operator-set config value, never user input.
            result = self._run(  # noqa: S604  # nosec B604
                self._command,
                shell=True,
                timeout=self._timeout,
                env=merged_env,
                capture_output=True,
            )
        except subprocess.TimeoutExpired:
            logger.warning("restart command timed out after %.0fs: %s", self._timeout, self._command)
            return False
        except OSError as exc:
            logger.warning("restart command could not run (%s): %s", exc, self._command)
            return False
        if result.returncode != 0:
            logger.warning("restart command exited %d: %s", result.returncode, self._command)
            return False
        return True


__all__ = [
    "CommandRestartHook",
    "CommandRunner",
]
