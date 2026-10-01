"""Health adapters: warm-up, liveness probe, restart hook, and the check loop.

    * :class:`._probe.HttpHealthProbe` - an HTTP GET liveness probe
      (:class:`~semdex.application.ports.HealthProbe`)
    * :class:`._restart.CommandRestartHook` - runs the operator's restart command
      (:class:`~semdex.application.ports.RestartHook`)
    * :func:`._warmup.warmup_embedding` / :func:`._warmup.warmup_summary` - the
      best-effort startup warm-up calls
    * :class:`._loop.HealthLoop` - the opt-in background check-and-restart loop

The two builder helpers (:func:`restart_hook_from_command`, :func:`probe_url_for`)
derive a hook and a probe URL from config, so the composition root and the
``serve`` command share one derivation.
"""

from __future__ import annotations

from ._loop import HealthLoop
from ._probe import HttpHealthProbe
from ._restart import CommandRestartHook
from ._warmup import warmup_embedding, warmup_summary

# Provider endpoint defaults, mirroring the embedding/summarizer adapters, so a
# probe built from an unset [health]/[embedding] endpoint targets the same host
# the provider actually calls.
_OLLAMA_DEFAULT_ENDPOINT = "http://127.0.0.1:11434"
_OPENAI_DEFAULT_ENDPOINT = "https://api.openai.com/v1"


def restart_hook_from_command(command: str | None, timeout: float) -> CommandRestartHook | None:
    """Build a restart hook from the configured command, or ``None`` when unset.

    An unset or empty ``[health].restart_command`` disables restart entirely, so
    the caller wraps nothing and a down backend is only surfaced, never restarted.
    """
    if not command:
        return None
    return CommandRestartHook(command, timeout=timeout)


def probe_url_for(service: str, endpoint: str | None) -> str:
    """Derive the liveness-probe URL for a server-backed provider.

    ``ollama`` probes its host root (a running ollama answers 200 "Ollama is
    running"); ``openai`` probes ``<base>/models`` (the OpenAI-compatible listing
    endpoint). ``endpoint`` ``None`` falls back to each provider's default host.
    """
    if service == "ollama":
        return (endpoint or _OLLAMA_DEFAULT_ENDPOINT).rstrip("/")
    return (endpoint or _OPENAI_DEFAULT_ENDPOINT).rstrip("/") + "/models"


__all__ = [
    "CommandRestartHook",
    "HealthLoop",
    "HttpHealthProbe",
    "probe_url_for",
    "restart_hook_from_command",
    "warmup_embedding",
    "warmup_summary",
]
