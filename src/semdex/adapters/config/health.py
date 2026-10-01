"""Health configuration model parsed from the ``[health]`` section.

Controls how ``serve`` keeps its server-backed providers (ollama / OpenAI-compat)
resident and reachable: a startup warm-up, an ollama ``keep_alive`` passthrough, a
reactive restart hook the providers call when a backend is down, and an opt-in
background health-check loop. Absent section keeps every existing default: warm-up
is best-effort (a no-op without an http provider), restart is disabled until a
command is set, and the loop is off.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from lib_layered_config import Config


class HealthConfig(BaseModel):
    """Validated, immutable health/keep-alive/restart settings.

    Example:
        >>> HealthConfig().warmup
        True
        >>> HealthConfig().restart_command is None
        True
        >>> HealthConfig().check_enabled
        False
    """

    model_config = ConfigDict(frozen=True)

    # warmup - at serve startup, issue one probe call per http-backed provider so
    # the model is resident before the first real query.
    #   Default: True (best-effort; a no-op when no ollama/openai provider is used).
    #   Effect: a throwaway embed/summarize per http provider warms the GPU.
    #   When to unset: set False to skip the warm-up (faster startup, first query
    #     pays the cold-load cost).
    #   Consequences: a failed warm-up is logged and ignored, never fatal.
    warmup: bool = True
    # keep_alive - ollama ``keep_alive`` passthrough for the http providers.
    #   Default: None (ollama's own default, roughly 5 minutes).
    #   Effect: how long ollama keeps the model in VRAM after a call ("30m",
    #     "0" = unload immediately, "-1" = keep forever).
    #   WARNING: do NOT set "-1" on a SHARED GPU. It pins the model in VRAM,
    #     defeats ollama's model time-sharing, and can OOM when embedding +
    #     summarizer + vision OCR contend for one card.
    #   Consequences: a longer keep_alive cuts cold-load latency but holds VRAM.
    keep_alive: str | None = None
    # restart_command - shell command semdex runs when a backend is judged down
    # (used by both the reactive per-call self-heal and the active loop).
    #   Default: None (restart disabled; a down backend is surfaced, not restarted).
    #   Effect: the operator's recovery command (e.g. restart the ollama service).
    #   Trust boundary: this is an OPERATOR-set config value, run through the
    #     shell; never derived from user input.
    #   Consequences: without it, a down backend cannot be auto-recovered.
    restart_command: str | None = None
    # restart_timeout - max seconds the restart command may run.
    #   Default: 120.0.
    #   Effect: caps the restart hook; a longer run is killed and counts as failed.
    #   When to change: raise it for a slow service restart.
    restart_timeout: float = Field(default=120.0, gt=0)
    # check_enabled - opt-in active background health-check loop.
    #   Default: False (the loop is off; only the reactive per-call self-heal runs).
    #   Effect: when True (and a restart_command is set) serve runs a thread that
    #     probes the endpoints every check_interval and restarts a down one.
    #   Consequences: notices an idle backend has died before the next query does;
    #     adds one background thread and periodic probe traffic.
    check_enabled: bool = False
    # check_interval - seconds between health-check-loop polls.
    #   Default: 30.0.
    #   Effect: how often the loop probes each endpoint.
    #   When to change: lower it for faster detection, raise it to cut probe load.
    check_interval: float = Field(default=30.0, gt=0)
    # recover_timeout - after a restart, max seconds to wait for the endpoint to
    # come back healthy (shared by the reactive self-heal and the loop).
    #   Default: 120.0.
    #   Effect: bounds the post-restart recovery wait before giving up.
    #   When to change: raise it for a model that is slow to load after a restart.
    recover_timeout: float = Field(default=120.0, ge=0)


def get_health_config(config: Config) -> HealthConfig:
    """Parse the ``[health]`` section into a HealthConfig.

    Falls back to the all-default (warm-up on, restart off, loop off) settings
    when the section is absent.

    Example:
        >>> from lib_layered_config import Config
        >>> get_health_config(Config({}, {})).warmup
        True
    """
    return HealthConfig.model_validate(config.get("health", {}))


__all__ = [
    "HealthConfig",
    "get_health_config",
]
