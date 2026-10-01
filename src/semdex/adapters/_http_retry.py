"""Shared retry-with-backoff for the adapters' HTTP transports.

The embedding, summarizer, and extractor transports all POST to a model or
document server that can drop a connection or return a transient 5xx/429
mid-run - a single blip would otherwise kill an hours-long embed pass.
``retry_http`` wraps ONLY the "post + raise_for_status" step of each transport
in exponential backoff: a transient failure is retried, anything else re-raised
at once, so each transport keeps its own response parsing and error mapping
OUTSIDE the retry (a schema/JSON failure is a real error, never retried). httpx
is imported lazily so this module stays importable without the optional HTTP
extras installed.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import httpx

# Total attempts for one request (not "retries after the first"): with the
# default 4, a request is tried up to 4 times before its last error propagates.
DEFAULT_RETRIES = 4
# First backoff pause in seconds; each subsequent pause doubles it (1, 2, 4, ...).
DEFAULT_BASE_DELAY = 1.0
# Upper bound (seconds) on any single backoff pause, so a long run of failures
# does not grow the wait without limit.
DEFAULT_MAX_DELAY = 30.0

# The smallest HTTP status treated as a server-side (transient) failure. A 429
# (rate limit / "retry later") is transient too; every other 4xx is the caller's
# fault, where retrying never helps.
_SERVER_ERROR_FLOOR = 500
_TOO_MANY_REQUESTS = 429


def is_transient(exc: BaseException) -> bool:
    """Whether ``exc`` is a blip worth retrying rather than a permanent failure.

    True for an httpx transport error (connect/read/write/pool timeout, a dropped
    connection, a protocol error) and for an HTTP 5xx or 429 status. False for a
    4xx status other than 429, a pydantic ``ValidationError``, a JSON decode
    error, or any other exception - retrying those never helps.
    """
    import httpx

    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == _TOO_MANY_REQUESTS or status >= _SERVER_ERROR_FLOOR
    return False


def retry_http(
    do_request: Callable[[], httpx.Response],
    *,
    tries: int = DEFAULT_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """Call ``do_request`` with exponential backoff on a transient failure.

    ``do_request`` must perform the POST and ``raise_for_status``; its ``Response``
    is returned. A transient failure (see :func:`is_transient`) is retried up to
    ``tries`` attempts, pausing ``base_delay`` seconds and doubling each time
    (capped at ``max_delay``); once the attempts are spent the last exception
    propagates. A non-transient failure propagates immediately, so the caller's
    own parsing and error mapping run on it unretried. ``sleep`` is injected so
    tests exercise the backoff without real delay.
    """
    import httpx

    attempt = 0
    delay = base_delay
    while True:
        try:
            return do_request()
        except httpx.HTTPError as exc:
            attempt += 1
            if attempt >= tries or not is_transient(exc):
                raise
            sleep(min(delay, max_delay))
            delay *= 2


__all__ = [
    "DEFAULT_BASE_DELAY",
    "DEFAULT_MAX_DELAY",
    "DEFAULT_RETRIES",
    "is_transient",
    "retry_http",
]
