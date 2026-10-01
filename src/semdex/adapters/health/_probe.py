"""HTTP liveness probe for a server-backed provider endpoint.

``HttpHealthProbe`` GETs a URL and reports whether the backend answered with a
success status. It NEVER raises: every httpx error (and a missing httpx) maps to
``False``, so a health check can only report "down", never introduce a new
failure. httpx is imported lazily (mirroring ``embedding/_base.py``) so the module
stays importable without the optional HTTP extras.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)

# Default per-probe timeout: a liveness GET should answer fast, and a slow probe
# would delay the loop's recovery polling, so it is shorter than a real request.
_DEFAULT_TIMEOUT = 5.0


class HttpHealthProbe:
    """Liveness :class:`~semdex.application.ports.HealthProbe` over HTTP GET.

    Owns its httpx client only when the caller injects none (tests inject a
    ``MockTransport`` client); an owned client is closed after each check so the
    probe holds no persistent connection between polls.
    """

    def __init__(self, url: str, *, timeout: float = _DEFAULT_TIMEOUT, client: httpx.Client | None = None) -> None:
        self._url = url
        self._timeout = timeout
        self._client = client

    def check(self) -> bool:
        """GET the URL; ``True`` on a success status, ``False`` on any failure.

        Catches every httpx error and a missing httpx dependency so a probe never
        raises - the caller treats a ``False`` as "backend down".
        """
        try:
            import httpx
        except ImportError:  # pragma: no cover - only without the httpx extra
            logger.warning("httpx is not installed; cannot probe %s", self._url)
            return False
        owns_client = self._client is None
        active = self._client if self._client is not None else httpx.Client(timeout=self._timeout)
        try:
            return active.get(self._url).is_success
        except httpx.HTTPError as exc:
            logger.warning("health probe of %s failed: %s", self._url, exc)
            return False
        finally:
            if owns_client:
                active.close()


__all__ = [
    "HttpHealthProbe",
]
