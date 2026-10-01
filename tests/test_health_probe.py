"""Unit tests for the HTTP liveness probe.

No server required: an ``httpx.MockTransport`` is injected so a success, a 5xx,
and a raised connect error are all exercised offline. The probe must NEVER raise -
every failure maps to ``False`` - so a health check cannot become a new failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from semdex.adapters.health import HttpHealthProbe

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.os_agnostic


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_success_status_is_healthy() -> None:
    """A 2xx response makes check() return True and hits the probed URL."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text="Ollama is running")

    probe = HttpHealthProbe("http://h:11434", client=_client(handler))

    assert probe.check() is True
    assert str(seen[0].url) == "http://h:11434"


def test_server_error_status_is_unhealthy() -> None:
    """A 5xx response is not a success, so check() returns False without raising."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    assert HttpHealthProbe("http://h:11434", client=_client(handler)).check() is False


def test_connect_error_is_unhealthy_not_raised() -> None:
    """A dropped connection is swallowed and reported as False, never raised."""

    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    assert HttpHealthProbe("http://h:11434", client=_client(handler)).check() is False


def test_openai_models_probe_url() -> None:
    """probe_url_for maps openai to <base>/models and ollama to the host root."""
    from semdex.adapters.health import probe_url_for

    assert probe_url_for("ollama", "http://h:11434/") == "http://h:11434"
    assert probe_url_for("openai", "http://h:8080/v1/") == "http://h:8080/v1/models"
    assert probe_url_for("ollama", None) == "http://127.0.0.1:11434"
