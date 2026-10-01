"""Shared building blocks for the LLM summarizer adapters.

``CallableSummarizer`` is the SummaryProvider used by the ollama and openai
summarizer adapters: it holds a model id and one injected summarize callable. The
heavy/optional HTTP work lives in each provider's ``load_*`` factory, which builds
the callable; this class stays a thin, unit-testable adapter.

``post_chat_request`` is the shared HTTP transport for the server-backed
summarizers (ollama ``/api/chat`` and the OpenAI-compatible
``/v1/chat/completions``): one place that lazily imports httpx, owns the client
lifecycle, and maps every transport/validation failure to
:class:`~semdex.domain.errors.SummaryError`. Each provider supplies only its URL,
request body, response model, and headers.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from ...domain.errors import SummaryError
from .._http_retry import DEFAULT_RETRIES, retry_http

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import httpx

_ResponseT = TypeVar("_ResponseT", bound=BaseModel)


def post_chat_request(
    *,
    url: str,
    payload: Mapping[str, Any],
    response_model: type[_ResponseT],
    timeout: float,
    client: httpx.Client | None,
    headers: Mapping[str, str] | None = None,
    service: str,
    install_extra: str,
    retries: int = DEFAULT_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
) -> _ResponseT:
    """POST ``payload`` to ``url`` and validate the reply into ``response_model``.

    The one HTTP path both server-backed summarizers share. ``service`` labels
    errors (``"ollama"`` / ``"openai"``) and ``install_extra`` names the optional
    extra to install if httpx is missing (``semdex[summary]``). Owns the httpx
    client only when the caller injects none (tests inject a ``MockTransport``
    client). The POST is retried up to ``retries`` attempts on a transient blip
    (dropped connection, 5xx, 429) with exponential backoff; ``sleep`` is
    injectable so tests skip the real wait. Raises
    :class:`~semdex.domain.errors.SummaryError` on any httpx, JSON, or schema
    failure so callers never see a raw httpx/pydantic error.
    """
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - only without the httpx extra
        raise SummaryError(f"httpx is not installed; install semdex[{install_extra}]") from exc
    owns_client = client is None
    active = client if client is not None else httpx.Client(timeout=timeout)
    request_headers = dict(headers) if headers else None

    def send() -> httpx.Response:
        response = active.post(url, json=dict(payload), headers=request_headers)
        response.raise_for_status()
        return response

    try:
        response = retry_http(send, tries=retries, sleep=sleep)
        return response_model.model_validate(_response_json(response, service))
    except httpx.HTTPError as exc:
        raise SummaryError(f"{service} summary request to {url} failed: {exc}") from exc
    except ValidationError as exc:
        raise SummaryError(f"{service} at {url} returned an unexpected response: {exc}") from exc
    finally:
        if owns_client:
            active.close()


def _response_json(response: Any, service: str) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise SummaryError(f"{service} returned invalid JSON: {exc}") from exc


class CallableSummarizer:
    """SummaryProvider built from an injected summarize callable plus a model id."""

    def __init__(self, *, model_id: str, summarize_fn: Callable[[str], str]) -> None:
        self._model_id = model_id
        self._summarize_fn = summarize_fn

    @property
    def model_id(self) -> str:
        return self._model_id

    def summarize(self, text: str) -> str:
        return self._summarize_fn(text)


# Static conformance assertion -- CallableSummarizer satisfies SummaryProvider.
if TYPE_CHECKING:
    from ...application.ports import SummaryProvider

    _assert: SummaryProvider = CallableSummarizer(model_id="", summarize_fn=lambda _t: "")


__all__ = [
    "CallableSummarizer",
    "post_chat_request",
]
