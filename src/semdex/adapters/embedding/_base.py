"""Shared building blocks for the vector-embedding adapters.

``CallableEmbedding`` is the EmbeddingProvider used by the fastembed, model2vec,
ollama, and openai adapters: it holds a model id, a dimension, and two injected
embed callables (passage and query). The heavy/optional library work lives in
each provider's ``load_*`` factory, which builds the callables; this class stays
a thin, unit-testable adapter. ``to_vector(s)`` normalizes a library's
numpy/array output into the domain ``Vector`` (a float tuple).

``post_embedding_request`` is the shared HTTP transport for the server-backed
providers (ollama ``/api/embed`` and the OpenAI-compatible ``/v1/embeddings``):
one place that lazily imports httpx, owns the client lifecycle, and maps every
transport/validation failure to :class:`~semdex.domain.errors.EmbeddingError`.
Each provider supplies only its URL, request body, response model, and headers.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from ...domain.errors import EmbeddingError
from .._http_retry import DEFAULT_RETRIES, retry_http

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    import httpx

    from ...domain.models import Vector

_ResponseT = TypeVar("_ResponseT", bound=BaseModel)


def post_embedding_request(
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

    The one HTTP path both server-backed embedding providers share. ``service``
    labels errors (``"ollama"`` / ``"openai"``) and ``install_extra`` names the
    optional extra to install if httpx is missing (``semdex[ollama]`` etc.).
    Owns the httpx client only when the caller injects none (tests inject a
    ``MockTransport`` client). The POST is retried up to ``retries`` attempts on a
    transient blip (dropped connection, 5xx, 429) with exponential backoff;
    ``sleep`` is injectable so tests skip the real wait. Raises
    :class:`~semdex.domain.errors.EmbeddingError` on any httpx, JSON, or schema
    failure so callers never see a raw httpx/pydantic error.
    """
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - only without the httpx extra
        raise EmbeddingError(f"httpx is not installed; install semdex[{install_extra}]") from exc
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
        raise EmbeddingError(f"{service} embed request to {url} failed: {exc}") from exc
    except ValidationError as exc:
        raise EmbeddingError(f"{service} at {url} returned an unexpected response: {exc}") from exc
    finally:
        if owns_client:
            active.close()


def _response_json(response: Any, service: str) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise EmbeddingError(f"{service} returned invalid JSON: {exc}") from exc


def to_vector(row: Any) -> Vector:
    """Normalize one embedding row (numpy array or sequence) to a float tuple."""
    values = row.tolist() if hasattr(row, "tolist") else row
    return tuple(float(value) for value in values)


def to_vectors(rows: Iterable[Any]) -> list[Vector]:
    """Normalize an iterable of embedding rows to a list of float tuples."""
    return [to_vector(row) for row in rows]


class CallableEmbedding:
    """EmbeddingProvider from injected passage/query embed callables plus a dim."""

    def __init__(
        self,
        *,
        model_id: str,
        dim: int,
        passages_fn: Callable[[Sequence[str]], list[Vector]],
        query_fn: Callable[[str], Vector],
    ) -> None:
        self._model_id = model_id
        self._dim = dim
        self._passages_fn = passages_fn
        self._query_fn = query_fn

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return self._dim

    def embed_passages(self, texts: Sequence[str]) -> list[Vector]:
        return self._passages_fn(texts)

    def embed_query(self, text: str) -> Vector:
        return self._query_fn(text)


# Static conformance assertion -- CallableEmbedding satisfies EmbeddingProvider.
if TYPE_CHECKING:
    from ...application.ports import EmbeddingProvider

    _assert: EmbeddingProvider = CallableEmbedding(
        model_id="", dim=1, passages_fn=lambda _t: [], query_fn=lambda _t: ()
    )


__all__ = [
    "CallableEmbedding",
    "post_embedding_request",
    "to_vector",
    "to_vectors",
]
