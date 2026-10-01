"""OpenAI-compatible embedding provider (``POST /v1/embeddings`` over HTTP).

One provider for every server that speaks the OpenAI embeddings API: hosted
OpenAI, a local ollama server's OpenAI-compat ``/v1``, and llama.cpp's
``llama-server``. Like the native ``ollama`` provider it carries no ML
dependency - only httpx (``semdex[openai]``) - and probes the model's dimension
once at load time. ``endpoint`` is the OpenAI *base URL* (ends in ``/v1``); the
provider appends ``/embeddings``. ``api_key`` (loaded from the environment, never
inlined) becomes a ``Bearer`` header; local ollama/llama.cpp need none.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from ...domain.errors import EmbeddingError
from .._http_retry import DEFAULT_RETRIES
from ._base import CallableEmbedding, post_embedding_request

if TYPE_CHECKING:
    from collections.abc import Sequence

    import httpx

    from ...application.ports import EmbeddingProvider
    from ...domain.models import Vector

_DEFAULT_MODEL = "text-embedding-3-small"
_DEFAULT_ENDPOINT = "https://api.openai.com/v1"


class _OpenAIEmbedRequest(BaseModel):
    """OpenAI ``/v1/embeddings`` request body."""

    model_config = ConfigDict(protected_namespaces=())

    model: str
    input: list[str]


class _OpenAIEmbedItem(BaseModel):
    """One row of the ``/v1/embeddings`` response (extra fields ignored)."""

    embedding: list[float]
    index: int


class _OpenAIEmbedResponse(BaseModel):
    """OpenAI ``/v1/embeddings`` response envelope (extra fields ignored)."""

    data: list[_OpenAIEmbedItem]


def load_openai_embedding(
    model_id: str | None = None,
    *,
    endpoint: str | None = None,
    timeout: float = 60.0,
    api_key: str | None = None,
    retries: int = DEFAULT_RETRIES,
    query_prefix: str = "",
    passage_prefix: str = "",
    client: httpx.Client | None = None,
) -> EmbeddingProvider:
    """Wrap an OpenAI-compatible embeddings server as an EmbeddingProvider.

    Probes the model's dimension once. ``endpoint`` is the base URL ending in
    ``/v1`` (default hosted OpenAI); ``/embeddings`` is appended. ``retries`` caps
    how many times a single embed request is retried on a transient blip (dropped
    connection, 5xx, 429). Raises :class:`~semdex.domain.errors.EmbeddingError` if
    httpx is missing or the server is unreachable.

    ``query_prefix`` and ``passage_prefix`` are prepended before the text is sent.
    ``/v1/embeddings`` has no notion of query-versus-passage, so a model trained to
    receive an instruction only gets one if the client sends it. Both default to
    empty, leaving the request unchanged.
    """
    resolved_model = model_id or _DEFAULT_MODEL
    url = (endpoint or _DEFAULT_ENDPOINT).rstrip("/") + "/embeddings"
    # Bearer auth only when a key is supplied; local ollama/llama.cpp need none.
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None

    def _embed(inputs: list[str]) -> list[Vector]:
        """Send already-prefixed text. Neither public path can be written in terms of the other:
        routing a query through ``passages_fn`` would give it the PASSAGE prefix."""
        payload = _OpenAIEmbedRequest(model=resolved_model, input=inputs).model_dump()
        body = post_embedding_request(
            url=url,
            payload=payload,
            response_model=_OpenAIEmbedResponse,
            timeout=timeout,
            client=client,
            headers=headers,
            service="openai",
            install_extra="openai",
            retries=retries,
        )
        if len(body.data) != len(inputs):
            raise EmbeddingError(f"openai returned {len(body.data)} embeddings for {len(inputs)} inputs at {url}")
        # The API may return rows out of order; ``index`` is the request position.
        ordered = sorted(body.data, key=lambda item: item.index)
        return [tuple(item.embedding) for item in ordered]

    def passages_fn(texts: Sequence[str]) -> list[Vector]:
        return _embed([passage_prefix + text for text in texts])

    def query_fn(text: str) -> Vector:
        return _embed([query_prefix + text])[0]

    dim = len(query_fn("dimension probe"))
    return CallableEmbedding(model_id=resolved_model, dim=dim, passages_fn=passages_fn, query_fn=query_fn)


__all__ = [
    "load_openai_embedding",
]
