"""Cohere native embedding provider (``embed-v4.0`` over HTTP).

Calls ``POST https://api.cohere.com/v2/embed``. A paid CLOUD service: ``api_key``
is REQUIRED (env-only, sent as an ``Authorization: Bearer`` header), and the model
is strong on multilingual / German content. Like the openai/ollama providers it
carries no ML dependency - only httpx (``semdex[cohere]``) - and probes the
model's dimension once at load time. Retrieval is asymmetric: passages use the
``search_document`` input type and queries ``search_query``. A large batch is
split into slices of ``_MAX_BATCH`` texts per call and the per-slice results
concatenated in order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from ...domain.errors import EmbeddingError
from .._http_retry import DEFAULT_RETRIES
from ._base import CallableEmbedding, post_embedding_request

if TYPE_CHECKING:
    from collections.abc import Sequence

    import httpx

    from ...application.ports import EmbeddingProvider
    from ...domain.models import Vector

_DEFAULT_MODEL = "embed-v4.0"
_DEFAULT_ENDPOINT = "https://api.cohere.com/v2"
# The /v2/embed endpoint caps a single call at 96 texts; larger batches sub-batch.
_MAX_BATCH = 96
# Asymmetric retrieval input types: passages are documents, queries are queries.
_TASK_DOCUMENT = "search_document"
_TASK_QUERY = "search_query"


class _CohereEmbedRequest(BaseModel):
    """Cohere ``/v2/embed`` request body."""

    model_config = ConfigDict(protected_namespaces=())

    model: str
    texts: list[str]
    input_type: str
    embedding_types: list[str]


class _CohereEmbeddings(BaseModel):
    """The ``embeddings`` object of the response (one list per requested type)."""

    # ``float`` is the wire key; aliased off the shadowed builtin name.
    float_: list[list[float]] = Field(alias="float")


class _CohereResponse(BaseModel):
    """Cohere ``/v2/embed`` response envelope (extra fields ignored)."""

    embeddings: _CohereEmbeddings


def load_cohere_embedding(
    model_id: str | None = None,
    *,
    endpoint: str | None = None,
    timeout: float = 60.0,
    api_key: str | None = None,
    retries: int = DEFAULT_RETRIES,
    client: httpx.Client | None = None,
) -> EmbeddingProvider:
    """Wrap Cohere's embedding API as an EmbeddingProvider.

    Probes the model's dimension once. ``endpoint`` is the API base URL ending in
    ``/v2`` (default hosted Cohere); ``/embed`` is appended. ``api_key`` is
    REQUIRED (a paid cloud API) and sent as ``Authorization: Bearer``. The default
    model's native 1536-dim output is used (``output_dimension`` is never sent).
    ``retries`` caps how many times a single embed request is retried on a
    transient blip (dropped connection, 5xx, 429). Raises
    :class:`~semdex.domain.errors.EmbeddingError` if httpx is missing, no api_key
    is given, or the server is unreachable.
    """
    if not api_key:
        raise EmbeddingError("cohere requires an api_key (set SEMDEX___EMBEDDING__API_KEY, never inline)")
    resolved_model = model_id or _DEFAULT_MODEL
    url = (endpoint or _DEFAULT_ENDPOINT).rstrip("/") + "/embed"
    headers = {"Authorization": f"Bearer {api_key}"}

    def _embed(texts: Sequence[str], *, task: str) -> list[Vector]:
        inputs = list(texts)
        vectors: list[Vector] = []
        for start in range(0, len(inputs), _MAX_BATCH):
            chunk = inputs[start : start + _MAX_BATCH]
            payload = _CohereEmbedRequest(
                model=resolved_model, texts=chunk, input_type=task, embedding_types=["float"]
            ).model_dump()
            body = post_embedding_request(
                url=url,
                payload=payload,
                response_model=_CohereResponse,
                timeout=timeout,
                client=client,
                headers=headers,
                service="cohere",
                install_extra="cohere",
                retries=retries,
            )
            rows = body.embeddings.float_
            if len(rows) != len(chunk):
                raise EmbeddingError(f"cohere returned {len(rows)} embeddings for {len(chunk)} inputs at {url}")
            vectors.extend(tuple(row) for row in rows)
        return vectors

    def passages_fn(texts: Sequence[str]) -> list[Vector]:
        return _embed(texts, task=_TASK_DOCUMENT)

    def query_fn(text: str) -> Vector:
        return _embed([text], task=_TASK_QUERY)[0]

    dim = len(query_fn("dimension probe"))
    return CallableEmbedding(model_id=resolved_model, dim=dim, passages_fn=passages_fn, query_fn=query_fn)


__all__ = [
    "load_cohere_embedding",
]
