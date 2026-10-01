"""Google Gemini native embedding provider (``gemini-embedding-001`` over HTTP).

Calls ``POST .../v1beta/models/<model>:batchEmbedContents`` on Google's
Generative Language API. A paid CLOUD service: ``api_key`` is REQUIRED (env-only,
sent as the ``x-goog-api-key`` header), and the model is strong on multilingual /
German content. Like the openai/ollama providers it carries no ML dependency -
only httpx (``semdex[gemini]``) - and probes the model's dimension once at load
time. Retrieval is asymmetric: passages use the ``RETRIEVAL_DOCUMENT`` task type
and queries ``RETRIEVAL_QUERY``. A large batch is split into slices of
``_MAX_BATCH`` requests per call and the per-slice results concatenated in order.
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

_DEFAULT_MODEL = "gemini-embedding-001"
_DEFAULT_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"
# batchEmbedContents caps a single call at 100 requests; larger batches sub-batch.
_MAX_BATCH = 100
# Asymmetric retrieval task types: passages are documents, queries are queries.
_TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
_TASK_QUERY = "RETRIEVAL_QUERY"


class _GeminiPart(BaseModel):
    """One ``content.parts`` entry: a single text span to embed."""

    text: str


class _GeminiContent(BaseModel):
    """The ``content`` of one embed request (a list of text parts)."""

    parts: list[_GeminiPart]


class _GeminiRequestItem(BaseModel):
    """One entry of the ``batchEmbedContents`` ``requests`` array."""

    model_config = ConfigDict(protected_namespaces=())

    # The per-request model field carries the ``models/`` prefix, unlike the URL
    # path which uses the bare model id.
    model: str
    content: _GeminiContent
    task_type: str = Field(serialization_alias="taskType")


class _GeminiBatchRequest(BaseModel):
    """The ``batchEmbedContents`` request body."""

    requests: list[_GeminiRequestItem]


class _GeminiEmbedding(BaseModel):
    """One embedding row of the response (extra fields ignored)."""

    values: list[float]


class _GeminiResponse(BaseModel):
    """``batchEmbedContents`` response envelope (extra fields ignored)."""

    embeddings: list[_GeminiEmbedding]


def load_gemini_embedding(
    model_id: str | None = None,
    *,
    endpoint: str | None = None,
    timeout: float = 60.0,
    api_key: str | None = None,
    retries: int = DEFAULT_RETRIES,
    client: httpx.Client | None = None,
) -> EmbeddingProvider:
    """Wrap Google's Gemini embedding API as an EmbeddingProvider.

    Probes the model's dimension once. ``endpoint`` is the API base URL (default
    the public Generative Language endpoint); the URL becomes
    ``{base}/models/{model}:batchEmbedContents``. ``api_key`` is REQUIRED (a paid
    cloud API) and sent as ``x-goog-api-key``. ``retries`` caps how many times a
    single embed request is retried on a transient blip (dropped connection, 5xx,
    429). Raises :class:`~semdex.domain.errors.EmbeddingError` if httpx is missing,
    no api_key is given, or the server is unreachable.
    """
    if not api_key:
        raise EmbeddingError("gemini requires an api_key (set SEMDEX___EMBEDDING__API_KEY, never inline)")
    resolved_model = model_id or _DEFAULT_MODEL
    url = (endpoint or _DEFAULT_ENDPOINT).rstrip("/") + f"/models/{resolved_model}:batchEmbedContents"
    headers = {"x-goog-api-key": api_key}
    # The default 3072-dim output IS L2-normalized; a reduced outputDimensionality
    # is NOT, which would break cosine scaling. So we never send it and keep the
    # normalized native width.
    request_model = f"models/{resolved_model}"

    def _embed(texts: Sequence[str], *, task: str) -> list[Vector]:
        inputs = list(texts)
        vectors: list[Vector] = []
        for start in range(0, len(inputs), _MAX_BATCH):
            chunk = inputs[start : start + _MAX_BATCH]
            items = [
                _GeminiRequestItem(
                    model=request_model, content=_GeminiContent(parts=[_GeminiPart(text=text)]), task_type=task
                )
                for text in chunk
            ]
            payload = _GeminiBatchRequest(requests=items).model_dump(by_alias=True)
            body = post_embedding_request(
                url=url,
                payload=payload,
                response_model=_GeminiResponse,
                timeout=timeout,
                client=client,
                headers=headers,
                service="gemini",
                install_extra="gemini",
                retries=retries,
            )
            if len(body.embeddings) != len(chunk):
                raise EmbeddingError(
                    f"gemini returned {len(body.embeddings)} embeddings for {len(chunk)} inputs at {url}"
                )
            vectors.extend(tuple(row.values) for row in body.embeddings)
        return vectors

    def passages_fn(texts: Sequence[str]) -> list[Vector]:
        return _embed(texts, task=_TASK_DOCUMENT)

    def query_fn(text: str) -> Vector:
        return _embed([text], task=_TASK_QUERY)[0]

    dim = len(query_fn("dimension probe"))
    return CallableEmbedding(model_id=resolved_model, dim=dim, passages_fn=passages_fn, query_fn=query_fn)


__all__ = [
    "load_gemini_embedding",
]
