"""ollama embedding provider (local embedding server over HTTP).

Calls a running ollama server's ``POST /api/embed`` (``nomic-embed-text`` by
default), so no ML dependency lives in semdex - it mirrors the pgvector/extractor
server-backend pattern. Opt-in via ``semdex[ollama]`` (httpx); needs a reachable
ollama host (``[embedding].endpoint``, default ``http://127.0.0.1:11434``). The
dimension is probed once at load time.
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

_DEFAULT_MODEL = "nomic-embed-text"
_DEFAULT_ENDPOINT = "http://127.0.0.1:11434"


class _OllamaEmbedRequest(BaseModel):
    """ollama ``/api/embed`` request body."""

    model_config = ConfigDict(protected_namespaces=())

    model: str
    input: list[str]
    # ollama VRAM-residency window ("30m", "0", "-1"); omitted from the wire body
    # when None so the default request is unchanged. See [health].keep_alive.
    keep_alive: str | None = None
    # Runtime options passed through to ollama (``num_batch``); omitted when None
    # so the default request is unchanged.
    options: dict[str, int] | None = None


class _OllamaEmbedResponse(BaseModel):
    """ollama ``/api/embed`` response (extra fields ignored)."""

    embeddings: list[list[float]]


def load_ollama_embedding(
    model_id: str | None = None,
    *,
    endpoint: str | None = None,
    timeout: float = 60.0,
    retries: int = DEFAULT_RETRIES,
    keep_alive: str | None = None,
    num_batch: int | None = None,
    query_prefix: str = "",
    passage_prefix: str = "",
    client: httpx.Client | None = None,
) -> EmbeddingProvider:
    """Wrap a running ollama embedding server as an EmbeddingProvider.

    Probes the model's dimension once. ``retries`` caps how many times a single
    embed request is retried on a transient blip (dropped connection, 5xx, 429).
    ``keep_alive`` (None keeps ollama's default) sets how long the model stays
    resident in VRAM. Raises :class:`~semdex.domain.errors.EmbeddingError` if httpx
    is missing or the server is unreachable.

    ``num_batch`` is ollama's PHYSICAL batch size in tokens (None keeps ollama's
    default of 2048). It is a correctness knob, not a tuning one: an input longer
    than the physical batch cannot be processed, and ollama's own recovery is to
    silently re-run it truncated to that batch - so with the default, every text
    over 2048 tokens is embedded from its first 2048 tokens only, with a 200 and no
    warning (measured on ollama 0.31.1). In a many-input request that recovery can
    fail outright and the whole call returns 400, which is a client error and is
    therefore NOT retried. Set this to at least the served context length to embed
    long texts in full.

    ``query_prefix`` and ``passage_prefix`` are prepended to the text before it is
    sent. ollama's ``/api/embed`` applies no template of its own, so a model trained
    to receive an instruction only receives one if the client sends it: without this
    a query and a passage went to the server as identical text. Qwen3-Embedding
    wants ``"Instruct: <task>\\nQuery:"`` on the query and NOTHING on the passage,
    E5 wants ``"query: "`` and ``"passage: "``. Both default to empty, so the
    request is unchanged unless a prefix is configured.
    """
    resolved_model = model_id or _DEFAULT_MODEL
    url = (endpoint or _DEFAULT_ENDPOINT).rstrip("/") + "/api/embed"
    options = {"num_batch": num_batch} if num_batch is not None else None

    def _embed(inputs: list[str]) -> list[Vector]:
        """Send already-prefixed text. The two public paths prefix differently, so neither can
        be written in terms of the other: routing a query through ``passages_fn`` would give it
        the PASSAGE prefix, which is exactly the asymmetry this exists to provide."""
        # exclude_none drops keep_alive/options from the body when unset, so the
        # default request is byte-for-byte unchanged.
        payload = _OllamaEmbedRequest(
            model=resolved_model, input=inputs, keep_alive=keep_alive, options=options
        ).model_dump(exclude_none=True)
        body = post_embedding_request(
            url=url,
            payload=payload,
            response_model=_OllamaEmbedResponse,
            timeout=timeout,
            client=client,
            service="ollama",
            install_extra="ollama",
            retries=retries,
        )
        if len(body.embeddings) != len(inputs):
            raise EmbeddingError(f"ollama returned {len(body.embeddings)} embeddings for {len(inputs)} inputs at {url}")
        return [tuple(row) for row in body.embeddings]

    def passages_fn(texts: Sequence[str]) -> list[Vector]:
        return _embed([passage_prefix + text for text in texts])

    def query_fn(text: str) -> Vector:
        return _embed([query_prefix + text])[0]

    dim = len(query_fn("dimension probe"))
    return CallableEmbedding(model_id=resolved_model, dim=dim, passages_fn=passages_fn, query_fn=query_fn)


__all__ = [
    "load_ollama_embedding",
]
