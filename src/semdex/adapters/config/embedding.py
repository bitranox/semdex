"""Embedding configuration model parsed from the ``[embedding]`` section.

Selects which embedding provider turns text into vectors. The composition root
turns this into a concrete adapter behind the ``EmbeddingProvider`` port. The
default is the light ``fastembed`` provider; a real model dictates its own
dimension, so ``[index].embedding_dim`` applies only to the placeholder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from ...domain.enums import EmbeddingBackend

if TYPE_CHECKING:
    from lib_layered_config import Config


class EmbeddingConfig(BaseModel):
    """Validated, immutable embedding provider selection.

    Example:
        >>> EmbeddingConfig().provider.value
        'fastembed'
        >>> EmbeddingConfig().model is None
        True
    """

    model_config = ConfigDict(frozen=True)

    provider: EmbeddingBackend = EmbeddingBackend.FASTEMBED
    # Model id for the selected provider. ``None`` uses that provider's own
    # built-in default (fastembed -> bge-small, model2vec -> potion, ollama ->
    # nomic-embed-text, openai -> text-embedding-3-small). Ignored by placeholder.
    model: str | None = None
    # Base URL of the embedding server; used by the HTTP providers (``ollama``,
    # ``openai``). For ``openai`` it is the OpenAI base URL ending in ``/v1``.
    endpoint: str | None = None
    # Thread count for the fastembed (onnxruntime) provider; None uses all CPUs.
    # Set explicitly to silence onnxruntime CPU-affinity warnings in containers.
    threads: int | None = Field(default=None, gt=0)
    # Passages per ``embed_passages`` call at index time; None embeds all of a source's
    # chunks in one call (today's behavior). Set it to cap peak memory / request size on a
    # constrained embed server, or to bound a slow CPU/HTTP batch. The ollama loaders already
    # sub-batch internally, so this is mainly a cap for the in-process providers.
    batch: int | None = Field(default=None, gt=0)
    # Per-request HTTP timeout (seconds) for the ``ollama``/``openai`` providers;
    # None keeps the loader default (60 s). Raise it for slow CPU embed servers.
    timeout: float | None = Field(default=None, gt=0)
    # ollama PHYSICAL batch size in tokens; None keeps ollama's default (2048).
    # Correctness, not tuning: ollama cannot process an input longer than this and
    # silently re-runs it truncated to the batch (HTTP 200, no warning), or fails the
    # whole many-input call with a 400 that is a client error and so is never retried.
    # Set it to at least the served context length to embed long texts in full.
    num_batch: int | None = Field(default=None, gt=0)
    # Total attempts per embed request for the ``ollama``/``openai`` providers
    # before the error propagates (transient blips only: dropped connection, 5xx,
    # 429), with exponential backoff. None keeps the loader default (4).
    retries: int | None = Field(default=None, ge=1)
    # Bearer token for the ``openai`` provider (hosted OpenAI or a secured
    # gateway). Local ollama/llama.cpp need none. SECRET: set it from the
    # environment (e.g. ``SEMDEX___EMBEDDING__API_KEY``), never inline in a
    # committed config file.
    api_key: str | None = None
    # Instruction prefixes for models trained to receive one, applied by the
    # ``ollama``, ``openai`` and ``sentence_transformers`` providers. Those speak
    # raw text to a server that applies no template, so an instruction-tuned model
    # gets an instruction only if it is sent. Empty leaves the text untouched.
    # Qwen3-Embedding wants "Instruct: <task>\nQuery:" on the query and NOTHING on
    # the passage; E5 wants "query: " / "passage: ". Changing either changes the
    # vectors, so reindex afterwards.
    query_prefix: str = ""
    passage_prefix: str = ""


def get_embedding_config(config: Config) -> EmbeddingConfig:
    """Parse the ``[embedding]`` section into an EmbeddingConfig.

    Falls back to the fastembed provider defaults when the section is absent.

    Example:
        >>> from lib_layered_config import Config
        >>> get_embedding_config(Config({}, {})).provider.value
        'fastembed'
    """
    return EmbeddingConfig.model_validate(config.get("embedding", {}))


__all__ = [
    "EmbeddingConfig",
    "get_embedding_config",
]
