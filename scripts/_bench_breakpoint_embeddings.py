"""A chonkie BaseEmbeddings that reuses semdex's own embedding providers as the SEMANTIC
chunker's breakpoint model - so the sweep can compare breakpoint models (local static, or a
remote GPU model over an OpenAI-compatible / ollama endpoint) WITHOUT chonkie's OpenAIEmbeddings
extra (catsu/openai).

Bench-only (scripts/); not part of the semdex package. The retrieval embedding still goes through
the package's own providers - this only decides WHERE the semantic chunker cuts.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from chonkie.embeddings.base import BaseEmbeddings

from semdex.composition import build_embedding
from semdex.domain.enums import EmbeddingBackend

# Bench breakpoint-model registry: label -> (backend, model_id, needs_endpoint).
# Static model2vec runs locally (CPU); the rest embed over an endpoint (ollama /v1 or the ST shim).
BREAKPOINT_MODELS: dict[str, tuple[EmbeddingBackend, str, bool]] = {
    "potion-base-32M": (EmbeddingBackend.MODEL2VEC, "minishlab/potion-base-32M", False),
    "potion-multilingual-128M": (EmbeddingBackend.MODEL2VEC, "minishlab/potion-multilingual-128M", False),
    "bge-m3": (EmbeddingBackend.OLLAMA, "bge-m3", True),
    "qwen3-0.6b": (EmbeddingBackend.OLLAMA, "qwen3-embedding:0.6b", True),
    # e5-large + jina-v3 are served by the sentence-transformers OpenAI shim (OPENAI backend, /v1).
    "e5-large": (EmbeddingBackend.OPENAI, "intfloat/multilingual-e5-large", True),
    "jina-v3": (EmbeddingBackend.OPENAI, "jinaai/jina-embeddings-v3", True),
}


# Per-request HTTP budget for an endpoint-backed breakpoint model. Not a magic number: one
# request carries one document's full sentence list, so it must cover the slowest document in the
# corpus, not an average call. 900 s covers the mldr_en long-document sets with headroom while
# still failing eventually if the endpoint really is wedged.
_DEFAULT_TIMEOUT = 900.0

# ollama's physical batch size in tokens for the breakpoint models it serves. Its default of 2048
# is BELOW the 4096 context it serves bge-m3/qwen3-0.6b with, and an input in that gap is neither
# truncatable nor processable: ollama silently re-runs it truncated to 2048 (so the breakpoint
# decision would come from a fraction of a merged sentence group), and in a many-input request the
# call can fail with a 400 - a client error, never retried, which killed a 7h chunk run at 94%.
_DEFAULT_NUM_BATCH = 4096


def _resolve_num_batch(explicit: int | None) -> int:
    """ollama physical batch: explicit argument, else ``SEMDEX_BREAKPOINT_NUM_BATCH``, else default."""
    if explicit is not None:
        return int(explicit)
    raw = os.environ.get("SEMDEX_BREAKPOINT_NUM_BATCH", "").strip()
    if not raw:
        return _DEFAULT_NUM_BATCH
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"SEMDEX_BREAKPOINT_NUM_BATCH must be an integer, got {raw!r}") from None
    if value <= 0:
        raise SystemExit(f"SEMDEX_BREAKPOINT_NUM_BATCH must be > 0, got {value}")
    return value


def _resolve_timeout(explicit: float | None) -> float:
    """Per-request timeout: explicit argument, else ``SEMDEX_BREAKPOINT_TIMEOUT``, else the default."""
    if explicit is not None:
        return float(explicit)
    raw = os.environ.get("SEMDEX_BREAKPOINT_TIMEOUT", "").strip()
    if not raw:
        return _DEFAULT_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        raise SystemExit(f"SEMDEX_BREAKPOINT_TIMEOUT must be a number, got {raw!r}") from None
    if value <= 0:
        raise SystemExit(f"SEMDEX_BREAKPOINT_TIMEOUT must be > 0, got {value}")
    return value


class SemdexBreakpointEmbeddings(BaseEmbeddings):
    """Adapt a semdex EmbeddingProvider to chonkie's BaseEmbeddings interface.

    Subclasses chonkie's BaseEmbeddings (SemanticChunker isinstance-checks it) and implements the
    three abstract members (dimension, embed, get_tokenizer) plus a batched embed_batch, delegating
    the actual embedding to a semdex provider. A gpt2 token counter (shared with the chunk sizing)
    is handed to chonkie so chunk_size stays in the same token unit as the rest of the sweep.
    """

    def __init__(self, provider: Any, dim: int, token_counter: Any) -> None:
        # Skip BaseEmbeddings.__init__ (it runs an optional-dependency availability probe for the
        # built-in handlers); this adapter carries its own already-built provider.
        self._provider = provider
        self._dim = int(dim)
        self._token_counter = token_counter

    # chonkie BaseEmbeddings surface -------------------------------------------------
    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        return np.asarray(self._provider.embed_query(text), dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        vecs = self._provider.embed_passages(list(texts))
        return [np.asarray(v, dtype=np.float32) for v in vecs]

    def get_tokenizer(self) -> Any:
        return self._token_counter

    # chonkie also probes these on some paths; keep them harmless.
    def get_tokenizer_or_token_counter(self) -> Any:
        return self._token_counter

    def similarity(self, u: np.ndarray, v: np.ndarray) -> np.floating:
        denom = float(np.linalg.norm(u) * np.linalg.norm(v)) or 1.0
        return np.float32(float(u @ v) / denom)

    def __repr__(self) -> str:
        return f"SemdexBreakpointEmbeddings(dim={self._dim})"


def build_breakpoint_embeddings(
    label: str,
    *,
    endpoint: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
    num_batch: int | None = None,
) -> Any:
    """Build the chonkie breakpoint embedder for a registry ``label``.

    ``endpoint`` (an ollama base URL or the ST-shim /v1 URL) is required for the endpoint-backed
    models; the static model2vec ones ignore it. Returns a SemdexBreakpointEmbeddings ready to hand
    to ChonkieChunker(semantic_model=...).

    ``timeout`` is the per-request HTTP budget for the endpoint-backed models, defaulting to
    ``SEMDEX_BREAKPOINT_TIMEOUT`` and then to :data:`_DEFAULT_TIMEOUT`. It is deliberately far
    above the loaders' own 60 s: chonkie's SemanticChunker embeds the sentences of a WHOLE
    document in ONE ``embed_batch`` call, so a long document is a single very large request whose
    wall time scales with the document, not with any batch size we control here. At 60 s the
    mldr_en long-document sets timed out and lost hours of GPU work per set.

    ``num_batch`` (``SEMDEX_BREAKPOINT_NUM_BATCH``, then :data:`_DEFAULT_NUM_BATCH`) reaches only
    the ollama-backed models; the shim-backed ones ignore it. It must cover the served context,
    otherwise ollama truncates every longer merged group to its physical batch without saying so.
    """
    from semdex.adapters.chunker.chonkie import resolve_tokenizer

    if label not in BREAKPOINT_MODELS:
        raise SystemExit(f"unknown breakpoint model {label!r}; known: {sorted(BREAKPOINT_MODELS)}")
    backend, model_id, needs_endpoint = BREAKPOINT_MODELS[label]
    ep = endpoint if needs_endpoint else None
    if needs_endpoint and not ep:
        raise SystemExit(f"breakpoint model {label!r} needs an endpoint (SEMDEX_PREEMBED_SEMANTIC_ENDPOINT)")
    provider = build_embedding(
        backend,
        model=model_id,
        endpoint=ep,
        api_key=api_key or os.environ.get("SEMDEX_BREAKPOINT_API_KEY"),
        timeout=_resolve_timeout(timeout),
        num_batch=_resolve_num_batch(num_batch),
        allow_fallback=False,
    )
    counter = resolve_tokenizer("gpt2")
    return SemdexBreakpointEmbeddings(provider, dim=provider.dim, token_counter=counter)
