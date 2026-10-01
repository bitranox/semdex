"""Embedding provider adapters, each behind the ``EmbeddingProvider`` port.

    * :mod:`.fastembed` - ONNX, no torch, the default provider (light, offline
      after first download)
    * :mod:`.model2vec` - static embeddings, the lightest real provider
    * :mod:`.ollama` - a local ollama server's native ``/api/embed`` over HTTP
    * :mod:`.openai` - any OpenAI-compatible ``/v1/embeddings`` server over HTTP
    * :mod:`.gemini` - Google's native ``batchEmbedContents`` cloud API over HTTP
    * :mod:`.cohere` - Cohere's native ``/v2/embed`` cloud API over HTTP
    * :mod:`.sentence_transformers` - the heavy (torch) opt-in
    * :class:`._base.CallableEmbedding` - the shared adapter the HTTP/local
      providers build (``post_embedding_request`` is their shared transport)

The deterministic token-hash placeholder lives in :mod:`..memory.index`.
"""

from __future__ import annotations

from ._base import CallableEmbedding
from .cohere import load_cohere_embedding
from .fastembed import load_fastembed_embedding
from .gemini import load_gemini_embedding
from .model2vec import load_model2vec_embedding
from .ollama import load_ollama_embedding
from .openai import load_openai_embedding
from .sentence_transformers import SentenceTransformerEmbedding, load_sentence_transformer_embedding

__all__ = [
    "CallableEmbedding",
    "SentenceTransformerEmbedding",
    "load_cohere_embedding",
    "load_fastembed_embedding",
    "load_gemini_embedding",
    "load_model2vec_embedding",
    "load_ollama_embedding",
    "load_openai_embedding",
    "load_sentence_transformer_embedding",
]
