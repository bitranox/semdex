"""LLM summarizer adapters, each behind the ``SummaryProvider`` port.

    * :mod:`.ollama` - a local ollama server's native ``/api/chat`` over HTTP
    * :mod:`.openai` - any OpenAI-compatible ``/v1/chat/completions`` server over HTTP
    * :class:`._base.CallableSummarizer` - the shared adapter both HTTP providers
      build (``post_chat_request`` is their shared transport)

The tier is off by default (``SummaryBackend.NONE``); the composition root builds
one of these only when the summary tier is enabled.
"""

from __future__ import annotations

from ._base import CallableSummarizer
from .ollama import load_ollama_summarizer
from .openai import load_openai_summarizer

__all__ = [
    "CallableSummarizer",
    "load_ollama_summarizer",
    "load_openai_summarizer",
]
