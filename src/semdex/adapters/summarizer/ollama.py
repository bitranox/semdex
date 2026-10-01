"""ollama LLM summarizer (local chat server over HTTP).

Calls a running ollama server's ``POST /api/chat`` with a chat/instruct model
(``llama3.2`` by default), so no LLM dependency lives in semdex - it mirrors the
ollama embedding provider. Opt-in via ``semdex[summary]`` (httpx); needs a
reachable ollama host (default ``http://127.0.0.1:11434``). Produces one
per-document summary at index time (``stream`` off, ``temperature`` 0 for a
stable result). The input is clipped to ``max_input_chars`` so a huge document
does not blow the model's context window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from .._http_retry import DEFAULT_RETRIES
from ._base import CallableSummarizer, post_chat_request

if TYPE_CHECKING:
    import httpx

    from ...application.ports import SummaryProvider

_DEFAULT_MODEL = "llama3.2"
_DEFAULT_ENDPOINT = "http://127.0.0.1:11434"
_DEFAULT_MAX_INPUT_CHARS = 8000
_DEFAULT_PROMPT = (
    "Summarize the document below in 1-2 sentences so a reader can decide, without "
    "opening it, whether it is relevant to their question. Start with a short topic "
    "label, then the summary. Write plain text only, no preamble or code fences."
)


class _OllamaChatMessage(BaseModel):
    """One chat message (system prompt or the document text)."""

    role: str
    content: str


class _OllamaChatRequest(BaseModel):
    """ollama ``/api/chat`` request body (non-streaming, deterministic)."""

    model_config = ConfigDict(protected_namespaces=())

    model: str
    messages: list[_OllamaChatMessage]
    stream: bool = False
    options: dict[str, float] = Field(default_factory=lambda: {"temperature": 0.0})
    # ollama VRAM-residency window ("30m", "0", "-1"); omitted from the wire body
    # when None so the default request is unchanged. See [health].keep_alive.
    keep_alive: str | None = None


class _OllamaChatResponseMessage(BaseModel):
    """The assistant message of an ollama chat reply (extra fields ignored)."""

    content: str


class _OllamaChatResponse(BaseModel):
    """ollama ``/api/chat`` response envelope (extra fields ignored)."""

    message: _OllamaChatResponseMessage


def load_ollama_summarizer(
    model: str | None = None,
    *,
    endpoint: str | None = None,
    timeout: float = 60.0,
    retries: int = DEFAULT_RETRIES,
    keep_alive: str | None = None,
    prompt: str | None = None,
    max_input_chars: int = _DEFAULT_MAX_INPUT_CHARS,
    client: httpx.Client | None = None,
) -> SummaryProvider:
    """Wrap a running ollama chat server as a SummaryProvider.

    ``prompt`` (the system message) controls the summary's length and tone;
    ``max_input_chars`` clips the document text sent to the model. ``retries`` caps
    how many times a single summary request is retried on a transient blip (dropped
    connection, 5xx, 429). ``keep_alive`` (None keeps ollama's default) sets how
    long the model stays resident in VRAM. Raises
    :class:`~semdex.domain.errors.SummaryError` if httpx is missing or the server
    is unreachable.
    """
    resolved_model = model or _DEFAULT_MODEL
    resolved_prompt = prompt or _DEFAULT_PROMPT
    url = (endpoint or _DEFAULT_ENDPOINT).rstrip("/") + "/api/chat"

    def summarize_fn(text: str) -> str:
        # exclude_none drops keep_alive from the body when unset, so the default
        # request is byte-for-byte unchanged.
        payload = _OllamaChatRequest(
            model=resolved_model,
            messages=[
                _OllamaChatMessage(role="system", content=resolved_prompt),
                _OllamaChatMessage(role="user", content=text[:max_input_chars]),
            ],
            keep_alive=keep_alive,
        ).model_dump(exclude_none=True)
        body = post_chat_request(
            url=url,
            payload=payload,
            response_model=_OllamaChatResponse,
            timeout=timeout,
            client=client,
            service="ollama",
            install_extra="summary",
            retries=retries,
        )
        return body.message.content.strip()

    return CallableSummarizer(model_id=resolved_model, summarize_fn=summarize_fn)


__all__ = [
    "load_ollama_summarizer",
]
