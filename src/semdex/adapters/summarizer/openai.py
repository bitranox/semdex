"""OpenAI-compatible LLM summarizer (``POST /v1/chat/completions`` over HTTP).

One summarizer for every server that speaks the OpenAI chat API: hosted OpenAI, a
local ollama server's OpenAI-compat ``/v1``, or llama.cpp's ``llama-server``. Like
the ollama summarizer it carries no LLM dependency - only httpx
(``semdex[summary]``). ``endpoint`` is the OpenAI *base URL* (ends in ``/v1``); the
summarizer appends ``/chat/completions``. ``api_key`` (loaded from the environment,
never inlined) becomes a ``Bearer`` header; local ollama/llama.cpp need none. The
input is clipped to ``max_input_chars`` so a huge document does not overflow the
model's context window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from .._http_retry import DEFAULT_RETRIES
from ._base import CallableSummarizer, post_chat_request

if TYPE_CHECKING:
    import httpx

    from ...application.ports import SummaryProvider

_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_ENDPOINT = "https://api.openai.com/v1"
_DEFAULT_MAX_INPUT_CHARS = 8000
_DEFAULT_PROMPT = (
    "Summarize the document below in 1-2 sentences so a reader can decide, without "
    "opening it, whether it is relevant to their question. Start with a short topic "
    "label, then the summary. Write plain text only, no preamble or code fences."
)


class _OpenAIChatMessage(BaseModel):
    """One chat message (system prompt or the document text)."""

    role: str
    content: str


class _OpenAIChatRequest(BaseModel):
    """OpenAI ``/v1/chat/completions`` request body (deterministic)."""

    model_config = ConfigDict(protected_namespaces=())

    model: str
    messages: list[_OpenAIChatMessage]
    temperature: float = 0.0


class _OpenAIChoiceMessage(BaseModel):
    """The assistant message of one chat choice (extra fields ignored)."""

    content: str


class _OpenAIChoice(BaseModel):
    """One choice of an OpenAI chat-completions response."""

    message: _OpenAIChoiceMessage


class _OpenAIChatResponse(BaseModel):
    """OpenAI ``/v1/chat/completions`` response envelope (extra fields ignored)."""

    choices: list[_OpenAIChoice]


def load_openai_summarizer(
    model: str | None = None,
    *,
    endpoint: str | None = None,
    timeout: float = 60.0,
    api_key: str | None = None,
    retries: int = DEFAULT_RETRIES,
    prompt: str | None = None,
    max_input_chars: int = _DEFAULT_MAX_INPUT_CHARS,
    client: httpx.Client | None = None,
) -> SummaryProvider:
    """Wrap an OpenAI-compatible chat server as a SummaryProvider.

    ``endpoint`` is the base URL ending in ``/v1`` (default hosted OpenAI);
    ``/chat/completions`` is appended. ``prompt`` (the system message) controls
    the summary's length and tone; ``max_input_chars`` clips the document text.
    ``retries`` caps how many times a single summary request is retried on a
    transient blip (dropped connection, 5xx, 429). Raises
    :class:`~semdex.domain.errors.SummaryError` if httpx is missing or the
    server is unreachable.
    """
    resolved_model = model or _DEFAULT_MODEL
    resolved_prompt = prompt or _DEFAULT_PROMPT
    url = (endpoint or _DEFAULT_ENDPOINT).rstrip("/") + "/chat/completions"
    # Bearer auth only when a key is supplied; local ollama/llama.cpp need none.
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None

    def summarize_fn(text: str) -> str:
        payload = _OpenAIChatRequest(
            model=resolved_model,
            messages=[
                _OpenAIChatMessage(role="system", content=resolved_prompt),
                _OpenAIChatMessage(role="user", content=text[:max_input_chars]),
            ],
        ).model_dump()
        body = post_chat_request(
            url=url,
            payload=payload,
            response_model=_OpenAIChatResponse,
            timeout=timeout,
            client=client,
            headers=headers,
            service="openai",
            install_extra="summary",
            retries=retries,
        )
        if not body.choices:
            from ...domain.errors import SummaryError

            raise SummaryError(f"openai at {url} returned no choices")
        return body.choices[0].message.content.strip()

    return CallableSummarizer(model_id=resolved_model, summarize_fn=summarize_fn)


__all__ = [
    "load_openai_summarizer",
]
