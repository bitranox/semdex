"""Summary configuration model parsed from the ``[summary]`` section.

Selects the LLM (if any) that generates the opt-in per-document summary attached
to search hits. The composition root turns this into a concrete adapter behind
the ``SummaryProvider`` port. The default provider is ``NONE`` - the tier is OFF,
so hits carry no summary and behaviour is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from ...domain.enums import SummaryBackend

if TYPE_CHECKING:
    from lib_layered_config import Config

# The default triage prompt (kept in sync with the summarizer adapters' own
# fallbacks). It controls the summary's length and tone: composition always
# passes this through to the loader.
_DEFAULT_PROMPT = (
    "Summarize the document below in 1-2 sentences so a reader can decide, without "
    "opening it, whether it is relevant to their question. Start with a short topic "
    "label, then the summary. Write plain text only, no preamble or code fences."
)


class SummaryConfig(BaseModel):
    """Validated, immutable summary-tier selection.

    Example:
        >>> SummaryConfig().provider.value
        'none'
        >>> SummaryConfig().max_input_chars
        8000
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    provider: SummaryBackend = SummaryBackend.NONE
    # Chat model id for the selected provider. ``None`` uses that provider's own
    # built-in default (ollama -> llama3.2, openai -> gpt-4o-mini). Ignored by NONE.
    model: str | None = None
    # Base URL of the chat server; used by the HTTP providers (``ollama``,
    # ``openai``). For ``openai`` it is the OpenAI base URL ending in ``/v1``.
    endpoint: str | None = None
    # Bearer token for the ``openai`` provider (hosted OpenAI or a secured
    # gateway). Local ollama/llama.cpp need none. SECRET: set it from the
    # environment (e.g. ``SEMDEX___SUMMARY__API_KEY``), never inline in a
    # committed config file.
    api_key: str | None = None
    # Per-request HTTP timeout (seconds) for the ``ollama``/``openai`` providers;
    # None keeps the loader default (60 s). Raise it for slow CPU chat servers.
    timeout: float | None = Field(default=None, gt=0)
    # Total attempts per summary request for the ``ollama``/``openai`` providers
    # before the error propagates (transient blips only: dropped connection, 5xx,
    # 429), with exponential backoff. None keeps the loader default (4).
    retries: int | None = Field(default=None, ge=1)
    # Clip the document text sent to the model, so a huge source does not overflow
    # the model's context window (one summary per document, not per chunk).
    max_input_chars: int = Field(default=8000, gt=0)
    # The system prompt: the length/tone control. Ask for a leading category to
    # fold a classification into the summary (there is no separate tag column).
    prompt: str = _DEFAULT_PROMPT


def get_summary_config(config: Config) -> SummaryConfig:
    """Parse the ``[summary]`` section into a SummaryConfig.

    Falls back to the tier-off default (``provider = none``) when the section is
    absent.

    Example:
        >>> from lib_layered_config import Config
        >>> get_summary_config(Config({}, {})).provider.value
        'none'
    """
    return SummaryConfig.model_validate(config.get("summary", {}))


__all__ = [
    "SummaryConfig",
    "get_summary_config",
]
