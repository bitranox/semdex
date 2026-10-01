"""Composition factory dispatch for the opt-in summarizer (build_summarizer)."""

from __future__ import annotations

import pytest

from semdex.composition import build_summarizer
from semdex.domain.enums import SummaryBackend

pytestmark = pytest.mark.os_agnostic


def test_none_provider_returns_none() -> None:
    """NONE is the tier-off signal: no summarizer is built."""
    assert build_summarizer(SummaryBackend.NONE) is None


def test_ollama_provider_builds_without_touching_the_network() -> None:
    """The ollama loader constructs lazily (no probe), so the model id is set with no server."""
    summarizer = build_summarizer(SummaryBackend.OLLAMA, model="llama3.2", endpoint="http://h:11434")
    assert summarizer is not None
    assert summarizer.model_id == "llama3.2"


def test_openai_provider_builds_without_touching_the_network() -> None:
    """The openai loader constructs lazily (no probe), so the model id is set with no server."""
    summarizer = build_summarizer(SummaryBackend.OPENAI, model="gpt-4o-mini", endpoint="http://h:8080/v1")
    assert summarizer is not None
    assert summarizer.model_id == "gpt-4o-mini"


def test_provider_default_model_is_used_when_model_is_none() -> None:
    """model=None falls back to the provider's own default chat model."""
    summarizer = build_summarizer(SummaryBackend.OLLAMA)
    assert summarizer is not None
    assert summarizer.model_id == "llama3.2"
