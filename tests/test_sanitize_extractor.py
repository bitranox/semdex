"""Unit tests for the SanitizingExtractor decorator (dry_run flags, strip removes)."""

from __future__ import annotations

from typing import Literal

import pytest

from semdex.adapters.config.sanitize import SanitizeConfig
from semdex.adapters.sanitize import SanitizingExtractor
from semdex.domain.models import ExtractedDocument, SourceRef

pytestmark = pytest.mark.os_agnostic

_SRC = SourceRef(uri="/x/doc.md", label="", content_hash="", mtime=0.0)
_PROSE = (
    "The library provides a small, dependency-light interface for semantic search over local files. "
    "It watches a folder and keeps a vector index in sync without ever mutating the sources. "
) * 6
_JS = (
    "!function(t){function n(e){if(r[e])return r[e].exports;var i=r[e]={i:e,l:!1,exports:{}};"
    "return t[e].call(i.exports,i,i.exports,n),i.l=!0,i.exports}var r={};n.m=t,n.c=r;" * 8
)


def _cfg(mode: Literal["dry_run", "strip"]) -> SanitizeConfig:
    return SanitizeConfig(enabled=True, mode=mode)  # thresholds keep their documented defaults


class _FakeInner:
    def __init__(self, text: str) -> None:
        self._text = text

    def __call__(self, source: SourceRef) -> ExtractedDocument:
        return ExtractedDocument(source=source, text=self._text)


def test_dry_run_keeps_text_unchanged() -> None:
    text = _PROSE + _JS + _PROSE
    out = SanitizingExtractor(_FakeInner(text), _cfg("dry_run"))(_SRC)
    assert out.text == text  # dry_run reports but never mutates


def test_strip_removes_the_junk_keeps_prose() -> None:
    from semdex.adapters.sanitize import machine_content_spans

    text = _PROSE + _JS + _PROSE
    cfg = _cfg("strip")
    out = SanitizingExtractor(_FakeInner(text), cfg)(_SRC)
    assert "semantic search over local files" in out.text  # prose kept
    assert len(out.text) < len(text)  # the JS bulk is gone
    assert out.source == _SRC  # provenance preserved
    # the functional contract: NO machine-content span survives (no giant-chunk risk); a tiny
    # sub-min-span boundary fragment may remain, which is by design (not over-aggressive).

    assert (
        machine_content_spans(
            out.text,
            window_chars=cfg.window_chars,
            min_span_chars=cfg.min_span_chars,
            whitespace_ratio_min=cfg.whitespace_ratio_min,
            symbol_ratio_max=cfg.symbol_ratio_max,
            word_length_max=cfg.word_length_max,
            min_detectors=cfg.min_detectors,
        )
        == []
    )


def test_clean_document_passes_through_unchanged() -> None:
    out = SanitizingExtractor(_FakeInner(_PROSE), _cfg("strip"))(_SRC)
    assert out.text == _PROSE  # nothing flagged
