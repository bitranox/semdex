"""Unit tests for the machine-content detector (minified JS/CSS, base64, dense blobs)."""

from __future__ import annotations

import pytest

from semdex.adapters.sanitize.detectors import machine_content_spans

pytestmark = pytest.mark.os_agnostic


def _spans(text: str) -> list[tuple[int, int]]:
    """machine_content_spans with the config defaults (typed, so no int|float **-unpack)."""
    return machine_content_spans(
        text,
        window_chars=400,
        min_span_chars=400,
        whitespace_ratio_min=0.08,
        symbol_ratio_max=0.35,
        word_length_max=15.0,
        min_detectors=2,
    )


_PROSE_DE = (
    "Das Unternehmen liefert seit vielen Jahren zuverlaessige Loesungen fuer den Mittelstand. "
    "Unsere Kunden schaetzen die persoenliche Beratung und die kurzen Reaktionszeiten. "
) * 6
_PROSE_EN = (
    "The library provides a small, dependency-light interface for semantic search over local files. "
    "It watches a folder and keeps a vector index in sync without ever mutating the sources. "
) * 6
_MINIFIED_JS = (
    "!function(t){function n(e){if(r[e])return r[e].exports;var i=r[e]={i:e,l:!1,exports:{}};"
    "return t[e].call(i.exports,i,i.exports,n),i.l=!0,i.exports}var r={};n.m=t,n.c=r;" * 8
)
_BASE64 = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZnaGlqa2xtbm9wcXJzdHV2d3h5eg" * 12


def test_prose_is_not_flagged() -> None:
    assert _spans(_PROSE_DE) == []
    assert _spans(_PROSE_EN) == []


def test_minified_js_is_flagged() -> None:
    spans = _spans(_MINIFIED_JS)
    assert spans, "expected the minified-JS blob to be flagged as machine content"
    # the flagged span covers (most of) the blob
    covered = sum(e - s for s, e in spans)
    assert covered >= len(_MINIFIED_JS) * 0.5


def test_base64_blob_is_flagged() -> None:
    assert _spans(_BASE64), "expected a long base64 blob to be flagged"


def test_short_junk_below_min_span_is_left_alone() -> None:
    # a tiny code snippet embedded in prose stays (below min_span_chars)
    text = _PROSE_EN[:300] + " `a={b:1};` " + _PROSE_EN[:300]
    assert _spans(text) == []


def test_junk_span_inside_prose_is_isolated_not_the_whole_doc() -> None:
    text = _PROSE_EN + _MINIFIED_JS + _PROSE_EN
    spans = _spans(text)
    assert spans
    # the prose ends are NOT inside any flagged span
    assert not any(s <= 5 for s, _e in spans)
