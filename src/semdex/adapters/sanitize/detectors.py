"""Deterministic machine-content detectors (no ML, no tokenizer - fully CI-testable).

Scores fixed-size character windows on cheap heuristics and returns the ``(start, end)`` char
spans that look like machine content (minified JS/CSS, base64/data blobs): low whitespace, high
symbol density, or huge unbroken runs. Adjacent junk windows merge; only spans at least
``min_span_chars`` long are returned, so a short code snippet in real prose is left alone.
"""

from __future__ import annotations


def _detectors_tripped(
    window: str, *, whitespace_ratio_min: float, symbol_ratio_max: float, word_length_max: float
) -> int:
    """How many machine-content signals a window trips (higher = more junk-like)."""
    n = len(window)
    if n == 0:
        return 0
    whitespace_ratio = sum(1 for c in window if c.isspace()) / n
    symbol_ratio = sum(1 for c in window if not c.isalnum() and not c.isspace()) / n
    words = window.split()
    mean_word_length = (sum(len(w) for w in words) / len(words)) if words else float(n)
    tripped = 0
    if whitespace_ratio < whitespace_ratio_min:  # minified code / base64: almost no spaces
        tripped += 1
    if symbol_ratio > symbol_ratio_max:  # code: many braces/operators/punctuation
        tripped += 1
    if mean_word_length > word_length_max:  # base64 / minified: huge unbroken runs
        tripped += 1
    return tripped


def _merge_adjacent(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge contiguous ``(start, end)`` windows into spans."""
    merged: list[tuple[int, int]] = []
    for start, end in windows:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def machine_content_spans(  # noqa: PLR0913 - each arg is a distinct, independently-tunable threshold
    text: str,
    *,
    window_chars: int,
    min_span_chars: int,
    whitespace_ratio_min: float,
    symbol_ratio_max: float,
    word_length_max: float,
    min_detectors: int,
) -> list[tuple[int, int]]:
    """Return the char spans of ``text`` that look like machine content, longest-safe first.

    A window is machine content when at least ``min_detectors`` heuristics trip. Adjacent junk
    windows merge; spans shorter than ``min_span_chars`` are dropped (a short snippet in prose
    is not junk). Empty list means nothing to sanitize.
    """
    junk: list[tuple[int, int]] = []
    for start in range(0, len(text), window_chars):
        window = text[start : start + window_chars]
        tripped = _detectors_tripped(
            window,
            whitespace_ratio_min=whitespace_ratio_min,
            symbol_ratio_max=symbol_ratio_max,
            word_length_max=word_length_max,
        )
        if tripped >= min_detectors:
            junk.append((start, min(start + window_chars, len(text))))
    return [(s, e) for s, e in _merge_adjacent(junk) if e - s >= min_span_chars]
