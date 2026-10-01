"""``SanitizingExtractor`` - an ``Extract`` decorator that removes machine-content junk.

Wraps any extractor (same ``Extract`` port), runs the machine-content detectors on the returned
text, and - per ``mode`` - flags (``dry_run``: text unchanged, only a logged report) or strips
(``strip``: excise the spans) before the text reaches the chunker. Read-only w.r.t. the source
file (only the derived extraction text is transformed); every action is logged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...domain.models import ExtractedDocument
from .detectors import machine_content_spans

if TYPE_CHECKING:
    from ...adapters.config.sanitize import SanitizeConfig
    from ...application.ports import Extract
    from ...domain.models import SourceRef

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SanitizeReport:
    """What the sanitizer found/did for one document (for the log + a dry-run preview)."""

    source_uri: str
    spans: tuple[tuple[int, int], ...]
    removed_chars: int
    mode: str


def strip_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Remove the (merged, ascending) spans from ``text``, keeping everything else verbatim."""
    kept: list[str] = []
    prev = 0
    for start, end in spans:
        kept.append(text[prev:start])
        prev = end
    kept.append(text[prev:])
    return "".join(kept)


class SanitizingExtractor:
    """Decorate an ``Extract`` to flag/strip machine-content spans from its output."""

    def __init__(self, inner: Extract, config: SanitizeConfig) -> None:
        self._inner = inner
        self._config = config

    def __call__(self, source: SourceRef) -> ExtractedDocument:
        document = self._inner(source)
        cfg = self._config
        spans = machine_content_spans(
            document.text,
            window_chars=cfg.window_chars,
            min_span_chars=cfg.min_span_chars,
            whitespace_ratio_min=cfg.whitespace_ratio_min,
            symbol_ratio_max=cfg.symbol_ratio_max,
            word_length_max=cfg.word_length_max,
            min_detectors=cfg.min_detectors,
        )
        if not spans:
            return document
        removed = sum(end - start for start, end in spans)
        report = SanitizeReport(source.uri, tuple(spans), removed, cfg.mode)
        logger.info(
            "sanitize %s: %d machine-content span(s), %d chars, mode=%s",
            report.source_uri,
            len(report.spans),
            report.removed_chars,
            report.mode,
        )
        if cfg.mode == "strip":
            return ExtractedDocument(source=document.source, text=strip_spans(document.text, spans))
        return document  # dry_run: report only, text unchanged (never silently lose content)
