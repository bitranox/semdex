"""Content sanitizer: strip machine-content junk (minified JS/CSS, base64) from extracted text.

An ``Extract`` decorator (``SanitizingExtractor``) that scans the derived extraction text for
machine-content spans and, per mode, flags (dry-run) or strips them before chunking. Read-only
w.r.t. the source; opt-in; never silently loses content (dry-run + a logged report).
"""

from .detectors import machine_content_spans
from .extractor import SanitizingExtractor

__all__ = [
    "SanitizingExtractor",
    "machine_content_spans",
]
