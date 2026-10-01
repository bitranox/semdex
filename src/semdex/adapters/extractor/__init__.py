"""Extractor adapters: convert source files to text/markdown.

Every adapter implements the ``Extract`` port. ``TextExtractor`` is the embedded,
zero-dependency default. The four doc converters each run as their own Docker
container reached over HTTP (so semdex avoids their heavy deps) and are opt-in
via a matching extra:

    * :class:`.text.TextExtractor` - embedded plain text/markdown reader
    * :class:`.markitdown.MarkitdownExtractor` - markitdown-mcp (MCP)
    * :class:`.xberg.XbergExtractor` - Xberg REST (OCR, CPU)
    * :class:`.docling.DoclingExtractor` - docling-serve REST (CPU image)
    * :class:`.mineru.MineruExtractor` - MinerU REST (GPU host; classic or VLM)
    * :class:`.vision_ocr.VisionOcrExtractor` - OpenAI-vision OCR (olmOCR / any VLM)
    * :class:`.router.RoutingExtractor` - per-file backend routing (subtree x
      filetype x scan-detection)
"""

from __future__ import annotations

from .docling import DoclingExtractor
from .markitdown import MarkitdownExtractor
from .mineru import MineruExtractor
from .router import ExtractorFactory, RoutingExtractor
from .text import TextExtractor
from .vision_ocr import VisionOcrExtractor
from .xberg import XbergExtractor

__all__ = [
    "DoclingExtractor",
    "ExtractorFactory",
    "MarkitdownExtractor",
    "MineruExtractor",
    "RoutingExtractor",
    "TextExtractor",
    "VisionOcrExtractor",
    "XbergExtractor",
]
