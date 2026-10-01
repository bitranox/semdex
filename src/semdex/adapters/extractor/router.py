"""Per-file extractor routing.

Net routing model per file = directory (subtree) x filetype x scan-detection,
all config-driven with sensible defaults. ``RoutingExtractor`` IS an ``Extract``
(same ``__call__(source) -> ExtractedDocument`` port), so it drops into the index
/ reconcile use cases unchanged - it just resolves WHICH backend a file uses and
delegates:

1. filetype defaults - ``.md``/``.txt``/code -> ``text``; office + ``.pdf`` ->
   ``markitdown`` (text layer); images -> OCR.
2. subtree override - the longest-prefix-matching ``[[dataset.route]]`` under the
   dataset's roots picks the base extractor (else the dataset default, else the
   filetype default).
3. scan-escalation - an image is always a scan; a ``.pdf`` is a scan when its base
   extraction yields fewer than ``scan_min_chars`` characters. A scan routes to
   the route's ``on_scan`` (a vision-OCR backend) when set.

Each named backend is built lazily and cached (one instance per backend a dataset
references), so only the extractors actually used spin up. The per-backend
``build_extractor`` factory is injected by composition (this adapter never imports
composition).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from ...domain.enums import ExtractorBackend
from ...domain.errors import ExtractionError
from ..discovery.location import from_uri

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ...application.ports import Extract
    from ...domain.models import ExtractedDocument, SourceRef
    from ..config.dataset import ExtractorEndpoint, Route

# Plain text / markdown / code: read straight off disk, no converter.
_TEXT_EXTS = frozenset(
    {".md", ".markdown", ".txt", ".text", ".rst", ".log", ".csv", ".tsv", ".json", ".yaml", ".yml", ".toml", ".ini"}
)
# Office + rich documents: a doc converter reads the embedded text layer.
_OFFICE_EXTS = frozenset({".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".html", ".htm", ".epub"})
# Images: always a scan -> OCR / vision.
_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"})
# Backends that can read a scan/image (the escalation targets).
_OCR_BACKENDS = frozenset(
    {
        ExtractorBackend.XBERG,
        ExtractorBackend.DOCLING,
        ExtractorBackend.MINERU,
        ExtractorBackend.OLMOCR,
        ExtractorBackend.OPENAI_VISION,
    }
)


class ExtractorFactory(Protocol):
    """Builds one Extract for a backend + its endpoint/OCR options (``build_extractor``)."""

    def __call__(  # noqa: PLR0913 - mirrors build_extractor's DI seam: backend + endpoint + OCR/vision options
        self,
        backend: ExtractorBackend,
        *,
        endpoint: str | None = ...,
        force_ocr: bool = ...,
        ocr_language: str | None = ...,
        model: str | None = ...,
        api_key: str | None = ...,
        mineru_backend: str | None = ...,
    ) -> Extract: ...


def _filetype_default(suffix: str) -> ExtractorBackend:
    """The zero-config default backend for a non-image file by extension."""
    if suffix in _OFFICE_EXTS:
        return ExtractorBackend.MARKITDOWN
    return ExtractorBackend.TEXT


class RoutingExtractor:
    """An Extract that routes each file to a backend by subtree x filetype x scan."""

    def __init__(  # noqa: PLR0913 - DI seam: routing inputs (roots + default + routes + endpoints + cutoff + factory)
        self,
        *,
        roots: Sequence[Path],
        default_backend: ExtractorBackend,
        routes: Sequence[Route],
        endpoints: Mapping[ExtractorBackend, ExtractorEndpoint],
        scan_min_chars: int,
        factory: ExtractorFactory,
    ) -> None:
        self._roots = tuple(root.resolve() for root in roots)
        self._default = default_backend
        self._routes = tuple(routes)
        self._endpoints = dict(endpoints)
        self._scan_min_chars = scan_min_chars
        self._factory = factory
        self._cache: dict[ExtractorBackend, Extract] = {}

    def __call__(self, source: SourceRef) -> ExtractedDocument:
        path = from_uri(source.uri).resolve()
        suffix = path.suffix.lower()
        route = self._match_route(path)
        on_scan = route.on_scan if route is not None else None

        if suffix in _IMAGE_EXTS:
            backend = self._image_backend(route, on_scan, source)
            return self._extractor(backend)(source)

        base = self._base_backend(suffix, route)
        base_extract = self._extractor(base)
        if suffix == ".pdf" and on_scan is not None:
            document = base_extract(source)
            if len(document.text.strip()) < self._scan_min_chars:
                return self._extractor(on_scan)(source)
            return document
        return base_extract(source)

    def _image_backend(
        self, route: Route | None, on_scan: ExtractorBackend | None, source: SourceRef
    ) -> ExtractorBackend:
        """Resolve the OCR/vision backend for an image (always a scan)."""
        if on_scan is not None:
            return on_scan
        explicit = route.extractor if route is not None and route.extractor is not None else self._default
        if explicit in _OCR_BACKENDS:
            return explicit
        raise ExtractionError(
            f"{source.uri}: an image file needs an OCR/vision extractor - set a [[dataset.route]] "
            "on_scan/extractor or the dataset 'extractor' to an OCR backend (xberg/docling/mineru/olmocr/openai_vision)"
        )

    def _base_backend(self, suffix: str, route: Route | None) -> ExtractorBackend:
        if route is not None and route.extractor is not None:
            return route.extractor
        if self._default is not ExtractorBackend.TEXT:
            return self._default
        return _filetype_default(suffix)

    def _match_route(self, path: Path) -> Route | None:
        """Return the route whose ``subtree`` is the longest prefix of ``path`` under a root."""
        best: Route | None = None
        best_len = -1
        for root in self._roots:
            try:
                rel_parts = path.relative_to(root).parts
            except ValueError:
                continue
            for route in self._routes:
                sub_parts = Path(route.subtree).parts if route.subtree else ()
                if rel_parts[: len(sub_parts)] == sub_parts and len(sub_parts) > best_len:
                    best, best_len = route, len(sub_parts)
        return best

    def _extractor(self, backend: ExtractorBackend) -> Extract:
        cached = self._cache.get(backend)
        if cached is not None:
            return cached
        cfg = self._endpoints.get(backend)
        built = self._factory(
            backend,
            endpoint=cfg.endpoint if cfg is not None else None,
            force_ocr=cfg.force_ocr if cfg is not None else False,
            ocr_language=cfg.ocr_language if cfg is not None else None,
            model=cfg.model if cfg is not None else None,
            api_key=cfg.api_key if cfg is not None else None,
            mineru_backend=cfg.mineru_backend if cfg is not None else None,
        )
        self._cache[backend] = built
        return built


# Static conformance assertion -- pyright verifies RoutingExtractor satisfies Extract.
if TYPE_CHECKING:
    _assert_extract: Extract = RoutingExtractor(
        roots=(),
        default_backend=ExtractorBackend.TEXT,
        routes=(),
        endpoints={},
        scan_min_chars=24,
        factory=None,  # type: ignore[arg-type]
    )


__all__ = [
    "ExtractorFactory",
    "RoutingExtractor",
]
