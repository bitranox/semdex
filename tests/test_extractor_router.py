"""Unit tests for the per-file extractor router.

The router resolves a backend from the file PATH (subtree x filetype x scan) and
delegates - it never reads the file itself - so these tests inject recording fake
extractors and assert the routing decisions with paths that need not exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from semdex.adapters.config.dataset import ExtractorEndpoint, Route
from semdex.adapters.discovery.location import to_uri
from semdex.adapters.extractor.router import RoutingExtractor
from semdex.domain.enums import ExtractorBackend
from semdex.domain.errors import ExtractionError
from semdex.domain.models import ExtractedDocument, SourceRef

if TYPE_CHECKING:
    from collections.abc import Mapping

pytestmark = pytest.mark.os_agnostic

B = ExtractorBackend


class _Fake:
    """A fake Extract that records its calls and returns a fixed text."""

    def __init__(self, backend: ExtractorBackend, text: str) -> None:
        self.backend = backend
        self.text = text
        self.calls: list[str] = []

    def __call__(self, source: SourceRef) -> ExtractedDocument:
        self.calls.append(source.uri)
        return ExtractedDocument(source=source, text=self.text)


class _Factory:
    """An ExtractorFactory that builds (once) a recording fake per backend."""

    def __init__(self, texts: Mapping[ExtractorBackend, str]) -> None:
        self._texts = dict(texts)
        self.built: dict[ExtractorBackend, _Fake] = {}
        self.build_count: dict[ExtractorBackend, int] = {}

    def __call__(
        self,
        backend: ExtractorBackend,
        *,
        endpoint: str | None = None,
        force_ocr: bool = False,
        ocr_language: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        mineru_backend: str | None = None,
    ) -> _Fake:
        self.build_count[backend] = self.build_count.get(backend, 0) + 1
        fake = _Fake(backend, self._texts.get(backend, "text"))
        self.built[backend] = fake
        return fake


def _router(
    root: Path,
    *,
    default: ExtractorBackend = B.TEXT,
    routes: tuple[Route, ...] = (),
    texts: dict[ExtractorBackend, str] | None = None,
    scan_min_chars: int = 10,
) -> tuple[RoutingExtractor, _Factory]:
    factory = _Factory(texts or {})
    router = RoutingExtractor(
        roots=[root],
        default_backend=default,
        routes=routes,
        endpoints={},
        scan_min_chars=scan_min_chars,
        factory=factory,
    )
    return router, factory


def _src(root: Path, rel: str) -> SourceRef:
    return SourceRef(uri=to_uri(root / rel), label="", content_hash="", mtime=0.0)


def test_filetype_default_text_and_office(tmp_path: Path) -> None:
    """With the default TEXT backend: .md -> text, .pdf -> markitdown (text layer)."""
    router, factory = _router(tmp_path)
    router(_src(tmp_path, "notes/a.md"))
    router(_src(tmp_path, "docs/b.pdf"))
    assert factory.built[B.TEXT].calls == [to_uri(tmp_path / "notes/a.md")]
    assert factory.built[B.MARKITDOWN].calls == [to_uri(tmp_path / "docs/b.pdf")]


def test_dataset_default_overrides_filetype(tmp_path: Path) -> None:
    """An explicit non-TEXT dataset default applies to every file, incl .md."""
    router, factory = _router(tmp_path, default=B.MARKITDOWN)
    router(_src(tmp_path, "a.md"))
    assert set(factory.built) == {B.MARKITDOWN}


def test_subtree_longest_prefix_wins(tmp_path: Path) -> None:
    """The longest matching [[dataset.route]] subtree selects the backend."""
    routes = (
        Route(subtree="a", extractor=B.DOCLING),
        Route(subtree="a/b", extractor=B.MINERU),
    )
    router, factory = _router(tmp_path, routes=routes)
    router(_src(tmp_path, "a/x.pdf"))
    router(_src(tmp_path, "a/b/y.pdf"))
    assert factory.built[B.DOCLING].calls == [to_uri(tmp_path / "a/x.pdf")]
    assert factory.built[B.MINERU].calls == [to_uri(tmp_path / "a/b/y.pdf")]


def test_image_escalates_to_on_scan(tmp_path: Path) -> None:
    """An image (always a scan) routes to the route's on_scan vision OCR."""
    routes = (Route(subtree="scans", extractor=B.MINERU, on_scan=B.OLMOCR),)
    router, factory = _router(tmp_path, routes=routes)
    router(_src(tmp_path, "scans/photo.png"))
    assert set(factory.built) == {B.OLMOCR}
    assert factory.built[B.OLMOCR].calls == [to_uri(tmp_path / "scans/photo.png")]


def test_image_without_ocr_raises(tmp_path: Path) -> None:
    """An image under a TEXT default with no OCR route is a clear error, not garbage."""
    router, _ = _router(tmp_path)
    with pytest.raises(ExtractionError):
        router(_src(tmp_path, "loose/photo.jpg"))


def test_scanned_pdf_escalates_when_text_layer_is_thin(tmp_path: Path) -> None:
    """A .pdf whose base extraction yields < scan_min_chars escalates to on_scan."""
    routes = (Route(subtree="docs", extractor=B.MARKITDOWN, on_scan=B.OLMOCR),)
    texts = {B.MARKITDOWN: "", B.OLMOCR: "OCR TEXT"}
    router, factory = _router(tmp_path, routes=routes, texts=texts, scan_min_chars=10)
    doc = router(_src(tmp_path, "docs/scan.pdf"))
    assert doc.text == "OCR TEXT"
    assert factory.built[B.OLMOCR].calls  # vision OCR ran


def test_text_layer_pdf_keeps_base_extractor(tmp_path: Path) -> None:
    """A .pdf with a real text layer (>= scan_min_chars) uses the base, not on_scan."""
    routes = (Route(subtree="docs", extractor=B.MARKITDOWN, on_scan=B.OLMOCR),)
    texts = {B.MARKITDOWN: "this is a full text layer with plenty of characters"}
    router, factory = _router(tmp_path, routes=routes, texts=texts, scan_min_chars=10)
    doc = router(_src(tmp_path, "docs/report.pdf"))
    assert doc.text.startswith("this is a full text layer")
    assert B.OLMOCR not in factory.built  # no escalation


def test_backend_extractor_is_built_once_and_cached(tmp_path: Path) -> None:
    """Each backend is built once even across many files (per-backend cache)."""
    router, factory = _router(tmp_path, default=B.MARKITDOWN)
    for i in range(4):
        router(_src(tmp_path, f"d/f{i}.pdf"))
    assert factory.build_count[B.MARKITDOWN] == 1
    assert len(factory.built[B.MARKITDOWN].calls) == 4


def test_path_outside_roots_falls_back_to_filetype_default(tmp_path: Path) -> None:
    """A file under no configured root ignores routes and uses the filetype default."""
    routes = (Route(subtree="scans", extractor=B.MINERU),)
    router, factory = _router(tmp_path, routes=routes)
    router(_src(tmp_path.parent, "elsewhere/a.pdf"))  # not under tmp_path
    assert set(factory.built) == {B.MARKITDOWN}  # filetype default, route ignored


def test_endpoints_thread_through_to_the_factory(tmp_path: Path) -> None:
    """extractor_config for a backend is passed to the factory (endpoint/model/flags)."""
    seen: dict[str, object] = {}

    def factory(
        backend: ExtractorBackend,
        *,
        endpoint: str | None = None,
        force_ocr: bool = False,
        ocr_language: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        mineru_backend: str | None = None,
    ) -> _Fake:
        seen.update({"backend": backend, "endpoint": endpoint, "model": model, "mineru_backend": mineru_backend})
        return _Fake(backend, "x")

    endpoints = {
        B.MINERU: ExtractorEndpoint(backend=B.MINERU, endpoint="http://ocr:8000", mineru_backend="vlm-transformers")
    }
    router = RoutingExtractor(
        roots=[tmp_path],
        default_backend=B.MINERU,
        routes=(),
        endpoints=endpoints,
        scan_min_chars=10,
        factory=factory,
    )
    router(_src(tmp_path, "a.pdf"))
    assert seen == {
        "backend": B.MINERU,
        "endpoint": "http://ocr:8000",
        "model": None,
        "mineru_backend": "vlm-transformers",
    }


def test_dataset_config_parses_route_alias_and_extractor_config() -> None:
    """A [[dataset.route]] (singular TOML key) + [[dataset.extractor_config]] parse cleanly."""
    from semdex.adapters.config.dataset import DatasetConfig

    cfg = DatasetConfig.model_validate(
        {
            "name": "kb",
            "sources": ["~/kb"],
            "extractor": "text",
            "scan_min_chars": 40,
            "extractor_config": [
                {"backend": "mineru", "endpoint": "http://ocr:8000", "mineru_backend": "vlm-transformers"},
                {"backend": "olmocr", "endpoint": "http://vllm:8100/v1", "model": "olmocr"},
            ],
            "route": [
                {"subtree": "scans", "extractor": "mineru", "on_scan": "olmocr"},
                {"subtree": "notes", "extractor": "text"},
            ],
        }
    )
    assert cfg.scan_min_chars == 40
    assert [r.subtree for r in cfg.routes] == ["scans", "notes"]
    assert cfg.routes[0].on_scan is B.OLMOCR
    assert {e.backend for e in cfg.extractor_config} == {B.MINERU, B.OLMOCR}


def test_writable_dataset_rejects_routes() -> None:
    """A writable knowledge dataset extracts nothing, so routes are a config error."""
    from pydantic import ValidationError

    from semdex.adapters.config.dataset import DatasetConfig

    with pytest.raises(ValidationError):
        DatasetConfig.model_validate({"name": "kb", "writable": True, "route": [{"subtree": "x"}]})


def test_duplicate_extractor_config_backend_rejected() -> None:
    """Two extractor_config entries for the same backend is a config error."""
    from pydantic import ValidationError

    from semdex.adapters.config.dataset import DatasetConfig

    with pytest.raises(ValidationError):
        DatasetConfig.model_validate(
            {
                "name": "kb",
                "extractor_config": [
                    {"backend": "mineru", "endpoint": "http://a"},
                    {"backend": "mineru", "endpoint": "http://b"},
                ],
            }
        )
