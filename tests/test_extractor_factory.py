"""Composition extractor factory: selects an adapter by ExtractorBackend."""

from __future__ import annotations

from pathlib import Path

import pytest

from semdex.adapters.extractor import (
    DoclingExtractor,
    MarkitdownExtractor,
    MineruExtractor,
    TextExtractor,
    XbergExtractor,
)
from semdex.composition import build_extractor, build_index_production
from semdex.domain.enums import ExtractorBackend
from semdex.domain.errors import ExtractionError

pytestmark = pytest.mark.os_agnostic

_HTTP_BACKENDS: list[tuple[ExtractorBackend, type[object]]] = [
    (ExtractorBackend.MARKITDOWN, MarkitdownExtractor),
    (ExtractorBackend.XBERG, XbergExtractor),
    (ExtractorBackend.DOCLING, DoclingExtractor),
    (ExtractorBackend.MINERU, MineruExtractor),
]


def test_text_backend_is_default_and_needs_no_endpoint() -> None:
    """The embedded text reader is built without an endpoint."""
    assert isinstance(build_extractor(ExtractorBackend.TEXT), TextExtractor)


@pytest.mark.parametrize(("backend", "expected_cls"), _HTTP_BACKENDS)
def test_http_backend_selection(backend: ExtractorBackend, expected_cls: type[object]) -> None:
    """Each HTTP backend maps to its adapter when an endpoint is given."""
    assert isinstance(build_extractor(backend, endpoint="http://host:9"), expected_cls)


@pytest.mark.parametrize(("backend", "_expected_cls"), _HTTP_BACKENDS)
def test_http_backend_without_endpoint_fails(backend: ExtractorBackend, _expected_cls: type[object]) -> None:
    """Selecting an HTTP backend without an endpoint raises a clear error."""
    with pytest.raises(ExtractionError, match="requires"):
        build_extractor(backend)


def test_build_index_production_wires_selected_extractor(tmp_path: Path) -> None:
    """build_index_production wires the extractor chosen by extractor_backend."""
    services = build_index_production(
        tmp_path, extractor_backend=ExtractorBackend.DOCLING, extractor_endpoint="http://host:5001"
    )
    assert isinstance(services.extract, DoclingExtractor)


def test_build_index_production_defaults_to_text(tmp_path: Path) -> None:
    """With no extractor selection, the embedded text reader is wired."""
    services = build_index_production(tmp_path)
    assert isinstance(services.extract, TextExtractor)


def test_render_dpi_reaches_the_vision_backends() -> None:
    """The resolution a page is rasterised at decides what it COSTS a vision model.

    Vision tokens scale with pixel count, so this is not cosmetic: at the adapter default of 200
    a dense page produced an input over a 16k context and exhausted the spare VRAM of a 16 GB
    card mid-inference, killing the server. A knob that cannot be set from the factory cannot be
    lowered by a caller that needs to.
    """
    from semdex.adapters.extractor.vision_ocr import VisionOcrExtractor

    lowered = build_extractor(ExtractorBackend.OLMOCR, endpoint="http://x/v1", model="m", render_dpi=110)
    # isinstance rather than a bare attribute read: it narrows the Extract protocol to the
    # concrete adapter for the type checker AND asserts the factory really returns that adapter.
    assert isinstance(lowered, VisionOcrExtractor)
    assert lowered.render_dpi == 110

    default = build_extractor(ExtractorBackend.OLMOCR, endpoint="http://x/v1", model="m")
    assert isinstance(default, VisionOcrExtractor)
    assert default.render_dpi == 200, "omitting it must keep the adapter default, not force one"


def test_render_dpi_is_ignored_by_the_non_vision_backends() -> None:
    """Only the vision backends rasterise; passing it elsewhere must not blow up a caller.

    The routing extractor builds every backend through this one factory, so a parameter that only
    some backends understand has to be harmless to the rest.
    """
    text = build_extractor(ExtractorBackend.TEXT, render_dpi=110)

    assert text is not None
