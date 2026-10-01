"""Docker integration tests for the CPU-fine document-extractor backends.

local_only: each spins up the extractor's real container and drives it through
the adapter. docling-serve, Xberg, and markitdown-mcp all run CPU-only, so
these need Docker but no GPU; they are skipped when Docker is absent. MinerU
(GPU-required) lives in ``test_extractor_mineru_docker.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from semdex.adapters.extractor import DoclingExtractor, MarkitdownExtractor, XbergExtractor
from semdex.domain.models import SourceRef

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = [pytest.mark.local_only, pytest.mark.os_agnostic]


def _write(tmp_path: Path, name: str, data: bytes) -> SourceRef:
    path = tmp_path / name
    path.write_bytes(data)
    return SourceRef(uri=str(path), label="curated", content_hash="h", mtime=1.0)


def _http_ready(port: int, path: str) -> bool:
    import httpx

    try:
        return httpx.get(f"http://127.0.0.1:{port}{path}", timeout=2).status_code < 500
    except httpx.HTTPError:
        return False


def _docling_ready(port: int) -> bool:
    return _http_ready(port, "/health")


def _xberg_ready(port: int) -> bool:
    return _http_ready(port, "/")


def _markitdown_ready(port: int) -> bool:
    """Ready only once the MCP ASGI app answers on ``/mcp/`` (see the matrix test).

    A bare TCP connect succeeds the instant uvicorn binds the port, before the
    Streamable-HTTP session manager finishes its lifespan startup; a GET to
    ``/mcp/`` returns an HTTP status only once the app is truly serving.
    """
    return _http_ready(port, "/mcp/")


@pytest.fixture(scope="module")
def docling_endpoint(service_container: Callable[..., int]) -> str:
    port = service_container(
        image="quay.io/docling-project/docling-serve:latest",
        container_port=5001,
        ready=_docling_ready,
    )
    return f"http://127.0.0.1:{port}"


@pytest.fixture(scope="module")
def xberg_endpoint(service_container: Callable[..., int]) -> str:
    port = service_container(
        image="goldziher/kreuzberg:core",
        container_port=8000,
        ready=_xberg_ready,
    )
    return f"http://127.0.0.1:{port}"


@pytest.fixture(scope="module")
def markitdown_endpoint(service_container: Callable[..., int]) -> str:
    port = service_container(
        image="mcp/markitdown:latest",
        container_port=3001,
        command=["--http", "--host", "0.0.0.0", "--port", "3001"],  # noqa: S104 - container must bind all interfaces
        ready=_markitdown_ready,
    )
    return f"http://127.0.0.1:{port}"


def test_docling_extracts_markdown(docling_endpoint: str, tmp_path: Path) -> None:
    doc = DoclingExtractor(docling_endpoint)(_write(tmp_path, "note.md", b"# Title\n\nHello world."))
    assert "Hello world" in doc.text


def test_xberg_extracts_text(xberg_endpoint: str, tmp_path: Path) -> None:
    doc = XbergExtractor(xberg_endpoint)(_write(tmp_path, "note.txt", b"Hello world."))
    assert "Hello world" in doc.text


def test_xberg_ocrs_scanned_image_with_force_ocr(xberg_endpoint: str) -> None:
    """With force_ocr the v4 image runs Tesseract on a scanned image and recovers its text."""
    scan = Path(__file__).parent / "fixtures" / "extract" / "scan.png"
    src = SourceRef(uri=str(scan), label="curated", content_hash="h", mtime=1.0)
    doc = XbergExtractor(xberg_endpoint, force_ocr=True, ocr_language="eng")(src)
    assert "quick brown fox" in doc.text.lower()


def test_markitdown_extracts_markdown(markitdown_endpoint: str, tmp_path: Path) -> None:
    doc = MarkitdownExtractor(markitdown_endpoint)(_write(tmp_path, "note.txt", b"Hello world."))
    assert "Hello world" in doc.text
