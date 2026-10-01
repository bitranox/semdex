"""Docker integration test for the MinerU extractor backend (GPU-required).

local_only + GPU: MinerU's official image is CUDA-only, so this test needs an
NVIDIA GPU (``nvidia-smi`` present) AND the image name supplied via the
``SEMDEX_MINERU_IMAGE`` environment variable (there is no single canonical public
tag). It is skipped otherwise. The adapter's request/response handling is covered
CPU-only by ``test_extractor_http.py``; this test exercises the real GPU pipeline.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from semdex.adapters.extractor import MineruExtractor
from semdex.domain.models import SourceRef

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = [pytest.mark.local_only, pytest.mark.os_agnostic]

_MINERU_IMAGE = os.environ.get("SEMDEX_MINERU_IMAGE")

if shutil.which("nvidia-smi") is None:
    pytest.skip("MinerU needs an NVIDIA GPU (nvidia-smi not found)", allow_module_level=True)
if not _MINERU_IMAGE:
    pytest.skip("set SEMDEX_MINERU_IMAGE to the MinerU API image to run", allow_module_level=True)


def _mineru_ready(port: int) -> bool:
    import httpx

    try:
        return httpx.get(f"http://127.0.0.1:{port}/docs", timeout=2).status_code < 500
    except httpx.HTTPError:
        return False


# A one-page PDF (minimal but valid) so MinerU has something to parse.
_MINIMAL_PDF = (
    b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 4 0 R"
    b"/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
    b"4 0 obj<</Length 44>>stream\nBT /F1 24 Tf 20 100 Td (Hello world) Tj ET\nendstream endobj\n"
    b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
    b"trailer<</Root 1 0 R>>\n%%EOF"
)


@pytest.fixture(scope="module")
def mineru_endpoint(service_container: Callable[..., int]) -> str:
    port = service_container(
        image=_MINERU_IMAGE,
        container_port=8000,
        extra_args=["--gpus", "all"],
        ready=_mineru_ready,
    )
    return f"http://127.0.0.1:{port}"


def test_mineru_extracts_markdown(mineru_endpoint: str, tmp_path: Path) -> None:
    path = tmp_path / "hello.pdf"
    path.write_bytes(_MINIMAL_PDF)
    source = SourceRef(uri=str(path), label="curated", content_hash="h", mtime=1.0)
    doc = MineruExtractor(mineru_endpoint)(source)
    assert "Hello world" in doc.text
