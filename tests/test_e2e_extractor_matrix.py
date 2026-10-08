# pyright: basic
"""Document-extraction fidelity benchmark matrix (local_only).

Measures how well each extractor adapter (the embedded ``TextExtractor`` plus the
docling/xberg/markitdown/mineru containers) recovers known text from a given
document FORMAT: phrase_recall (fraction of expected phrases recovered, case-
insensitive and whitespace-normalized) plus latency, against the self-authored
fixtures in ``tests/fixtures/extract/`` (see its README.md + expected.toml). An
opt-in tier additionally runs the container extractors against real labeled
PDF/image corpora (olmOCR-Bench, OmniDocBench) for a second, less-controlled
fidelity signal. Results print as tables and append to the shared
``benchmark-report.md`` artifact (same file ``test_e2e_matrix.py`` writes to).

This file wraps httpx/mcp/huggingface_hub, all optional and largely untyped from
pyright's point of view, so it opts down to pyright basic mode (like
``test_e2e_matrix.py``). The container cells need Docker and self-skip (recorded
as "skip" in the coverage grid) when it is absent; mineru additionally needs an
NVIDIA GPU and ``SEMDEX_MINERU_IMAGE`` (see ``test_extractor_mineru_docker.py``).
Env knobs:
  SEMDEX_BENCH_EXTRACTORS       comma list of text,markitdown,xberg,docling,mineru (default all)
  SEMDEX_BENCH_EXTRACT_CORPORA  comma list of fixtures,olmocr,omnidocbench (default fixtures)
  SEMDEX_BENCH_OMNIDOCBENCH     set to "1" to opt into OmniDocBench (research-only/non-commercial)
  SEMDEX_BENCH_MAX_DOCS         cap the olmocr/omnidocbench sample (default 5)
  SEMDEX_MINERU_IMAGE           MinerU's GPU image, for the mineru matrix cells
  SEMDEX_BENCH_REPORT           report path (default benchmark-report.md)
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import rtoml
from _benchmark_report import ResultStatus, image_digest, record

from semdex.adapters.discovery.location import to_uri
from semdex.adapters.extractor import (
    DoclingExtractor,
    MarkitdownExtractor,
    MineruExtractor,
    TextExtractor,
    XbergExtractor,
)
from semdex.domain.errors import ExtractionError
from semdex.domain.models import SourceRef

if TYPE_CHECKING:
    from semdex.application.ports import Extract

pytestmark = [pytest.mark.local_only, pytest.mark.os_agnostic]

_REPORT = Path(os.environ.get("SEMDEX_BENCH_REPORT", "benchmark-report.md"))
_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "extract"
# rtoml (a dev dep, works on every Python version) instead of stdlib tomllib, which is 3.11+
# only - on 3.10 the tomllib import broke both pyright and pytest collection.
_EXPECTED: dict[str, Any] = rtoml.load(_FIXTURES_DIR / "expected.toml")["fixtures"]

_FORMAT_FIXTURES: dict[str, str] = {
    "md": "sample.md",
    "txt": "sample.txt",
    "html": "sample.html",
    "pdf": "sample.pdf",
    "docx": "sample.docx",
    "pptx": "sample.pptx",
    "xlsx": "sample.xlsx",
    "image-ocr": "scan.png",
}
_FORMATS = tuple(_FORMAT_FIXTURES)

_SUPPORTED_FORMATS: dict[str, frozenset[str]] = {
    "text": frozenset({"md", "txt"}),
    "markitdown": frozenset({"md", "txt", "html", "pdf", "docx", "pptx", "xlsx", "image-ocr"}),
    "xberg": frozenset({"md", "txt", "html", "pdf", "docx", "pptx", "xlsx", "image-ocr"}),
    "docling": frozenset({"md", "pdf", "docx", "pptx", "xlsx", "html"}),
    "mineru": frozenset({"pdf"}),
}
_EXTRACTORS = tuple(_SUPPORTED_FORMATS)  # declaration order: text, markitdown, xberg, docling, mineru


def _selected_extractors() -> tuple[str, ...]:
    """Restrict the extractor axis to SEMDEX_BENCH_EXTRACTORS (comma list); empty = all.

    Lets a local baseline seed run only the cached, cheap extractors (e.g. text,markitdown)
    without pulling the multi-GB docling/xberg images, which the scheduled CI covers.
    """
    raw = os.environ.get("SEMDEX_BENCH_EXTRACTORS", "").strip()
    if not raw:
        return _EXTRACTORS
    wanted = {name.strip() for name in raw.split(",")}
    return tuple(name for name in _EXTRACTORS if name in wanted)


# --------------------------------------------------------------------------- metrics


def _normalize(text: str) -> str:
    """Collapse whitespace and lowercase, so extraction formatting never counts against recall."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _phrase_recall(text: str, phrases: list[str]) -> float:
    """Fraction of *phrases* recovered in *text* (case-insensitive, whitespace-normalized)."""
    if not phrases:
        return 1.0
    normalized = _normalize(text)
    hits = sum(1 for phrase in phrases if _normalize(phrase) in normalized)
    return hits / len(phrases)


def _similarity(text: str, reference: str) -> float:
    """Normalized edit-distance similarity (0..1) between extracted text and a reference."""
    return difflib.SequenceMatcher(None, _normalize(text), _normalize(reference)).ratio()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _append_report(title: str, header: str, rows: list[str]) -> None:
    with _REPORT.open("a", encoding="utf-8") as fh:
        fh.write(f"\n## {title}\n\n{header}\n")
        for row in rows:
            fh.write(row + "\n")


# --------------------------------------------------------------------------- containers


def _docker_available() -> bool:
    """Whether the Docker CLI is present and its daemon is reachable."""
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True, check=False).returncode == 0


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
    """Ready only once the MCP ASGI app answers on ``/mcp/``.

    A bare TCP connect succeeds the instant uvicorn binds the port, before the
    Streamable-HTTP session manager finishes its ASGI lifespan startup; calling
    the tool in that window fails. A GET to ``/mcp/`` returns an HTTP status
    (4xx for the missing MCP session) only once the app is actually serving, so
    it is a truthful readiness signal where TCP is a false-positive one.
    """
    return _http_ready(port, "/mcp/")


def _mineru_ready(port: int) -> bool:
    return _http_ready(port, "/docs")


def _build_docling(service_container: Callable[..., int]) -> Extract:
    port = service_container(
        image="quay.io/docling-project/docling-serve:latest", container_port=5001, ready=_docling_ready
    )
    return DoclingExtractor(f"http://127.0.0.1:{port}")


def _build_xberg(service_container: Callable[..., int]) -> Extract:
    port = service_container(image="goldziher/kreuzberg:core", container_port=8000, ready=_xberg_ready)
    # force_ocr so the scanned-image cell exercises OCR (v4 returns only metadata otherwise);
    # measured to hold 100% on every document format too, so it costs nothing here.
    return XbergExtractor(f"http://127.0.0.1:{port}", force_ocr=True, ocr_language="eng")


def _build_markitdown(service_container: Callable[..., int]) -> Extract:
    port = service_container(
        image="mcp/markitdown:latest",
        container_port=3001,
        command=["--http", "--host", "0.0.0.0", "--port", "3001"],  # noqa: S104 - container must bind all interfaces
        ready=_markitdown_ready,
    )
    return MarkitdownExtractor(f"http://127.0.0.1:{port}")


def _build_mineru(service_container: Callable[..., int]) -> Extract | None:
    """Build the MinerU extractor, or None without a GPU + SEMDEX_MINERU_IMAGE (see test_extractor_mineru_docker.py)."""
    image = os.environ.get("SEMDEX_MINERU_IMAGE")
    if shutil.which("nvidia-smi") is None or not image:
        return None
    port = service_container(image=image, container_port=8000, extra_args=["--gpus", "all"], ready=_mineru_ready)
    return MineruExtractor(f"http://127.0.0.1:{port}")


_CONTAINER_BUILDERS: dict[str, Callable[[Callable[..., int]], Extract]] = {
    "docling": _build_docling,
    "xberg": _build_xberg,
    "markitdown": _build_markitdown,
}

# Container image per extractor, so a recorded result pins the exact image digest it ran
# against (an upstream ``:latest`` rebuild is a prime suspect for an extraction regression).
_EXTRACTOR_IMAGES: dict[str, str] = {
    "docling": "quay.io/docling-project/docling-serve:latest",
    "xberg": "goldziher/kreuzberg:core",
    "markitdown": "mcp/markitdown:latest",
}


def _build_extractor(name: str, service_container: Callable[..., int]) -> Extract | None:
    """Build the named extractor, or None if its backend is unavailable (self-skip)."""
    if name == "text":
        return TextExtractor()
    if name == "mineru":
        return _build_mineru(service_container)
    if not _docker_available():
        return None
    return _CONTAINER_BUILDERS[name](service_container)


# --------------------------------------------------------------------------- fixture matrix


def _extract_and_score(extractor: Extract, fixture_name: str) -> tuple[float, float]:
    """Extract one fixture and MEASURE (phrase_recall, elapsed_seconds) - no assertions.

    This is the coverage grid's scoring primitive: it records whatever an extractor recovers,
    including 0.0 (e.g. xberg returns image metadata, not OCR text, for a scanned PNG). The
    per-extractor floor ("at least one phrase survives") is asserted by the targeted tests that
    call this, not here - so one weak cell is data, never a grid abort.
    """
    source = SourceRef(uri=to_uri(_FIXTURES_DIR / fixture_name), label="", content_hash="", mtime=0.0)
    phrases = _EXPECTED[fixture_name]["phrases"]
    started = time.perf_counter()
    text = extractor(source).text
    elapsed = time.perf_counter() - started
    return _phrase_recall(text, phrases), elapsed


def test_text_extractor_on_supported_fixtures() -> None:
    """The embedded TextExtractor needs no container - proves the fixtures + metric work standalone."""
    extractor = TextExtractor()
    for fmt in _FORMATS:
        if fmt not in _SUPPORTED_FORMATS["text"]:
            continue
        recall, _elapsed = _extract_and_score(extractor, _FORMAT_FIXTURES[fmt])
        assert recall > 0.0, f"text/{fmt}: no expected phrases recovered"


def _score_extractor_row(name: str, extractor: Extract | None, latency_rows: list[str]) -> dict[str, str]:
    """Score one extractor across every format: 'n/a' unsupported, 'skip' unavailable, else recall%.

    Every cell is also recorded to ``benchmark-results.json`` (status + phrase-recall/latency)
    so the coverage grid is machine-comparable against the baseline. Container extractors pin
    their image digest once per row.
    """
    supported = _SUPPORTED_FORMATS[name]
    images = None
    if extractor is not None and name in _EXTRACTOR_IMAGES:
        images = {name: image_digest(_EXTRACTOR_IMAGES[name])}
    row: dict[str, str] = {}
    for fmt in _FORMATS:
        combo = f"{name}@{fmt}"
        if fmt not in supported:
            row[fmt] = "n/a"
            record("extract", combo, ResultStatus.NA)
            continue
        if extractor is None:
            row[fmt] = "skip"
            record("extract", combo, ResultStatus.SKIP)
            continue
        try:
            recall, elapsed = _extract_and_score(extractor, _FORMAT_FIXTURES[fmt])
        except ExtractionError:
            # A single supported-format failure (e.g. markitdown cannot OCR this image with
            # the pinned container) is DATA for the coverage grid, not a reason to abort the
            # whole matrix. Record it as "fail" and carry on; the regression harness then
            # tracks fail<->ok transitions per cell like any other state.
            row[fmt] = "fail"
            record("extract", combo, ResultStatus.FAIL, images=images)
            continue
        row[fmt] = f"{recall * 100:.0f}%"
        record(
            "extract", combo, ResultStatus.OK, {"phrase_recall": recall, "latency_ms": elapsed * 1000}, images=images
        )
        latency_rows.append(f"| {name:10s} | {fmt:9s} | {recall * 100:5.1f}% | {elapsed * 1000:8.1f} |")
    return row


def _grid_table(grid: dict[str, dict[str, str]], names: tuple[str, ...]) -> tuple[str, list[str]]:
    """Render the extractor x format grid as a markdown table (header, rows)."""
    header = "| extractor   | " + " | ".join(f"{fmt:9s}" for fmt in _FORMATS) + " |"
    divider = "|---|" + "---|" * len(_FORMATS)
    rows = [
        "| " + f"{name:11s}" + " | " + " | ".join(f"{grid[name][fmt]:9s}" for fmt in _FORMATS) + " |" for name in names
    ]
    return header + "\n" + divider, rows


def test_extractor_fixture_matrix(service_container: Callable[..., int]) -> None:
    """Coverage grid: phrase-recall + latency per (extractor x fixture format)."""
    grid: dict[str, dict[str, str]] = {}
    latency_rows: list[str] = []
    selected = _selected_extractors()
    for name in selected:
        extractor = _build_extractor(name, service_container)
        grid[name] = _score_extractor_row(name, extractor, latency_rows)

    header, grid_rows = _grid_table(grid, selected)
    print("\nextractor fidelity coverage grid (recall% / n/a / skip):")
    print(header)
    for row in grid_rows:
        print("  " + row)
    _append_report("Extractor fidelity coverage grid", header, grid_rows)
    _append_report(
        "Extractor fidelity latency",
        "| extractor | format | recall | latency_ms |\n|---|---|---|---|",
        latency_rows,
    )

    ran_any = any(cell.endswith("%") for row in grid.values() for cell in row.values())
    assert ran_any, "no extractor produced a successful result; check docker availability"


# --------------------------------------------------------------------------- pdf corpora

_PDF_CORPORA = ("olmocr", "omnidocbench")
_DEFAULT_MAX_DOCS = 5
_OLMOCR_REPO = "allenai/olmOCR-bench"
_OLMOCR_JSONL = "bench_data/old_scans.jsonl"  # 'present'-type text assertions on real scanned PDFs
_OMNIDOCBENCH_REPO = "opendatalab/OmniDocBench"
# mineru is deliberately excluded here: it is pdf-only, GPU-gated, and already has its own
# dedicated integration test (test_extractor_mineru_docker.py); the three CPU containers
# cover both the olmocr (pdf) and omnidocbench (image) corpora uniformly.
_PDF_CORPUS_EXTRACTORS = ("markitdown", "xberg", "docling")

# A corpus reference is either a list of must-appear phrases (olmocr's per-page assertions)
# or a single full reference string (omnidocbench's reading-order block text).
Reference = list[str] | str


def _cap(name: str) -> int:
    """Read an integer env cap, defaulting to _DEFAULT_MAX_DOCS when unset or zero."""
    n = int(os.environ.get(name, "0"))
    return n or _DEFAULT_MAX_DOCS


def _load_olmocr_sample(max_docs: int) -> list[tuple[Path, Reference]]:
    """Download a small olmOCR-Bench sample: real scanned PDFs + their 'present'-type text assertions.

    olmOCR-Bench ships per-page unit-test assertions (present/absent/order/table/math), not a
    full-page transcript, so the reference here is a page's 'present'-type text snippets - a
    defensible phrase_recall target, though not the benchmark's own scoring harness.
    """
    from huggingface_hub import hf_hub_download

    jsonl_path = Path(hf_hub_download(repo_id=_OLMOCR_REPO, repo_type="dataset", filename=_OLMOCR_JSONL))
    by_pdf: dict[str, list[str]] = {}
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("type") == "present":
                by_pdf.setdefault(row["pdf"], []).append(row["text"])

    docs: list[tuple[Path, Reference]] = []
    for pdf_rel, phrases in by_pdf.items():
        if len(docs) >= max_docs:
            break
        pdf_path = hf_hub_download(repo_id=_OLMOCR_REPO, repo_type="dataset", filename=f"bench_data/pdfs/{pdf_rel}")
        docs.append((Path(pdf_path), phrases))
    return docs


def _omnidocbench_reference(record: dict[str, Any]) -> str:
    """Join a page's layout blocks (text entries) in reading order into one reference string."""
    blocks = sorted(record["layout_dets"], key=lambda b: b.get("order") if isinstance(b.get("order"), int) else 10**9)
    return " ".join(block["text"] for block in blocks if "text" in block)


def _load_omnidocbench_sample(max_docs: int) -> list[tuple[Path, Reference]]:
    """Download a small OmniDocBench sample: page images + their reading-order block text.

    OmniDocBench's ground truth is per-block (text/latex + reading order), not a rendered
    markdown file, so the reference here is those blocks' text joined in reading order.
    """
    from huggingface_hub import hf_hub_download

    index_path = Path(hf_hub_download(repo_id=_OMNIDOCBENCH_REPO, repo_type="dataset", filename="OmniDocBench.json"))
    records = json.loads(index_path.read_text(encoding="utf-8"))

    docs: list[tuple[Path, Reference]] = []
    for entry in records:
        if len(docs) >= max_docs:
            break
        reference = _omnidocbench_reference(entry)
        if not reference.strip():
            continue
        image_path = entry["page_info"]["image_path"]
        img_path = hf_hub_download(repo_id=_OMNIDOCBENCH_REPO, repo_type="dataset", filename=f"images/{image_path}")
        docs.append((Path(img_path), reference))
    return docs


_CORPUS_LOADERS: dict[str, Callable[[int], list[tuple[Path, Reference]]]] = {
    "olmocr": _load_olmocr_sample,
    "omnidocbench": _load_omnidocbench_sample,
}


def _selected_corpora(env_value: str) -> set[str]:
    return {corpus.strip() for corpus in env_value.split(",") if corpus.strip()}


def _score_corpus_extractor(
    corpus_id: str,
    name: str,
    docs: list[tuple[Path, Reference]],
    service_container: Callable[..., int],
) -> tuple[str, float] | None:
    """Score one container extractor across the corpus sample: its table row and mean similarity.

    Returns None if the extractor's backend is unavailable. A zero score is a result, not an
    error: olmOCR-Bench's old_scans PDFs have no text layer, so an extractor without OCR
    (markitdown) recovers nothing from them, and that is the row the table should show. The
    caller asserts only that SOME extractor read the corpus, as the fixture grid does.
    """
    extractor = _build_extractor(name, service_container)
    if extractor is None:
        return None

    recalls: list[float] = []
    similarities: list[float] = []
    for path, reference in docs:
        source = SourceRef(uri=to_uri(path), label="", content_hash="", mtime=0.0)
        text = extractor(source).text
        if isinstance(reference, list):
            recalls.append(_phrase_recall(text, reference))
            similarities.append(_similarity(text, " ".join(reference)))
        else:
            similarities.append(_similarity(text, reference))

    mean_similarity = _mean(similarities)
    recall_label = f"{_mean(recalls) * 100:.1f}%" if recalls else "n/a"
    row = f"| {corpus_id:12s} | {name:10s} | {recall_label:>6s} | {mean_similarity:.3f} | {len(docs)} |"
    return row, mean_similarity


@pytest.mark.parametrize("corpus_id", _PDF_CORPORA)
def test_extractor_pdf_corpora(corpus_id: str, service_container: Callable[..., int]) -> None:
    """Container-extractor fidelity against a real labeled PDF/image corpus (opt-in, skipped by default)."""
    selected = _selected_corpora(os.environ.get("SEMDEX_BENCH_EXTRACT_CORPORA", "fixtures"))
    if corpus_id not in selected:
        pytest.skip(f"{corpus_id} is opt-in; add it to SEMDEX_BENCH_EXTRACT_CORPORA to run")
    if corpus_id == "omnidocbench" and os.environ.get("SEMDEX_BENCH_OMNIDOCBENCH") != "1":
        pytest.skip("omnidocbench is research-only/non-commercial; set SEMDEX_BENCH_OMNIDOCBENCH=1 to opt in")
    pytest.importorskip("huggingface_hub")

    max_docs = _cap("SEMDEX_BENCH_MAX_DOCS")
    try:
        docs = _CORPUS_LOADERS[corpus_id](max_docs)
    except Exception as exc:
        pytest.skip(f"{corpus_id}: could not download the corpus sample: {exc}")
    if not docs:
        pytest.skip(f"{corpus_id}: no usable documents in the downloaded sample")

    rows: list[str] = []
    similarities: dict[str, float] = {}
    for name in _PDF_CORPUS_EXTRACTORS:
        scored = _score_corpus_extractor(corpus_id, name, docs, service_container)
        if scored is not None:
            rows.append(scored[0])
            similarities[name] = scored[1]
    if not rows:
        pytest.skip(f"{corpus_id}: no container extractor available (docker absent?)")

    print(f"\n[{corpus_id}] PDF corpus fidelity ({len(docs)} docs): extractor | phrase_recall | edit_similarity")
    for row in rows:
        print("  " + row)
    _append_report(
        f"PDF corpus fidelity - {corpus_id}",
        "| corpus | extractor | phrase_recall | edit_similarity | n_docs |\n|---|---|---|---|---|",
        rows,
    )
    assert any(score > 0.0 for score in similarities.values()), (
        f"{corpus_id}: no extractor recovered any text from the corpus: {similarities}"
    )
