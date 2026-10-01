#!/usr/bin/env python3
"""Score the document extractors against OmniDocBench, on documents nobody here wrote.

The extraction grid on these pages scores each converter against eight fixtures authored in this
repo to exercise the formats. That measures agreement with our own expectations: the fixtures say
what we thought a converter should find, and a converter that finds it scores 100 percent. It
cannot say how any of them behaves on a document from the wild, which is the only thing a reader
choosing an extractor actually cares about.

OmniDocBench is 1,651 real document pages - books, papers, slides, magazines, newspapers, exam
papers, handwritten notes - with human layout annotations carrying the text of every block, across
English, Chinese and mixed pages. Scoring against it replaces "did it find the phrases we planted"
with "did it read the page".

Two numbers per page, because they fail differently:

* ``block_recall`` - the share of annotated text blocks that appear in the extraction at all. This
  is the same currency as the fixture grid's ``phrase_recall``, so the two tables can be read
  against each other, and it is indifferent to reading order and formatting.
* ``edit_similarity`` - similarity of the whole extracted text to the whole annotated text. This
  one DOES punish dropped, duplicated and reordered content, so a converter that finds every block
  while mangling the page around them scores well on the first and badly on this.

Every extractor is handed the identical file: the page image wrapped in a single-page PDF. Feeding
images to the two backends that accept them and PDFs to the two that do not would compare
different inputs.

Usage::

    python scripts/score_extraction_omnidocbench.py --pages 120
    python scripts/score_extraction_omnidocbench.py --extractors xberg --pages 20
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semdex.composition import build_extractor
from semdex.domain.enums import ExtractorBackend
from semdex.domain.models import SourceRef

_BENCH = Path.home() / "bench-ocr" / "datasets" / "OmniDocBench"
_OUT = Path(__file__).resolve().parents[1] / "tests" / "benchmarks" / "raw" / "extraction-omnidocbench.json"

# The annotated categories that carry running text. Headers, footers, page numbers and the
# `abandon` class are page furniture that converters legitimately drop, and equations/tables are
# scored by their own metrics upstream; including any of them would measure a formatting policy
# rather than whether the page was read.
_TEXT_CATEGORIES = frozenset({"text_block", "title"})

# A block must be at least this long to count. Median annotated block length is 70 characters, but
# the tail runs down to single words like "that", which any extraction contains by accident - and
# counting those inflates recall for every backend equally and silently.
_MIN_BLOCK_CHARS = 20

# OmniDocBench's language tag -> the tesseract pack the xberg container carries (eng, chi_sim,
# chi_tra). Half the benchmark is Chinese, and OCR'ing a Chinese page with the English pack
# returns confident nonsense rather than an error, so this mapping is load-bearing.
_OCR_LANGUAGE = {
    "english": "eng",
    "simplified_chinese": "chi_sim",
    "traditional_chinese": "chi_tra",
    "en_ch_mixed": "eng+chi_sim",
}
_DEFAULT_OCR_LANGUAGE = "eng"

# Default endpoints. The CPU converters run in local containers; the GPU-backed ones
# (mineru, olmocr) run on the shared GPU host, so their defaults point there and both are
# overridable with --endpoint name=url.
_ENDPOINTS = {
    "markitdown": "http://127.0.0.1:13001",
    "xberg": "http://127.0.0.1:18000",
    "docling": "http://127.0.0.1:15001",
    "mineru": "http://px-semdex-test:8001",
    "olmocr": "http://px-semdex-test:8000/v1",
}
# Vision-LLM backends need a model name; the others ignore it.
_VISION_MODEL = {"olmocr": "olmocr"}

# The resolution a vision backend rasterises a page at, and the knob that decides what a page
# COSTS: vision tokens scale with the pixel count. semdex ships 200, which on these pages
# produced inputs over a 16k context and, on the dense ones, exhausted the spare VRAM of a 16 GB
# card mid-inference - killing the vLLM engine and taking the server down with it. 150 is about
# 56 percent of the pixels and is the usual floor for OCR quality. Recorded in the output,
# because a vision score is meaningless without the resolution it was measured at.
_VISION_DPI = 150


@dataclass(frozen=True, slots=True)
class Page:
    """One annotated page: where its image is, what it is, and what it says."""

    image: Path
    source: str
    language: str
    blocks: list[str]

    @property
    def text(self) -> str:
        return "\n".join(self.blocks)


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def normalize(text: str) -> str:
    """Collapse whitespace and case, so formatting differences are not scored as errors."""
    return " ".join(text.split()).lower()


def ground_truth_blocks(layout: Sequence[dict[str, Any]]) -> list[str]:
    """The annotated running text of a page, in reading order, long blocks only."""
    kept = [
        item
        for item in layout
        if item.get("category_type") in _TEXT_CATEGORIES
        and not item.get("ignore")
        and len(str(item.get("text") or "").strip()) >= _MIN_BLOCK_CHARS
    ]
    kept.sort(key=lambda item: item.get("order", 0))
    return [str(item["text"]).strip() for item in kept]


# Latin words OR single CJK characters. Half of OmniDocBench is Chinese, which has no spaces, so
# splitting on whitespace would score every Chinese page as one enormous token and report a flat
# zero for every backend.
_TOKEN = re.compile(r"[0-9a-z]+|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def tokenize(text: str) -> list[str]:
    """Words for alphabetic scripts, characters for CJK."""
    return _TOKEN.findall(text.lower())


def word_recall(blocks: Sequence[str], extracted: str) -> float:
    """Share of the annotated text the extraction contains, counted as a token multiset.

    An earlier version asked whether each annotated BLOCK appeared verbatim in the extraction.
    That measured formatting, not reading: docling read a slide correctly and scored 0.033,
    because the annotation bullets each line with a filled circle and docling emits a hyphen.
    Counting tokens is indifferent to bullets, punctuation, line breaks and reading order, which
    is what this metric is supposed to be indifferent to.

    Multiset, so a word occurring three times in the page and once in the extraction counts once.
    A converter could in principle inflate this by emitting extra text; `extracted_chars` is
    recorded beside it so that is visible, and `edit_similarity` punishes it directly.
    """
    reference = Counter(token for block in blocks for token in tokenize(block))
    if not reference:
        return 1.0
    found = Counter(tokenize(extracted))
    hit = sum(min(count, found[token]) for token, count in reference.items())
    return hit / sum(reference.values())


def edit_similarity(reference: str, extracted: str) -> float:
    """Similarity of the whole page text to the whole annotation.

    Unlike block recall this punishes dropped, duplicated and reordered content, so the pair
    separates "found the words" from "reproduced the page".
    """
    left, right = normalize(reference), normalize(extracted)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def load_pages(bench: Path) -> list[Page]:
    """Every annotated page whose image is on disk and which carries scorable text."""
    doc = json.loads((bench / "OmniDocBench.json").read_text(encoding="utf-8"))
    images = bench / "images"
    pages: list[Page] = []
    for entry in doc:
        info = entry["page_info"]
        attribute = info.get("page_attribute", {})
        image = images / Path(info["image_path"]).name
        blocks = ground_truth_blocks(entry.get("layout_dets", []))
        if not image.exists() or not blocks:
            continue
        pages.append(
            Page(
                image=image,
                source=str(attribute.get("data_source", "unknown")),
                language=str(attribute.get("language", "unknown")),
                blocks=blocks,
            )
        )
    return pages


def stratified_sample(pages: Sequence[Page], total: int, seed: int) -> list[Page]:
    """A sample spread across document sources, so no single type dominates the headline.

    OmniDocBench is not balanced - `book` has 276 pages and `historical_document` five - so a
    uniform random draw would report mostly books and call it "documents in the wild".
    """
    by_source: dict[str, list[Page]] = defaultdict(list)
    for page in pages:
        by_source[page.source].append(page)
    # Reproducibility, not secrecy: the same seed must pick the same pages so a re-run is
    # comparable. noqa S311 for that reason.
    rng = random.Random(seed)  # noqa: S311
    per_source = max(1, total // max(1, len(by_source)))
    picked: list[Page] = []
    for source in sorted(by_source):
        bucket = sorted(by_source[source], key=lambda p: p.image.name)
        picked.extend(rng.sample(bucket, min(per_source, len(bucket))))
    return picked[:total]


def as_single_page_pdf(image: Path, destination: Path) -> Path:
    """Wrap a page image in a one-page PDF, so every backend is handed the same file."""
    from PIL import Image

    with Image.open(image) as handle:
        page = handle.convert("RGB")
        page.save(destination, "PDF", resolution=float(page.info.get("dpi", (150, 150))[0]))
    return destination


def _extractor_for(name: str, ocr_language: str) -> Any:
    """Build one converter, with OCR turned on where the backend needs telling.

    xberg returns an empty string for an image-only PDF unless force_ocr is set - it looks for a
    text layer and finds none. The e2e extractor matrix configures it that way too; not doing so
    here scored it a flat zero on every page and would have published that as a fidelity result.
    """
    return build_extractor(
        ExtractorBackend(name),
        endpoint=_ENDPOINTS[name],
        timeout=600.0 if name in _VISION_MODEL else 180.0,
        force_ocr=True,
        ocr_language=ocr_language,
        model=_VISION_MODEL.get(name),
        render_dpi=_VISION_DPI if name in _VISION_MODEL else None,
    )


def score_page(extract: Any, page: Page, pdf: Path) -> dict[str, Any]:
    """Run one extractor over one page and score both metrics from the same output."""
    started = time.perf_counter()
    document = extract(SourceRef(uri=pdf.resolve().as_uri(), label="", content_hash="h", mtime=0.0))
    elapsed = (time.perf_counter() - started) * 1000
    return {
        "word_recall": round(word_recall(page.blocks, document.text), 4),
        "edit_similarity": round(edit_similarity(page.text, document.text), 4),
        "latency_ms": round(elapsed, 1),
        "extracted_chars": len(document.text),
    }


def summarize(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    """Mean of each metric over rows grouped by ``keys``, with the group size."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    out: list[dict[str, Any]] = []
    for group, members in sorted(grouped.items(), key=lambda item: [str(v) for v in item[0]]):
        scored = [m for m in members if m.get("ok")]
        out.append(
            {
                **dict(zip(keys, group, strict=True)),
                "pages": len(members),
                "failed": len(members) - len(scored),
                "word_recall": round(sum(m["word_recall"] for m in scored) / len(scored), 4) if scored else 0.0,
                "edit_similarity": round(sum(m["edit_similarity"] for m in scored) / len(scored), 4) if scored else 0.0,
                "latency_p50_ms": _median([m["latency_ms"] for m in scored]),
            }
        )
    return out


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[len(ordered) // 2], 1)


def text_layer_control(names: Sequence[str]) -> dict[str, int]:
    """Characters each converter returns for a PDF that HAS a text layer.

    A zero on this benchmark has two possible causes - the converter cannot OCR a scanned page,
    or the harness never reached it - and they look identical in the results. Running the repo's
    own text-layer fixture through the same code path separates them: a converter that reads this
    and returns nothing for the benchmark has no OCR, which is a finding; one that returns nothing
    for both is a broken wiring, which is not.
    """
    fixture = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "extract" / "sample.pdf"
    source = SourceRef(uri=fixture.resolve().as_uri(), label="", content_hash="h", mtime=0.0)
    out: dict[str, int] = {}
    for name in names:
        try:
            out[name] = len(_extractor_for(name, _DEFAULT_OCR_LANGUAGE)(source).text)
        except Exception:
            out[name] = -1
    return out


def merge_payload(existing: dict[str, Any], fresh: dict[str, Any], names: Sequence[str]) -> dict[str, Any]:
    """Fold a run into an earlier one, replacing only the extractors this run measured.

    The GPU-backed converters need the shared card to themselves, so they are run separately from
    the CPU containers; without this the second run would publish a table with three of the five
    backends missing. Rows are replaced per extractor rather than appended, so a re-measurement
    updates in place instead of leaving two rows a reader would average.
    """
    merged = dict(fresh)
    replaced = set(names)
    for section in ("overall", "by_source", "by_language"):
        previous: list[dict[str, Any]] = list(existing.get(section) or [])
        kept = [row for row in previous if row["extractor"] not in replaced]
        incoming: list[dict[str, Any]] = list(fresh.get(section) or [])
        merged[section] = sorted(
            kept + incoming,
            key=lambda row: [str(row.get(k, "")) for k in ("extractor", "source", "language")],
        )
    control = dict(existing.get("text_layer_control_chars") or {})
    control.update(fresh.get("text_layer_control_chars") or {})
    merged["text_layer_control_chars"] = control
    return merged


def _environment() -> dict[str, Any]:
    return {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "semdex_git_sha": _git_sha(),
        "host": platform.node(),
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "benchmark": "OmniDocBench (opendatalab), page images wrapped as single-page PDFs",
        "text_categories": sorted(_TEXT_CATEGORIES),
        "min_block_chars": _MIN_BLOCK_CHARS,
        "ocr_languages": _OCR_LANGUAGE,
        "vision_render_dpi": _VISION_DPI,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extractors", default="markitdown,xberg,docling")
    parser.add_argument("--pages", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--bench", type=Path, default=_BENCH)
    parser.add_argument("--work", type=Path, default=Path(tempfile.gettempdir()) / "semdex-odb-pdfs")
    parser.add_argument("--out", type=Path, default=_OUT)
    parser.add_argument(
        "--endpoint",
        action="append",
        default=[],
        metavar="NAME=URL",
        help="override an extractor endpoint, e.g. olmocr=http://host:8000/v1 (repeatable)",
    )
    parser.add_argument("--merge", action="store_true", help="fold into an existing --out")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    for override in args.endpoint:
        name, _, url = override.partition("=")
        if name not in _ENDPOINTS:
            print(f"[odb] unknown extractor in --endpoint: {name}", file=sys.stderr)
            return 2
        _ENDPOINTS[name] = url
    pages = load_pages(args.bench)
    if not pages:
        print(f"[odb] no annotated pages under {args.bench}", file=sys.stderr)
        return 2
    sample = stratified_sample(pages, args.pages, args.seed)
    print(f"[odb] {len(pages)} scorable pages, sampling {len(sample)}", flush=True)

    args.work.mkdir(parents=True, exist_ok=True)
    pdfs = [as_single_page_pdf(page.image, args.work / f"{i:04d}.pdf") for i, page in enumerate(sample)]

    names = [e.strip() for e in args.extractors.split(",") if e.strip()]
    rows: list[dict[str, Any]] = []
    for name in names:
        # One extractor per OCR language, since the tesseract pack has to match the page.
        by_language: dict[str, Any] = {}
        for page, pdf in zip(sample, pdfs, strict=True):
            ocr = _OCR_LANGUAGE.get(page.language, _DEFAULT_OCR_LANGUAGE)
            extract = by_language.setdefault(ocr, _extractor_for(name, ocr))
            base = {"extractor": name, "source": page.source, "language": page.language}
            try:
                rows.append({**base, "ok": True, **score_page(extract, page, pdf)})
            except Exception as exc:  # a converter refusing a page is a result, not a sweep abort
                rows.append(
                    {
                        **base,
                        "ok": False,
                        "error": type(exc).__name__,
                        "word_recall": 0.0,
                        "edit_similarity": 0.0,
                        "latency_ms": 0.0,
                        "extracted_chars": 0,
                    }
                )
        done = [r for r in rows if r["extractor"] == name]
        ok = [r for r in done if r["ok"]]
        print(
            f"[odb] {name:11s} {len(ok)}/{len(done)} pages  "
            f"word_recall={sum(r['word_recall'] for r in ok) / max(1, len(ok)):.3f}  "
            f"edit_sim={sum(r['edit_similarity'] for r in ok) / max(1, len(ok)):.3f}",
            flush=True,
        )

    payload = {
        **_environment(),
        "pages_scored": len(sample),
        "text_layer_control_chars": text_layer_control(names),
        "overall": summarize(rows, ["extractor"]),
        "by_source": summarize(rows, ["extractor", "source"]),
        "by_language": summarize(rows, ["extractor", "language"]),
    }
    if args.merge and args.out.exists():
        payload = merge_payload(json.loads(args.out.read_text(encoding="utf-8")), payload, names)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
