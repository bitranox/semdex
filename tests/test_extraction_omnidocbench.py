"""Scoring and sampling logic for scripts/score_extraction_omnidocbench.py.

The benchmark's whole claim is that it measures fidelity on real documents rather than agreement
with fixtures we wrote. That claim rests on three pure pieces: which annotated text counts as
ground truth, how an extraction is scored against it, and how pages are sampled. Each has already
been wrong once in a way that produced a plausible number rather than an error.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_extraction_omnidocbench.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("score_extraction_omnidocbench", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def odb() -> Any:
    return _load()


# --- what counts as ground truth --------------------------------------------------------------


def _block(text: str, category: str = "text_block", order: int = 0, ignore: int = 0) -> dict[str, Any]:
    return {"category_type": category, "text": text, "order": order, "ignore": ignore}


def test_only_running_text_categories_are_scored(odb: Any) -> None:
    """Headers, page numbers and the `abandon` class are furniture a converter may drop.

    Scoring them would measure a formatting policy rather than whether the page was read, and
    would mark every backend down for the same non-failure.
    """
    layout = [
        _block("A" * 40, "text_block"),
        _block("B" * 40, "title"),
        _block("C" * 40, "header"),
        _block("D" * 40, "page_number"),
        _block("E" * 40, "abandon"),
        _block("F" * 40, "table"),
    ]

    kept = odb.ground_truth_blocks(layout)

    assert kept == ["A" * 40, "B" * 40]


def test_blocks_are_returned_in_reading_order(odb: Any) -> None:
    """edit_similarity compares whole pages, so a shuffled reference would punish a correct read."""
    layout = [_block("third" * 8, order=3), _block("first" * 8, order=1), _block("second" * 8, order=2)]

    kept = odb.ground_truth_blocks(layout)

    assert [k[:5] for k in kept] == ["first", "secon", "third"]


def test_annotations_marked_ignore_are_skipped(odb: Any) -> None:
    layout = [_block("kept" * 10), _block("dropped" * 10, ignore=1)]

    assert odb.ground_truth_blocks(layout) == ["kept" * 10]


def test_very_short_blocks_do_not_count(odb: Any) -> None:
    """The annotation tail runs down to single words like "that".

    Any extraction of any page contains those by accident, so counting them inflates every
    backend's score equally and invisibly - a metric that cannot go below its own noise.
    """
    layout = [_block("that"), _block("x" * 25)]

    assert odb.ground_truth_blocks(layout) == ["x" * 25]


# --- scoring an extraction --------------------------------------------------------------------


def test_a_perfect_extraction_scores_one(odb: Any) -> None:
    blocks = ["the quick brown fox", "jumps over the lazy dog"]

    assert odb.word_recall(blocks, "the quick brown fox jumps over the lazy dog") == 1.0


def test_recall_ignores_bullets_and_punctuation(odb: Any) -> None:
    """The failure that made the first version of this benchmark meaningless.

    docling read a slide correctly and scored 0.033, because the annotation bullets each line
    with a filled circle and docling emits a hyphen. A metric that calls that a miss is measuring
    markup, not reading.
    """
    blocks = ["◎ Promote parental and community participation."]

    assert odb.word_recall(blocks, "- Promote parental and community participation") == 1.0


def test_recall_ignores_reading_order(odb: Any) -> None:
    """A two-column page emitted in the wrong order was still read."""
    blocks = ["alpha beta", "gamma delta"]

    assert odb.word_recall(blocks, "gamma delta alpha beta") == 1.0


def test_recall_counts_the_missing_share(odb: Any) -> None:
    blocks = ["alpha beta gamma delta"]

    assert odb.word_recall(blocks, "alpha beta") == pytest.approx(0.5)


def test_an_empty_extraction_scores_zero(odb: Any) -> None:
    """A converter with no OCR returns nothing; that has to read as zero, not as an error."""
    assert odb.word_recall(["alpha beta gamma"], "") == 0.0


def test_chinese_pages_are_scored_by_character(odb: Any) -> None:
    """Half of OmniDocBench is Chinese, which has no spaces.

    Splitting on whitespace makes a Chinese page one enormous token, which is present or absent
    as a whole - so every backend scores 0.0 or 1.0 and the benchmark says nothing about them.
    """
    blocks = ["中文文字识别"]

    assert odb.word_recall(blocks, "中文文字") == pytest.approx(4 / 6)
    assert odb.word_recall(blocks, "中文文字识别") == 1.0


def test_repeated_words_are_counted_as_a_multiset(odb: Any) -> None:
    """Emitting one "the" must not satisfy three of them, or padding would buy recall."""
    assert odb.word_recall(["the the the"], "the") == pytest.approx(1 / 3)


def test_edit_similarity_punishes_what_recall_forgives(odb: Any) -> None:
    """The pair only earns its keep if the two metrics disagree on reordering.

    If both were order-insensitive, the second column would be decoration.
    """
    reference = "alpha beta gamma delta epsilon zeta"
    shuffled = "zeta epsilon delta gamma beta alpha"

    assert odb.word_recall([reference], shuffled) == 1.0
    assert odb.edit_similarity(reference, shuffled) < 0.6


def test_edit_similarity_of_an_empty_extraction_is_zero(odb: Any) -> None:
    assert odb.edit_similarity("alpha beta", "") == 0.0


# --- sampling ----------------------------------------------------------------------------------


def _page(odb: Any, source: str, name: str) -> Any:
    return odb.Page(image=Path(f"/nonexistent/{name}.png"), source=source, language="english", blocks=["x" * 30])


def test_the_sample_is_spread_across_document_sources(odb: Any) -> None:
    """OmniDocBench is not balanced: `book` has 276 pages and `historical_document` five.

    A uniform draw would return mostly books and be published as "documents in the wild".
    """
    pages = [_page(odb, "book", f"b{i}") for i in range(50)] + [_page(odb, "note", f"n{i}") for i in range(50)]

    picked = odb.stratified_sample(pages, 10, seed=1)

    assert sum(1 for p in picked if p.source == "book") == 5
    assert sum(1 for p in picked if p.source == "note") == 5


def test_a_small_source_contributes_everything_it_has(odb: Any) -> None:
    """Asking for more pages than a source owns must take all of them, not fail or repeat."""
    pages = [_page(odb, "book", f"b{i}") for i in range(50)] + [_page(odb, "historical", "h0")]

    picked = odb.stratified_sample(pages, 20, seed=1)

    assert sum(1 for p in picked if p.source == "historical") == 1


def test_the_sample_is_reproducible_for_a_given_seed(odb: Any) -> None:
    """A re-run has to score the same pages, or two runs cannot be compared at all."""
    pages = [_page(odb, "book", f"b{i}") for i in range(40)]

    first = [p.image.name for p in odb.stratified_sample(pages, 8, seed=7)]
    second = [p.image.name for p in odb.stratified_sample(pages, 8, seed=7)]
    other = [p.image.name for p in odb.stratified_sample(pages, 8, seed=8)]

    assert first == second
    assert first != other


# --- folding the GPU run into the CPU run ------------------------------------------------------


def _summary(extractor: str, source: str, recall: float) -> dict[str, Any]:
    return {"extractor": extractor, "source": source, "word_recall": recall, "pages": 1, "failed": 0}


def test_merging_keeps_extractors_the_new_run_did_not_measure(odb: Any) -> None:
    """The GPU backends need the card to themselves, so they run separately from the CPU ones.

    Without this the second run would publish a table holding two of the five backends and read
    as though the other three had not been measured at all.
    """
    existing = {"by_source": [_summary("xberg", "book", 0.8), _summary("docling", "book", 0.7)]}
    fresh = {"by_source": [_summary("olmocr", "book", 0.95)]}

    merged = odb.merge_payload(existing, fresh, ["olmocr"])

    assert {r["extractor"] for r in merged["by_source"]} == {"xberg", "docling", "olmocr"}


def test_re_measuring_an_extractor_replaces_its_rows(odb: Any) -> None:
    """Two rows for one extractor would be silently averaged by any reader of the table."""
    existing = {"by_source": [_summary("xberg", "book", 0.10)]}
    fresh = {"by_source": [_summary("xberg", "book", 0.80)]}

    merged = odb.merge_payload(existing, fresh, ["xberg"])

    assert len(merged["by_source"]) == 1
    assert merged["by_source"][0]["word_recall"] == 0.80


def test_the_text_layer_control_accumulates_across_runs(odb: Any) -> None:
    """A backend's zero is only readable beside its control, so the control must survive a merge."""
    existing = {"text_layer_control_chars": {"markitdown": 108, "xberg": 108}}
    fresh = {"text_layer_control_chars": {"olmocr": 96}}

    merged = odb.merge_payload(existing, fresh, ["olmocr"])

    assert merged["text_layer_control_chars"] == {"markitdown": 108, "xberg": 108, "olmocr": 96}
