"""A recursive chunk set whose count moves under overlap was re-cut, and its cells are void.

``recursive`` overlap is context appended to chunks whose boundaries are already fixed
(chonkie's OverlapRefinery), so the chunk count must not change with the overlap level. Four
cap512 sets built under an earlier size-guard grew by 16 to 22 percent and carried 10 to 15
percent crumb chunks of 16 tokens or fewer; the audit had filed that as the overlap axis being
"real at 512", which reads as a finding rather than the corruption it is. This makes it an
integrity violation, which the export then voids.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "audit_chunk_dimensions.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("audit_chunk_dimensions_integrity", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_chunk_dimensions_integrity"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit() -> Any:
    return _load()


def _row(profile: str, *, rows: int, corpus: str = "c", **axes: Any) -> dict[str, Any]:
    base = {
        "strategy": "recursive",
        "max_tokens": 256,
        "overlap_tokens": 0,
        "tokenizer": "gpt2",
        "breakpoint_model": None,
    }
    base.update(axes)
    return {
        "corpus": corpus,
        "profile": profile,
        "axes": base,
        "rows": rows,
        "chars_total": 1000,
        "tokens_total": 250,
        "chars_per_token": 4.0,
        "cap_enforced": True,
        "true_cap": 256,
        "token_max": 256,
        "over_true_cap": 0,
        "meta_count": rows,
        "meta_count_stale": False,
        "boundary_sha256": f"{corpus}-{profile}",
    }


def test_a_recursive_set_that_grew_under_overlap_is_an_integrity_violation(audit: Any) -> None:
    rows = [
        _row("recursive-t512-o0-gpt2", rows=68090, max_tokens=512),
        _row("recursive-t512-o10-gpt2", rows=78808, max_tokens=512, overlap_tokens=10),
        _row("recursive-t512-o15-gpt2", rows=83001, max_tokens=512, overlap_tokens=15),
    ]
    moved = audit._integrity(rows)["moved_boundaries_under_overlap"]
    assert [m["profile"] for m in moved] == ["recursive-t512-o10-gpt2", "recursive-t512-o15-gpt2"]
    assert moved[0]["rows_at_overlap_0"] == 68090
    assert moved[0]["rows"] == 78808


def test_a_recursive_set_whose_count_holds_is_clean(audit: Any) -> None:
    rows = [
        _row("recursive-t256-o0-gpt2", rows=148008),
        _row("recursive-t256-o51-gpt2", rows=148013, overlap_tokens=51),  # five chunks over a thousand: trimming
    ]
    assert audit._integrity(rows)["moved_boundaries_under_overlap"] == []


def test_the_re_splitting_strategies_are_allowed_to_move(audit: Any) -> None:
    # fast and markdown re-cut the text into overlapping windows, so their count rises by design.
    rows = [
        _row("fast-t256-o0-gpt2", rows=150311, strategy="fast"),
        _row("fast-t256-o51-gpt2", rows=156759, strategy="fast", overlap_tokens=51),
    ]
    assert audit._integrity(rows)["moved_boundaries_under_overlap"] == []


def test_a_set_with_no_overlap_0_sibling_cannot_be_judged_and_is_not_flagged(audit: Any) -> None:
    rows = [_row("recursive-t256-o51-gpt2", rows=148013, overlap_tokens=51)]
    assert audit._integrity(rows)["moved_boundaries_under_overlap"] == []


def test_siblings_are_matched_within_a_corpus_and_a_cap(audit: Any) -> None:
    rows = [
        _row("recursive-t256-o0-gpt2", rows=148008, corpus="en"),
        _row("recursive-t256-o0-gpt2", rows=201824, corpus="de"),
        _row("recursive-t256-o10-gpt2", rows=201824, corpus="de", overlap_tokens=10),
    ]
    assert audit._integrity(rows)["moved_boundaries_under_overlap"] == []


def test_the_true_cap_adds_overlap_only_for_the_strategy_that_appends_it(audit: Any) -> None:
    # recursive appends overlap on top of a capped chunk, so its true cap is cap plus overlap; the
    # Rust splitters re-cut the text into windows of the cap, so overlap never lifts theirs. The
    # shape table printed "True cap 282" for fast at 26 tokens of overlap while every fast set
    # holds 256, which made "cap held" a check that could not fail for them.
    assert audit.true_cap({"strategy": "recursive", "max_tokens": 256, "overlap_tokens": 26}) == 282
    assert audit.true_cap({"strategy": "fast", "max_tokens": 256, "overlap_tokens": 26}) == 256
    assert audit.true_cap({"strategy": "markdown", "max_tokens": 256, "overlap_tokens": 51}) == 256
    assert audit.true_cap({"strategy": "semantic", "max_tokens": 256, "overlap_tokens": 0}) == 256
    assert audit.true_cap({"strategy": "recursive", "max_tokens": None, "overlap_tokens": 26}) == 0
