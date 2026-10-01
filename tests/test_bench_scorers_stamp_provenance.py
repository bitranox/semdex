"""Every scorer that writes cells into the scores cache must stamp them as it measures them.

Wiring four scorers fixes four scorers. The shape underneath is "a caller has to remember to do
X, and forgetting is silent": a fifth sweep added later would write unstamped cells, the export
would report them as ``not_recorded``, and nothing would say the scorer was simply never wired.

So the guard is derived, not a hand-kept list of four. A scorer that builds a
``<cache>/scores/<name>.json`` results path is a cell producer by construction - that is the file
the exporter reads cells out of - and every one of them must stamp what it measures.

It checks the CALL, not the import. An import-only check passes a scorer that imports the stamp
and forgets to apply it, which is the very omission this exists to catch - a guard that reads
stronger than it is gets trusted at its stated strength. So the rule is that a ``stamped(...)``
call must appear inside a statement that writes into a results dict, which is where a cell is
actually recorded.

The detector is checked both ways rather than believed. Four other ``score_*.py`` scripts write a
whole payload to ``args.out`` instead of producing cells, and must NOT be selected; and the
call-site rule is exercised against synthetic sources that are wrong in each of the two ways a
real scorer could be wrong.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"

pytestmark = pytest.mark.os_agnostic

# A results file inside the scores cache: `<something> / "scores" / "<name>.json"`.
_PRODUCES_CELLS = re.compile(r'/\s*"scores"\s*/\s*"[^"]+\.json"')

# Scorers that write a whole payload elsewhere. They are the negative control: the detector must
# keep answering "no" for them, or it is no longer detecting cell production.
_NOT_CELL_PRODUCERS = {
    "score_ann_frontier.py",
    "score_extraction_omnidocbench.py",
    "score_span_integrity.py",
    "score_qwen3_instruction.py",
}


def _writes_into_a_dict(node: ast.AST) -> bool:
    """``results[key] = ...`` or ``results.update(...)`` - where a measured cell gets recorded.

    Takes any node because it is fed straight from ``ast.walk``, and answers False for everything
    that is not one of those two statement shapes.
    """
    if isinstance(node, ast.Assign):
        return any(isinstance(target, ast.Subscript) for target in node.targets)
    if isinstance(node, ast.Expr):
        call = node.value
        return isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "update"
    return False


def _calls_stamped(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "stamped"
        for child in ast.walk(node)
    )


def stamps_at_a_write_site(source: str) -> bool:
    """True when a ``stamped(...)`` call sits inside a statement that records a cell.

    Deliberately not satisfied by importing the stamp, nor by calling it anywhere at all: both
    leave the cell itself unstamped while looking wired.
    """
    return any(_writes_into_a_dict(node) and _calls_stamped(node) for node in ast.walk(ast.parse(source)))


def _scorers() -> list[Path]:
    return sorted(_SCRIPTS.glob("score_*.py"))


def _cell_producers() -> list[Path]:
    return [path for path in _scorers() if _PRODUCES_CELLS.search(path.read_text(encoding="utf-8"))]


def test_the_detector_finds_the_known_cell_producers() -> None:
    """A detector that selected nothing would pass the real test vacuously."""
    found = {path.name for path in _cell_producers()}
    assert found == {
        "score_chunk_sweep.py",
        "score_hybrid_sweep.py",
        "score_product_k.py",
        "score_rerank_sweep.py",
        "score_through_store.py",
    }, f"cell producers detected: {sorted(found)}"


def test_the_detector_rejects_scorers_that_write_no_cells() -> None:
    """The known negative: these mention scores but produce no cell for the exporter to read."""
    produced = {path.name for path in _cell_producers()}
    present = {path.name for path in _scorers()}
    for name in sorted(_NOT_CELL_PRODUCERS & present):
        assert name not in produced, f"{name} was selected as a cell producer, so the rule is wrong"


def test_importing_the_stamp_without_applying_it_is_not_enough() -> None:
    """The hole an import-only check leaves open, stated as a case."""
    source = "from _provenance import stamped\nresults[cell] = row\n"
    assert not stamps_at_a_write_site(source)


def test_calling_the_stamp_away_from_the_write_site_is_not_enough() -> None:
    """Calling it somewhere proves nothing about the cell that gets recorded."""
    source = "from _provenance import stamped\nprint(stamped(row))\nresults[cell] = row\n"
    assert not stamps_at_a_write_site(source)


def test_a_subscript_assignment_wrapped_in_the_stamp_counts() -> None:
    assert stamps_at_a_write_site("results[cell] = stamped(row)\n")


def test_an_update_wrapped_in_the_stamp_counts() -> None:
    assert stamps_at_a_write_site("results.update({k: stamped(r) for k, r in rows.items()})\n")


def test_every_cell_producer_stamps_what_it_measures() -> None:
    producers = _cell_producers()
    assert producers, "no cell producers found: the detector, not the scorers, is broken"
    unstamped = [path.name for path in producers if not stamps_at_a_write_site(path.read_text(encoding="utf-8"))]
    assert unstamped == [], f"scorers recording cells without stamping provenance: {unstamped}"
