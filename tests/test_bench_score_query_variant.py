"""A query variant must never land in the files the published tables are built from.

The overlap effect on GerDaLIR tracks QUERY LENGTH (see the chunking page), and the causal test
for that is to re-score the same cells with every query cut to its first N words: only the query
vectors change, the corpus vectors are reused. That re-score produces rows keyed by the same
(corpus, profile, embedder) as the real ones, so written into the default results file or the
default per-query directory it would silently overwrite the published measurement with a
truncated-query one. The variant therefore refuses to run unless both output locations are named
explicitly, and every row it does write says which variant it is.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "score_chunk_sweep.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("score_chunk_sweep_variant", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["score_chunk_sweep_variant"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scorer() -> Any:
    return _load()


def test_truncation_keeps_the_first_n_words_and_leaves_short_queries_alone(scorer: Any) -> None:
    queries = {"q1": "one two three four five", "q2": "short one", "q3": "  spaced   out  words here "}
    cut = scorer.truncate_queries(queries, 3)
    assert cut == {"q1": "one two three", "q2": "short one", "q3": "spaced out words"}
    assert queries["q1"] == "one two three four five", "the caller's mapping must not be mutated"


def test_truncation_to_fewer_than_one_word_is_refused(scorer: Any) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        scorer.truncate_queries({"q": "a b"}, 0)


def test_a_variant_run_is_refused_unless_both_output_locations_are_explicit(scorer: Any) -> None:
    with pytest.raises(SystemExit, match="SEMDEX_SCORE_OUT"):
        scorer.variant_guard(20, out_set=False, perquery_set=True)
    with pytest.raises(SystemExit, match="SEMDEX_SCORE_PERQUERY"):
        scorer.variant_guard(20, out_set=True, perquery_set=False)


def test_a_variant_run_with_both_locations_named_is_allowed(scorer: Any) -> None:
    scorer.variant_guard(20, out_set=True, perquery_set=True)


def test_no_variant_needs_no_explicit_locations(scorer: Any) -> None:
    scorer.variant_guard(None, out_set=False, perquery_set=False)


def test_the_variant_is_read_from_the_environment(scorer: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEMDEX_SCORE_QUERY_WORDS", raising=False)
    assert scorer.query_words() is None
    monkeypatch.setenv("SEMDEX_SCORE_QUERY_WORDS", "20")
    assert scorer.query_words() == 20
