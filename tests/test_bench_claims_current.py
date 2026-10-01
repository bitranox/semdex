"""Every figure quoted in benchmark PROSE must still equal its data.

The generated tables cannot drift, because ``scripts/gen_bench_tables.py`` renders them from
``tests/benchmarks/raw/*.json`` and ``tests/test_bench_tables_current.py`` re-renders and compares.
The prose around them had no such gate: rewriting ``docs/benchmarks/03-chunking.md`` published 42
hand-transcribed figures, one of which was already wrong (0.0895 published as "0.090"). That is the
same drift the generated tables exist to prevent, arriving through the sentences instead.

A claim file pairs each published figure with the derivation that produces it, so the checker knows
what a number MEANS rather than merely finding one. What it cannot do is notice a figure nobody
wrote a claim for, which is why the coverage question is answered by reading the claim file beside
the page rather than by this test.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "check_bench_claims.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("check_bench_claims", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker() -> Any:
    return _load()


# ---------------------------------------------------------------- the gate


def test_every_seeded_claim_still_equals_its_data(checker: Any) -> None:
    examined, failures = checker.check_all()
    assert not failures, "\n".join(f"{f.claim_id}: {f.reason}" for f in failures)
    assert examined > 0


def test_the_gate_examined_the_page_it_was_seeded_with(checker: Any) -> None:
    """A claim file emptied by accident must not read as a clean page."""
    examined, _failures = checker.check_all()
    assert examined >= 40, f"only {examined} claims examined; 03-chunking alone was seeded with 40+"


def test_a_run_that_examined_nothing_is_a_failure(checker: Any, tmp_path: Path) -> None:
    """A gate that checks nothing must not be able to print a pass."""
    examined, failures = checker.check_all(claims_dir=tmp_path)
    assert examined == 0
    assert failures, "an empty claims directory reported success"
    assert "no claims" in failures[0].reason


def test_every_claim_id_is_unique(checker: Any) -> None:
    seen: list[str] = []
    for path in checker.claim_files():
        seen.extend(claim.id for claim in checker.load_claim_file(path).claims)
    duplicated = sorted({name for name in seen if seen.count(name) > 1})
    assert not duplicated, f"claim ids used twice: {duplicated}"


# ---------------------------------------------------------------- published-number parsing


@pytest.mark.parametrize(
    ("published", "value", "tolerance"),
    [
        ("41", 41.0, 0.5),
        ("12,298", 12298.0, 0.5),
        ("+0.0143", 0.0143, 0.00005),
        ("-0.0103", -0.0103, 0.00005),
        ("36.0", 36.0, 0.05),
        ("nine", 9.0, 0.5),
        ("once", 1.0, 0.5),
        ("twice", 2.0, 0.5),
    ],
)
def test_a_published_figure_carries_its_own_tolerance(
    checker: Any, published: str, value: float, tolerance: float
) -> None:
    """The default tolerance is "the published number is the correct rounding of the derived one"."""
    parsed = checker.parse_published(published)
    assert parsed.value == pytest.approx(value)
    assert parsed.tolerance == pytest.approx(tolerance)


# ---------------------------------------------------------------- the derivation engine


def _effect(**overrides: Any) -> dict[str, Any]:
    """One paired comparison, shaped like a row of chunk-knob-effects.json."""
    effect: dict[str, Any] = {
        "axis": "overlap_tokens",
        "corpus": "long_corpus",
        "embedding": "model-a",
        "held_fixed": {"strategy": "recursive", "max_tokens": 256},
        "from_level": 0,
        "to_level": 26,
        "mean_delta": 0.01,
        "ci_lo": 0.005,
        "ci_hi": 0.015,
        "wins": 10,
        "losses": 2,
        "resolved": True,
    }
    effect.update(overrides)
    return effect


def _claim(checker: Any, **overrides: Any) -> Any:
    fields: dict[str, Any] = {
        "id": "probe",
        "quote": "the probe sentence says 3 things",
        "published": "3",
        "rows": "effects",
        "reduce": "count",
        "where": {},
        "why": "a probe",
    }
    fields.update(overrides)
    return checker.Claim(**fields)


def test_count_counts_only_the_matching_rows(checker: Any) -> None:
    rows = [_effect(resolved=True), _effect(resolved=True), _effect(resolved=False)]
    claim = _claim(checker, reduce="count", where={"resolved": True})
    assert checker.derive(claim, rows, rows, {}) == 2


def test_a_list_in_a_filter_means_one_of(checker: Any) -> None:
    rows = [_effect(corpus="a"), _effect(corpus="b"), _effect(corpus="c")]
    claim = _claim(checker, reduce="count", where={"corpus": ["a", "c"]})
    assert checker.derive(claim, rows, rows, {}) == 2


def test_a_named_set_expands_in_a_filter(checker: Any) -> None:
    """The fit-corpora list is repeated by a dozen claims; naming it once is what keeps it right."""
    rows = [_effect(corpus="a"), _effect(corpus="b")]
    claim = _claim(checker, reduce="count", where={"corpus": "@fit"})
    assert checker.derive(claim, rows, rows, {"fit": ["a"]}) == 1


def test_a_dotted_filter_key_reaches_a_nested_field(checker: Any) -> None:
    rows = [_effect(), _effect(held_fixed={"strategy": "semantic", "max_tokens": 256})]
    claim = _claim(checker, reduce="count", where={"held_fixed.strategy": "recursive"})
    assert checker.derive(claim, rows, rows, {}) == 1


def test_a_filter_key_no_row_carries_is_refused(checker: Any) -> None:
    """A typo in a filter key silently widens the selection, which is how a wrong claim goes green."""
    rows = [_effect()]
    claim = _claim(checker, reduce="count", where={"held_fixed.stratgy": "recursive"})
    with pytest.raises(checker.DerivationError, match="no row carries"):
        checker.derive(claim, rows, rows, {})


def test_a_selection_that_matches_nothing_is_refused(checker: Any) -> None:
    """Zero rows reduce to zero, which reads as a measurement rather than as an empty selection."""
    rows = [_effect()]
    claim = _claim(checker, reduce="mean", field="mean_delta", where={"corpus": "absent"})
    with pytest.raises(checker.DerivationError, match="matched no rows"):
        checker.derive(claim, rows, rows, {})


def test_value_refuses_a_selection_that_is_not_a_single_row(checker: Any) -> None:
    rows = [_effect(), _effect()]
    claim = _claim(checker, reduce="value", field="mean_delta", where={})
    with pytest.raises(checker.DerivationError, match="exactly one row"):
        checker.derive(claim, rows, rows, {})


def test_mean_max_and_abs_compose(checker: Any) -> None:
    rows = [_effect(mean_delta=-0.08), _effect(mean_delta=-0.02)]
    claim = _claim(checker, reduce="max", field="mean_delta", transform="abs", where={})
    assert checker.derive(claim, rows, rows, {}) == pytest.approx(0.08)


def test_scale_turns_a_stored_fraction_into_published_percent(checker: Any) -> None:
    rows = [_effect(mean_delta=0.0717)]
    claim = _claim(checker, reduce="value", field="mean_delta", scale=100, where={})
    assert checker.derive(claim, rows, rows, {}) == pytest.approx(7.17)


def test_percent_change_reads_two_selections(checker: Any) -> None:
    rows = [_effect(corpus="hi", mean_delta=110.0), _effect(corpus="lo", mean_delta=100.0)]
    claim = _claim(
        checker,
        reduce="percent_change",
        field="mean_delta",
        where={"corpus": "hi"},
        against={"corpus": "lo"},
    )
    assert checker.derive(claim, rows, rows, {}) == pytest.approx(10.0)


def test_against_field_compares_two_fields_of_one_row(checker: Any) -> None:
    """true_cap against nominal_cap on the same row is how the overlap percentage is published."""
    rows = [{"true_cap": 266, "nominal_cap": 256}]
    claim = _claim(
        checker,
        reduce="percent_change",
        field="true_cap",
        against_field="nominal_cap",
        where={},
    )
    assert checker.derive(claim, rows, rows, {}) == pytest.approx(3.90625)


def test_orientation_uses_the_generators_own_low_to_high_rule(checker: Any) -> None:
    """chunk-knob-effects stores the HIGHER level as from_level, so an unoriented read inverts.

    The rule lives in gen_bench_tables._oriented and is reused rather than restated, because a
    second implementation of it is exactly the drift this whole gate exists to stop.
    """
    descending = _effect(from_level=51, to_level=0, mean_delta=-0.0141)
    oriented = checker.orient_rows([descending])[0]
    assert (oriented["from_level"], oriented["to_level"]) == (0, 51)
    assert oriented["mean_delta"] == pytest.approx(0.0141)
    assert oriented["favoured_level"] == 51
    assert oriented["unfavoured_level"] == 0
    assert oriented["favoured_end"] == "high"
    assert oriented["levels"] == [0, 51]


def test_a_filter_against_a_list_field_asks_whether_it_contains_the_value(checker: Any) -> None:
    """The "18 ties" claims are about comparisons INVOLVING a strategy, on either side of the pair."""
    rows = checker.orient_rows([_effect(from_level="fast", to_level="recursive", mean_delta=0.01)])
    claim = _claim(checker, reduce="count", where={"levels": "fast"})
    assert checker.derive(claim, rows, rows, {}) == 1
    missing = _claim(checker, reduce="count", where={"levels": "semantic"})
    assert checker.derive(missing, rows, rows, {}) == 0


# ---------------------------------------------------------------- claim-to-doc binding


def _claim_file(checker: Any, tmp_path: Path, doc_text: str, claim_toml: str, rows: list[Any]) -> Any:
    """A complete claim file over a throwaway doc and a throwaway raw file."""
    (tmp_path / "raw").mkdir(exist_ok=True)
    (tmp_path / "raw" / "probe.json").write_text(json.dumps({"effects": rows}))
    doc = tmp_path / "probe.md"
    doc.write_text(doc_text)
    claims = tmp_path / "probe.toml"
    claims.write_text(
        f'doc = "{doc.name}"\n\n[rowsets.effects]\nsource = "probe.json"\npath = "effects"\n\n{claim_toml}'
    )
    return checker.load_claim_file(claims, doc_root=tmp_path, raw_dir=tmp_path / "raw")


_ONE_CLAIM = """
[[claim]]
id = "probe"
quote = "exactly 2 comparisons resolved"
published = "2"
rows = "effects"
reduce = "count"
where = { resolved = true }
why = "a probe"
"""


def test_a_correct_figure_passes(checker: Any, tmp_path: Path) -> None:
    claim_file = _claim_file(
        checker,
        tmp_path,
        "Prose saying exactly 2 comparisons resolved.\n",
        _ONE_CLAIM,
        [_effect(), _effect(), _effect(resolved=False)],
    )
    assert checker.check_claim_file(claim_file) == []


def test_a_wrong_figure_is_reported(checker: Any, tmp_path: Path) -> None:
    """The whole point: change the data under a published number and the gate goes red."""
    claim_file = _claim_file(
        checker,
        tmp_path,
        "Prose saying exactly 2 comparisons resolved.\n",
        _ONE_CLAIM,
        [_effect(), _effect(), _effect()],
    )
    failures = checker.check_claim_file(claim_file)
    assert [f.claim_id for f in failures] == ["probe"]
    assert "published 2" in failures[0].reason
    assert "derived 3" in failures[0].reason


def test_a_quote_the_doc_does_not_carry_is_reported(checker: Any, tmp_path: Path) -> None:
    """A claim whose sentence was rewritten away is stale, not satisfied."""
    claim_file = _claim_file(checker, tmp_path, "The sentence was rewritten.\n", _ONE_CLAIM, [_effect(), _effect()])
    failures = checker.check_claim_file(claim_file)
    assert "does not appear" in failures[0].reason


def test_a_quote_appearing_twice_is_reported(checker: Any, tmp_path: Path) -> None:
    """Two occurrences mean the claim cannot say which figure it guards."""
    doc = "Prose saying exactly 2 comparisons resolved.\nAnd exactly 2 comparisons resolved again.\n"
    claim_file = _claim_file(checker, tmp_path, doc, _ONE_CLAIM, [_effect(), _effect()])
    failures = checker.check_claim_file(claim_file)
    assert "appears 2 times" in failures[0].reason


def test_a_quote_inside_a_generated_block_is_reported(checker: Any, tmp_path: Path) -> None:
    """Generated tables have their own gate; covering one here would inflate the claim count."""
    doc = (
        "<!-- BEGIN GENERATED probe (scripts/gen_bench_tables.py) -->\n"
        "Prose saying exactly 2 comparisons resolved.\n"
        "<!-- END GENERATED probe -->\n"
    )
    claim_file = _claim_file(checker, tmp_path, doc, _ONE_CLAIM, [_effect(), _effect()])
    failures = checker.check_claim_file(claim_file)
    assert "generated block" in failures[0].reason


def test_a_published_number_missing_from_its_quote_is_reported(checker: Any, tmp_path: Path) -> None:
    """Otherwise a claim can guard a sentence that never states the number it checks."""
    claim = _ONE_CLAIM.replace('quote = "exactly 2 comparisons resolved"', 'quote = "comparisons resolved"')
    claim_file = _claim_file(checker, tmp_path, "Prose saying comparisons resolved.\n", claim, [_effect(), _effect()])
    failures = checker.check_claim_file(claim_file)
    assert "not in the quote" in failures[0].reason


def test_an_unknown_key_in_a_claim_is_refused(checker: Any, tmp_path: Path) -> None:
    """A misspelt filter key would otherwise leave the selection silently unfiltered."""
    claim = _ONE_CLAIM.replace("where = { resolved = true }", "wheree = { resolved = true }")
    with pytest.raises(checker.ClaimFileError, match="unknown key"):
        _claim_file(checker, tmp_path, "exactly 2 comparisons resolved\n", claim, [_effect()])


def test_a_claim_naming_an_undeclared_rowset_is_refused(checker: Any, tmp_path: Path) -> None:
    claim = _ONE_CLAIM.replace('rows = "effects"', 'rows = "effect"')
    with pytest.raises(checker.ClaimFileError, match="undeclared rowset"):
        _claim_file(checker, tmp_path, "exactly 2 comparisons resolved\n", claim, [_effect()])


# --------------------------------------------------------------------------
# A count small enough to spell, and a filter stated against another field
# --------------------------------------------------------------------------


def test_a_spelled_count_is_read_as_its_integer(checker: Any) -> None:
    """These pages spell a small count, and the figures most worth guarding are among them: the
    dense-to-hybrid result reads "of the ten ... nine resolve". Refusing a word leaves those
    unguardable, or makes the prose write digits for the gate's convenience."""
    assert checker.parse_published("nine").value == 9.0
    assert checker.parse_published("Twelve").value == 12.0
    assert checker.parse_published("nine").tolerance == 0.5, "a spelled number is exact"


def test_a_word_that_is_not_a_numeral_is_still_refused(checker: Any) -> None:
    """The control. Reading words must not turn a typo into a silently guessed figure."""
    with pytest.raises(checker.ClaimFileError):
        checker.parse_published("eleventy")


def test_a_filter_can_be_stated_against_another_field_of_the_row(checker: Any) -> None:
    """A method against its OWN reranked variant is a relation, not a list of values."""
    rows = [
        {"from_level": "dense@20", "to_level": "dense@20+rerank"},
        {"from_level": "bm25@20", "to_level": "bm25@20+rerank"},
        {"from_level": "dense@20", "to_level": "bm25@20+rerank"},
    ]
    where = {"to_level": {"field": "from_level", "suffix": "+rerank"}}

    matched = [row for row in rows if checker._matches(row, where, {})]

    assert [row["from_level"] for row in matched] == ["dense@20", "bm25@20"]


def test_two_lists_take_the_cross_product_which_is_why_the_relation_exists(checker: Any) -> None:
    """The control that names what the relative filter is FOR. Stating the same intent as two
    lists matches the mismatched pair as well, so a count of "each base against its own rerank"
    comes out too high while looking like a filter that says exactly that."""
    rows = [
        {"from_level": "dense@20", "to_level": "dense@20+rerank"},
        {"from_level": "dense@20", "to_level": "bm25@20+rerank"},
    ]
    where = {"from_level": ["dense@20", "bm25@20"], "to_level": ["dense@20+rerank", "bm25@20+rerank"]}

    assert len([row for row in rows if checker._matches(row, where, {})]) == 2, "both, including the mismatched pair"


def test_a_relative_filter_that_names_no_field_is_refused(checker: Any) -> None:
    """A malformed relation must refuse rather than match nothing: matching nothing reads as a real
    answer of zero, which is the failure mode the missing-key guard already exists for."""
    with pytest.raises(checker.DerivationError):
        checker._matches({"to_level": "x"}, {"to_level": {"suffix": "+rerank"}}, {})
    with pytest.raises(checker.DerivationError):
        checker._matches({"to_level": "x"}, {"to_level": {"field": "absent"}}, {})


# --------------------------------------------------------------------------
# A path naming a single record, not a list
# --------------------------------------------------------------------------


def test_a_path_to_a_single_record_is_a_one_row_rowset(checker: Any, tmp_path: Path) -> None:
    """A raw file's scalar facts (a sample's mean document size, a shipped threshold) live in a
    table, not a list; a rowset naming that table is the one row it holds."""
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "probe.json").write_text(json.dumps({"sample": {"mean_bytes": 1639, "documents": 500}}))
    rowset = checker.RowSet(name="sample", source="probe.json", path="sample")
    rows = checker.load_rows(rowset, raw_dir=tmp_path / "raw")
    assert rows == [{"mean_bytes": 1639, "documents": 500}]
    claim = _claim(checker, reduce="value", field="mean_bytes", where={"documents": 500})
    assert checker.derive(claim, rows, rows, {}) == 1639


def test_a_path_to_a_scalar_is_still_refused(checker: Any, tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "probe.json").write_text(json.dumps({"threshold": 100000}))
    rowset = checker.RowSet(name="threshold", source="probe.json", path="threshold")
    with pytest.raises(checker.ClaimFileError, match="not a list of records"):
        checker.load_rows(rowset, raw_dir=tmp_path / "raw")


@pytest.mark.parametrize(
    ("published", "derived"),
    [
        ("82.4", 82.34999999999999),
        ("19.2", 19.149999999999999),
        ("82.3", 82.35000000000001),
    ],
)
def test_a_derived_value_on_the_rounding_boundary_is_correctly_published_either_way(
    checker: Any, published: str, derived: float
) -> None:
    """A stored 0.8235 is 82.35 in print and rounds to 82.4, but the float product lands a hair
    under the boundary and a strict half-unit test rejects it; the raw file already rounded once,
    so a value sitting on the boundary is published correctly by either neighbour."""
    assert checker.within_tolerance(derived, checker.parse_published(published))


def test_a_derived_value_past_the_rounding_boundary_is_still_reported(checker: Any) -> None:
    assert not checker.within_tolerance(82.44, checker.parse_published("82.3"))
