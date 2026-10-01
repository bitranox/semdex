#!/usr/bin/env python
"""Check every figure quoted in benchmark PROSE against a derivation from the committed data.

``scripts/gen_bench_tables.py`` made the published TABLES a rendering of
``tests/benchmarks/raw/*.json``, so they cannot drift. The sentences around them stayed
hand-transcribed: rewriting ``docs/benchmarks/03-chunking.md`` published 42 such figures and one of
them was already wrong. This closes that half.

A regex that merely finds numbers in prose was considered and rejected. It cannot know what a
number refers to, so it can neither confirm nor refute one. Instead each published figure is paired,
in a claim file, with the derivation that produces it - which rowset, which filter, which
reduction - and the checker recomputes it. Being explicit is the point: the claim file is what
records the MEANING of the number, and reading one beside its page is how a reviewer sees whether a
figure is guarded at all.

Claim files live in ``tests/benchmarks/claims/<page>.toml``, one per documentation page. See that
directory's README for the format.

Usage:
  python scripts/check_bench_claims.py          # check every claim file, exit 1 on a mismatch
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import rtoml

_ROOT = Path(__file__).resolve().parent.parent
_RAW = _ROOT / "tests" / "benchmarks" / "raw"
_CLAIMS = _ROOT / "tests" / "benchmarks" / "claims"

# The sibling table generator owns the low-to-high orientation of a paired comparison and the
# generated-block markers. Both are reused rather than restated here: a second implementation of
# either is precisely the duplicated-number problem this gate exists to stop.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

Row = dict[str, Any]

_CLAIM_KEYS = frozenset(
    {
        "id",
        "quote",
        "published",
        "rows",
        "reduce",
        "field",
        "where",
        "against",
        "against_rows",
        "against_field",
        "transform",
        "scale",
        "tolerance",
        "why",
    }
)
_FILE_KEYS = frozenset({"doc", "sets", "rowsets", "claim"})
_ROWSET_KEYS = frozenset({"source", "path", "orient"})
_SINGLE_REDUCERS = frozenset({"count", "sum", "mean", "min", "max", "value"})
_PAIR_REDUCERS = frozenset({"ratio", "difference", "percent_change"})
_TRANSFORMS = frozenset({"abs", "negate"})

# A count small enough to spell is spelled, in these pages and in ordinary prose: "of the ten
# dense-to-hybrid comparisons, nine resolve". Refusing those leaves the most load-bearing figures
# on a page unguardable, or forces the prose to write digits for the gate's convenience. The range
# stops at twenty because past that the pages use digits anyway, and a word this table does not
# hold still raises rather than being guessed at.
_WORD_NUMERALS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    # A frequency word is a count too: "wins once" and "wins twice" are counts of resolved wins.
    "once": 1,
    "twice": 2,
}

__all__ = [
    "Claim",
    "ClaimFile",
    "ClaimFileError",
    "DerivationError",
    "Failure",
    "Published",
    "RowSet",
    "check_all",
    "check_claim_file",
    "claim_files",
    "derive",
    "load_claim_file",
    "load_rows",
    "main",
    "orient_rows",
    "parse_published",
    "strip_generated",
]


class ClaimFileError(Exception):
    """A claim file is malformed, so nothing it contains can be trusted to check anything."""


class DerivationError(Exception):
    """A derivation could not be evaluated, which is a defect in the claim, not a mismatch."""


# ---------------------------------------------------------------- the generator, reused


def _generator() -> Any:
    """The table generator module, imported late because ``scripts`` reaches sys.path above."""
    # Deferred: the sibling script is importable only after the sys.path insert at module load.
    import gen_bench_tables

    return gen_bench_tables


def orient_rows(rows: Sequence[Row]) -> list[Row]:
    """Orient paired comparisons low level to high level and name the level each one favours.

    The raw file stores the HIGHER level as ``from_level`` in about half its rows, so a filter
    written against the published table ("0 to 51") would miss them and an unoriented sign would
    read backwards.

    Args:
        rows: Paired-comparison rows carrying from_level, to_level and mean_delta.

    Returns:
        New rows, oriented, each carrying four fields a claim can filter on: ``favoured_level``,
        ``unfavoured_level``, ``favoured_end`` (``"low"`` or ``"high"``, for a claim about the
        DIRECTION rather than the level) and ``levels`` (both levels, for "comparisons involving X").
    """
    generator = _generator()
    oriented: list[Row] = []
    for row in rows:
        # Reaching into the generator rather than restating its rule: a second implementation of
        # the orientation is exactly the duplicated-number problem this gate exists to stop.
        item: Row = dict(generator._oriented(row))
        item["favoured_level"] = generator._favoured_level(item)
        won_high = item["favoured_level"] == item["to_level"]
        item["unfavoured_level"] = item["from_level"] if won_high else item["to_level"]
        item["favoured_end"] = "high" if won_high else "low"
        item["levels"] = [item["from_level"], item["to_level"]]
        oriented.append(item)
    return oriented


def strip_generated(text: str) -> str:
    """The prose alone, with every generated table removed.

    A figure inside a generated block is already gated by ``test_bench_tables_current``; letting a
    claim point at one would inflate the count of guarded prose figures without guarding any.
    """
    # The generator's own marker pattern, so the block format has one definition.
    return _generator()._BLOCK_RE.sub("", text)


# ---------------------------------------------------------------- the published figure


class Published:
    """A figure exactly as the page prints it, with the tolerance its own precision implies."""

    def __init__(self, *, text: str, value: float, tolerance: float) -> None:
        self.text = text
        self.value = value
        self.tolerance = tolerance


def parse_published(text: str, override: float | None = None) -> Published:
    """Read a published figure and the tolerance that its rounding allows.

    The default tolerance states the honest claim: the printed number is the correct rounding of
    the derived one. A figure rounded more coarsely than its digits suggest ("33,700" for 33,748)
    needs an explicit tolerance, which forces whoever writes it to say so.

    A count spelled as a word is read as the integer it names, so a sentence that says "nine
    resolve" can be guarded without rewriting it to say "9". The tolerance is an integer's, because
    a spelled number is exact by construction: nothing rounds to "nine".

    Args:
        text: The figure as printed, e.g. ``"+0.0143"``, ``"12,298"``, ``"36.0"``, ``"nine"``.
        override: An explicit absolute tolerance, replacing the implied one.

    Returns:
        The parsed figure.

    Raises:
        ClaimFileError: The text is not a number this checker can read.

    Examples:
        >>> parse_published("36.0").tolerance
        0.05
        >>> parse_published("nine").value
        9.0
    """
    spelled = _WORD_NUMERALS.get(text.strip().lower())
    if spelled is not None:
        return Published(text=text, value=float(spelled), tolerance=override if override is not None else 0.5)
    bare = text.replace(",", "").removeprefix("+")
    try:
        value = float(bare)
    except ValueError as exc:
        raise ClaimFileError(f"published figure {text!r} is not a number") from exc
    places = len(bare.partition(".")[2])
    return Published(text=text, value=value, tolerance=override if override is not None else 0.5 * 10**-places)


# ---------------------------------------------------------------- claim-file model


class RowSet:
    """One list of records inside one committed raw file, named so claims can be short."""

    def __init__(self, *, name: str, source: str, path: str | None = None, orient: bool = False) -> None:
        self.name = name
        self.source = source
        self.path = path
        self.orient = orient


class Claim:
    """One published figure and the derivation that must reproduce it."""

    def __init__(
        self,
        *,
        id: str,  # noqa: A002 - the claim's own name in the file; "claim_id" would read wrong there
        quote: str,
        published: str,
        rows: str,
        reduce: str,
        why: str,
        where: Mapping[str, Any] | None = None,
        field: str | None = None,
        against: Mapping[str, Any] | None = None,
        against_rows: str | None = None,
        against_field: str | None = None,
        transform: str | None = None,
        scale: float = 1.0,
        tolerance: float | None = None,
    ) -> None:
        self.id = id
        self.quote = quote
        self.published = parse_published(published, tolerance)
        self.rows = rows
        self.reduce = reduce
        self.why = why
        self.where: Mapping[str, Any] = where or {}
        self.field = field
        self.against = against
        self.against_rows = against_rows
        self.against_field = against_field
        self.transform = transform
        self.scale = scale


class ClaimFile:
    """Every claim made about one documentation page."""

    def __init__(
        self,
        *,
        path: Path,
        doc: Path,
        sets: Mapping[str, list[Any]],
        rowsets: Mapping[str, RowSet],
        claims: Sequence[Claim],
        raw_dir: Path,
    ) -> None:
        self.path = path
        self.doc = doc
        self.sets = sets
        self.rowsets = rowsets
        self.claims = tuple(claims)
        self.raw_dir = raw_dir


class Failure:
    """One claim that did not hold, named so the message points at the sentence to fix."""

    def __init__(self, claim_id: str, reason: str) -> None:
        self.claim_id = claim_id
        self.reason = reason


# ---------------------------------------------------------------- loading


def _require_keys(table: Mapping[str, Any], allowed: frozenset[str], what: str) -> None:
    """Refuse an unrecognised key, because a misspelt one is silently ignored otherwise."""
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ClaimFileError(f"{what}: unknown key(s) {unknown}; allowed are {sorted(allowed)}")


def _rowset_from(name: str, table: Mapping[str, Any]) -> RowSet:
    _require_keys(table, _ROWSET_KEYS, f"rowset {name!r}")
    if "source" not in table:
        raise ClaimFileError(f"rowset {name!r}: no source file named")
    return RowSet(name=name, source=str(table["source"]), path=table.get("path"), orient=bool(table.get("orient")))


def _claim_from(table: Mapping[str, Any], rowsets: Mapping[str, RowSet], where: str) -> Claim:
    _require_keys(table, _CLAIM_KEYS, f"{where}")
    missing = sorted({"id", "quote", "published", "rows", "reduce", "why"} - set(table))
    if missing:
        raise ClaimFileError(f"{where}: missing required key(s) {missing}")
    for key in ("rows", "against_rows"):
        named = table.get(key)
        if named is not None and named not in rowsets:
            raise ClaimFileError(f"{where}: {key} names undeclared rowset {named!r}")
    _validate_reduce(table, where)
    return Claim(**dict(table))


def _validate_reduce(table: Mapping[str, Any], where: str) -> None:
    """A reduction the engine does not implement must fail at load, not silently at use."""
    reduce = table["reduce"]
    if reduce not in _SINGLE_REDUCERS | _PAIR_REDUCERS:
        raise ClaimFileError(f"{where}: unknown reduce {reduce!r}")
    if reduce != "count" and not table.get("field"):
        raise ClaimFileError(f"{where}: reduce {reduce!r} needs a field")
    transform = table.get("transform")
    if transform is not None and transform not in _TRANSFORMS:
        raise ClaimFileError(f"{where}: unknown transform {transform!r}")


def load_claim_file(path: Path, doc_root: Path = _ROOT, raw_dir: Path = _RAW) -> ClaimFile:
    """Parse and validate one claim file.

    Args:
        path: The ``.toml`` claim file.
        doc_root: Root the file's ``doc`` path is resolved against.
        raw_dir: Directory holding the raw JSON its rowsets name.

    Returns:
        The parsed claim file.

    Raises:
        ClaimFileError: The file is malformed in any way that would weaken a check.
    """
    data: dict[str, Any] = rtoml.load(path)
    _require_keys(data, _FILE_KEYS, f"{path.name}")
    if "doc" not in data:
        raise ClaimFileError(f"{path.name}: no doc named")
    rowsets = {name: _rowset_from(name, table) for name, table in dict(data.get("rowsets", {})).items()}
    claims = [
        _claim_from(table, rowsets, f"{path.name} claim #{index}")
        for index, table in enumerate(list(data.get("claim", [])), start=1)
    ]
    return ClaimFile(
        path=path,
        doc=doc_root / str(data["doc"]),
        sets=dict(data.get("sets", {})),
        rowsets=rowsets,
        claims=claims,
        raw_dir=raw_dir,
    )


def claim_files(claims_dir: Path = _CLAIMS) -> list[Path]:
    """Every claim file, in a stable order."""
    return sorted(claims_dir.glob("*.toml")) if claims_dir.is_dir() else []


def load_rows(rowset: RowSet, raw_dir: Path = _RAW) -> list[Row]:
    """The records a rowset names, oriented if it declares itself a set of paired comparisons.

    A path that names a single table rather than a list is a one-record rowset, so a scalar fact
    the raw file keeps beside its rows can be claimed with ``reduce = "value"``.

    Args:
        rowset: The declared rowset.
        raw_dir: Directory holding the committed raw JSON.

    Returns:
        The rows, ready to filter.

    Raises:
        ClaimFileError: The source file or the named list is missing.
    """
    source = raw_dir / rowset.source
    if not source.exists():
        raise ClaimFileError(f"rowset {rowset.name!r}: no such raw file {rowset.source}")
    document: Any = json.loads(source.read_text())
    rows: Any = document if rowset.path is None else document.get(rowset.path)
    if isinstance(rows, dict):
        # A raw file's scalar facts (a sample's mean document size, a shipped threshold) sit in a
        # table rather than a list; naming that table is naming the one record it holds.
        rows = [rows]
    if not isinstance(rows, list):
        raise ClaimFileError(f"rowset {rowset.name!r}: {rowset.source}[{rowset.path}] is not a list of records")
    records = cast(list[Row], rows)
    return orient_rows(records) if rowset.orient else records


# ---------------------------------------------------------------- the derivation engine

_MISSING = object()


def _get(row: Row, dotted: str) -> Any:
    """Read a field, following dots into nested tables like ``held_fixed.strategy``."""
    value: Any = row
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return _MISSING
        table = cast(Row, value)
        if part not in table:
            return _MISSING
        value = table[part]
    return value


def _expand(wanted: Any, sets: Mapping[str, list[Any]]) -> Any:
    """Resolve a ``"@name"`` reference to the list declared under ``[sets]``."""
    if isinstance(wanted, str) and wanted.startswith("@"):
        name = wanted[1:]
        if name not in sets:
            raise DerivationError(f"filter names undeclared set {wanted!r}")
        return sets[name]
    return wanted


def _hit(actual: Any, wanted: Any) -> bool:
    """One filter term.

    A list on the WANTED side means "one of"; a list on the ROW side (the ``levels`` of a paired
    comparison) means "contains", which is how a claim says "comparisons involving recursive".
    """
    if isinstance(wanted, list):
        return actual in wanted
    return wanted in actual if isinstance(actual, list) else actual == wanted


_RELATIVE_KEYS = frozenset({"field", "prefix", "suffix"})


def _relative(row: Row, key: str, spec: Mapping[str, Any]) -> bool:
    """One filter term stated against ANOTHER field of the same row.

    Some relations live in the data rather than in a list of values: a method compared against its
    own reranked variant is `to_level == from_level + "+rerank"`, which no enumeration of levels
    expresses. Listing the three bases and the three rerank levels separately takes the CROSS
    product and silently matches nine pairs where the page counts three.

    Deliberately not an expression language: a named field with an optional prefix or suffix, so a
    claim file stays declarative and a reviewer can see what it compares.
    """
    unknown = set(spec) - _RELATIVE_KEYS
    if unknown or "field" not in spec:
        raise DerivationError(f"filter {key!r} must name a field and may add a prefix or suffix, got {dict(spec)!r}")
    other = _get(row, str(spec["field"]))
    if other is _MISSING:
        raise DerivationError(f"filter {key!r} names field {spec['field']!r}, which the row lacks")
    wanted = f"{spec.get('prefix', '')}{other}{spec.get('suffix', '')}"
    return _get(row, key) == wanted


def _matches(row: Row, where: Mapping[str, Any], sets: Mapping[str, list[Any]]) -> bool:
    for key, wanted in where.items():
        # `wanted` is Any, so isinstance alone narrows it to Mapping[Unknown, Unknown]; the cast
        # states the shape a claim file's TOML table actually has.
        hit = (
            _relative(row, key, cast("Mapping[str, Any]", wanted))
            if isinstance(wanted, Mapping)
            else _hit(_get(row, key), _expand(wanted, sets))
        )
        if not hit:
            return False
    return True


def _select(rows: Sequence[Row], where: Mapping[str, Any], sets: Mapping[str, list[Any]], rowset: str) -> list[Row]:
    """Filter rows, refusing a key that no row in the rowset carries.

    A misspelt key matches nothing, so without this the filter silently drops out and the claim is
    evaluated over the whole rowset.
    """
    for key in where:
        if all(_get(row, key) is _MISSING for row in rows):
            raise DerivationError(f"filter key {key!r} is one no row carries in rowset {rowset!r}")
    return [row for row in rows if _matches(row, where, sets)]


def _values(rows: Sequence[Row], field: str | None, transform: str | None) -> list[float]:
    if field is None:
        raise DerivationError("no field named")
    if not rows:
        raise DerivationError("the filter matched no rows, so there is nothing to reduce")
    numbers: list[float] = []
    for row in rows:
        value = _get(row, field)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise DerivationError(f"field {field!r} is {value!r}, which is not a number")
        numbers.append(float(value))
    if transform == "abs":
        return [abs(number) for number in numbers]
    return [-number for number in numbers] if transform == "negate" else numbers


def _only(values: Sequence[float]) -> float:
    if len(values) != 1:
        raise DerivationError(f"the filter must select exactly one row here, it selected {len(values)}")
    return values[0]


def _reduce_single(claim: Claim, rows: Sequence[Row]) -> float:
    if claim.reduce == "count":
        return float(len(rows))
    values = _values(rows, claim.field, claim.transform)
    if claim.reduce == "value":
        return _only(values)
    if claim.reduce == "mean":
        return sum(values) / len(values)
    return {"sum": sum, "min": min, "max": max}[claim.reduce](values)


def _reduce_pair(claim: Claim, left: Sequence[Row], right: Sequence[Row]) -> float:
    a = _only(_values(left, claim.field, claim.transform))
    b = _only(_values(right, claim.against_field or claim.field, claim.transform))
    if claim.reduce == "difference":
        return a - b
    if b == 0:
        raise DerivationError("the comparison row's value is zero, so the ratio is undefined")
    return a / b if claim.reduce == "ratio" else (a - b) / b * 100


def derive(claim: Claim, rows: Sequence[Row], against_rows: Sequence[Row], sets: Mapping[str, list[Any]]) -> float:
    """Evaluate a claim's derivation against the rows it names.

    Args:
        claim: The claim to evaluate.
        rows: Rows of the claim's own rowset.
        against_rows: Rows of the comparison rowset, for a two-selection reduction.
        sets: The named lists a filter may reference as ``"@name"``.

    Returns:
        The derived figure, already scaled.

    Raises:
        DerivationError: The derivation cannot be evaluated as written.
    """
    selected = _select(rows, claim.where, sets, claim.rows)
    if claim.reduce not in _PAIR_REDUCERS:
        return _reduce_single(claim, selected) * claim.scale
    other_where = claim.against if claim.against is not None else claim.where
    other = _select(against_rows, other_where, sets, claim.against_rows or claim.rows)
    return _reduce_pair(claim, selected, other) * claim.scale


# ---------------------------------------------------------------- checking


def within_tolerance(derived: float, published: Published) -> bool:
    """Whether a derived value is published correctly at the figure's own precision.

    The raw files hold values already rounded once (a hit rate stored as ``0.8235``), so a derived
    value can sit exactly on a rounding boundary, where either neighbour is its correct printed
    form; the float product then lands a hair to one side and a strict half-unit test rejects a
    figure that is right. The slack is a millionth of the tolerance, far below any real drift.

    Args:
        derived: The value the claim's derivation produced.
        published: The figure as the page prints it, with its implied tolerance.

    Returns:
        True when the difference is within the tolerance, boundary included.

    Examples:
        >>> within_tolerance(82.34999999999999, parse_published("82.4"))
        True
        >>> within_tolerance(82.44, parse_published("82.3"))
        False
    """
    return abs(derived - published.value) <= published.tolerance * (1 + 1e-6)


def _quote_failure(claim: Claim, text: str, prose: str, doc_name: str) -> str | None:
    """Whether the claim still points at exactly one prose sentence stating its figure.

    Counted over the PROSE, because a table note generated from the same data legitimately repeats
    a sentence the prose also states, and that repetition leaves the claim unambiguous.
    """
    occurrences = prose.count(claim.quote)
    if occurrences == 0 and claim.quote in text:
        return "quote sits inside a generated block, which test_bench_tables_current already gates"
    if occurrences == 0:
        return f"quote does not appear in {doc_name}: {claim.quote!r}"
    if occurrences > 1:
        return f"quote appears {occurrences} times in {doc_name}, so it names no single figure"
    if claim.published.text not in claim.quote:
        return f"published figure {claim.published.text!r} is not in the quote {claim.quote!r}"
    return None


def _check_claim(
    claim: Claim, claim_file: ClaimFile, text: str, prose: str, rows_by_name: dict[str, list[Row]]
) -> str | None:
    reason = _quote_failure(claim, text, prose, claim_file.doc.name)
    if reason:
        return reason
    try:
        derived = derive(
            claim,
            rows_by_name[claim.rows],
            rows_by_name[claim.against_rows or claim.rows],
            claim_file.sets,
        )
    except DerivationError as exc:
        return f"derivation failed: {exc}"
    if not within_tolerance(derived, claim.published):
        return (
            f"published {claim.published.text} but derived {derived:.6g} "
            f"(tolerance {claim.published.tolerance:g}); derivation: {claim.why}"
        )
    return None


def check_claim_file(claim_file: ClaimFile) -> list[Failure]:
    """Check every claim in one file against its page and the committed data.

    Args:
        claim_file: The parsed claim file.

    Returns:
        One failure per claim that did not hold; empty when the page is clean.
    """
    text = claim_file.doc.read_text()
    prose = strip_generated(text)
    rows_by_name = {name: load_rows(rowset, claim_file.raw_dir) for name, rowset in claim_file.rowsets.items()}
    failures: list[Failure] = []
    for claim in claim_file.claims:
        reason = _check_claim(claim, claim_file, text, prose, rows_by_name)
        if reason:
            failures.append(Failure(claim.id, reason))
    return failures


def check_all(claims_dir: Path = _CLAIMS) -> tuple[int, list[Failure]]:
    """Check every claim file under ``claims_dir``.

    Args:
        claims_dir: Directory holding the per-page claim files.

    Returns:
        How many claims were examined, and every failure found. Examining nothing is itself a
        failure: a gate that checks nothing must not be able to report a pass.
    """
    examined = 0
    failures: list[Failure] = []
    for path in claim_files(claims_dir):
        claim_file = load_claim_file(path)
        examined += len(claim_file.claims)
        failures.extend(check_claim_file(claim_file))
    if examined == 0:
        failures.append(Failure("<none>", f"no claims were examined; {claims_dir} holds no claim file"))
    return examined, failures


def main() -> None:
    examined, failures = check_all()
    for path in claim_files():
        print(f"[claims] {path.relative_to(_ROOT)}: {len(load_claim_file(path).claims)} claims", flush=True)
    for failure in failures:
        print(f"[claims] FAIL {failure.claim_id}: {failure.reason}", flush=True)
    print(f"[claims] {examined} claims examined, {len(failures)} failing", flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
