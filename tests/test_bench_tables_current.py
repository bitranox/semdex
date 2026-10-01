"""Every published benchmark table must still equal its data.

The previous documentation set hand-transcribed about sixty tables from a results file outside the
repository and checked exactly one of them, which is how it came to state a store's quality lever
as 0.71/0.81/0.83 where the data said 0.6534/0.7933/0.8257. Generating the tables removes the
second copy that could drift; this test is what keeps them generated.

It reads only committed JSON under ``tests/benchmarks/raw`` and the docs, so it runs in CI on a
machine with no vector cache.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "gen_bench_tables.py"
_DOCS = _ROOT / "docs"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("gen_bench_tables", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator() -> Any:
    return _load()


@pytest.fixture(scope="module")
def tables(generator: Any) -> dict[str, Any]:
    return generator.collect_table_data()


def _docs_with_blocks() -> list[Path]:
    return sorted(p for p in _DOCS.rglob("*.md") if "BEGIN GENERATED" in p.read_text())


def test_at_least_one_table_is_registered(tables: dict[str, Any]) -> None:
    """A guard against the whole suite passing vacuously because the data went missing."""
    assert tables, "no tables registered: tests/benchmarks/raw is empty or unreadable"


def test_every_generated_block_matches_a_fresh_render(generator: Any, tables: dict[str, Any]) -> None:
    stale: list[str] = []
    for path in _docs_with_blocks():
        text = path.read_text()
        rendered, _written, _unknown = generator._splice(text, tables)
        if rendered != text:
            stale.append(str(path.relative_to(_ROOT)))
    assert not stale, f"stale generated tables in {stale}. Run: python scripts/gen_bench_tables.py"


def test_no_doc_references_an_unregistered_table(generator: Any, tables: dict[str, Any]) -> None:
    """A marker whose id nobody generates would silently keep whatever text sits inside it."""
    unknown: list[str] = []
    for path in _docs_with_blocks():
        _rendered, _written, missing = generator._splice(path.read_text(), tables)
        unknown.extend(f"{path.name}:{m}" for m in missing)
    assert not unknown, f"docs reference table ids that no generator produces: {unknown}"


def test_markers_are_balanced_and_unique(generator: Any) -> None:
    """An unbalanced marker makes the splice silently skip a table rather than fail."""
    for path in _docs_with_blocks():
        text = path.read_text()
        opens = re.findall(r"BEGIN GENERATED ([a-z0-9_]+)", text)
        closes = re.findall(r"END GENERATED ([a-z0-9_]+)", text)
        assert opens == closes, f"{path.name}: markers do not pair up ({opens} vs {closes})"
        assert len(opens) == len(set(opens)), f"{path.name}: a table id appears twice in one file: {opens}"

    # A table MAY appear on more than one page: the resolve-rate summary is evidence for both the
    # chunking and the embedding decision. That is safe precisely because the copies are
    # generated - they cannot drift apart, which is the reason duplication is normally avoided.


def test_the_manifest_is_current(generator: Any, tables: dict[str, Any]) -> None:
    """Catches raw data changing without the tables being regenerated."""
    import json

    manifest_path = _DOCS / "benchmarks" / "generated-tables.manifest.json"
    assert manifest_path.exists(), "run: python scripts/gen_bench_tables.py"
    assert json.loads(manifest_path.read_text()) == generator.manifest(tables), (
        "the manifest does not match the current data. Run: python scripts/gen_bench_tables.py"
    )


def test_the_chunking_page_summary_carries_only_the_chunk_axes(tables: dict[str, Any]) -> None:
    """The chunking page's headline table must not carry the embedding and retrieval-method axes.

    Those rows belong to the embedding and retrieval pages, which render the full summary. On the
    chunking page they outweigh every chunking axis and nothing in its prose explains them.
    """
    axes = {row[0].strip("`") for row in tables["chunk_knob_summary_chunking"]["rows"]}
    assert axes, "the chunk-axes summary rendered no rows"
    assert axes <= {"strategy", "max_tokens", "overlap_tokens", "breakpoint_model"}, sorted(axes)
    full = {row[0].strip("`") for row in tables["chunk_knob_summary"]["rows"]}
    assert {"embedding", "method"} <= full, "the full summary lost the rows the other pages rely on"


def test_a_rendered_table_is_not_empty_for_data_that_exists(tables: dict[str, Any]) -> None:
    """A table that silently renders zero rows would read as 'nothing to report'."""
    empty = [table_id for table_id, spec in tables.items() if not spec["rows"]]
    assert not empty, f"registered tables rendered no rows: {empty}"


def test_generated_tables_carry_their_caveat(tables: dict[str, Any]) -> None:
    """Every table states what it does and does not support, next to the numbers.

    A number's caveat has to travel with it. Left to prose above the table it gets separated the
    first time somebody quotes the table on its own.
    """
    missing = [table_id for table_id, spec in tables.items() if not spec.get("note")]
    assert not missing, f"tables with no explanatory note: {missing}"


def _effect(**overrides: Any) -> dict[str, Any]:
    """One paired comparison, shaped like a row of chunk-knob-effects.json."""
    effect: dict[str, Any] = {
        "axis": "overlap_tokens",
        "corpus": "long_corpus",
        "embedding": "model-a",
        "held_fixed": {"strategy": "recursive", "max_tokens": 256, "breakpoint_model": None},
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
    # A comparison is material only if it resolved AND clears the 0.005 nDCG@10 floor the real
    # exporter judges against (see _score_stats.MATERIAL_FLOOR); default it that way so a test
    # that sets only "resolved" and "mean_delta" (most of them) still gets a self-consistent row,
    # rather than one the exporter could never actually produce.
    effect.setdefault("material", effect["resolved"] and abs(effect["mean_delta"]) >= 0.005)
    return effect


def _audit_row(corpus: str, chunks_per_doc: float, profile: str = "recursive-t256-o0-gpt2") -> dict[str, Any]:
    return {"corpus": corpus, "profile": profile, "chunks_per_doc": chunks_per_doc}


def test_a_corpus_the_audit_never_judged_is_named(generator: Any) -> None:
    """The silent drop that removed every GerDaLIR comparison from every published table.

    A stale audit has no row for a new corpus, so the fitness filter cannot see it and drops it
    exactly as it drops a corpus of one-chunk documents - with no output either way.
    """
    effects = {"effects": [_effect(corpus="new_corpus")]}
    audit = {"chunk_sets": [_audit_row("old_corpus", 18.5)]}
    assert generator.unjudged_corpora(effects, audit) == ["new_corpus"]


def test_a_corpus_judged_and_rejected_is_not_named(generator: Any) -> None:
    """The filter doing its job must stay silent, or the warning is noise nobody reads."""
    effects = {"effects": [_effect(corpus="short_corpus")]}
    audit = {"chunk_sets": [_audit_row("short_corpus", 1.2)]}
    assert generator.unjudged_corpora(effects, audit) == []


def test_a_corpus_judged_only_on_another_profile_is_named(generator: Any) -> None:
    """Fitness is read off ONE reference profile, so a row under any other profile is not a verdict."""
    effects = {"effects": [_effect(corpus="new_corpus")]}
    audit = {"chunk_sets": [_audit_row("new_corpus", 18.5, profile="semantic-t256-gpt2")]}
    assert generator.unjudged_corpora(effects, audit) == ["new_corpus"]


def test_an_axis_the_fitness_filter_does_not_govern_is_not_named(generator: Any) -> None:
    """Embedding and retrieval-method comparisons are measurable on one-chunk documents."""
    effects = {"effects": [_effect(axis="embedding", corpus="short_corpus")]}
    assert generator.unjudged_corpora(effects, {"chunk_sets": []}) == []


def _rung_rows(generator: Any, effects: list[dict[str, Any]]) -> list[list[str]]:
    return generator._rung_table({"effects": effects}, None)["rows"]


def _ceiling_rows(generator: Any, effects: list[dict[str, Any]]) -> list[list[str]]:
    return generator._rung_ceiling_table({"effects": effects}, None)["rows"]


def test_a_ladder_still_resolving_at_its_last_step_says_so(generator: Any) -> None:
    """The reading the whole table exists for: the sweep stopped, the data did not."""
    effects = [
        _effect(from_level=0, to_level=26, resolved=True, mean_delta=0.004),
        _effect(from_level=26, to_level=51, resolved=True, mean_delta=0.003),
        _effect(from_level=51, to_level=77, resolved=True, mean_delta=0.005),
    ]
    (row,) = _ceiling_rows(generator, effects)
    assert row[3] == "3/3 (1 material)"  # only the 0.005 step clears the material floor
    assert row[4] == "51 to 77"
    assert row[5].startswith("51 to 77")
    assert row[6] == "+0.0050 material"


def test_a_ladder_whose_top_steps_stopped_resolving_says_so(generator: Any) -> None:
    effects = [
        _effect(from_level=0, to_level=26, resolved=True, mean_delta=0.004),
        _effect(from_level=26, to_level=51, resolved=False, mean_delta=0.0004),
        _effect(from_level=51, to_level=77, resolved=False, mean_delta=0.0001),
    ]
    (row,) = _ceiling_rows(generator, effects)
    assert row[3] == "1/3 (0 material)"  # the one resolved step's 0.004 delta is under the floor
    assert row[4] == "none", "no step here clears the material floor"
    assert row[6] == "+0.0001 unresolved"


def test_two_embedders_disagreeing_produce_two_rows_not_one_average(generator: Any) -> None:
    """The split. Pooled, these two average to a small gain describing neither."""
    effects = [
        _effect(embedding="stopped", from_level=0, to_level=26, resolved=True, mean_delta=0.004),
        _effect(embedding="stopped", from_level=26, to_level=51, resolved=False, mean_delta=0.0002),
        _effect(embedding="stopped", from_level=51, to_level=77, resolved=False, mean_delta=0.0001),
        _effect(embedding="climbing", from_level=0, to_level=26, resolved=True, mean_delta=0.004),
        _effect(embedding="climbing", from_level=26, to_level=51, resolved=True, mean_delta=0.003),
        _effect(embedding="climbing", from_level=51, to_level=77, resolved=True, mean_delta=0.0052),
    ]
    rows = _ceiling_rows(generator, effects)
    assert len(rows) == 2
    by_embedder = {row[2]: row for row in rows}
    assert by_embedder["`climbing`"][6] == "+0.0052 material"
    assert by_embedder["`stopped`"][6] == "+0.0001 unresolved"


def test_a_two_level_ladder_has_no_consecutive_step_and_is_dropped(generator: Any) -> None:
    """Two levels are one comparison, which the effect table already reports."""
    assert _ceiling_rows(generator, [_effect(from_level=0, to_level=26)]) == []


def test_the_ceiling_row_reads_the_same_whichever_way_the_pair_was_stored(generator: Any) -> None:
    """chunk-knob-effects stores the HIGHER level as from_level, so orientation is load-bearing."""
    forward = [
        _effect(from_level=0, to_level=26, resolved=True, mean_delta=0.004),
        _effect(from_level=26, to_level=51, resolved=True, mean_delta=0.003),
    ]
    reversed_pairs = [
        _effect(from_level=26, to_level=0, resolved=True, mean_delta=-0.004),
        _effect(from_level=51, to_level=26, resolved=True, mean_delta=-0.003),
    ]
    assert _ceiling_rows(generator, forward) == _ceiling_rows(generator, reversed_pairs)


def test_a_comparison_against_a_knobs_default_is_labelled_default_not_none(generator: Any) -> None:
    """A plain semantic profile sets no breakpoint model, so the exporter stores that level as null.

    The table must name it as the default rather than print Python's None, which reads as a
    missing value: "bge-m3 to None" says nothing about what the default model was compared with.
    """
    rows = generator._knob_rows(
        [_effect(axis="breakpoint_model", from_level="bge-m3", to_level=None, held_fixed={"strategy": "semantic"})],
        "breakpoint_model",
        None,
    )
    assert rows[0][3] == "bge-m3 to default"


def test_the_rung_ladder_steps_between_neighbours_only(generator: Any) -> None:
    """0 to 51 is the whole ladder, not a step; publishing it as one double-counts the climb."""
    effects = [
        _effect(from_level=0, to_level=26),
        _effect(from_level=26, to_level=51),
        _effect(from_level=0, to_level=51),
    ]
    assert [row[2] for row in _rung_rows(generator, effects)] == ["0 to 26", "26 to 51"]


def test_every_embedder_at_a_rung_step_is_counted(generator: Any) -> None:
    """One level pair is measured once per embedder, and all of them make the mean."""
    effects = [
        _effect(from_level=0, to_level=26, embedding="model-a", mean_delta=0.02, resolved=True),
        _effect(from_level=0, to_level=26, embedding="model-b", mean_delta=0.00, resolved=False),
        _effect(from_level=26, to_level=51, embedding="model-a", mean_delta=0.01),
        _effect(from_level=26, to_level=51, embedding="model-b", mean_delta=0.01),
    ]
    first = _rung_rows(generator, effects)[0]
    assert first[4] == "1/2", "the resolved count lost an embedder"
    assert first[5] == "+0.0100", "the mean was taken over one embedder, not both"


def test_a_two_level_axis_renders_no_ladder(generator: Any) -> None:
    """Two levels are a single comparison, which the effect table above already shows."""
    assert _rung_rows(generator, [_effect(from_level=0, to_level=26)]) == []


def test_a_rung_step_reads_the_same_whichever_way_the_pair_was_stored(generator: Any) -> None:
    """The raw file stores the HIGHER level as from_level, so an unoriented read inverts the sign."""
    climbing = [
        _effect(from_level=0, to_level=26, mean_delta=0.02),
        _effect(from_level=26, to_level=51, mean_delta=0.02),
    ]
    descending = [
        _effect(from_level=26, to_level=0, mean_delta=-0.02, ci_lo=-0.025, ci_hi=-0.015),
        _effect(from_level=51, to_level=26, mean_delta=-0.02, ci_lo=-0.025, ci_hi=-0.015),
    ]
    assert _rung_rows(generator, climbing) == _rung_rows(generator, descending)


def test_a_profile_label_shows_the_overlap_of_every_strategy_that_honours_it(generator: Any) -> None:
    """Two cells that differ only in overlap must not print the same profile name.

    markdown and fast re-split the text into overlapping windows, so their levels are different
    chunk sets. Labelling only recursive's overlap printed three GerDaLIR ranking rows as the
    identical "fast hint256" carrying three different scores.
    """
    labels = {
        generator._display_profile({"strategy": "fast", "max_tokens": 256, "overlap_tokens": overlap})
        for overlap in (0, 26, 51)
    }
    assert len(labels) == 3, f"overlap levels collapse onto one label: {sorted(labels)}"


def test_a_profile_label_claims_no_overlap_for_a_strategy_that_ignores_it(generator: Any) -> None:
    """chonkie drops the setting for semantic and late, so printing ov0tok there would be a lie."""
    label = generator._display_profile({"strategy": "semantic", "max_tokens": 256, "overlap_tokens": 26})
    assert "ov" not in label, label


def test_every_ranking_row_is_named_uniquely(generator: Any, tables: dict[str, Any]) -> None:
    """A ranking whose rows share a name cannot be acted on, whatever the numbers beside them."""
    collisions = {}
    for table_id, spec in tables.items():
        if not table_id.startswith("ranking_"):
            continue
        names = [f"{row[1]} {row[2]}" for row in spec["rows"]]
        duplicated = sorted({name for name in names if names.count(name) > 1})
        if duplicated:
            collisions[table_id] = duplicated
    assert not collisions, f"ranking rows that do not name a distinct configuration: {collisions}"


def test_the_fit_set_is_exactly_the_corpora_that_can_carry_a_chunking_claim(generator: Any) -> None:
    """Which corpora survive the fitness filter is pinned, because dropping one is silent.

    unjudged_corpora() catches a corpus the audit never JUDGED, which is the stale-audit case that
    once removed all 162 GerDaLIR comparisons from every published table. It cannot catch a corpus
    dropped because the threshold or the reference profile is itself wrong - that path judges the
    corpus and rejects it, exactly as it rejects the one-chunk-per-document corpora on purpose.
    This asserts the outcome instead, so any change to _FIT_CHUNKS_PER_DOC, to
    _FITNESS_REFERENCE_PROFILE, or to the audit data fails here and names the difference.

    Update the expected set deliberately when a corpus is added or a threshold is revised. That
    edit is the review this test exists to force.
    """
    audit = generator._load("chunk-dimension-audit.json")
    assert audit, "chunk-dimension-audit.json is missing; run scripts/audit_chunk_dimensions.py"
    assert generator._fit_corpora(audit) == {
        "gerdalir_de_12k_slice",
        "mldr_de_3k_slice",
        "mldr_en_8k_slice",
    }


def test_fit_corpora_excludes_a_rendering_on_the_real_committed_data(generator: Any) -> None:
    """The shared filter, called directly, drops every declared rendering.

    mldr_en_8k_md_slice is mldr_en_8k_slice re-rendered with its section headings marked: same
    documents, same queries and qrels. tests/benchmarks/raw/markdown-structure-effect.json
    declares that in its ``slice`` object, and the shared filter must read it and exclude the
    rendering from the fit set even though its chunks/doc passes the threshold.
    """
    audit = generator._load("chunk-dimension-audit.json")
    renderings = generator.declared_renderings()
    assert "mldr_en_8k_md_slice" in renderings, "the committed rendering declaration was not found"
    assert generator.fit_corpora(audit, renderings) == {
        "gerdalir_de_12k_slice",
        "mldr_de_3k_slice",
        "mldr_en_8k_slice",
    }


def test_fit_corpora_excludes_a_rendering_by_declaration_not_by_threshold(generator: Any) -> None:
    """A rendering is excluded because it is DECLARED a rendering, never because of its size.

    A synthetic corpus that clears the chunks/doc threshold is dropped when a rendering
    declaration names it, and kept when nothing declares it - proving the exclusion tracks the
    declaration, not the threshold that would otherwise let it in on its own merits.
    """
    audit = {
        "chunk_sets": [
            {"corpus": "widget_corpus", "profile": generator._FITNESS_REFERENCE_PROFILE, "chunks_per_doc": 5},
        ]
    }
    assert generator.fit_corpora(audit, {"widget_corpus"}) is None
    assert generator.fit_corpora(audit, set()) == {"widget_corpus"}


def test_declared_renderings_reads_the_slice_source_corpus_shape(generator: Any, tmp_path: Path) -> None:
    """The reader scans committed raw files for a ``slice`` object naming its source corpus.

    Passed a directory (never hardcoded to the real raw dir), so a future rendering declared the
    same way is picked up with no code change, and this test can prove it on a throwaway fixture.
    """
    (tmp_path / "some-effect.json").write_text(
        json.dumps({"slice": {"corpus": "widget_md_slice", "source_corpus": "widget_slice"}})
    )
    (tmp_path / "unrelated.json").write_text(json.dumps({"effects": []}))
    (tmp_path / "a-list.json").write_text(json.dumps([1, 2, 3]))
    assert generator.declared_renderings(tmp_path) == {"widget_md_slice"}


def test_declared_renderings_ignores_a_slice_that_is_not_an_object(generator: Any, tmp_path: Path) -> None:
    """A ``slice`` value that is not an object declares nothing, and must not crash the scan."""
    (tmp_path / "string-slice.json").write_text(json.dumps({"slice": "widget_md_slice"}))
    (tmp_path / "half-slice.json").write_text(json.dumps({"slice": {"corpus": "widget_md_slice"}}))
    assert generator.declared_renderings(tmp_path) == set()


def test_declared_renderings_refuses_a_raw_file_it_cannot_parse(generator: Any, tmp_path: Path) -> None:
    """A broken raw file must stop the generator, never be skipped.

    Skipping fails in the dangerous direction: if the file that declares a rendering is the broken
    one, the rendering silently re-enters every cross-corpus aggregate and double-counts its source.
    Every other raw reader in the generator raises on a malformed file; this one must too.
    """
    (tmp_path / "some-effect.json").write_text("{not valid json")
    with pytest.raises(json.JSONDecodeError):
        generator.declared_renderings(tmp_path)
