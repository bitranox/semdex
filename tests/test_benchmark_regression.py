# pyright: basic
"""Fast-CI tests for the benchmark regression machinery.

The E2E matrices that produce ``benchmark-results.json`` are ``local_only`` (Docker + real
models), but the *comparator* that turns those results into a pass/fail regression verdict
is the safety net and must be covered in ordinary CI. These tests exercise
``scripts/benchmark_compare.py`` (status-flip and metric-drop detection, tolerance, the
reported-only timing carve-out, not-run/new bucketing, schema guard, ``--update``) and the
round-trip that the recorder in ``_benchmark_report.py`` writes is one the comparator reads.

``# pyright: basic`` because the comparator is loaded by path at runtime (``scripts/`` is not
an importable package), so its attributes are not statically resolvable here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import _benchmark_report as recorder
import pytest

pytestmark = pytest.mark.os_agnostic


def _load_comparator():
    """Load scripts/benchmark_compare.py by path (scripts/ is not an importable package)."""
    path = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_compare.py"
    spec = importlib.util.spec_from_file_location("benchmark_compare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so Pydantic can resolve the module's `from __future__ import
    # annotations` string forward-refs (e.g. ResultStatus) via its module namespace.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bc = _load_comparator()


def _results(cells):
    """Build a BenchmarkResults from a ``{combo: (status, metrics)}`` mapping."""
    results = {
        combo: bc.BenchmarkResult(status=bc.ResultStatus(status), metrics=metrics)
        for combo, (status, metrics) in cells.items()
    }
    return bc.BenchmarkResults(generated_utc="2026-01-01T00:00:00+00:00", versions={}, results=results)


def test_status_flip_ok_to_fail_is_a_regression():
    base = _results({"quality/recursive": ("ok", {"ndcg@10": 0.6})})
    cur = _results({"quality/recursive": ("fail", {})})
    cmp = bc.compare(base, cur, bc.DEFAULT_TOLERANCE)
    assert cmp.regressions and "status ok -> fail" in cmp.regressions[0]


def test_status_flip_ok_to_skip_is_a_regression():
    base = _results({"store/pgvector": ("ok", {"ndcg@10": 0.6})})
    cur = _results({"store/pgvector": ("skip", {})})
    assert bc.compare(base, cur, bc.DEFAULT_TOLERANCE).regressions


def test_metric_drop_beyond_tolerance_is_a_regression():
    base = _results({"quality/recursive": ("ok", {"ndcg@10": 0.62})})
    cur = _results({"quality/recursive": ("ok", {"ndcg@10": 0.50})})
    cmp = bc.compare(base, cur, bc.DEFAULT_TOLERANCE)
    assert cmp.regressions and "ndcg@10" in cmp.regressions[0]


def test_metric_drop_within_tolerance_is_ok():
    base = _results({"quality/recursive": ("ok", {"ndcg@10": 0.62})})
    cur = _results({"quality/recursive": ("ok", {"ndcg@10": 0.60})})  # 0.02 < 0.03 tolerance
    cmp = bc.compare(base, cur, bc.DEFAULT_TOLERANCE)
    assert not cmp.regressions and cmp.ok


def test_metric_improvement_is_not_a_regression():
    base = _results({"quality/recursive": ("ok", {"ndcg@10": 0.62})})
    cur = _results({"quality/recursive": ("ok", {"ndcg@10": 0.71})})
    assert not bc.compare(base, cur, bc.DEFAULT_TOLERANCE).regressions


def test_timing_and_size_metrics_do_not_gate():
    """Machine-dependent metrics are reported, never gated - in the direction that could gate.

    Every metric here is compared higher-is-better, so only a metric that goes DOWN can be
    called a regression. For latency and size, down is faster and smaller: a run on a quicker
    machine is exactly what the carve-out has to keep from turning the gate red. Asserting on
    an increase instead proves nothing, because an increase cannot gate for any metric.
    """
    base = _results({"store/pgvector": ("ok", {"search_p50_ms": 60.0, "store_mb": 50.0, "ndcg@10": 0.62})})
    cur = _results({"store/pgvector": ("ok", {"search_p50_ms": 6.0, "store_mb": 5.0, "ndcg@10": 0.62})})
    assert not bc.compare(base, cur, bc.DEFAULT_TOLERANCE).regressions


def test_quality_drop_still_gates_when_timing_also_changes():
    base = _results({"store/pgvector": ("ok", {"search_p50_ms": 6.0, "ndcg@10": 0.62})})
    cur = _results({"store/pgvector": ("ok", {"search_p50_ms": 6.0, "ndcg@10": 0.40})})
    assert bc.compare(base, cur, bc.DEFAULT_TOLERANCE).regressions


def test_missing_key_is_not_run_not_a_regression():
    base = _results({"quality/a": ("ok", {"ndcg@10": 0.6}), "quality/b": ("ok", {"ndcg@10": 0.6})})
    cur = _results({"quality/a": ("ok", {"ndcg@10": 0.6})})
    cmp = bc.compare(base, cur, bc.DEFAULT_TOLERANCE)
    assert not cmp.regressions
    assert cmp.not_run == ["quality/b"]


def test_new_key_is_reported_not_a_regression():
    base = _results({"quality/a": ("ok", {"ndcg@10": 0.6})})
    cur = _results({"quality/a": ("ok", {"ndcg@10": 0.6}), "quality/c": ("ok", {"ndcg@10": 0.7})})
    cmp = bc.compare(base, cur, bc.DEFAULT_TOLERANCE)
    assert not cmp.regressions and cmp.new == ["quality/c"]


def test_na_cells_never_regress():
    base = _results({"extract/text@pdf": ("n/a", {})})
    cur = _results({"extract/text@pdf": ("n/a", {})})
    assert not bc.compare(base, cur, bc.DEFAULT_TOLERANCE).regressions


def test_main_returns_1_on_regression(tmp_path: Path):
    base_path = tmp_path / "baseline.json"
    cur_path = tmp_path / "current.json"
    base_path.write_text(_results({"quality/r": ("ok", {"ndcg@10": 0.62})}).model_dump_json(), encoding="utf-8")
    cur_path.write_text(_results({"quality/r": ("ok", {"ndcg@10": 0.40})}).model_dump_json(), encoding="utf-8")
    assert bc.main([str(cur_path), "--baseline", str(base_path)]) == 1


def test_main_returns_0_without_regression(tmp_path: Path):
    base_path = tmp_path / "baseline.json"
    cur_path = tmp_path / "current.json"
    base_path.write_text(_results({"quality/r": ("ok", {"ndcg@10": 0.62})}).model_dump_json(), encoding="utf-8")
    cur_path.write_text(_results({"quality/r": ("ok", {"ndcg@10": 0.63})}).model_dump_json(), encoding="utf-8")
    assert bc.main([str(cur_path), "--baseline", str(base_path)]) == 0


def test_main_update_writes_baseline(tmp_path: Path):
    base_path = tmp_path / "sub" / "baseline.json"
    cur_path = tmp_path / "current.json"
    cur_path.write_text(_results({"quality/r": ("ok", {"ndcg@10": 0.7})}).model_dump_json(), encoding="utf-8")
    assert bc.main([str(cur_path), "--baseline", str(base_path), "--update"]) == 0
    assert base_path.exists()
    assert bc.main([str(cur_path), "--baseline", str(base_path)]) == 0  # now compares clean


def test_schema_version_mismatch_is_rejected(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema_version": 999, "generated_utc": null, "versions": {}, "results": {}}', encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        bc.main([str(bad)])
    assert exc.value.code == 2


def test_recorder_writes_file_the_comparator_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The recorder's benchmark-results.json validates against the comparator's schema."""
    results_path = tmp_path / "benchmark-results.json"
    monkeypatch.setattr(recorder, "_RESULTS_PATH", results_path)
    monkeypatch.setattr(recorder, "_results", {})
    monkeypatch.setattr(recorder, "_versions", recorder._capture_versions())

    recorder.record("quality", "recursive+fastembed", recorder.ResultStatus.OK, {"ndcg@10": 0.62}, corpus="nfcorpus")
    recorder.record("extract", "text@md", recorder.ResultStatus.OK, {"phrase_recall": 1.0})

    doc = bc.BenchmarkResults.model_validate_json(results_path.read_text(encoding="utf-8"))
    assert doc.schema_version == bc.SCHEMA_VERSION
    assert set(doc.results) == {"quality/recursive+fastembed@nfcorpus", "extract/text@md"}
    assert doc.results["quality/recursive+fastembed@nfcorpus"].metrics["ndcg@10"] == pytest.approx(0.62)
    assert doc.versions["python"]  # captured a python version


# --- paired per-query comparison -------------------------------------------------------------
#
# The gated slice is 20 queries. Comparing two MEANS over 20 queries carries a 95 percent
# half-width near 0.13 on nDCG@10, so the 0.03 tolerance was both blind (a real 0.10 drop sits
# inside the noise) and trigger-happy (a fire at 0.03 proves nothing). Both runs answer the SAME
# queries, so the difference can be taken per query, which removes query difficulty - nearly all
# of that variance. These cover what the pairing must and must not do.


def _paired(cells):
    """Build a BenchmarkResults from ``{combo: (status, metrics, per_query)}``."""
    results = {
        combo: bc.BenchmarkResult(status=bc.ResultStatus(status), metrics=metrics, per_query=per_query)
        for combo, (status, metrics, per_query) in cells.items()
    }
    return bc.BenchmarkResults(generated_utc="2026-01-01T00:00:00+00:00", versions={}, results=results)


# Twenty queries whose difficulty varies the way real ones do. This spread is the reason an
# unpaired mean cannot resolve a small consistent shift.
_SPREAD = [
    0.21,
    0.95,
    0.44,
    0.80,
    0.12,
    0.67,
    0.99,
    0.33,
    0.58,
    0.72,
    0.05,
    0.88,
    0.40,
    0.61,
    0.29,
    0.93,
    0.50,
    0.77,
    0.16,
    0.84,
]


def _cell(scores, *, metric="ndcg@10"):
    per_query = {metric: {f"q{i}": s for i, s in enumerate(scores)}}
    return ("ok", {metric: sum(scores) / len(scores)}, per_query)


def test_a_consistent_drop_too_small_for_the_unpaired_tolerance_is_caught_when_paired():
    """The regression the old gate could not see: every query down 0.02, tolerance 0.03.

    This is the gap. Unpaired, the means differ by less than the tolerance and the gate passes
    a run in which EVERY query got worse. Paired, the same data is unambiguous.
    """
    base = _paired({"quality/recursive": _cell(_SPREAD)})
    cur = _paired({"quality/recursive": _cell([s - 0.02 for s in _SPREAD])})

    unpaired_only = bc.compare(
        _results({"quality/recursive": ("ok", base.results["quality/recursive"].metrics)}),
        _results({"quality/recursive": ("ok", cur.results["quality/recursive"].metrics)}),
        bc.DEFAULT_TOLERANCE,
    )
    assert not unpaired_only.regressions, "precondition: the mean comparison misses this drop"

    cmp = bc.compare(base, cur, bc.DEFAULT_TOLERANCE)

    assert cmp.regressions, "paired comparison must catch a drop on every single query"
    assert "paired" in cmp.regressions[0]
    assert "20 worse, 0 better" in cmp.regressions[0]


def test_queries_swinging_far_wider_than_the_tolerance_are_not_a_regression():
    """Individual queries move a lot between runs; that is not a trend and must not gate.

    Each query shifts by 0.15 - five times the tolerance - but half go up and half go down, so
    the behaviour did not get worse. A gate that fired here would be unusable.
    """
    swung = [s + (0.15 if i % 2 else -0.15) for i, s in enumerate(_SPREAD)]
    cmp = bc.compare(_paired({"q/r": _cell(_SPREAD)}), _paired({"q/r": _cell(swung)}), bc.DEFAULT_TOLERANCE)

    assert not cmp.regressions


def test_one_collapsed_query_among_improvements_is_not_a_regression():
    """A single query dragging the mean down is a query that changed, not a trend.

    On a 20-query slice one query is 5 percent of the mean, so a single collapse can outweigh
    many small gains. Requiring more queries worse than better keeps that from gating.
    """
    scores = [s + 0.01 for s in _SPREAD]
    scores[3] = 0.0  # one query collapses from 0.80
    cmp = bc.compare(_paired({"q/r": _cell(_SPREAD)}), _paired({"q/r": _cell(scores)}), bc.DEFAULT_TOLERANCE)

    assert not cmp.regressions
    assert cmp.ok == ["q/r"]


def test_an_improvement_on_every_query_is_never_a_regression():
    cur = [min(1.0, s + 0.05) for s in _SPREAD]
    assert not bc.compare(
        _paired({"q/r": _cell(_SPREAD)}), _paired({"q/r": _cell(cur)}), bc.DEFAULT_TOLERANCE
    ).regressions


def test_only_queries_both_runs_scored_are_compared():
    """A query added or dropped between runs has no counterpart, so it cannot be differenced.

    Differencing against a missing query would compare a score to nothing. The shared ids here
    are all worse, and the extra ones on either side must not change the verdict.
    """
    base = _paired({"q/r": ("ok", {"ndcg@10": 0.5}, {"ndcg@10": {"a": 0.8, "b": 0.6, "gone": 0.1}})})
    cur = _paired({"q/r": ("ok", {"ndcg@10": 0.5}, {"ndcg@10": {"a": 0.7, "b": 0.5, "added": 0.99}})})

    cmp = bc.compare(base, cur, bc.DEFAULT_TOLERANCE)

    assert cmp.regressions and "2 queries" in cmp.regressions[0]


def test_a_baseline_without_per_query_data_still_gates_through_the_unpaired_path():
    """Baselines captured before per-query recording existed must keep working.

    Otherwise adding the pairing would silently disable the gate for every existing baseline -
    the gate would go green because it stopped checking, which is the worst failure mode here.
    """
    base = _results({"q/r": ("ok", {"ndcg@10": 0.62})})
    cur = _paired({"q/r": _cell([0.50] * 20)})

    cmp = bc.compare(base, cur, bc.DEFAULT_TOLERANCE)

    assert cmp.regressions and "unpaired" in cmp.regressions[0]


def test_a_current_run_without_per_query_data_still_gates_through_the_unpaired_path():
    base = _paired({"q/r": _cell([0.62] * 20)})
    cur = _results({"q/r": ("ok", {"ndcg@10": 0.50})})

    cmp = bc.compare(base, cur, bc.DEFAULT_TOLERANCE)

    assert cmp.regressions and "unpaired" in cmp.regressions[0]


def test_timing_metrics_never_gate_even_with_per_query_data():
    """The reported-only carve-out is about the metric, not about how it was compared.

    Pairing makes a consistent shift trivially detectable, so a timing metric that improves on
    every query is precisely the shape the paired path fires on. The carve-out has to be
    checked before the comparison, not inside it.
    """
    fast = {"search_ms": {f"q{i}": 10.0 for i in range(20)}}
    slow = {"search_ms": {f"q{i}": 90.0 for i in range(20)}}
    base = _paired({"q/r": ("ok", {"search_ms": 90.0}, slow)})
    cur = _paired({"q/r": ("ok", {"search_ms": 10.0}, fast)})

    assert not bc.compare(base, cur, bc.DEFAULT_TOLERANCE).regressions


def test_the_paired_floor_is_what_decides_a_borderline_drop():
    """The floor is the knob; make it explicit that it is doing the deciding, not the tolerance."""
    cur = [s - 0.004 for s in _SPREAD]
    args = (_paired({"q/r": _cell(_SPREAD)}), _paired({"q/r": _cell(cur)}), bc.DEFAULT_TOLERANCE)

    assert not bc.compare(*args).regressions  # below the default floor of 0.005
    assert bc.compare(*args, 0.001).regressions  # a stricter floor resolves it


def test_the_recorder_and_the_comparator_agree_on_the_result_schema():
    """The writer lives in tests/ and the reader in scripts/, each with its own model.

    A field added to one and not the other does not fail loudly: Pydantic ignores what it does
    not know, so the comparator would keep gating with the per-query data silently discarded and
    report green. Compare the field sets rather than trusting them to be edited together.
    """
    assert set(bc.BenchmarkResult.model_fields) == set(recorder.BenchmarkResult.model_fields)


# --- baseline updates fold in, never replace ---------------------------------------------------


def test_updating_from_a_partial_run_keeps_the_cells_it_did_not_measure(tmp_path):
    """A run covering part of the matrix must not delete the rest of the baseline.

    CI measures one corpus and two extractors out of four and eight. If ``--update`` replaced
    the file, adopting such a run would drop every cell it did not cover, and the next
    comparison would pass because there was nothing left to check.
    """
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        _results(
            {"quality/a@nfcorpus": ("ok", {"ndcg@10": 0.60}), "extract/docling@pdf": ("ok", {"chars": 900.0})}
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    partial = tmp_path / "current.json"
    partial.write_text(
        _results({"quality/a@nfcorpus": ("ok", {"ndcg@10": 0.65})}).model_dump_json(indent=2), encoding="utf-8"
    )

    assert bc.main([str(partial), "--baseline", str(baseline), "--update"]) == 0

    after = bc.BenchmarkResults.model_validate_json(baseline.read_text(encoding="utf-8"))
    assert set(after.results) == {"quality/a@nfcorpus", "extract/docling@pdf"}
    assert after.results["quality/a@nfcorpus"].metrics["ndcg@10"] == 0.65  # the measured cell moved
    assert after.results["extract/docling@pdf"].metrics["chars"] == 900.0  # the unmeasured one survived


def test_updating_onto_a_missing_baseline_captures_the_run(tmp_path):
    """First capture: there is nothing to fold into, and that is not an error."""
    baseline = tmp_path / "nested" / "baseline.json"
    current = tmp_path / "current.json"
    current.write_text(_results({"quality/a": ("ok", {"ndcg@10": 0.6})}).model_dump_json(indent=2), encoding="utf-8")

    assert bc.main([str(current), "--baseline", str(baseline), "--update"]) == 0
    assert "quality/a" in bc.BenchmarkResults.model_validate_json(baseline.read_text(encoding="utf-8")).results


def test_the_merge_reports_what_it_did(capsys, tmp_path):
    """Coverage kept from an older run is the number that must not quietly go to zero."""
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        _results({"a": ("ok", {"m": 1.0}), "b": ("ok", {"m": 1.0}), "c": ("ok", {"m": 1.0})}).model_dump_json(),
        encoding="utf-8",
    )
    current = tmp_path / "current.json"
    current.write_text(_results({"a": ("ok", {"m": 2.0}), "d": ("ok", {"m": 2.0})}).model_dump_json(), encoding="utf-8")

    bc.main([str(current), "--baseline", str(baseline), "--update"])

    assert "1 new, 1 updated, 2 kept from earlier runs, 4 total" in capsys.readouterr().out


def test_per_query_data_survives_a_baseline_update(tmp_path):
    """The paired comparison is only possible if the update carries per-query scores through."""
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    current.write_text(_paired({"q/r": _cell(_SPREAD)}).model_dump_json(indent=2), encoding="utf-8")

    bc.main([str(current), "--baseline", str(baseline), "--update"])

    after = bc.BenchmarkResults.model_validate_json(baseline.read_text(encoding="utf-8"))
    assert len(after.results["q/r"].per_query["ndcg@10"]) == len(_SPREAD)


def test_a_merged_baseline_records_which_run_each_cell_came_from(tmp_path):
    """A merged baseline spans several runs while its versions block names only the latest.

    Without a per-cell stamp, a regression in a cell kept from an older run would be attributed
    to the dependency versions recorded at the top of the file, which that cell was never
    measured under.
    """
    baseline = tmp_path / "baseline.json"
    old = _results({"kept": ("ok", {"m": 1.0}), "remeasured": ("ok", {"m": 1.0})})
    baseline.write_text(
        old.model_copy(update={"generated_utc": "2026-01-01T00:00:00+00:00"}).model_dump_json(), encoding="utf-8"
    )
    current = tmp_path / "current.json"
    fresh = _results({"remeasured": ("ok", {"m": 1.0}), "added": ("ok", {"m": 1.0})})
    current.write_text(
        fresh.model_copy(update={"generated_utc": "2026-06-06T00:00:00+00:00"}).model_dump_json(), encoding="utf-8"
    )

    bc.main([str(current), "--baseline", str(baseline), "--update"])

    after = bc.BenchmarkResults.model_validate_json(baseline.read_text(encoding="utf-8"))
    assert after.results["kept"].measured_utc == "2026-01-01T00:00:00+00:00"
    assert after.results["remeasured"].measured_utc == "2026-06-06T00:00:00+00:00"
    assert after.results["added"].measured_utc == "2026-06-06T00:00:00+00:00"


def test_a_cells_vintage_survives_a_later_merge_that_does_not_remeasure_it(tmp_path):
    """Two merges later, a cell must still name the run that actually produced it."""
    baseline = tmp_path / "baseline.json"
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        _results({"a": ("ok", {"m": 1.0})})
        .model_copy(update={"generated_utc": "2026-01-01T00:00:00+00:00"})
        .model_dump_json(),
        encoding="utf-8",
    )
    second.write_text(
        _results({"b": ("ok", {"m": 1.0})})
        .model_copy(update={"generated_utc": "2026-02-02T00:00:00+00:00"})
        .model_dump_json(),
        encoding="utf-8",
    )

    bc.main([str(first), "--baseline", str(baseline), "--update"])
    bc.main([str(second), "--baseline", str(baseline), "--update"])

    after = bc.BenchmarkResults.model_validate_json(baseline.read_text(encoding="utf-8"))
    assert after.results["a"].measured_utc == "2026-01-01T00:00:00+00:00"
    assert after.results["b"].measured_utc == "2026-02-02T00:00:00+00:00"
