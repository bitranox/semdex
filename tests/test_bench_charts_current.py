"""Guard: the committed benchmark charts must not drift from the raw data.

scripts/gen_bench_charts.py derives each chart's plotted values from the raw JSON and writes
their hashes to docs/benchmarks/img/charts.manifest.json. This test re-derives the values (matplotlib-free -
it imports only the pure ``collect_chart_data``) and fails if the committed manifest is stale, i.e.
the data changed but the charts were not regenerated. Fix: ``python scripts/gen_bench_charts.py``
then commit docs/benchmarks/img/*.png + charts.manifest.json.
"""

# pyright: basic
# Loads the generator by path (it lives in scripts/, not the package) and reads its module globals.

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_GEN = _ROOT / "scripts" / "gen_bench_charts.py"
_MANIFEST = _ROOT / "docs" / "benchmarks" / "img" / "charts.manifest.json"
_RAW = _ROOT / "tests" / "benchmarks" / "raw"


def _load_generator() -> Any:  # a dynamically loaded module - its attrs are not statically known
    spec = importlib.util.spec_from_file_location("gen_bench_charts", _GEN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_by_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.os_agnostic
def test_committed_charts_match_current_raw_data() -> None:
    """The manifest hashes equal the hashes freshly derived from the raw JSON."""
    gen = _load_generator()
    fresh = gen._manifest(gen.collect_chart_data())  # both pure, no matplotlib
    committed = json.loads(_MANIFEST.read_text())
    assert committed == fresh, (
        "benchmark charts are STALE vs the raw data - run `python scripts/gen_bench_charts.py` "
        "and commit docs/benchmarks/img/*.png + docs/benchmarks/img/charts.manifest.json"
    )


@pytest.mark.os_agnostic
def test_two_corpora_sharing_a_language_get_distinct_chart_labels() -> None:
    """The forest plot is the evidence for a CORPUS split, so its rows must name the corpus.

    Labelling by language alone printed GerDaLIR German and MLDR German as the same "de", and
    those two are exactly the pair whose overlap results point opposite ways.
    """
    gen = _load_generator()
    labels = {gen._corpus_label(c) for c in ("gerdalir_de_12k_slice", "mldr_de_3k_slice", "mldr_en_8k_slice")}
    assert len(labels) == 3, f"corpus labels collapse onto each other: {sorted(labels)}"


@pytest.mark.os_agnostic
def test_the_overlap_forest_plot_names_both_german_corpora() -> None:
    """End to end through the real data, not just the label helper."""
    gen = _load_generator()
    rows = gen.collect_chart_data()["chunk_knob_effects"]["rows"].get("overlap_tokens", [])
    corpora = {row["label"].split()[0] for row in rows}
    assert {"gerdalir-de", "mldr-de", "mldr-en"} <= corpora, f"overlap chart labels: {sorted(corpora)}"


@pytest.mark.os_agnostic
def test_the_breakpoint_forest_plot_names_the_default_model_not_none() -> None:
    """The plain semantic profiles carry a null breakpoint level; the chart must call it the default."""
    gen = _load_generator()
    rows = gen.collect_chart_data()["chunk_knob_effects"]["rows"].get("breakpoint_model", [])
    changes = {row["change"] for row in rows}
    assert not {c for c in changes if "None" in c}, f"a null level leaked into the chart: {sorted(changes)}"
    assert any(c.endswith(" to default") for c in changes), f"no default-level comparison charted: {sorted(changes)}"


@pytest.mark.os_agnostic
def test_the_markdown_structure_chart_labels_strategy_and_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One row per pair, named by strategy and embedder, ordered by delta."""
    gen = _load_generator()
    raw = tmp_path / "raw"
    # Seed the fixture dir from the real raw tree so every collector has all its input files;
    # several collectors have no missing-file guard and crash on a fixture-only dir.
    shutil.copytree(gen._RAW, raw)
    raw.joinpath("markdown-structure-effect.json").write_text(
        json.dumps(
            {
                "effects": [
                    {
                        "strategy": "fast",
                        "embedding": "fastembed:bge-base",
                        "mean_delta": 0.001,
                        "ci_lo": -0.002,
                        "ci_hi": 0.004,
                        "resolved": False,
                    },
                    {
                        "strategy": "markdown",
                        "embedding": "fastembed:bge-base",
                        "mean_delta": 0.012,
                        "ci_lo": 0.005,
                        "ci_hi": 0.019,
                        "resolved": True,
                    },
                ]
            }
        )
    )
    monkeypatch.setattr(gen, "_RAW", raw)
    d = gen._markdown_structure()
    assert [r["label"] for r in d["rows"]] == ["fast bge-base", "markdown bge-base"]
    assert d["rows"][1]["resolved"] is True
    collected = gen.collect_chart_data()["markdown_structure"]
    expected_labels = ["fast bge-base", "markdown bge-base"]
    actual_labels = [r["label"] for r in collected["rows"]]
    assert actual_labels == expected_labels, "the collector is wired into the manifest"


@pytest.mark.os_agnostic
def test_the_product_k_budget_chart_has_one_series_per_embedder_and_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delivered nDCG@k against cap, one line per embedder, one panel per budget."""
    gen = _load_generator()
    raw = tmp_path / "raw"
    # Seed the fixture dir from the real raw tree so every collector has all its input files;
    # several collectors have no missing-file guard and crash on a fixture-only dir.
    shutil.copytree(gen._RAW, raw)

    def row(cap: int, k: int, budget: int, delivered: float) -> dict[str, Any]:
        return {
            "corpus": "mldr_en_8k_slice",
            "profile": f"recursive-t{cap}-o0-gpt2",
            "embedding": "fastembed:bge-base",
            "axes": {"strategy": "recursive", "max_tokens": cap, "overlap_tokens": 0, "breakpoint_model": None},
            "k": k,
            "budgets": [budget],
            "ndcg_delivered": delivered,
        }

    raw.joinpath("product-k.json").write_text(
        json.dumps(
            {
                "product_k": 5,
                "budgets": [1280, 2560],
                "rows": [
                    row(64, 20, 1280, 0.41),
                    row(64, 40, 2560, 0.44),
                    row(256, 5, 1280, 0.50),
                    row(256, 10, 2560, 0.53),
                ],
            }
        )
    )
    monkeypatch.setattr(gen, "_RAW", raw)
    d = gen._product_k_budget()
    assert d["caps"] == [64, 128, 256, 512] and d["budgets"] == [1280, 2560]
    assert d["series"]["1280"]["fastembed:bge-base"] == [0.41, None, 0.50, None]
    assert d["series"]["2560"]["fastembed:bge-base"] == [0.44, None, 0.53, None]
    collected = gen.collect_chart_data()["product_k_budget"]
    assert collected["series"]["1280"]["fastembed:bge-base"][0] == 0.41, "the collector is wired into the manifest"


@pytest.mark.os_agnostic
def test_the_product_k_budget_chart_draws_all_embedders_beyond_palette_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A seventh embedder is drawn, not dropped when palette has six colors."""
    # This one renders a real figure, so it needs the charts extra; CI installs only .[dev].
    pytest.importorskip("matplotlib")
    gen = _load_generator()

    # Build data with 7 embedders (exceeds 6-color palette)
    d = {
        "caps": [64, 128, 256, 512],
        "budgets": [1280],
        "series": {"1280": {f"ollama:model-{n}": [0.4 + n / 100, None, 0.5 + n / 100, None] for n in range(7)}},
    }

    # Capture rendered figures instead of writing them
    captured: list[Any] = []
    monkeypatch.setattr(gen, "_save", lambda fig, name: captured.append(fig))

    gen.render_product_k_budget(d)

    # Verify one figure was rendered
    assert len(captured) == 1
    fig = captured[0]

    # Get the axes and check line count
    ax = fig.axes[0]
    assert len(ax.get_lines()) == 7, f"expected 7 lines, got {len(ax.get_lines())}"

    # Verify all embedder names appear in the legend
    legend = ax.get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert labels == [f"model-{n}" for n in range(7)], f"expected all 7 embedders in legend, got {labels}"


@pytest.mark.os_agnostic
def test_the_fit_set_agrees_across_tables_charts_and_explorer_on_real_data() -> None:
    """One fitness filter, called from three generators, must answer the same question.

    scripts/gen_bench_tables.py, scripts/gen_bench_charts.py and scripts/gen_bench_explorer.py each
    need the fit set; only gen_bench_tables.py defines it. This proves the other two delegate to
    it rather than restating the profile and threshold literals, which is how a rendering (or any
    future exclusion) could enter one generator's aggregates and not another's.
    """
    tables = _load_by_path("gen_bench_tables", _ROOT / "scripts" / "gen_bench_tables.py")
    charts = _load_generator()
    explorer = _load_by_path("gen_bench_explorer", _ROOT / "scripts" / "gen_bench_explorer.py")

    audit = tables._load("chunk-dimension-audit.json")
    from_tables = tables._fit_corpora(audit)
    from_charts = charts._fit_corpora()
    from_explorer = set(explorer._fit_corpora(audit))

    assert from_tables == from_charts == from_explorer
