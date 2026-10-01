"""An export must not be able to publish a corpus it does not claim to cover.

Every ``chunk-sweep-*.json`` carries a prose ``note`` that tells the reader which body the numbers
were measured on, and whether that body can carry a chunking claim at all - the MIRACL export says
in so many words that it is VOID for chunk-parameter claims. That note is written per export, but
the rows come from whatever the source file happens to hold, and nothing checked that the two
agree.

They can disagree by accident, because ``score_chunk_sweep.py`` defaults ``SEMDEX_SCORE_OUT`` to
``chunk_sweep_scores.json`` - the MIRACL file. Score any other corpus without setting that variable
and its cells land there, and the next export publishes them under the MIRACL note. That already
nearly happened with 44 MLDR cells.

So the guard belongs in the export, not in the operator's memory: a spec declares the corpora it
may contain, and a foreign row is a hard failure naming the cells.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "export_bench_raw.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("export_bench_raw", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so decorators that look the module up by __module__ resolve.
    sys.modules["export_bench_raw"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exporter() -> Any:
    return _load()


def _write_source(cache_root: Path, name: str, cells: dict[str, dict[str, Any]]) -> None:
    scores = cache_root / "scores"
    scores.mkdir(parents=True, exist_ok=True)
    (scores / name).write_text(json.dumps(cells))


def _cell(corpus: str) -> dict[str, Any]:
    return {
        "corpus": corpus,
        "profile": "recursive-t256-o0-gpt2",
        "embedding": "ollama:bge-m3",
        "dim": 1024,
        "n_queries": 100,
        "ndcg@10": 0.5,
    }


def test_every_export_spec_declares_the_corpora_it_may_contain(exporter: Any) -> None:
    """A new export must not be able to opt out of the guard by forgetting the key."""
    missing = [name for name, spec in exporter._EXPORTS.items() if not spec.get("corpora")]
    assert missing == [], f"exports without a declared corpora allowlist: {missing}"


def test_build_export_refuses_a_row_from_an_undeclared_corpus(
    exporter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The near-miss itself: an MLDR cell sitting in the MIRACL source file."""
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    _write_source(
        tmp_path,
        "only_source.json",
        {
            "miracl_en_100k_slice__recursive-t256-o0-gpt2__ollama-bge-m3": _cell("miracl_en_100k_slice"),
            "mldr_de_3k_slice__recursive-t256-o0-gpt2__ollama-bge-m3": _cell("mldr_de_3k_slice"),
        },
    )
    spec = {
        "sources": ["only_source.json"],
        "note": "n",
        "fit_for_chunk_claims": False,
        "corpora": ["miracl_en_100k_slice", "miracl_de_100k_slice"],
    }
    with pytest.raises(ValueError) as excinfo:
        exporter.build_export("chunk-sweep-miracl.json", spec)
    message = str(excinfo.value)
    assert "mldr_de_3k_slice" in message, message
    assert "chunk-sweep-miracl.json" in message, message


def test_build_export_accepts_rows_from_every_declared_corpus(
    exporter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard must not reject the legitimate multi-corpus case it has to allow."""
    monkeypatch.setenv("CACHE_ROOT", str(tmp_path))
    _write_source(
        tmp_path,
        "only_source.json",
        {
            "mldr_en_8k_slice__recursive-t256-o0-gpt2__ollama-bge-m3": _cell("mldr_en_8k_slice"),
            "mldr_de_3k_slice__recursive-t256-o0-gpt2__ollama-bge-m3": _cell("mldr_de_3k_slice"),
        },
    )
    spec = {
        "sources": ["only_source.json"],
        "note": "n",
        "fit_for_chunk_claims": True,
        "corpora": ["mldr_en_8k_slice", "mldr_de_3k_slice"],
    }
    payload = exporter.build_export("chunk-sweep-mldr.json", spec)
    assert payload is not None
    assert payload["summary"]["corpora"] == ["mldr_de_3k_slice", "mldr_en_8k_slice"]


def test_gerdalir_has_its_own_export_and_source_file(exporter: Any) -> None:
    """The German legal corpus the CPU chain is grinding needs somewhere to be published."""
    spec = exporter._EXPORTS.get("chunk-sweep-gerdalir.json")
    assert spec is not None, "no export spec for GerDaLIR"
    assert spec["corpora"] == ["gerdalir_de_12k_slice"]
    assert "gerdalir_chunk_scores.json" in spec["sources"]
    assert spec["fit_for_chunk_claims"] is True


def test_the_beir_export_s_note_names_every_corpus_it_publishes(exporter: Any) -> None:
    """The allowlist and the note have to move together, and this is what holds them together.

    The allowlist once said nfcorpus and scifact while the sources held four corpora, and the
    export was rejected on every run until that was settled. It was settled by widening the list,
    which is the dangerous direction: widening it to match whatever the data happens to contain,
    leaving the note behind, is how this guard gets defeated by the person it protects. So the
    test is no longer a literal list. It requires the note to NAME each corpus the export
    publishes, which a silent widening cannot satisfy.
    """
    spec = exporter._EXPORTS["chunk-sweep-beir.json"]
    assert sorted(spec["corpora"]) == ["cqadupstack", "fiqa", "nfcorpus", "scifact"]
    unnamed = [corpus for corpus in spec["corpora"] if corpus not in spec["note"]]
    assert not unnamed, f"published but absent from the note a reader gets: {unnamed}"
    assert spec["fit_for_chunk_claims"] is False


def test_the_miracl_export_still_declares_only_miracl(exporter: Any) -> None:
    """Regression anchor for the file the near-miss targeted."""
    spec = exporter._EXPORTS["chunk-sweep-miracl.json"]
    assert sorted(spec["corpora"]) == ["miracl_de_100k_slice", "miracl_en_100k_slice"]


def test_one_failing_export_does_not_block_the_others(
    exporter: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected export must still let every unrelated export be written, and still exit non-zero.

    The guard is only useful if it can be left switched on. If one bad export aborts the whole run,
    the pressure at the moment it fires is to widen its allowlist just to get the OTHER files out -
    which is how a guard gets defeated by the person it is protecting. So the run reports the
    failure, writes what it legitimately can, and exits non-zero so nothing downstream reads a
    partial set as a success.
    """
    cache = tmp_path / "cache"
    out = tmp_path / "out"
    monkeypatch.setenv("CACHE_ROOT", str(cache))
    monkeypatch.setenv("OUT_DIR", str(out))
    _write_source(
        cache,
        "mldr_chunk_scores.json",
        {
            "mldr_en_8k_slice__recursive-t256-o0-gpt2__ollama-bge-m3": _cell("mldr_en_8k_slice"),
        },
    )
    # An MLDR cell in the BEIR source file: foreign to that export whatever the BEIR allowlist
    # grows to hold, so this stays a rejection and does not quietly go green the next time a
    # corpus is legitimately added.
    _write_source(
        cache,
        "beir_chunk_scores.json",
        {
            "mldr_de_3k_slice__recursive-t256-o0-gpt2__ollama-bge-m3": _cell("mldr_de_3k_slice"),
        },
    )

    with pytest.raises(SystemExit) as excinfo:
        exporter.main()

    assert excinfo.value.code != 0
    assert (out / "chunk-sweep-mldr.json").exists(), "a clean export was blocked by an unrelated failure"
    assert not (out / "chunk-sweep-beir.json").exists(), "the rejected export must not be written"


def test_the_mldr_export_publishes_the_marked_twin_from_its_own_source_file(exporter: Any) -> None:
    spec = exporter._EXPORTS["chunk-sweep-mldr.json"]
    assert "mldr_md_chunk_scores.json" in spec["sources"]
    assert "mldr_en_8k_md_slice" in spec["corpora"]
    assert "heading" in spec["note"], "the note must say the third corpus is the marked twin, not a new body"
