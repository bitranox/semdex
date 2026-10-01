"""The scorer must not merge a run into a results file measured on a different corpus.

``score_chunk_sweep.py`` reads its output file first and merges new cells into it, and its default
output is ``chunk_sweep_scores.json`` - the file the MIRACL export publishes. Score any other
corpus without setting ``SEMDEX_SCORE_OUT`` and the cells land there silently, to be published
under a note that calls the body VOID for chunk-parameter claims.

The export refuses such a file, but that is the second line: by then the results file is already
polluted and has to be restored from a backup nobody was required to take. This is the first line -
refuse before scoring, while the file is still clean.

There is no override. The one file that looked like a legitimate merge, the BEIR results, turned out to
be a note-versus-contents mismatch instead of a precedent, so a real merge is a deliberate code change.
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
    spec = importlib.util.spec_from_file_location("score_chunk_sweep", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["score_chunk_sweep"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scorer() -> Any:
    return _load()


def _results(*corpora: str) -> dict[str, Any]:
    return {f"{c}__recursive-t256-o0-gpt2__ollama-bge-m3": {"corpus": c, "ndcg@10": 0.5} for c in corpora}


def test_scoring_a_new_corpus_into_an_existing_results_file_is_refused(scorer: Any) -> None:
    """The exact near-miss: MLDR scored with the default output, which is the MIRACL file."""
    existing = _results("miracl_en_100k_slice", "miracl_de_100k_slice")
    with pytest.raises(ValueError) as excinfo:
        scorer._reject_corpus_mixing(
            Path("/embeddings/scores/chunk_sweep_scores.json"),
            existing,
            ["mldr_en_8k_slice", "mldr_de_3k_slice"],
        )
    message = str(excinfo.value)
    assert "mldr_en_8k_slice" in message, message
    assert "SEMDEX_SCORE_OUT" in message, message


def test_rescoring_a_corpus_the_file_already_holds_is_allowed(scorer: Any) -> None:
    """Re-running MIRACL into the MIRACL file is the normal, idempotent case."""
    existing = _results("miracl_en_100k_slice", "miracl_de_100k_slice")
    scorer._reject_corpus_mixing(Path("x.json"), existing, ["miracl_en_100k_slice"])


def test_a_fresh_results_file_accepts_any_corpus(scorer: Any) -> None:
    """A new file has nothing to disagree with, so the first run must not be blocked."""
    scorer._reject_corpus_mixing(Path("new.json"), {}, ["gerdalir_de_12k_slice"])


def test_no_environment_variable_can_switch_the_guard_off(scorer: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """There is deliberately no override.

    The one workflow that looked like a legitimate merge was the BEIR results file, and that
    turned out to be the note-versus-contents mismatch this guard exists to catch rather than a
    precedent for it. A flag whose only purpose is to disable a safety check is the thing someone
    sets to get past a block and then copies into a script, so the check has no off switch: a real
    merge is a code change, made deliberately, with the export's note updated alongside it.
    """
    monkeypatch.setenv("SEMDEX_SCORE_ALLOW_MIXED_CORPORA", "1")
    with pytest.raises(ValueError):
        scorer._reject_corpus_mixing(Path("beir.json"), _results("nfcorpus"), ["scifact"])
