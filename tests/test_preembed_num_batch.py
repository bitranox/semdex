"""ollama's physical batch for pre-embedding, from SEMDEX_PREEMBED_NUM_BATCH in scripts/preembed_vectors.py.

ollama embeds an input longer than its physical batch (default 2048 tokens) from its first 2048
tokens and still answers HTTP 200, so a long chunk set (cap1024 plus overlap) can be embedded from a
prefix with no error anywhere. The pre-embed driver therefore takes the batch from the environment
and hands it to the ollama provider; unset keeps the server default and the request body unchanged.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preembed_vectors.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("preembed_vectors_num_batch", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def preembed() -> Any:
    return _load()


def test_unset_keeps_the_server_default(preembed: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEMDEX_PREEMBED_NUM_BATCH", raising=False)
    assert preembed._num_batch_from_env() is None


def test_a_positive_integer_is_passed_on(preembed: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEMDEX_PREEMBED_NUM_BATCH", "4096")
    assert preembed._num_batch_from_env() == 4096


@pytest.mark.parametrize("bad", ["0", "-1", "abc", "", "4096.0"])
def test_anything_else_is_refused(preembed: Any, monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setenv("SEMDEX_PREEMBED_NUM_BATCH", bad)
    with pytest.raises(SystemExit, match="SEMDEX_PREEMBED_NUM_BATCH"):
        preembed._num_batch_from_env()
