"""Per-request HTTP timeout for the breakpoint-model embedder in scripts/_bench_breakpoint_embeddings.py.

chonkie's SemanticChunker embeds a WHOLE document's sentences in ONE ``embed_batch`` call, so the
request's wall time scales with the longest document in the corpus. The loaders' own 60 s default
is far too small for that and silently cost hours of GPU work per chunk set, so the breakpoint
builder resolves its own, much larger budget - configurable rather than hardcoded.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "_bench_breakpoint_embeddings.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("_bench_breakpoint_embeddings", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def bench(monkeypatch: pytest.MonkeyPatch) -> Any:
    # A stray host value must not leak into the assertions.
    monkeypatch.delenv("SEMDEX_BREAKPOINT_TIMEOUT", raising=False)
    return _load()


def test_default_timeout_is_far_above_the_loader_default(bench: Any) -> None:
    """Unset: the generous default applies, not the loaders' 60 s."""
    assert bench._resolve_timeout(None) == bench._DEFAULT_TIMEOUT
    assert bench._DEFAULT_TIMEOUT > 60.0


def test_env_overrides_the_default(bench: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEMDEX_BREAKPOINT_TIMEOUT", "1800")
    assert bench._resolve_timeout(None) == 1800.0


def test_explicit_argument_wins_over_the_env(bench: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEMDEX_BREAKPOINT_TIMEOUT", "1800")
    assert bench._resolve_timeout(120.0) == 120.0


def test_blank_env_falls_back_to_the_default(bench: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """An exported-but-empty variable is 'unset', not an error."""
    monkeypatch.setenv("SEMDEX_BREAKPOINT_TIMEOUT", "   ")
    assert bench._resolve_timeout(None) == bench._DEFAULT_TIMEOUT


@pytest.mark.parametrize("bad", ["abc", "0", "-30"])
def test_unusable_env_value_fails_loudly(bench: Any, monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    """A typo must not silently degrade to a 60 s request budget mid-sweep."""
    monkeypatch.setenv("SEMDEX_BREAKPOINT_TIMEOUT", bad)
    with pytest.raises(SystemExit):
        bench._resolve_timeout(None)


def test_builder_passes_the_resolved_timeout_to_build_embedding(bench: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The resolved budget must actually reach the provider, which is where it broke before."""
    seen: dict[str, Any] = {}

    class _Provider:
        dim = 3

    def _fake_build_embedding(backend: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        seen["backend"] = backend
        return _Provider()

    monkeypatch.setattr(bench, "build_embedding", _fake_build_embedding)
    monkeypatch.setenv("SEMDEX_BREAKPOINT_TIMEOUT", "1234")
    bench.build_breakpoint_embeddings("jina-v3", endpoint="http://shim.invalid/v1")

    assert seen["timeout"] == 1234.0
    assert seen["endpoint"] == "http://shim.invalid/v1"


def test_default_num_batch_covers_the_served_context(bench: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """ollama's own default (2048) is below the 4096 context it serves, so we must raise it."""
    monkeypatch.delenv("SEMDEX_BREAKPOINT_NUM_BATCH", raising=False)
    assert bench._resolve_num_batch(None) == bench._DEFAULT_NUM_BATCH
    assert bench._DEFAULT_NUM_BATCH >= 4096


def test_num_batch_env_and_argument_override(bench: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEMDEX_BREAKPOINT_NUM_BATCH", "8192")
    assert bench._resolve_num_batch(None) == 8192
    assert bench._resolve_num_batch(2048) == 2048


@pytest.mark.parametrize("bad", ["abc", "0", "-1", "4096.5"])
def test_unusable_num_batch_fails_loudly(bench: Any, monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    """A typo must not silently restore ollama's truncating 2048 default."""
    monkeypatch.setenv("SEMDEX_BREAKPOINT_NUM_BATCH", bad)
    with pytest.raises(SystemExit):
        bench._resolve_num_batch(None)


def test_builder_passes_num_batch_to_build_embedding(bench: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The resolved batch must actually reach the provider - the link that was missing before."""
    seen: dict[str, Any] = {}

    class _Provider:
        dim = 3

    def _fake_build_embedding(backend: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return _Provider()

    monkeypatch.setattr(bench, "build_embedding", _fake_build_embedding)
    monkeypatch.setenv("SEMDEX_BREAKPOINT_NUM_BATCH", "4096")
    bench.build_breakpoint_embeddings("bge-m3", endpoint="http://ollama.invalid")

    assert seen["num_batch"] == 4096
