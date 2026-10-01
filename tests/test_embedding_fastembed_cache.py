"""fastembed cache-dir resolution: default to a PERSISTENT location, not /tmp.

fastembed's own default cache is under the ephemeral system temp dir, so a reboot wipes
it and an offline load then silently falls back to the placeholder. The adapter resolves
its own persistent default instead; these tests pin the resolution order without loading
fastembed or touching the network.
"""

from __future__ import annotations

import pytest

from semdex.adapters.embedding.fastembed import resolve_cache_dir

pytestmark = pytest.mark.os_agnostic


def test_default_cache_dir_is_persistent_not_tmp(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no override the cache resolves to a persistent user-cache path, never /tmp."""
    monkeypatch.delenv("FASTEMBED_CACHE_PATH", raising=False)
    resolved = resolve_cache_dir(None)
    assert resolved  # a concrete path, not None (fastembed's None -> /tmp default)
    assert not resolved.startswith("/tmp/")
    assert resolved.endswith("fastembed")


def test_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """FASTEMBED_CACHE_PATH wins over the built-in persistent default."""
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", "/data/persistent/fastembed")
    assert resolve_cache_dir(None) == "/data/persistent/fastembed"


def test_explicit_arg_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit cache_dir argument takes precedence over the environment variable."""
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", "/env/dir")
    assert resolve_cache_dir("/explicit/dir") == "/explicit/dir"
