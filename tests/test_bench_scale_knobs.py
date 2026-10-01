"""Persistent-store env knobs for scripts/bench_msmarco_scale.py.

The scale harness normally spins throwaway Docker DBs and throwaway store dirs. These
knobs let a run target the permanent px-semdex-test servers / kept embedded-store dirs /
a named pre-embedded collection instead, so re-evaluation needs no re-ingest. The default
(all knobs unset) must stay the old spin-a-container behaviour so CI/local are unchanged.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from semdex.domain.enums import StoreBackend

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench_msmarco_scale.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("bench_msmarco_scale", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def bench(monkeypatch: pytest.MonkeyPatch) -> Any:
    # Start from a clean slate so the host environment cannot leak into the assertions.
    for var in (
        "SEMDEX_BENCH_PGVECTOR_DSN",
        "SEMDEX_BENCH_MARIADB_DSN",
        "SEMDEX_BENCH_STORE_DIR",
        "SEMDEX_BENCH_COLLECTION",
    ):
        monkeypatch.delenv(var, raising=False)
    return _load()


def test_server_spins_container_by_default(bench: Any) -> None:
    spec = bench._server(StoreBackend.PGVECTOR)
    assert spec is not None
    assert spec.get("external") is not True
    assert spec["image"] == "pgvector/pgvector:pg16"  # would spin a throwaway container


def test_server_targets_persistent_dsn_when_env_set(bench: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    dsn = "host=px-semdex-test port=5432 user=postgres password=x dbname=semdex_kept"
    monkeypatch.setenv("SEMDEX_BENCH_PGVECTOR_DSN", dsn)
    spec = bench._server(StoreBackend.PGVECTOR)
    assert spec == {"external": True, "dsn_fixed": dsn, "datadir": None}  # no container, no teardown


def test_mariadb_dsn_is_independent_of_pgvector(bench: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEMDEX_BENCH_PGVECTOR_DSN", "pg-dsn")
    # only pgvector is redirected; mariadb still spins its container
    assert bench._server(StoreBackend.PGVECTOR).get("external") is True
    assert bench._server(StoreBackend.MARIADB).get("external") is not True


def test_collection_defaults_to_bench_and_honors_override(bench: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    assert bench._bench_collection() == "bench"
    monkeypatch.setenv("SEMDEX_BENCH_COLLECTION", "msmarco_50000__fastembed-bge-small")
    assert _load()._bench_collection() == "msmarco_50000__fastembed-bge-small"


def test_store_dir_is_throwaway_by_default_persistent_when_env_set(
    bench: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    work = tmp_path / "work"
    d, persistent = bench._store_dir_for(StoreBackend.SQLITE_VEC, work)
    assert d == work / "sqlite_vec" and persistent is False

    kept = tmp_path / "kept"
    monkeypatch.setenv("SEMDEX_BENCH_STORE_DIR", str(kept))
    d2, persistent2 = _load()._store_dir_for(StoreBackend.SQLITE_VEC, work)
    assert d2 == kept / "sqlite_vec" and persistent2 is True  # kept, not under the throwaway workdir
