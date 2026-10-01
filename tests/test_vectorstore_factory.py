"""Composition vector-store factory: selects an adapter by StoreBackend."""

from __future__ import annotations

from pathlib import Path

import pytest

from semdex.adapters.vectorstore import JsonVectorStore
from semdex.composition import build_vector_store
from semdex.domain.enums import StoreBackend
from semdex.domain.errors import VectorStoreError


@pytest.mark.os_agnostic
def test_json_backend_returns_json_store(tmp_path: Path) -> None:
    """The JSON backend yields a persistent JsonVectorStore."""
    store = build_vector_store(StoreBackend.JSON, tmp_path)
    assert isinstance(store, JsonVectorStore)


@pytest.mark.os_agnostic
def test_sqlite_backend_returns_sqlite_store(tmp_path: Path) -> None:
    """The sqlite_vec backend yields a SqliteVecStore (needs semdex[sqlite])."""
    pytest.importorskip("sqlite_vec")
    from semdex.adapters.vectorstore import SqliteVecStore

    store = build_vector_store(StoreBackend.SQLITE_VEC, tmp_path)
    assert isinstance(store, SqliteVecStore)


@pytest.mark.os_agnostic
def test_lance_backend_returns_lance_store(tmp_path: Path) -> None:
    """The lancedb backend yields a LanceVectorStore (needs semdex[lance])."""
    pytest.importorskip("lancedb")
    from semdex.adapters.vectorstore import LanceVectorStore

    store = build_vector_store(StoreBackend.LANCEDB, tmp_path)
    assert isinstance(store, LanceVectorStore)


@pytest.mark.os_agnostic
@pytest.mark.parametrize("backend", [StoreBackend.PGVECTOR, StoreBackend.MARIADB])
def test_server_backend_without_dsn_fails_clearly(tmp_path: Path, backend: StoreBackend) -> None:
    """A server backend selected without a [vector_store].dsn raises a clear error."""
    with pytest.raises(VectorStoreError, match=backend.value):
        build_vector_store(backend, tmp_path)
