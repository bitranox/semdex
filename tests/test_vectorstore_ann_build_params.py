"""The build-time ANN knobs must reach the index definition, not just the config object.

``AnnParams`` declares six fields and ``resolve_ann_params`` merges all six, but for a long time
only the three query-time ones (``nprobes``, ``refine_factor``, ``ef_search``) were read by an
adapter. ``m``, ``ef_construction`` and ``num_partitions`` could be set, were documented as
"build-time", resolved correctly, and then silently discarded - a dead knob that a tuning sweep
would have reported as "this parameter does not matter".

These assert the DDL the adapter actually emits, not the helper that builds it: a helper nobody
calls is exactly the failure being guarded against. The connection is the true external edge, so
substituting a recorder for it is a seam, not a patched internal.
"""

from __future__ import annotations

from typing import Any

import pytest

from semdex.adapters.vectorstore.mariadb import MariaDbVectorStore
from semdex.adapters.vectorstore.pgvector import PgVectorStore
from semdex.domain.ann_tuning import AnnParams
from semdex.domain.models import Collection

pytestmark = pytest.mark.os_agnostic

_COLLECTION = Collection(name="c", model_id="m", dim=8)


class _Result:
    """A cursor-ish object: the id of the freshly inserted collection."""

    def __init__(self, row: tuple[Any, ...] | None) -> None:
        self._row = row

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._row


class _RecordingConnection:
    """Records SQL and answers the two queries ensure_collection makes."""

    def __init__(self) -> None:
        self.sql: list[str] = []

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> _Result:
        self.sql.append(sql)
        if sql.startswith("SELECT id FROM collections"):
            return _Result(None)  # not present yet, so ensure_collection proceeds
        if "RETURNING id" in sql:
            return _Result((1,))
        return _Result(None)

    # mariadb goes through a cursor and reads the new id off lastrowid rather than RETURNING
    lastrowid = 1

    def cursor(self) -> _RecordingConnection:
        return self

    def fetchone(self) -> tuple[Any, ...] | None:
        last = self.sql[-1] if self.sql else ""
        return None if last.startswith("SELECT id FROM collections") else (1,)


def _pg(ann: AnnParams) -> tuple[PgVectorStore, _RecordingConnection]:
    """A real PgVectorStore with the recorder injected at its one external edge."""
    connection = _RecordingConnection()
    return PgVectorStore("", ann_params=ann, connection=connection), connection


def _maria(ann: AnnParams) -> tuple[MariaDbVectorStore, _RecordingConnection]:
    connection = _RecordingConnection()
    return MariaDbVectorStore("", ann_params=ann, connection=connection), connection


def _ddl(connection: _RecordingConnection, needle: str) -> str:
    """The emitted statement, lowercased.

    Absence assertions are made against this, and SQL keywords are case-free: checking for a
    lowercase parameter name in mixed-case DDL would pass whatever the adapter emitted.
    """
    matches = [sql for sql in connection.sql if needle in sql]
    assert matches, f"no statement containing {needle!r} in {connection.sql}"
    return matches[-1].lower()


# --- pgvector: m / ef_construction on the HNSW index -------------------------------------------


def test_pgvector_index_carries_the_build_parameters() -> None:
    store, connection = _pg(AnnParams(m=48, ef_construction=200))

    store.ensure_collection(_COLLECTION)

    ddl = _ddl(connection, "USING hnsw")
    assert "m = 48" in ddl
    assert "ef_construction = 200" in ddl


def test_pgvector_index_is_left_at_the_driver_default_when_unset() -> None:
    """An empty WITH clause is a syntax error, and pgvector's own defaults are the right default."""
    store, connection = _pg(AnnParams())

    store.ensure_collection(_COLLECTION)

    ddl = _ddl(connection, "USING hnsw")
    assert " with (" not in ddl


def test_pgvector_accepts_one_build_parameter_without_the_other() -> None:
    store, connection = _pg(AnnParams(m=16))

    store.ensure_collection(_COLLECTION)

    ddl = _ddl(connection, "USING hnsw")
    assert "m = 16" in ddl
    assert "ef_construction" not in ddl


def test_pgvector_query_time_knob_is_not_baked_into_the_index() -> None:
    """ef_search is a session setting; putting it in the DDL would freeze it at build time."""
    store, connection = _pg(AnnParams(ef_search=200, m=16))

    store.ensure_collection(_COLLECTION)

    assert "ef_search" not in _ddl(connection, "USING hnsw")


# --- mariadb: M on the vector index ------------------------------------------------------------


def test_mariadb_index_carries_the_neighbour_count() -> None:
    store, connection = _maria(AnnParams(m=32))

    store.ensure_collection(_COLLECTION)

    assert "m=32" in _ddl(connection, "VECTOR INDEX")


def test_mariadb_index_is_left_at_the_driver_default_when_unset() -> None:
    store, connection = _maria(AnnParams())

    store.ensure_collection(_COLLECTION)

    ddl = _ddl(connection, "VECTOR INDEX")
    assert " m=" not in ddl
    assert "distance=cosine" in ddl  # the required setting must survive the change


def test_mariadb_ignores_ef_construction_which_it_has_no_equivalent_for() -> None:
    """Emitting a parameter MariaDB does not know would make the table creation fail outright."""
    store, connection = _maria(AnnParams(m=32, ef_construction=200))

    store.ensure_collection(_COLLECTION)

    assert "ef_construction" not in _ddl(connection, "VECTOR INDEX")


# --- the values are coerced, never interpolated as caller text --------------------------------


@pytest.mark.parametrize(
    ("factory", "needle"),
    [(_pg, "USING hnsw"), (_maria, "VECTOR INDEX")],
)
def test_build_parameters_reach_the_ddl_as_integers(factory: Any, needle: str) -> None:
    """These land in DDL by interpolation, so anything but an int must not survive as text."""
    store, connection = factory(AnnParams(m=True))  # bool is the sneakiest int subclass

    store.ensure_collection(_COLLECTION)

    ddl = _ddl(connection, needle)
    assert "true" not in ddl
    assert "1" in ddl
