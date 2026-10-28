"""Whether the sqlite-vec store can run on this interpreter, for skipping its tests.

Importable is not enough. Some CPython builds (the python.org macOS installer among them)
compile ``sqlite3`` without loadable-extension support, so ``sqlite_vec`` imports fine
while no connection can ever load it. Every sqlite-vec test asks this one module, so the
two conditions cannot drift apart between files.

The loadability check reads the interpreter directly rather than calling the store's own
loader: a regression in that loader must fail the tests, not skip them.
"""

from __future__ import annotations

import importlib.util
import sqlite3

import pytest

__all__ = ["SQLITE_VEC_UNAVAILABLE_REASON", "requires_sqlite_vec", "skip_unless_sqlite_vec"]


def _unavailable_reason() -> str | None:
    if importlib.util.find_spec("sqlite_vec") is None:
        return "needs semdex[sqlite]"
    probe = sqlite3.connect(":memory:")
    try:
        if not hasattr(probe, "enable_load_extension"):
            return "this Python's sqlite3 was built without SQLite extension loading"
    finally:
        probe.close()
    return None


SQLITE_VEC_UNAVAILABLE_REASON = _unavailable_reason()

requires_sqlite_vec = pytest.mark.skipif(
    SQLITE_VEC_UNAVAILABLE_REASON is not None,
    reason=SQLITE_VEC_UNAVAILABLE_REASON or "sqlite-vec is usable",
)


def skip_unless_sqlite_vec() -> None:
    """Skip the calling test when sqlite-vec is not importable or not loadable here."""
    if SQLITE_VEC_UNAVAILABLE_REASON is not None:
        pytest.skip(SQLITE_VEC_UNAVAILABLE_REASON)
