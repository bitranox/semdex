"""sqlite-vec on a Python whose sqlite3 cannot load extensions.

Some CPython builds (the python.org macOS installer among them) compile the ``sqlite3``
module without loadable-extension support, so ``sqlite3.Connection`` has no
``enable_load_extension`` at all. The store must refuse that with its own
``VectorStoreError`` naming the reason, never let an ``AttributeError`` escape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from semdex.adapters.vectorstore.sqlitevec import load_sqlite_vec
from semdex.domain.errors import VectorStoreError

if TYPE_CHECKING:
    import sqlite3


class _ConnectionWithoutExtensionLoading:
    """Stands in for a ``sqlite3.Connection`` from a build without extension loading.

    typeshed declares ``enable_load_extension`` unconditionally, so a real connection of
    that shape cannot be built on a Python that has the method; this object simply lacks it.
    """


@pytest.mark.os_agnostic
def test_missing_extension_loading_raises_vector_store_error() -> None:
    """A connection without enable_load_extension is refused with a VectorStoreError."""
    connection = cast("sqlite3.Connection", _ConnectionWithoutExtensionLoading())

    with pytest.raises(VectorStoreError, match="extension loading"):
        load_sqlite_vec(connection)
