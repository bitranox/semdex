"""Vector store adapters.

Contents:
    * :mod:`.jsonstore` - JsonVectorStore (persistent orjson-file embedded store)
    * :mod:`.sqlitevec` - SqliteVecStore (embedded sqlite-vec KNN store; needs semdex[sqlite])
"""

from __future__ import annotations

from .jsonstore import JsonVectorStore
from .lance import LanceVectorStore
from .mariadb import MariaDbVectorStore
from .pgvector import PgVectorStore
from .sqlitevec import SqliteVecStore

__all__ = [
    "JsonVectorStore",
    "LanceVectorStore",
    "MariaDbVectorStore",
    "PgVectorStore",
    "SqliteVecStore",
]
