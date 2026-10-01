"""Integration tests for the MariaDB native VECTOR store (Docker-backed).

integration: needs Docker and the semdex[mariadb] extra, so it runs in integration.yml
(ubuntu, Docker + all extras), not the cross-OS default CI. The [mariadb] deps are now in
the dev extra for type-checking, so the integration marker - not dep absence - is what keeps
this out of the Docker-less default CI. One MariaDB container is shared across the module;
each test uses its own collection.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

pytest.importorskip("pymysql")

from semdex.adapters.vectorstore import MariaDbVectorStore
from semdex.domain.errors import CollectionModelMismatchError, VectorStoreError
from semdex.domain.models import Chunk, Collection, SourceRef

pytestmark = [pytest.mark.integration, pytest.mark.os_agnostic]

_PASSWORD = "semdex"
_DATABASE = "semdex"


def _ready(port: int) -> bool:
    import pymysql

    try:
        pymysql.connect(
            host="127.0.0.1", port=port, user="root", password=_PASSWORD, database=_DATABASE, connect_timeout=2
        ).close()
    except Exception:
        return False
    return True


@pytest.fixture(scope="module")
def maria_dsn(service_container: Callable[..., int]) -> str:
    port = service_container(
        image="mariadb:11.8",
        container_port=3306,
        env={"MARIADB_ROOT_PASSWORD": _PASSWORD, "MARIADB_DATABASE": _DATABASE},
        ready=_ready,
    )
    return f"mysql://root:{_PASSWORD}@127.0.0.1:{port}/{_DATABASE}"


def _chunk(text: str, path: str) -> Chunk:
    source = SourceRef(uri=str(path), label="curated", content_hash="h", mtime=1.5)
    return Chunk(text=text, source=source, ordinal=0, token_count=len(text.split()))


def test_full_lifecycle_persists_and_ranks(maria_dsn: str) -> None:
    """Ensure, upsert, cosine-rank, delete and swap all work and persist."""
    store = MariaDbVectorStore(maria_dsn)
    store.ensure_collection(Collection(name="life", model_id="m", dim=3))
    store.upsert(
        collection="life",
        chunks=[_chunk("x-axis", "/x.md"), _chunk("y-axis", "/y.md")],
        vectors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
    )

    reader = MariaDbVectorStore(maria_dsn)
    assert reader.count(collection="life") == 2
    assert Collection(name="life", model_id="m", dim=3) in reader.collections()
    hits = reader.query(collection="life", vector=(0.9, 0.1, 0.0), k=2)
    assert [h.uri for h in hits] == ["/x.md", "/y.md"]
    assert hits[0].score > hits[1].score

    store.delete_by_source(collection="life", uri="/x.md")
    assert reader.count(collection="life") == 1

    store.ensure_collection(Collection(name="life2__staging", model_id="m", dim=3))
    store.upsert(collection="life2__staging", chunks=[_chunk("z", "/z.md")], vectors=[(0.0, 0.0, 1.0)])
    store.swap(staging="life2__staging", target="life2")
    assert reader.count(collection="life2") == 1


def test_cosine_query_uses_the_vector_index(maria_dsn: str) -> None:
    """The MHNSW index must be built for cosine, else the cosine query full-scans (21 s/q at 250K).

    Regression guard for the euclidean-default bug: EXPLAIN of the store's own query form must
    show the vector index (``type=index``), not a full table scan (``type=ALL``).
    """
    from urllib.parse import urlparse

    import pymysql

    store = MariaDbVectorStore(maria_dsn)
    store.ensure_collection(Collection(name="idx", model_id="m", dim=3))
    store.upsert(
        collection="idx",
        chunks=[_chunk(f"c{i}", f"/c{i}.md") for i in range(20)],
        vectors=[(float(i), 1.0, 0.0) for i in range(20)],
    )

    url = urlparse(maria_dsn)
    assert url.hostname is not None and url.port is not None  # dsn is fully specified by the fixture
    assert url.username is not None and url.password is not None
    conn = pymysql.connect(
        host=url.hostname,
        port=url.port,
        user=url.username,
        password=url.password,
        database=url.path.lstrip("/"),
        autocommit=True,
    )
    cur = conn.cursor()
    cur.execute("SELECT id FROM collections WHERE name = 'idx'")
    coll_row = cur.fetchone()
    assert coll_row is not None
    coll_id = int(coll_row[0])
    # coll_id is an int() we just read back from our own table - no injection surface.
    cur.execute(
        f"EXPLAIN SELECT chunk_text, VEC_DISTANCE_COSINE(embedding, VEC_FromText('[0.9,0.1,0.0]')) AS dist "  # noqa: S608
        f"FROM chunks_{coll_id} ORDER BY dist LIMIT 5"
    )
    plan = " ".join(str(c) for row in cur.fetchall() for c in row)
    assert "index" in plan.lower() and "ALL" not in plan.split(), f"cosine query not using the vector index: {plan}"


def test_dimension_mismatch_is_rejected(maria_dsn: str) -> None:
    """Querying with a wrong-dimension vector is a model mismatch."""
    store = MariaDbVectorStore(maria_dsn)
    store.ensure_collection(Collection(name="mm", model_id="m", dim=8))
    with pytest.raises(CollectionModelMismatchError):
        store.query(collection="mm", vector=(1.0, 0.0), k=1)


def test_unknown_collection_query_raises(maria_dsn: str) -> None:
    """Querying an unknown collection is a clear VectorStoreError."""
    with pytest.raises(VectorStoreError):
        MariaDbVectorStore(maria_dsn).query(collection="nope", vector=(1.0, 0.0, 0.0), k=1)
