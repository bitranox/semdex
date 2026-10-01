"""Integration tests for the Postgres + pgvector store (Docker-backed).

integration: needs Docker and the semdex[pg] extra, so it runs in integration.yml
(ubuntu, Docker + all extras), not the cross-OS default CI. The [pg] deps are now in
the dev extra for type-checking, so the integration marker - not dep absence - is what
keeps this out of the Docker-less default CI. One pgvector container is shared across
the module; each test uses its own collection.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

pytest.importorskip("psycopg")
pytest.importorskip("pgvector")

from semdex.adapters.vectorstore import PgVectorStore
from semdex.domain.errors import CollectionModelMismatchError, VectorStoreError
from semdex.domain.models import Chunk, Collection, SourceRef

pytestmark = [pytest.mark.integration, pytest.mark.os_agnostic]

_PASSWORD = "semdex"


def _conninfo(port: int) -> str:
    return f"host=127.0.0.1 port={port} user=postgres password={_PASSWORD} dbname=postgres"


def _ready(port: int) -> bool:
    import psycopg

    try:
        psycopg.connect(_conninfo(port), connect_timeout=2).close()
    except Exception:
        return False
    return True


@pytest.fixture(scope="module")
def pg_dsn(service_container: Callable[..., int]) -> str:
    port = service_container(
        image="pgvector/pgvector:pg16", container_port=5432, env={"POSTGRES_PASSWORD": _PASSWORD}, ready=_ready
    )
    return _conninfo(port)


def _chunk(text: str, path: str) -> Chunk:
    source = SourceRef(uri=str(path), label="curated", content_hash="h", mtime=1.5)
    return Chunk(text=text, source=source, ordinal=0, token_count=len(text.split()))


def test_full_lifecycle_persists_and_ranks(pg_dsn: str) -> None:
    """Ensure, upsert, cosine-rank, delete and swap all work and persist."""
    store = PgVectorStore(pg_dsn)
    store.ensure_collection(Collection(name="life", model_id="m", dim=3))
    store.upsert(
        collection="life",
        chunks=[_chunk("x-axis", "/x.md"), _chunk("y-axis", "/y.md")],
        vectors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
    )

    # A second connection sees the committed rows (persistence).
    reader = PgVectorStore(pg_dsn)
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


def test_dimension_mismatch_is_rejected(pg_dsn: str) -> None:
    """Querying with a wrong-dimension vector is a model mismatch."""
    store = PgVectorStore(pg_dsn)
    store.ensure_collection(Collection(name="mm", model_id="m", dim=8))
    with pytest.raises(CollectionModelMismatchError):
        store.query(collection="mm", vector=(1.0, 0.0), k=1)


def test_unknown_collection_query_raises(pg_dsn: str) -> None:
    """Querying an unknown collection is a clear VectorStoreError."""
    with pytest.raises(VectorStoreError):
        PgVectorStore(pg_dsn).query(collection="nope", vector=(1.0, 0.0, 0.0), k=1)
