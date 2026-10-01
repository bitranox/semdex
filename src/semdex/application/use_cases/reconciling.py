"""Reconcile use case: sync a collection to a source connector's current listing.

Makes indexing a true SYNC rather than an append-only pass: it prunes chunks
for sources that vanished, (re)indexes new and changed sources, and skips
unchanged ones (matched by ``content_hash``) so nothing is re-embedded
needlessly. Connector-agnostic - it only sees the opaque ``(uri, content_hash)``
facets, so the same logic serves the filesystem connector now and an email
connector later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .indexing import index_sources

if TYPE_CHECKING:
    from ..ports import ChunkText, EmbeddingProvider, Extract, SourceConnector, SummaryProvider, VectorStore

_DEFAULT_MAX_TOKENS = 256


@dataclass(frozen=True, slots=True)
class ReconcileReport:
    """Outcome of a reconcile run: sources (re)indexed, pruned, and left unchanged."""

    indexed: int
    pruned: int
    unchanged: int


def reconcile(
    *,
    connector: SourceConnector,
    extract: Extract,
    chunk: ChunkText,
    embedding: EmbeddingProvider,
    store: VectorStore,
    collection: str,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    summarize: SummaryProvider | None = None,
) -> ReconcileReport:
    """Sync ``collection`` to the connector's current listing.

    Diffs the connector's ``(uri, content_hash)`` against what the store has
    indexed: a uri gone from the listing is pruned (this also removes the old
    side of a move); a uri whose hash changed is re-indexed; a uri whose hash is
    unchanged is skipped (not re-embedded). A brand-new uri is indexed.
    """
    current = list(connector.sources())
    current_by_uri = {source.uri: source for source in current}
    existing = {coll.name for coll in store.collections()}
    indexed = store.source_hashes(collection=collection) if collection in existing else {}

    vanished = indexed.keys() - current_by_uri.keys()
    for uri in vanished:
        store.delete_by_source(collection=collection, uri=uri)

    to_index = [source for source in current if indexed.get(source.uri) != source.content_hash]
    index_sources(
        extract=extract,
        chunk=chunk,
        embedding=embedding,
        store=store,
        collection=collection,
        sources=to_index,
        max_tokens=max_tokens,
        summarize=summarize,
    )
    return ReconcileReport(indexed=len(to_index), pruned=len(vanished), unchanged=len(current) - len(to_index))


__all__ = [
    "ReconcileReport",
    "reconcile",
]
