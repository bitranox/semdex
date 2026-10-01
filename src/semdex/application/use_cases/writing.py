"""Write use cases: deposit (remember) and delete (forget) client-written knowledge.

A writable KNOWLEDGE dataset has no connector; its content is written directly by
the MCP client, not mirrored from a source. ``remember`` reuses ``index_sources``
(for already-provided text, extraction is the identity), so it inherits the
drop-first idempotency and the ANN compaction. ``forget`` drops a uri's chunks.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from ...domain.models import ExtractedDocument, SourceRef
from .indexing import index_sources

if TYPE_CHECKING:
    from ..ports import ChunkText, EmbeddingProvider, VectorStoreWriter
    from .indexing import IndexReport

_DEFAULT_MAX_TOKENS = 256


def remember(
    *,
    chunk: ChunkText,
    embedding: EmbeddingProvider,
    store: VectorStoreWriter,
    collection: str,
    uri: str,
    text: str,
    label: str = "",
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> IndexReport:
    """Write ``text`` into a knowledge ``collection`` under ``uri``.

    Replaces any prior chunks stored under the same ``uri`` (drop-first), then
    chunks and embeds ``text`` with the dataset's own chunker and model. Returns
    the index report (chunk count).
    """
    ref = SourceRef(uri=uri, label=label, content_hash=_sha256(text), mtime=0.0)

    def _extract(source: SourceRef) -> ExtractedDocument:
        return ExtractedDocument(source=source, text=text)

    return index_sources(
        extract=_extract,
        chunk=chunk,
        embedding=embedding,
        store=store,
        collection=collection,
        sources=[ref],
        max_tokens=max_tokens,
    )


def forget(*, store: VectorStoreWriter, collection: str, uri: str) -> None:
    """Delete every chunk stored under ``uri`` (idempotent) and fold the index."""
    store.delete_by_source(collection=collection, uri=uri)
    store.compact(collection=collection)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = ["forget", "remember"]
