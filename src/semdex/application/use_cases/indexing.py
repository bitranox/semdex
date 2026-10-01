"""Index use case: extract -> chunk -> embed -> upsert for a set of sources.

Pure orchestration over the ports; no framework or I/O types. Depends only on
the narrow slices it needs (an extractor, a chunker, an embedding provider, and
the vector store WRITER slice).
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...domain.models import Collection

if TYPE_CHECKING:
    from ...domain.models import CompactionPolicy, SourceRef
    from ..ports import ChunkText, EmbeddingProvider, Extract, SummaryProvider, VectorStoreWriter

_DEFAULT_MAX_TOKENS = 256


@dataclass(frozen=True, slots=True)
class IndexReport:
    """Outcome of an indexing run: how many sources and chunks were written."""

    sources_indexed: int
    chunks_indexed: int


def index_sources(
    *,
    extract: Extract,
    chunk: ChunkText,
    embedding: EmbeddingProvider,
    store: VectorStoreWriter,
    collection: str,
    sources: Sequence[SourceRef],
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    compaction: CompactionPolicy | None = None,
    summarize: SummaryProvider | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> IndexReport:
    """Index every source into ``collection``, replacing any prior chunks.

    Ensures the collection is pinned to the embedding model, then for each
    source: extract, chunk, drop the source's old chunks, and upsert the new
    ones. Dropping first makes reindexing idempotent and clears a source that
    no longer yields text.

    When ``summarize`` is given (the opt-in summary tier), a single per-DOCUMENT
    summary is generated ONCE per source - right after ``extract``, from the whole
    document text - and copied onto every chunk of that source before upsert; it
    is never generated per chunk. With no summarizer every chunk keeps
    ``summary=None`` and behaviour is unchanged.

    ``upsert`` only appends; index maintenance (folding an ANN store's unindexed
    tail) is deferred to ``store.compact``. A ``compaction`` policy bounds the
    tail DURING a large load - fold once ``after_records`` new chunks or once
    ``after_seconds`` elapse - and a final ``compact`` folds the remainder. With
    no policy only the final fold runs. IMPORTANT: without that final ``compact``
    an ANN store leaves every newly-added row unindexed and brute-force scanned,
    so callers that bypass this use case must compact themselves.
    """
    store.ensure_collection(Collection(name=collection, model_id=embedding.model_id, dim=embedding.dim))
    chunks_indexed = 0
    records_since_compact = 0
    last_compact = clock()
    for source in sources:
        document = extract(source)
        chunks = chunk(document, max_tokens=max_tokens)
        store.delete_by_source(collection=collection, uri=source.uri)
        if not chunks:
            continue
        if summarize is not None:
            # One LLM call for the whole document, reused across all its chunks.
            summary = summarize.summarize(document.text)
            chunks = [dataclasses.replace(piece, summary=summary) for piece in chunks]
        vectors = embedding.embed_passages([piece.text for piece in chunks])
        store.upsert(collection=collection, chunks=chunks, vectors=vectors)
        chunks_indexed += len(chunks)
        records_since_compact += len(chunks)
        if compaction is not None and compaction.due(
            records_since=records_since_compact, seconds_since=clock() - last_compact
        ):
            store.compact(collection=collection)
            records_since_compact = 0
            last_compact = clock()
    # Fold the final tail once, after the whole load - not per upsert (an ANN store
    # would otherwise re-optimize its index on every source).
    store.compact(collection=collection)
    return IndexReport(sources_indexed=len(sources), chunks_indexed=chunks_indexed)


__all__ = [
    "IndexReport",
    "index_sources",
]
