"""Application use cases: index_sources and search (orchestrate the ports)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from semdex.application.use_cases.indexing import IndexReport, index_sources
from semdex.application.use_cases.searching import search
from semdex.composition import IndexServices, build_index_testing
from semdex.domain.models import SourceRef

_COLLECTION = "memory"


class _FakeSummarizer:
    """A deterministic in-test SummaryProvider: one summary per document text.

    Records every text it is asked to summarize so a test can assert the summary
    is computed ONCE per source document, not once per chunk.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def model_id(self) -> str:
        return "fake-summarizer"

    def summarize(self, text: str) -> str:
        self.calls.append(text)
        return f"summary of: {text}"


def _sources(*paths: str) -> list[SourceRef]:
    return [SourceRef(uri=str(p), label="curated", content_hash="h", mtime=0.0) for p in paths]


def _index(services: IndexServices, sources: Sequence[SourceRef]) -> IndexReport:
    return index_sources(
        extract=services.extract,
        chunk=services.chunk,
        embedding=services.embedding,
        store=services.store_writer,
        collection=_COLLECTION,
        sources=sources,
        max_tokens=100,
    )


@pytest.mark.os_agnostic
def test_index_sources_reports_and_persists_chunks() -> None:
    """Indexing extracts, chunks, embeds and upserts every source."""
    services = build_index_testing(documents={Path("/mem/a.md"): "alpha beta", Path("/mem/b.md"): "gamma delta"})

    report = _index(services, _sources("file:///mem/a.md", "file:///mem/b.md"))

    assert report == IndexReport(sources_indexed=2, chunks_indexed=2)
    assert services.store_reader.count(collection=_COLLECTION) == 2


@pytest.mark.os_agnostic
def test_index_sources_is_idempotent_on_reindex() -> None:
    """Reindexing the same sources replaces chunks instead of duplicating them."""
    services = build_index_testing(documents={Path("/mem/a.md"): "alpha beta gamma"})
    sources = _sources("file:///mem/a.md")

    _index(services, sources)
    _index(services, sources)

    assert services.store_reader.count(collection=_COLLECTION) == 1


@pytest.mark.os_agnostic
def test_index_sources_skips_empty_document() -> None:
    """A source with no extractable text contributes zero chunks."""
    services = build_index_testing(documents={Path("/mem/a.md"): ""})

    report = _index(services, _sources("file:///mem/a.md"))

    assert report == IndexReport(sources_indexed=1, chunks_indexed=0)
    assert services.store_reader.count(collection=_COLLECTION) == 0


@pytest.mark.os_agnostic
def test_index_sources_without_summarizer_leaves_summary_none() -> None:
    """The tier is off by default: hits carry no summary and behaviour is unchanged."""
    services = build_index_testing(documents={Path("/mem/a.md"): "alpha beta"})
    _index(services, _sources("file:///mem/a.md"))

    hits = search(embedding=services.embedding, store=services.store_reader, collection=_COLLECTION, query="alpha", k=1)
    assert hits[0].summary is None


@pytest.mark.os_agnostic
def test_index_sources_attaches_one_document_summary_and_round_trips_it() -> None:
    """With a summarizer, one summary per DOCUMENT is generated and returned on every hit.

    The document chunks into several pieces (small max_tokens), yet the summarizer is
    called exactly ONCE for the source, and that one summary rides through the store to
    each hit.
    """
    services = build_index_testing(documents={Path("/mem/doc.md"): "alpha beta gamma delta epsilon zeta"})
    summarizer = _FakeSummarizer()

    report = index_sources(
        extract=services.extract,
        chunk=services.chunk,
        embedding=services.embedding,
        store=services.store_writer,
        collection=_COLLECTION,
        sources=_sources("file:///mem/doc.md"),
        max_tokens=2,  # force several chunks per document
        summarize=summarizer,
    )

    assert report.chunks_indexed > 1  # the document really did split into multiple chunks
    assert summarizer.calls == ["alpha beta gamma delta epsilon zeta"]  # ONE call, the whole document

    hits = search(embedding=services.embedding, store=services.store_reader, collection=_COLLECTION, query="alpha", k=5)
    assert hits  # something matched
    assert all(hit.summary == "summary of: alpha beta gamma delta epsilon zeta" for hit in hits)


@pytest.mark.os_agnostic
def test_search_ranks_matching_source_first() -> None:
    """Search embeds the query and returns the closest source's chunk first."""
    services = build_index_testing(
        documents={
            Path("/mem/fruit.md"): "banana cherry apple",
            Path("/mem/car.md"): "engine piston valve",
        }
    )
    _index(services, _sources("file:///mem/fruit.md", "file:///mem/car.md"))

    hits = search(
        embedding=services.embedding,
        store=services.store_reader,
        collection=_COLLECTION,
        query="apple banana",
        k=2,
    )

    assert hits[0].uri == "file:///mem/fruit.md"


@pytest.mark.os_agnostic
def test_search_respects_k() -> None:
    """Search returns at most k hits."""
    services = build_index_testing(
        documents={Path("/mem/a.md"): "one", Path("/mem/b.md"): "two", Path("/mem/c.md"): "three"}
    )
    _index(services, _sources("file:///mem/a.md", "file:///mem/b.md", "file:///mem/c.md"))

    hits = search(
        embedding=services.embedding,
        store=services.store_reader,
        collection=_COLLECTION,
        query="one",
        k=2,
    )

    assert len(hits) == 2
