"""Reconcile use case: make indexing a true sync against a connector listing.

Prune sources that vanished, (re)index new/changed ones, and skip unchanged
ones (no needless re-embed). Uses the in-memory adapters so it is offline and
deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from semdex.adapters.memory.index import (
    InMemoryEmbeddingProvider,
    InMemoryExtractor,
    InMemoryVectorStore,
    chunk_in_memory,
)
from semdex.application.use_cases import reconcile
from semdex.domain.models import SourceRef, Vector


class FakeConnector:
    """Source connector returning a fixed list of SourceRefs (the current items)."""

    def __init__(self, refs: Sequence[SourceRef]) -> None:
        self._refs = list(refs)

    def sources(self) -> list[SourceRef]:
        return list(self._refs)


class CountingEmbedding:
    """Wraps an embedding provider and counts how many passages it embeds."""

    def __init__(self, inner: InMemoryEmbeddingProvider) -> None:
        self._inner = inner
        self.passages_embedded = 0

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def dim(self) -> int:
        return self._inner.dim

    def embed_passages(self, texts: Sequence[str]) -> list[Vector]:
        self.passages_embedded += len(texts)
        return self._inner.embed_passages(texts)

    def embed_query(self, text: str) -> Vector:
        return self._inner.embed_query(text)


def _ref(uri: str, content_hash: str) -> SourceRef:
    return SourceRef(uri=uri, label="", content_hash=content_hash, mtime=0.0)


def _fixture() -> tuple[InMemoryVectorStore, CountingEmbedding, InMemoryExtractor]:
    store = InMemoryVectorStore()
    embed = CountingEmbedding(InMemoryEmbeddingProvider(dim=16))
    extract = InMemoryExtractor({Path("/a.md"): "alpha text", Path("/b.md"): "beta text", Path("/c.md"): "gamma text"})
    return store, embed, extract


def _run(
    store: InMemoryVectorStore, embed: CountingEmbedding, extract: InMemoryExtractor, refs: Sequence[SourceRef]
) -> None:
    reconcile(
        connector=FakeConnector(refs),
        extract=extract,
        chunk=chunk_in_memory,
        embedding=embed,
        store=store,
        collection="c",
    )


@pytest.mark.os_agnostic
def test_reconcile_indexes_new_and_prunes_vanished() -> None:
    """A source absent from the connector's next listing is pruned; a new one is indexed."""
    store, embed, extract = _fixture()
    _run(store, embed, extract, [_ref("file:///a.md", "ha"), _ref("file:///b.md", "hb")])
    assert store.source_hashes(collection="c") == {"file:///a.md": "ha", "file:///b.md": "hb"}

    _run(store, embed, extract, [_ref("file:///a.md", "ha"), _ref("file:///c.md", "hc")])  # /b.md vanished, /c.md new
    assert store.source_hashes(collection="c") == {"file:///a.md": "ha", "file:///c.md": "hc"}


@pytest.mark.os_agnostic
def test_reconcile_reindexes_a_changed_source() -> None:
    """A source whose content_hash changed is re-indexed at the new hash."""
    store, embed, extract = _fixture()
    _run(store, embed, extract, [_ref("file:///a.md", "ha")])
    _run(store, embed, extract, [_ref("file:///a.md", "ha2")])  # same uri, new hash
    assert store.source_hashes(collection="c") == {"file:///a.md": "ha2"}


@pytest.mark.os_agnostic
def test_reconcile_skips_unchanged_sources_without_reembedding() -> None:
    """An unchanged source (same uri, same hash) is not re-embedded."""
    store, embed, extract = _fixture()
    _run(store, embed, extract, [_ref("file:///a.md", "ha"), _ref("file:///b.md", "hb")])
    assert embed.passages_embedded == 2  # one chunk each, first index

    embed.passages_embedded = 0
    _run(store, embed, extract, [_ref("file:///a.md", "ha"), _ref("file:///b.md", "hb")])  # nothing changed
    assert embed.passages_embedded == 0
