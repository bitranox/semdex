"""The benchmark corpus plumbing indexes and recovers real corpus doc ids.

``test_e2e_matrix.py`` runs only on the scheduled workflow, so its id-to-source mapping is
pinned here, through the real ``index_sources`` and ``search`` use cases, where every push
runs it on every platform.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _bench_corpus import corpus_extractor, corpus_sources, ranked_doc_ids

from semdex.application.use_cases import index_sources, search
from semdex.composition import build_chunker, build_embedding, build_vector_store
from semdex.domain.enums import ChunkStrategy, EmbeddingBackend, StoreBackend

# Doc ids in the shapes real corpora use: BEIR NFCorpus ("MED-4972"), CQADupStack (bare
# digits), DBpedia-entity (angle brackets, a colon and a slash), plus the characters a URI
# encoding has to survive. Each text shares no word with another, so the lexical
# placeholder embedding ranks a document first for its own text.
_DOCS = {
    "MED-4972": "zebra stripes savanna grassland",
    "228054": "quantum lattice photon emission",
    "<dbpedia:AC/DC>": "guitar amplifier riff anthem",
    "50% off: a b#c?d": "glacier moraine crevasse ice",
    "Ümlaut-日本": "harbour lighthouse fog horn",
}


@pytest.mark.os_agnostic
def test_a_corpus_indexes_and_every_hit_maps_back_to_its_own_doc_id(tmp_path: Path) -> None:
    embedding = build_embedding(EmbeddingBackend.PLACEHOLDER)
    store = build_vector_store(StoreBackend.JSON, tmp_path / "store")

    report = index_sources(
        extract=corpus_extractor(_DOCS),
        chunk=build_chunker(ChunkStrategy.WHITESPACE),
        embedding=embedding,
        store=store,
        collection="bench",
        sources=corpus_sources(_DOCS),
    )

    assert report.sources_indexed == len(_DOCS)
    for doc_id, text in _DOCS.items():
        hits = search(embedding=embedding, store=store, collection="bench", query=text, k=10)
        ranked = ranked_doc_ids(hits, 10)
        assert ranked[0] == doc_id
        assert set(ranked) == set(_DOCS)


@pytest.mark.os_agnostic
def test_two_doc_ids_that_resolve_to_one_path_are_refused() -> None:
    # The in-memory extractor is keyed by path, and a path collapses a doubled separator,
    # so without the refusal the second document would silently replace the first.
    with pytest.raises(ValueError, match="same in-memory path"):
        corpus_extractor({"a//b": "first text", "a/b": "second text"})
