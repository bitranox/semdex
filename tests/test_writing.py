from __future__ import annotations

from pathlib import Path

import pytest

from semdex.adapters.config.dataset import DatasetConfig
from semdex.application.use_cases.searching import search
from semdex.application.use_cases.writing import forget, remember
from semdex.composition import DatasetServices, build_dataset_services
from semdex.domain.enums import EmbeddingBackend, Partition, StoreBackend


def _knowledge(tmp_path: Path) -> DatasetServices:
    dataset = DatasetConfig(
        name="kb",
        backend=StoreBackend.JSON,
        store_dir=str(tmp_path / "kb"),
        collection="kb",
        embedding_provider=EmbeddingBackend.PLACEHOLDER,
        writable=True,
    )
    return build_dataset_services(dataset, default_partition=Partition.TABLE)


@pytest.mark.os_agnostic
def test_remember_then_search_then_forget(tmp_path: Path) -> None:
    svc = _knowledge(tmp_path)
    report = remember(
        chunk=svc.chunk,
        embedding=svc.embedding,
        store=svc.store,
        collection=svc.collection,
        uri="semdex://kb/n1",
        text="banana cherry apple",
        label="fruit",
    )
    assert report.chunks_indexed >= 1
    hits = search(embedding=svc.embedding, store=svc.store, collection=svc.collection, query="banana apple", k=5)
    assert hits and hits[0].uri == "semdex://kb/n1" and hits[0].label == "fruit"
    forget(store=svc.store, collection=svc.collection, uri="semdex://kb/n1")
    assert search(embedding=svc.embedding, store=svc.store, collection=svc.collection, query="banana apple", k=5) == []


@pytest.mark.os_agnostic
def test_remember_same_uri_replaces(tmp_path: Path) -> None:
    svc = _knowledge(tmp_path)
    remember(
        chunk=svc.chunk,
        embedding=svc.embedding,
        store=svc.store,
        collection=svc.collection,
        uri="semdex://kb/n1",
        text="alpha",
        label="",
    )
    remember(
        chunk=svc.chunk,
        embedding=svc.embedding,
        store=svc.store,
        collection=svc.collection,
        uri="semdex://kb/n1",
        text="omega",
        label="",
    )
    # drop-first replace: only the second write's chunk(s) remain under that uri
    assert svc.store.count(collection="kb") == 1
