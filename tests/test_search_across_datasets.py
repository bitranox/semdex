from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import pytest

from semdex.application.use_cases.searching import DatasetSearchTarget, search_across_datasets
from semdex.domain.models import Hit, Vector

if TYPE_CHECKING:
    from semdex.domain.models import Collection


class _Embed:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.calls = 0

    @property
    def model_id(self) -> str:
        return self.tag

    @property
    def dim(self) -> int:
        return 1

    def embed_passages(self, texts: Sequence[str]) -> list[Vector]:
        return [(0.0,) for _ in texts]

    def embed_query(self, text: str) -> Vector:
        self.calls += 1
        return (float(len(self.tag)),)


class _Store:
    def __init__(self, hits: list[Hit]) -> None:
        self._hits = hits
        self.seen_vector: Vector | None = None

    def query(self, *, collection: str, vector: Vector, k: int) -> list[Hit]:
        self.seen_vector = vector
        return self._hits[:k]

    def collections(self) -> list[Collection]:
        return []

    def count(self, *, collection: str) -> int:
        return len(self._hits)

    def source_hashes(self, *, collection: str) -> dict[str, str]:
        return {}


def _hit(uri: str) -> Hit:
    return Hit(chunk_text=uri, score=0.5, uri=uri, ordinal=0, label="", collection="c")


@pytest.mark.os_agnostic
def test_embeds_once_per_distinct_model_and_fuses() -> None:
    shared_embed = _Embed("m1")  # two datasets share this exact provider object
    acc_store = _Store([_hit("acc-1")])
    shared_store = _Store([_hit("shared-1")])
    targets = [
        DatasetSearchTarget(name="acc", model_key="m1", embedding=shared_embed, store=acc_store, collection="a"),
        DatasetSearchTarget(name="shared", model_key="m1", embedding=shared_embed, store=shared_store, collection="s"),
    ]
    fused = search_across_datasets(targets=targets, query="q", k=5, rrf_k=60)
    # one embed for the shared model_key, reused for both stores
    assert shared_embed.calls == 1
    assert acc_store.seen_vector == shared_store.seen_vector
    assert {f.hit.uri for f in fused} == {"acc-1", "shared-1"}
    assert {f.dataset for f in fused} == {"acc", "shared"}


@pytest.mark.os_agnostic
def test_distinct_models_embed_separately() -> None:
    e1, e2 = _Embed("aa"), _Embed("bbbb")
    t = [
        DatasetSearchTarget(name="d1", model_key="k1", embedding=e1, store=_Store([_hit("x")]), collection="c"),
        DatasetSearchTarget(name="d2", model_key="k2", embedding=e2, store=_Store([_hit("y")]), collection="c"),
    ]
    search_across_datasets(targets=t, query="q", k=5, rrf_k=60)
    assert e1.calls == 1 and e2.calls == 1
