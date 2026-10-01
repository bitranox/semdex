"""Domain value objects and enums for the semantic index (pure, frozen)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from semdex.domain.enums import ChangeKind, StoreBackend, WatchMode
from semdex.domain.errors import (
    CollectionModelMismatchError,
    EmbeddingError,
    ExtractionError,
    InvalidCollectionError,
    VectorStoreError,
    WatchError,
)
from semdex.domain.models import (
    ChangeEvent,
    Chunk,
    Collection,
    CompactionPolicy,
    ExtractedDocument,
    Hit,
    SourceRef,
)


def _source_ref() -> SourceRef:
    return SourceRef(uri="/mem/facts/a.md", label="curated", content_hash="deadbeef", mtime=1.0)


# ----------------------------- enums -----------------------------


@pytest.mark.os_agnostic
def test_watch_mode_values() -> None:
    """WatchMode covers the auto/native/poll switch."""
    assert {m.value for m in WatchMode} == {"auto", "native", "poll"}


@pytest.mark.os_agnostic
def test_store_backend_values() -> None:
    """StoreBackend covers the embedded + server backends."""
    assert {b.value for b in StoreBackend} == {"json", "lancedb", "sqlite_vec", "pgvector", "mariadb"}


@pytest.mark.os_agnostic
def test_change_kind_values() -> None:
    """ChangeKind covers filesystem change events."""
    assert {c.value for c in ChangeKind} == {"created", "modified", "deleted"}


@pytest.mark.os_agnostic
def test_enums_are_str_backed_for_config_and_comparison() -> None:
    """Enums inherit str so config values and comparisons work directly."""
    assert StoreBackend.LANCEDB == "lancedb"
    assert WatchMode.POLL == "poll"


# ------------------------- value objects -------------------------


@pytest.mark.os_agnostic
def test_source_ref_holds_identity_fields() -> None:
    """SourceRef captures the identity of an indexed source file."""
    ref = _source_ref()
    assert ref.uri == "/mem/facts/a.md"
    assert ref.label == "curated"
    assert ref.content_hash == "deadbeef"
    assert ref.mtime == 1.0


@pytest.mark.os_agnostic
def test_value_objects_are_frozen() -> None:
    """Domain value objects are immutable (frozen dataclasses)."""
    ref = _source_ref()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.content_hash = "changed"  # type: ignore[misc]


@pytest.mark.os_agnostic
def test_extracted_document_carries_text_and_full_provenance() -> None:
    """ExtractedDocument pairs extracted text with the source's full SourceRef."""
    ref = _source_ref()
    doc = ExtractedDocument(source=ref, text="# hello")
    assert doc.source is ref
    assert doc.source.uri == "/mem/facts/a.md"
    assert doc.text == "# hello"


@pytest.mark.os_agnostic
def test_chunk_references_its_source() -> None:
    """A Chunk carries its text plus provenance back to the source file."""
    chunk = Chunk(text="body", source=_source_ref(), ordinal=0, token_count=2)
    assert chunk.text == "body"
    assert chunk.source.label == "curated"
    assert chunk.ordinal == 0
    assert chunk.token_count == 2


@pytest.mark.os_agnostic
def test_hit_carries_chunk_text_score_and_provenance() -> None:
    """A Hit returns matched chunk text, score, source path and label."""
    hit = Hit(
        chunk_text="body",
        score=0.87,
        uri="/mem/a.md",
        ordinal=3,
        label="global",
        collection="memory",
    )
    assert hit.chunk_text == "body"
    assert hit.score == pytest.approx(0.87)
    assert hit.uri == "/mem/a.md"
    assert hit.ordinal == 3
    assert hit.label == "global"
    assert hit.collection == "memory"


@pytest.mark.os_agnostic
def test_change_event_pairs_a_path_with_a_kind() -> None:
    """A ChangeEvent pairs a filesystem path with the kind of change."""
    event = ChangeEvent(path=Path("/mem/a.md"), kind=ChangeKind.MODIFIED)
    assert event.path == Path("/mem/a.md")
    assert event.kind is ChangeKind.MODIFIED


# -------------------- Collection (has validation) --------------------


@pytest.mark.os_agnostic
def test_collection_pins_a_name_to_a_model_and_dim() -> None:
    """A Collection binds a name to exactly one embedding model and dim."""
    coll = Collection(name="memory", model_id="bge-m3", dim=1024)
    assert (coll.name, coll.model_id, coll.dim) == ("memory", "bge-m3", 1024)


@pytest.mark.os_agnostic
@pytest.mark.parametrize(
    ("name", "model_id", "dim"),
    [
        ("", "bge-m3", 8),
        ("memory", "", 8),
        ("memory", "bge-m3", 0),
        ("memory", "bge-m3", -1),
    ],
)
def test_collection_rejects_invalid_fields(name: str, model_id: str, dim: int) -> None:
    """Empty name/model_id or non-positive dim is an invalid collection."""
    with pytest.raises(InvalidCollectionError):
        Collection(name=name, model_id=model_id, dim=dim)


@pytest.mark.os_agnostic
def test_invalid_collection_error_is_a_value_error() -> None:
    """InvalidCollectionError stays catchable as ValueError."""
    assert issubclass(InvalidCollectionError, ValueError)


# ----------------------------- errors -----------------------------


@pytest.mark.os_agnostic
def test_error_hierarchy() -> None:
    """Store/embedding/extraction/watch errors are typed for boundary handling."""
    assert issubclass(CollectionModelMismatchError, VectorStoreError)
    for err in (ExtractionError, EmbeddingError, VectorStoreError, WatchError):
        assert issubclass(err, Exception)


# ----------------------------- compaction policy -----------------------------


@pytest.mark.os_agnostic
def test_compaction_policy_due_on_records_or_time() -> None:
    """A fold is due once either the record or the time trigger is reached (with new rows)."""
    policy = CompactionPolicy(after_records=100, after_seconds=60.0)
    assert policy.due(records_since=100, seconds_since=0.0)  # record trigger
    assert policy.due(records_since=1, seconds_since=60.0)  # time trigger
    assert not policy.due(records_since=99, seconds_since=59.0)  # neither reached


@pytest.mark.os_agnostic
def test_compaction_policy_never_due_without_new_records() -> None:
    """No new rows means nothing to fold, even past the time bound."""
    policy = CompactionPolicy(after_records=1, after_seconds=1.0)
    assert not policy.due(records_since=0, seconds_since=1000.0)


@pytest.mark.os_agnostic
def test_compaction_policy_non_positive_disables_a_trigger() -> None:
    """A non-positive threshold turns its trigger off."""
    records_only = CompactionPolicy(after_records=10, after_seconds=0.0)
    assert records_only.due(records_since=10, seconds_since=1e9)  # only records can fire
    assert not records_only.due(records_since=9, seconds_since=1e9)
    time_only = CompactionPolicy(after_records=0, after_seconds=5.0)
    assert time_only.due(records_since=1, seconds_since=5.0)  # only time can fire
    assert not time_only.due(records_since=1_000_000, seconds_since=4.0)
