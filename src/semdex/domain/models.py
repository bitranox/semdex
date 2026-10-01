"""Pure domain value objects for the semantic index.

Frozen, slotted dataclasses with no I/O, logging, or framework dependencies.
Adapters map external formats to and from these types at the boundary; the
application layer and ports pass them across boundaries instead of raw dicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from .enums import ChangeKind
from .errors import InvalidCollectionError

# A dense embedding vector. A bare tuple of floats is the most convenient shape
# for the inner layers; adapters convert to and from their store's native form.
# TypeAlias (not the 3.12 ``type`` statement) keeps this valid on Python 3.10+.
Vector: TypeAlias = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Identity of an indexed source: its opaque location URI, label, and hash.

    ``uri`` is the source-type-blind location the store keys on and a :class:`Hit`
    points back to - ``file:///abs/path`` for a file, ``imap://mailbox/uid`` for a
    mail. Only the (type-aware) connector/extractor interprets the scheme; the
    store and search layers treat it as an opaque string. ``label`` is a free-form
    provenance tag chosen by the deployment (for the bitranox consumer it is the
    memory tier, e.g. ``"curated"``; a general deployment may use ``""`` or a
    namespace). ``content_hash`` is the content identity that drives incremental
    reindexing and move detection; ``mtime`` is the version token at index time.
    """

    uri: str
    label: str
    content_hash: str
    mtime: float


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """The text/markdown extracted from a single source file, with provenance.

    Carries the full :class:`SourceRef` (not just a path) so downstream chunking
    attaches correct provenance without re-deriving the label, hash, or mtime.
    """

    source: SourceRef
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    """A unit of text to embed, with provenance back to its source file.

    ``ordinal`` is the chunk's position within the source document;
    ``token_count`` is measured with the embedding model's own tokenizer.
    ``summary`` is the optional per-DOCUMENT LLM summary (the opt-in summary
    tier): generated once for the whole source and copied onto every chunk of it,
    ``None`` when the tier is off. It rides the chunk into the store and back out
    on a :class:`Hit`.
    """

    text: str
    source: SourceRef
    ordinal: int
    token_count: int
    summary: str | None = None


@dataclass(frozen=True, slots=True)
class Hit:
    """A search result: matched chunk text, similarity score, and provenance.

    Carries the source ``uri`` (the opaque location - a client opens ``file://``
    in an editor, ``imap://`` in a mail app), the matched chunk's ``ordinal`` (its
    position within the source document, so a caller can point into the source
    rather than only at it), its free-form label, and the owning collection so
    recall can inject a real snippet the way the keyword scan does.

    ``summary`` is the optional per-DOCUMENT summary from the opt-in summary tier
    (the whole source's summary, shared by every chunk of it), letting a calling
    LLM triage relevance without opening the source; ``None`` when the tier is off.
    """

    chunk_text: str
    score: float
    uri: str
    ordinal: int
    label: str
    collection: str
    summary: str | None = None


@dataclass(frozen=True, slots=True)
class FusedHit:
    """A search hit after cross-dataset Reciprocal Rank Fusion.

    Wraps the original :class:`Hit` (whose ``score`` stays the per-dataset cosine,
    NOT comparable across datasets) with the dataset that produced it and the
    fused ``rrf_score`` that actually ranks it.
    """

    hit: Hit
    dataset: str
    rrf_score: float


@dataclass(frozen=True, slots=True)
class ChangeEvent:
    """A filesystem change: a path paired with the kind of change observed."""

    path: Path
    kind: ChangeKind


@dataclass(frozen=True, slots=True)
class Collection:
    """A named vector space pinned to exactly one embedding model and dimension.

    Vectors from different models cannot be compared in one similarity search,
    so a searchable collection is single-model by construction. Changing a
    collection's model means re-embedding it into a new collection.
    """

    name: str
    model_id: str
    dim: int

    def __post_init__(self) -> None:
        if not self.name:
            raise InvalidCollectionError("collection name must be non-empty")
        if not self.model_id:
            raise InvalidCollectionError("collection model_id must be non-empty")
        if self.dim <= 0:
            raise InvalidCollectionError(f"collection dim must be positive, got {self.dim}")


@dataclass(frozen=True, slots=True)
class CompactionPolicy:
    """When a batching indexer should fold an ANN store's unindexed tail.

    An ANN store (lancedb) keeps rows added since its last compaction in an
    unindexed tail that is brute-force scanned per query. This policy bounds that
    tail: a fold is ``due`` after ``after_records`` new rows, OR after
    ``after_seconds`` have elapsed while new rows exist - whichever comes first.
    A non-positive ``after_records`` disables the record trigger; a non-positive
    ``after_seconds`` disables the time trigger. It is a value object with no
    clock of its own - the caller passes the elapsed measures.
    """

    after_records: int
    after_seconds: float

    def due(self, *, records_since: int, seconds_since: float) -> bool:
        """True if a fold is warranted given rows added and time elapsed since the last one."""
        if records_since <= 0:
            return False
        by_records = self.after_records > 0 and records_since >= self.after_records
        by_time = self.after_seconds > 0 and seconds_since >= self.after_seconds
        return by_records or by_time


__all__ = [
    "ChangeEvent",
    "Chunk",
    "Collection",
    "CompactionPolicy",
    "ExtractedDocument",
    "FusedHit",
    "Hit",
    "SourceRef",
    "Vector",
]
