"""Application ports — Protocol definitions adapters implement.

Two shapes coexist:

* **Callable** Protocols (one ``__call__``) for function-shaped, one-shot
  operations. Existing module-level functions satisfy them automatically via
  structural subtyping (PEP 544): ``GetConfig``, ``SendEmail``, ``Extract``, ...
* **Multi-method** Protocols for stateful resources with several operations
  (an embedding model, a vector store, a filesystem watcher). Kept narrow and,
  where consumers need different slices, split (``VectorStoreReader`` vs
  ``VectorStoreWriter``) so each use case depends only on the slice it uses.

System Role:
    Sits between domain and adapters. Domain value objects cross these ports;
    infrastructure types (``Config``, ``EmailConfig``) are imported under
    ``TYPE_CHECKING`` only so import-linter layer contracts hold at runtime.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from ..domain.enums import DeployTarget, OutputFormat
from ..domain.models import ChangeEvent, Chunk, Collection, ExtractedDocument, Hit, SourceRef, Vector

if TYPE_CHECKING:
    from lib_layered_config import Config

    from ..adapters.email.sender import EmailConfig


class GetConfig(Protocol):
    """Load layered configuration with application defaults."""

    def __call__(
        self, *, profile: str | None = ..., start_dir: str | None = ..., dotenv_path: str | None = ...
    ) -> Config: ...


class GetDefaultConfigPath(Protocol):
    """Return the path to the bundled default configuration file."""

    def __call__(self) -> Path: ...


class DeployConfiguration(Protocol):
    """Deploy default configuration to specified target layers."""

    def __call__(
        self,
        *,
        targets: Sequence[DeployTarget],
        force: bool = ...,
        profile: str | None = ...,
        set_permissions: bool = ...,
        dir_mode: int | None = ...,
        file_mode: int | None = ...,
    ) -> list[Path]: ...


class DisplayConfig(Protocol):
    """Display the provided configuration in the requested format."""

    def __call__(
        self, config: Config, *, output_format: OutputFormat = ..., section: str | None = ..., profile: str | None = ...
    ) -> None: ...


class SendEmail(Protocol):
    """Send an email using configured SMTP settings."""

    def __call__(
        self,
        *,
        config: EmailConfig,
        recipients: str | Sequence[str] | None = ...,
        subject: str,
        body: str = ...,
        body_html: str = ...,
        from_address: str | None = ...,
        attachments: Sequence[Path] | None = ...,
    ) -> bool: ...


class SendNotification(Protocol):
    """Send a simple plain-text notification email."""

    def __call__(
        self,
        *,
        config: EmailConfig,
        recipients: str | Sequence[str] | None = ...,
        subject: str,
        message: str,
        from_address: str | None = ...,
    ) -> bool: ...


class LoadEmailConfigFromDict(Protocol):
    """Load EmailConfig from a configuration dictionary."""

    def __call__(self, config_dict: Mapping[str, Any]) -> EmailConfig: ...


class InitLogging(Protocol):
    """Initialize lib_log_rich runtime with the provided configuration."""

    def __call__(self, config: Config) -> None: ...


# --- Semantic index ports (patterns ported from the document-mcp reference) ---


class Extract(Protocol):
    """Convert a source file to text/markdown (one-shot, function-shaped).

    Receives the source's :class:`~semdex.domain.models.SourceRef` (resolving
    ``source.uri`` to read the bytes) and threads that provenance into the
    returned document.
    Raises :class:`~semdex.domain.errors.ExtractionError` on unsupported
    formats or undecodable content.
    """

    def __call__(self, source: SourceRef) -> ExtractedDocument: ...


class ChunkText(Protocol):
    """Split an extracted document into embeddable chunks.

    ``max_tokens`` is measured with the embedding model's own tokenizer so
    chunk sizes match the target model's budget.
    """

    def __call__(self, document: ExtractedDocument, *, max_tokens: int = ...) -> list[Chunk]: ...


class EmbeddingProvider(Protocol):
    """A model that turns text into vectors, with query/passage asymmetry.

    Stateful (holds the loaded model), so a multi-method Protocol. ``model_id``
    and ``dim`` pin the collection the vectors belong to; query and passage
    embedding are separate because many models need different instruction
    prefixes. Raises :class:`~semdex.domain.errors.EmbeddingError` on failure.
    """

    @property
    def model_id(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_passages(self, texts: Sequence[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...


class SummaryProvider(Protocol):
    """An LLM that condenses a document into a short per-document summary.

    Stateful (holds the chosen chat model / endpoint), so a multi-method
    Protocol mirroring :class:`EmbeddingProvider`. ``model_id`` records which
    model produced the summary; ``summarize`` returns a single summary for the
    whole document text (called ONCE per source at index time, never per chunk).
    Raises :class:`~semdex.domain.errors.SummaryError` on failure.
    """

    @property
    def model_id(self) -> str: ...

    def summarize(self, text: str) -> str: ...


class HealthProbe(Protocol):
    """Liveness check for a server-backed provider endpoint.

    Used by the reactive self-heal wrapper and the opt-in health-check loop to
    tell whether a backend (an ollama / OpenAI-compatible server) is up. Never
    raises: a probe swallows every transport failure and returns ``False`` on any
    error, so a health check never becomes a new failure mode.
    """

    def check(self) -> bool: ...


class RestartHook(Protocol):
    """Invoke the operator's restart command for a backend judged down.

    The command is an OPERATOR-set config value (``[health].restart_command``),
    not user input. Returns ``True`` iff the command ran to a zero exit; never
    raises (a timeout or spawn failure is caught and reported as ``False``).
    """

    def restart(self) -> bool: ...


class SourceConnector(Protocol):
    """Enumerate a source type's current items as SourceRefs (the list side).

    Each :class:`~semdex.domain.models.SourceRef` bundles the three facets the
    reconcile use case needs: the opaque ``uri`` (location), ``content_hash``
    (content identity, for move/change detection), and ``mtime`` (version). The
    filesystem connector lists files; an email connector will list messages,
    behind this same port - so reconcile stays source-type-blind.
    """

    def sources(self) -> Sequence[SourceRef]: ...


class VectorStoreReader(Protocol):
    """Read slice of the vector store: similarity search and inspection.

    The search use case depends on this slice only. Implementations raise
    :class:`~semdex.domain.errors.CollectionModelMismatchError` if a query
    vector's dimension does not match the collection's.
    """

    def query(self, *, collection: str, vector: Vector, k: int) -> list[Hit]: ...

    def collections(self) -> list[Collection]: ...

    def count(self, *, collection: str) -> int: ...

    def source_hashes(self, *, collection: str) -> dict[str, str]:
        """Map each indexed source ``uri`` to its ``content_hash``.

        The reconcile use case diffs this against a connector's current listing to
        prune vanished sources and re-index changed ones (unchanged hashes are
        skipped, so nothing is re-embedded needlessly).
        """
        ...


class VectorStoreWriter(Protocol):
    """Write slice of the vector store: create, upsert, delete, blue-green swap.

    The index use case depends on this slice only. ``swap`` renames a freshly
    built staging collection over the live target atomically, so a model change
    re-embeds into staging then swaps with no destructive in-place rewrite.
    """

    def ensure_collection(self, collection: Collection) -> None: ...

    def upsert(self, *, collection: str, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None: ...

    def delete_by_source(self, *, collection: str, uri: str) -> None: ...

    def swap(self, *, staging: str, target: str) -> None: ...

    def compact(self, *, collection: str) -> None:
        """Run any deferred index maintenance for the collection.

        A batching write path (``upsert`` per source) keeps ``upsert`` cheap and
        defers index work here: the index use case calls ``compact`` once after a
        bulk load, and a long-running watcher calls it periodically. For an ANN
        store this folds the freshly-added, still-unindexed rows into the index
        (so queries stop brute-force scanning that tail); stores that need no
        maintenance implement it as a no-op.
        """
        ...


class VectorStore(VectorStoreReader, VectorStoreWriter, Protocol):
    """A store that is both readable and writable.

    Every backend adapter implements both slices; the composition store factory
    returns this combined type and wires it into both ``IndexServices`` fields.
    Use cases still depend on the narrow reader/writer slices, not this union.
    """

    def close(self) -> None:
        """Release any resources the store holds (DB connection, file handle).

        Called when a long-lived server (``semdex serve``) shuts down, so a
        server-backed store does not leak its connection. Stores that hold no
        persistent resource implement it as a no-op.
        """
        ...


class Watcher(Protocol):
    """Watch source roots and deliver change events to a callback.

    Stateful, long-lived resource (native OS events or a polling fallback), so
    a multi-method Protocol. Raises :class:`~semdex.domain.errors.WatchError`
    if a root cannot be observed.
    """

    def start(self, *, roots: Sequence[Path], on_change: Callable[[ChangeEvent], None]) -> None: ...

    def stop(self) -> None: ...


__all__ = [
    "ChunkText",
    "DeployConfiguration",
    "DisplayConfig",
    "EmbeddingProvider",
    "Extract",
    "GetConfig",
    "GetDefaultConfigPath",
    "HealthProbe",
    "InitLogging",
    "LoadEmailConfigFromDict",
    "RestartHook",
    "SendEmail",
    "SendNotification",
    "SummaryProvider",
    "VectorStore",
    "VectorStoreReader",
    "VectorStoreWriter",
    "Watcher",
]
