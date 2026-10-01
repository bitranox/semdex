"""Dataset configuration models parsed from the ``[[dataset]]`` array.

A dataset is a named binding of a store backend + collection + embedding +
source roots that the MCP server exposes and a tool call selects by name.
"Private vs shared" is purely which store/DB the binding points at and what the
database granted - not a code path here. Secrets (a ``dsn`` password) are loaded
at runtime from the environment/keyfile, never committed inline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...domain.enums import AnnRecall, EmbeddingBackend, ExtractorBackend, Partition, StoreBackend
from ...domain.errors import ConfigurationError

if TYPE_CHECKING:
    from lib_layered_config import Config

_SERVER_BACKENDS = frozenset({StoreBackend.PGVECTOR, StoreBackend.MARIADB})
# The scan text-layer cutoff default (chars): a PDF whose base extraction yields
# fewer real characters is treated as a scan and escalated to on_scan (magic
# number in config, not hardcoded in the router).
_DEFAULT_SCAN_MIN_CHARS = 24


class ExtractorEndpoint(BaseModel):
    """Where one extractor backend lives + its per-backend options (``[[dataset.extractor_config]]``).

    Routes name WHICH backend a file uses; this says WHERE that backend is and how
    to call it. The HTTP doc-converters (markitdown/xberg/docling/mineru) and the
    vision-OCR backends (olmocr/openai_vision) each need an ``endpoint``; the
    vision backends also take a ``model`` and (hosted only) an env-loaded
    ``api_key``. ``TEXT`` needs no entry.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    backend: ExtractorBackend
    endpoint: str | None = None
    # Vision-OCR model id (olmocr/openai_vision); None uses the adapter default.
    model: str | None = None
    # Bearer token for a hosted vision server. SECRET: set from the environment.
    api_key: str | None = None
    # OCR hints consumed by xberg (and passed through where supported).
    force_ocr: bool = False
    ocr_language: str | None = None
    # MinerU pipeline selector (e.g. "vlm-transformers" for MinerU's VLM mode);
    # None uses the server default (classic pipeline).
    mineru_backend: str | None = None


class Route(BaseModel):
    """A per-directory-tree extractor rule (``[[dataset.route]]``).

    ``subtree`` is a path prefix under the dataset's source roots (longest prefix
    wins). ``extractor`` is the base backend for files in that tree; ``on_scan``
    is the vision/OCR backend a SCAN escalates to (an image is always a scan; a
    PDF is a scan when its base extraction yields < ``scan_min_chars`` text).
    Each named backend resolves its endpoint/model from ``extractor_config``.
    """

    model_config = ConfigDict(frozen=True)

    subtree: str = ""
    extractor: ExtractorBackend | None = None
    on_scan: ExtractorBackend | None = None


class DatasetConfig(BaseModel):
    """A validated, immutable dataset binding.

    Example:
        >>> DatasetConfig(name="notes").partition is None
        True
    """

    # populate_by_name so ``routes`` accepts the ``[[dataset.route]]`` TOML key
    # (singular array-of-tables) via its alias AND the plural attribute name.
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    name: str
    # Where this dataset's collection lives. Embedded backends (json/sqlite_vec/
    # lancedb) use ``store_dir``; server backends (pgvector/mariadb) use ``dsn``.
    backend: StoreBackend = StoreBackend.JSON
    store_dir: str | None = None
    dsn: str | None = None
    collection: str = "default"
    # Embedding the collection is pinned to (flat, mirroring [index].embedding_model).
    embedding_provider: EmbeddingBackend = EmbeddingBackend.FASTEMBED
    embedding_model: str | None = None
    embedding_endpoint: str | None = None
    # Per-request HTTP timeout (s) for the ollama/openai providers; None -> 60 s.
    embedding_timeout: float | None = None
    # ollama physical batch in tokens; None -> ollama's default (2048), which SILENTLY
    # embeds any longer input from its first 2048 tokens only. See [embedding].num_batch.
    embedding_num_batch: int | None = None
    # Instruction prefixes for a model trained to receive one (ollama/openai/
    # sentence_transformers only). Changing either changes the vectors: reindex.
    # See [embedding].query_prefix.
    embedding_query_prefix: str = ""
    embedding_passage_prefix: str = ""
    # Bearer token for the openai provider. SECRET: set from the environment,
    # never inline in a committed dataset config.
    embedding_api_key: str | None = None
    # Source roots this dataset indexes (filesystem paths now; an IMAP mailbox later).
    sources: tuple[str, ...] = ()
    # ``None`` inherits [vector_store].default_partition (decision 11).
    partition: Partition | None = None
    # ``None`` inherits [vector_store].ann_recall; set it to run this dataset's searches at a
    # different recall/latency preset than the global default (fan-out searches each dataset at
    # its own recall). Only affects the approximate stores (lancedb/pgvector/mariadb).
    ann_recall: AnnRecall | None = None
    # Advisory UX hint: the server does not offer an index/reindex tool for a
    # read-only dataset. The database GRANT is the real enforcer.
    read_only: bool = False
    # A writable KNOWLEDGE dataset: the client writes its content directly via the
    # remember/forget tools; it has no connector and is never reconciled. Mutually
    # exclusive with sources and read_only. The DB GRANT is the real enforcer.
    writable: bool = False
    # Opt-in per-document summary tier for THIS dataset (default off). When true,
    # each source gets one LLM summary at index/reindex time, attached to every
    # search hit. The provider + secrets come from the global [summary] section;
    # these two optional fields override its model/endpoint for this dataset only
    # (falling back to [summary] when unset).
    summarize: bool = False
    summary_model: str | None = None
    summary_endpoint: str | None = None
    # Extractor routing (net model per file = subtree x filetype x scan-detection).
    # ``extractor`` is the dataset-wide default backend (TEXT reads plain
    # text/markdown; office/pdf default to markitdown, images to OCR when the
    # default is left at TEXT). ``routes`` override per subtree; ``extractor_config``
    # says where each named backend lives.
    extractor: ExtractorBackend = ExtractorBackend.TEXT
    routes: tuple[Route, ...] = Field(default=(), alias="route")
    extractor_config: tuple[ExtractorEndpoint, ...] = ()
    # Scan text-layer cutoff (chars): a PDF whose base extraction yields fewer
    # real characters escalates to the route's on_scan (vision OCR).
    scan_min_chars: int = Field(default=_DEFAULT_SCAN_MIN_CHARS, gt=0)

    @model_validator(mode="after")
    def _validate(self) -> DatasetConfig:
        if not self.name:
            raise ValueError("dataset name must be non-empty")
        if self.backend in _SERVER_BACKENDS and not self.dsn:
            raise ValueError(f"dataset '{self.name}': backend {self.backend.value} requires a dsn")
        if self.writable and self.sources:
            raise ValueError(f"dataset '{self.name}': a writable dataset has no sources (the client writes it)")
        if self.writable and self.read_only:
            raise ValueError(f"dataset '{self.name}': writable and read_only are mutually exclusive")
        if self.writable and self.routes:
            raise ValueError(
                f"dataset '{self.name}': a writable dataset has no extractor routes (nothing is extracted)"
            )
        backends = [entry.backend for entry in self.extractor_config]
        dupes = sorted({b.value for b in backends if backends.count(b) > 1})
        if dupes:
            raise ValueError(f"dataset '{self.name}': duplicate extractor_config backend(s): {', '.join(dupes)}")
        return self


def get_datasets(config: Config) -> tuple[DatasetConfig, ...]:
    """Parse the ``[[dataset]]`` array into validated, uniquely-named datasets.

    Returns an empty tuple when no datasets are configured.

    Example:
        >>> from lib_layered_config import Config
        >>> get_datasets(Config({}, {}))
        ()
    """
    entries: list[Any] = config.get("dataset", []) or []
    datasets = tuple(DatasetConfig.model_validate(entry) for entry in entries)
    names = [dataset.name for dataset in datasets]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ConfigurationError(f"duplicate dataset name(s): {', '.join(duplicates)}")
    return datasets


__all__ = [
    "DatasetConfig",
    "ExtractorEndpoint",
    "Route",
    "get_datasets",
]
