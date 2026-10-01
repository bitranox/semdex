"""Type-safe domain enums for output formats, deployment targets, and indexing."""

from __future__ import annotations

from enum import StrEnum


class WatchMode(StrEnum):
    """Filesystem watch strategy.

    ``AUTO`` selects native OS events with a polling fallback for network
    mounts; ``NATIVE`` forces OS events; ``POLL`` forces the polling observer.

    Example:
        >>> WatchMode.POLL.value
        'poll'
    """

    AUTO = "auto"
    NATIVE = "native"
    POLL = "poll"


class StoreBackend(StrEnum):
    """Vector store backend selection.

    ``JSON`` is the zero-dependency embedded default (an orjson file, best for
    small corpora). ``LANCEDB`` and ``SQLITE_VEC`` are embedded upgrades;
    ``PGVECTOR`` and ``MARIADB`` are opt-in server backends for large or shared
    corpora.

    Example:
        >>> StoreBackend.JSON.value
        'json'
    """

    JSON = "json"
    LANCEDB = "lancedb"
    SQLITE_VEC = "sqlite_vec"
    PGVECTOR = "pgvector"
    MARIADB = "mariadb"


class AnnRecall(StrEnum):
    """Portable approximate-nearest-neighbour recall/latency preset for the ANN stores.

    A query-time knob (no reindex) mapped per backend to its native search parameter
    (lancedb ``nprobes``/``refine_factor``, pgvector/mariadb ``ef_search``). ``BALANCED``
    keeps the driver default so it is non-breaking; ``FAST`` trades recall for latency,
    ``ACCURATE`` the reverse. The exact stores (``JSON``, ``SQLITE_VEC``) ignore it.

    Example:
        >>> AnnRecall.BALANCED.value
        'balanced'
    """

    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"


class Partition(StrEnum):
    """How a dataset's collections are separated within the store backend.

    ``TABLE`` puts many collections in one store/database (a ``chunks_<id>``
    table per collection; for an embedded store, many collections in one file).
    ``DATABASE`` gives a dataset its own database (server backends) or its own
    store file/dir (embedded), so DB-native grants, backups, and blast-radius
    apply per dataset. It is a deployment choice - which database a dataset's DSN
    names - not a store-adapter code path.

    Example:
        >>> Partition.TABLE.value
        'table'
    """

    TABLE = "table"
    DATABASE = "database"


class McpTransport(StrEnum):
    """MCP server transport. ``STDIO`` is local (the client spawns the process);
    ``HTTP`` is Streamable HTTP for the network (SSE is deprecated).

    Example:
        >>> McpTransport.STDIO.value
        'stdio'
    """

    STDIO = "stdio"
    HTTP = "http"


class McpAuthMode(StrEnum):
    """How the HTTP MCP endpoint is authenticated. ``NONE`` = unauthenticated
    (bind localhost / front with a reverse proxy). ``BEARER`` = static bearer
    token(s). ``OAUTH`` = OAuth 2.1 (JWT verification / OAuth proxy). Ignored by
    the stdio transport, which is bounded by the process it runs in.

    Example:
        >>> McpAuthMode.NONE.value
        'none'
    """

    NONE = "none"
    BEARER = "bearer"
    OAUTH = "oauth"


class ExtractorBackend(StrEnum):
    """File-to-text/markdown extractor selection.

    ``TEXT`` is the zero-dependency embedded default (reads plain text/markdown
    from disk). The other four are opt-in doc converters that run as their own
    Docker container and are reached over HTTP, so semdex stays free of their
    heavy (torch, OCR) dependencies: ``MARKITDOWN`` (markitdown-mcp, MCP),
    ``XBERG`` / ``DOCLING`` / ``MINERU`` (REST). Each needs its endpoint set
    via ``[extractor].endpoint`` and the matching optional extra installed.
    ``OLMOCR`` / ``OPENAI_VISION`` are vision-LLM OCR backends (an
    OpenAI-vision-compatible ``/v1/chat/completions`` server, e.g. vLLM serving
    olmOCR, or any hosted/local vision model): they render scans/images and read
    the text with a vision model, the highest-fidelity path for degraded scans.

    Example:
        >>> ExtractorBackend.TEXT.value
        'text'
    """

    TEXT = "text"
    MARKITDOWN = "markitdown"
    XBERG = "xberg"
    DOCLING = "docling"
    MINERU = "mineru"
    OLMOCR = "olmocr"
    OPENAI_VISION = "openai_vision"


class EmbeddingBackend(StrEnum):
    """Embedding provider selection.

    ``FASTEMBED`` is the default: a light ONNX model (no torch), offline after the
    first download, good quality. ``MODEL2VEC`` is an even lighter static-embedding
    option; ``OLLAMA`` reaches a local ollama server's native ``/api/embed``;
    ``OPENAI`` reaches any OpenAI-compatible ``/v1/embeddings`` server (hosted
    OpenAI, ollama's ``/v1``, or llama.cpp's ``llama-server``). ``GEMINI`` and
    ``COHERE`` reach their vendors' native cloud embedding APIs (Google
    gemini-embedding-001, Cohere embed-v4); both are paid, need an ``api_key``
    (env-only), and are strong on multilingual/German content.
    ``SENTENCE_TRANSFORMERS`` is the heavy (torch) opt-in for maximum model choice;
    ``PLACEHOLDER`` is the deterministic, dependency-free token-hash fallback used
    offline and in tests.

    Example:
        >>> EmbeddingBackend.FASTEMBED.value
        'fastembed'
    """

    PLACEHOLDER = "placeholder"
    FASTEMBED = "fastembed"
    MODEL2VEC = "model2vec"
    OLLAMA = "ollama"
    OPENAI = "openai"
    GEMINI = "gemini"
    COHERE = "cohere"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class SummaryBackend(StrEnum):
    """LLM summarizer provider selection for the opt-in per-document summary tier.

    ``NONE`` is the default and means the tier is OFF (search hits carry no
    summary). ``OLLAMA`` reaches a local ollama server's native ``/api/chat``;
    ``OPENAI`` reaches any OpenAI-compatible ``/v1/chat/completions`` server
    (hosted OpenAI, ollama's ``/v1``, or llama.cpp's ``llama-server``). A summary
    is generated once per source document at index time, never per query.

    Example:
        >>> SummaryBackend.NONE.value
        'none'
    """

    NONE = "none"
    OLLAMA = "ollama"
    OPENAI = "openai"


class ChunkStrategy(StrEnum):
    """Text chunking strategy selection.

    ``RECURSIVE`` (chonkie, markdown-aware) is the default: robust and
    token-accurate with no embedding spend. ``MARKDOWN``/``FAST`` are the
    semantic-text-splitter strategies (true CommonMark parsing / plain fast
    recursive). ``SEMANTIC``/``LATE`` are chonkie's embedding-driven quality
    tiers (worthwhile only with a real embedding provider). ``WHITESPACE`` is the
    zero-dependency fallback.

    Example:
        >>> ChunkStrategy.RECURSIVE.value
        'recursive'
    """

    WHITESPACE = "whitespace"
    RECURSIVE = "recursive"
    MARKDOWN = "markdown"
    FAST = "fast"
    SEMANTIC = "semantic"
    LATE = "late"


class ChangeKind(StrEnum):
    """The kind of filesystem change reported by the watcher.

    Example:
        >>> ChangeKind.MODIFIED.value
        'modified'
    """

    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"


class OutputFormat(StrEnum):
    """Output format options for configuration display.

    Defines valid output format choices for the config command.
    Inherits from str to allow direct string comparison and Click integration.

    Attributes:
        HUMAN: Human-readable TOML-like output format.
        JSON: Machine-readable JSON output format.

    Example:
        >>> OutputFormat.HUMAN.value
        'human'
        >>> OutputFormat.JSON == "json"
        True
    """

    HUMAN = "human"
    JSON = "json"


class DeployTarget(StrEnum):
    """Configuration deployment target layers.

    Defines valid target layers for configuration file deployment.
    Inherits from str to allow direct string comparison and Click integration.

    Attributes:
        APP: System-wide application configuration (requires privileges).
        HOST: System-wide host-specific configuration (requires privileges).
        USER: User-specific configuration (~/.config on Linux).

    Example:
        >>> DeployTarget.USER.value
        'user'
        >>> DeployTarget.APP == "app"
        True
    """

    APP = "app"
    HOST = "host"
    USER = "user"


__all__ = [
    "ChangeKind",
    "ChunkStrategy",
    "DeployTarget",
    "EmbeddingBackend",
    "ExtractorBackend",
    "McpAuthMode",
    "McpTransport",
    "OutputFormat",
    "Partition",
    "StoreBackend",
    "SummaryBackend",
    "WatchMode",
]
