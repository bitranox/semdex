"""Composition root wiring adapters to application ports."""

from __future__ import annotations

import importlib.util
import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from ..adapters.config.deploy import deploy_configuration
from ..adapters.config.display import display_config

# Configuration services
from ..adapters.config.loader import get_config, get_default_config_path

# Email services
from ..adapters.email.sender import (
    load_email_config_from_dict,
    send_email,
    send_notification,
)

# Logging services
from ..adapters.logging.setup import init_logging
from ..application.batching import BatchingEmbedding
from ..domain.ann_tuning import AnnParams
from ..domain.enums import (
    ChunkStrategy,
    EmbeddingBackend,
    ExtractorBackend,
    Partition,
    StoreBackend,
    SummaryBackend,
    WatchMode,
)
from ..domain.errors import ConfigurationError

logger = logging.getLogger(__name__)

# Static conformance assertions - pyright verifies that each adapter function
# structurally satisfies its corresponding Protocol at type-check time.
if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from ..adapters.config.dataset import DatasetConfig
    from ..adapters.config.health import HealthConfig
    from ..adapters.config.sanitize import SanitizeConfig
    from ..adapters.config.summary import SummaryConfig
    from ..adapters.memory.email import EmailSpy
    from ..application.ports import (
        ChunkText,
        DeployConfiguration,
        DisplayConfig,
        EmbeddingProvider,
        Extract,
        GetConfig,
        GetDefaultConfigPath,
        InitLogging,
        LoadEmailConfigFromDict,
        SendEmail,
        SendNotification,
        SourceConnector,
        SummaryProvider,
        VectorStore,
        VectorStoreReader,
        VectorStoreWriter,
        Watcher,
    )

    _assert_get_config: GetConfig = get_config
    _assert_get_default_config_path: GetDefaultConfigPath = get_default_config_path
    _assert_deploy_configuration: DeployConfiguration = deploy_configuration
    _assert_display_config: DisplayConfig = display_config
    _assert_send_email: SendEmail = send_email
    _assert_send_notification: SendNotification = send_notification
    _assert_load_email_config_from_dict: LoadEmailConfigFromDict = load_email_config_from_dict
    _assert_init_logging: InitLogging = init_logging


@dataclass(frozen=True, slots=True)
class AppServices:
    """Frozen container holding all application port implementations."""

    get_config: GetConfig
    get_default_config_path: GetDefaultConfigPath
    deploy_configuration: DeployConfiguration
    display_config: DisplayConfig
    send_email: SendEmail
    send_notification: SendNotification
    load_email_config_from_dict: LoadEmailConfigFromDict
    init_logging: InitLogging


def build_production() -> AppServices:
    """Wire production adapters into an AppServices container."""
    return AppServices(
        get_config=get_config,
        get_default_config_path=get_default_config_path,
        deploy_configuration=deploy_configuration,
        display_config=display_config,
        send_email=send_email,
        send_notification=send_notification,
        load_email_config_from_dict=load_email_config_from_dict,
        init_logging=init_logging,
    )


def build_testing(*, spy: EmailSpy | None = None) -> AppServices:
    """Wire in-memory adapters into an AppServices container.

    Args:
        spy: Optional EmailSpy instance for capturing email operations.
            When None, a fresh EmailSpy is created. Pass your own spy
            to assert on captured emails in tests.

    Returns:
        AppServices container with in-memory adapters.
    """
    from ..adapters.memory import (
        EmailSpy,
        deploy_configuration_in_memory,
        display_config_in_memory,
        get_config_in_memory,
        get_default_config_path_in_memory,
        init_logging_in_memory,
        load_email_config_from_dict_in_memory,
    )

    email_spy = spy if spy is not None else EmailSpy()

    return AppServices(
        get_config=get_config_in_memory,
        get_default_config_path=get_default_config_path_in_memory,
        deploy_configuration=deploy_configuration_in_memory,
        display_config=display_config_in_memory,
        send_email=email_spy.send_email,
        send_notification=email_spy.send_notification,
        load_email_config_from_dict=load_email_config_from_dict_in_memory,
        init_logging=init_logging_in_memory,
    )


@dataclass(frozen=True, slots=True)
class IndexServices:
    """Frozen container holding the semantic-index port implementations.

    Kept separate from :class:`AppServices` (the CLI-app config/logging/email
    services) because the index is a distinct concern with its own lifecycle.
    ``store_reader`` and ``store_writer`` are the narrow read/write slices of one
    vector store, so a use case depends only on the slice it needs.
    """

    extract: Extract
    chunk: ChunkText
    embedding: EmbeddingProvider
    store_reader: VectorStoreReader
    store_writer: VectorStoreWriter
    watcher: Watcher
    # The opt-in per-document summarizer, or None when the summary tier is off.
    summarize: SummaryProvider | None = None


def build_index_testing(*, documents: Mapping[Path, str] | None = None) -> IndexServices:
    """Wire the in-memory index adapters into an IndexServices container.

    Args:
        documents: Optional ``path -> text`` map the in-memory extractor serves.

    Returns:
        IndexServices whose ``store_reader`` and ``store_writer`` share one
        backing store, so writes through the writer are visible to the reader.

    Note:
        ``build_index_production`` (LanceDB + a local embedding provider, etc.)
        lands with the phase-2 adapters; only the testing wiring exists today.
    """
    from ..adapters.memory.index import (
        InMemoryEmbeddingProvider,
        InMemoryExtractor,
        InMemoryVectorStore,
        InMemoryWatcher,
        chunk_in_memory,
    )

    store = InMemoryVectorStore()
    return IndexServices(
        extract=InMemoryExtractor(documents or {}),
        chunk=chunk_in_memory,
        embedding=InMemoryEmbeddingProvider(),
        store_reader=store,
        store_writer=store,
        watcher=InMemoryWatcher(),
    )


def build_vector_store(
    backend: StoreBackend,
    store_dir: Path,
    *,
    dsn: str | None = None,
    lance_index_threshold: int | None = None,
    ann_params: AnnParams | None = None,
) -> VectorStore:
    """Select the vector-store adapter for the configured backend.

    ``JSON`` is the zero-dependency embedded default. Other backends land
    incrementally behind this factory (each behind the same VectorStore ports);
    selecting one before it is implemented fails with a clear error.
    ``lance_index_threshold`` (None keeps the adapter default) is consumed only
    by the lancedb backend: the row count at which its ANN index is built.
    ``ann_params`` (None keeps each driver default) is the resolved ANN
    recall/latency knob, consumed only by the approximate stores (lancedb /
    pgvector / mariadb); the exact stores (json / sqlite_vec) ignore it.
    """
    from ..domain.errors import VectorStoreError

    if backend is StoreBackend.JSON:
        from ..adapters.vectorstore import JsonVectorStore

        return JsonVectorStore(store_dir)
    if backend is StoreBackend.SQLITE_VEC:
        from ..adapters.vectorstore import SqliteVecStore

        return SqliteVecStore(store_dir)
    if backend is StoreBackend.LANCEDB:
        from ..adapters.vectorstore import LanceVectorStore

        if lance_index_threshold is None:
            return LanceVectorStore(store_dir, ann_params=ann_params)
        return LanceVectorStore(store_dir, index_threshold=lance_index_threshold, ann_params=ann_params)
    if backend is StoreBackend.PGVECTOR:
        from ..adapters.vectorstore import PgVectorStore

        if dsn is None:
            raise VectorStoreError("pgvector backend requires [vector_store].dsn")
        return PgVectorStore(dsn, ann_params=ann_params)
    if backend is StoreBackend.MARIADB:
        from ..adapters.vectorstore import MariaDbVectorStore

        if dsn is None:
            raise VectorStoreError("mariadb backend requires [vector_store].dsn")
        return MariaDbVectorStore(dsn, ann_params=ann_params)
    raise VectorStoreError(f"vector store backend '{backend.value}' is not implemented yet")


def build_extractor(
    backend: ExtractorBackend,
    *,
    endpoint: str | None = None,
    max_bytes: int | None = None,
    timeout: float = 120.0,
    force_ocr: bool = False,
    ocr_language: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    mineru_backend: str | None = None,
    render_dpi: int | None = None,
) -> Extract:
    """Select the Extract adapter for the configured extractor backend.

    ``TEXT`` is the embedded, zero-dependency default. The doc converters
    (markitdown/xberg/docling/mineru) and the vision-OCR backends
    (olmocr/openai_vision) each run as their own container/server reached over
    HTTP and land behind this factory (each behind the same ``Extract`` port);
    ``endpoint`` names it. ``max_bytes`` (None keeps the adapter default) bounds
    the read. ``force_ocr``/``ocr_language`` are OCR hints (xberg). ``model``/
    ``api_key`` configure the vision-OCR backends; ``mineru_backend`` selects
    MinerU's pipeline (e.g. ``vlm-transformers`` for its VLM mode). ``render_dpi``
    (None keeps the adapter default of 200) is the resolution the vision backends
    rasterise a PDF page at, and it is the knob that decides how many vision
    tokens a page costs: at 200 a dense page exceeded a 16k context and exhausted
    a 16 GB card's spare VRAM mid-inference. This function is the injected
    ``ExtractorFactory`` the RoutingExtractor calls per backend.
    """
    from ..adapters.extractor import (
        DoclingExtractor,
        MarkitdownExtractor,
        MineruExtractor,
        TextExtractor,
        VisionOcrExtractor,
        XbergExtractor,
    )
    from ..adapters.extractor._http import DEFAULT_MAX_BYTES

    if backend is ExtractorBackend.TEXT:
        return TextExtractor(max_bytes=max_bytes) if max_bytes is not None else TextExtractor()
    resolved_max = max_bytes if max_bytes is not None else DEFAULT_MAX_BYTES
    if backend is ExtractorBackend.MARKITDOWN:
        return MarkitdownExtractor(endpoint, max_bytes=resolved_max, timeout=timeout)
    if backend is ExtractorBackend.XBERG:
        return XbergExtractor(
            endpoint, max_bytes=resolved_max, timeout=timeout, force_ocr=force_ocr, ocr_language=ocr_language
        )
    if backend is ExtractorBackend.DOCLING:
        return DoclingExtractor(endpoint, max_bytes=resolved_max, timeout=timeout)
    if backend is ExtractorBackend.MINERU:
        return MineruExtractor(endpoint, max_bytes=resolved_max, timeout=timeout, backend=mineru_backend)
    # OLMOCR / OPENAI_VISION share the OpenAI-vision adapter; only the default
    # model differs (olmOCR is the vLLM-served olmocr model).
    default_model = "olmocr" if backend is ExtractorBackend.OLMOCR else model
    return VisionOcrExtractor(
        endpoint,
        model=model or default_model,
        api_key=api_key,
        max_bytes=resolved_max,
        timeout=timeout,
        dpi=render_dpi,
    )


def build_chunker(
    strategy: ChunkStrategy,
    *,
    recipe: str = "markdown",
    overlap: int = 0,
    tokenizer: str = "gpt2",
    semantic_model: str | None = None,
    enforce_max_tokens: bool = True,
) -> ChunkText:
    """Select the chunker adapter for the configured strategy.

    ``RECURSIVE`` (chonkie, markdown-aware) is the recommended default. The
    chonkie (recursive/semantic/late) and semantic-text-splitter (markdown/fast)
    strategies need ``semdex[chunk]``; if the library is missing this degrades to
    the zero-dependency whitespace chunker with a warning (per coding-resilience),
    so a missing extra never makes indexing fail.
    """
    from ..adapters.chunker import WhitespaceChunker

    if strategy is ChunkStrategy.WHITESPACE:
        return WhitespaceChunker()
    if strategy in (ChunkStrategy.RECURSIVE, ChunkStrategy.SEMANTIC, ChunkStrategy.LATE):
        if importlib.util.find_spec("chonkie") is None:
            logger.warning(
                "chonkie missing (install semdex[chunk]); chunker %r falls back to whitespace", strategy.value
            )
            return WhitespaceChunker()
        from ..adapters.chunker import ChonkieChunker

        return ChonkieChunker(
            strategy,
            recipe=recipe,
            overlap=overlap,
            tokenizer=tokenizer,
            semantic_model=semantic_model,
            enforce_max_tokens=enforce_max_tokens,
        )
    if importlib.util.find_spec("semantic_text_splitter") is None:
        logger.warning(
            "semantic-text-splitter missing (install semdex[chunk]); chunker %r falls back to whitespace",
            strategy.value,
        )
        return WhitespaceChunker()
    from ..adapters.chunker import StsChunker

    return StsChunker(strategy, overlap=overlap, tokenizer=tokenizer)


def _maybe_batch(embedding: EmbeddingProvider, batch: int | None) -> EmbeddingProvider:
    """Wrap the embedding to batch embed_passages at index time when ``[embedding].batch`` is set."""
    return BatchingEmbedding(embedding, batch=batch) if batch else embedding


def build_index_production(
    store_dir: Path,
    *,
    embedding_provider: EmbeddingBackend = EmbeddingBackend.PLACEHOLDER,
    embedding_model: str | None = None,
    embedding_endpoint: str | None = None,
    embedding_threads: int | None = None,
    embedding_batch: int | None = None,
    embedding_timeout: float | None = None,
    embedding_retries: int | None = None,
    embedding_num_batch: int | None = None,
    embedding_query_prefix: str = "",
    embedding_passage_prefix: str = "",
    embedding_api_key: str | None = None,
    max_file_bytes: int | None = None,
    embedding_dim: int | None = None,
    backend: StoreBackend = StoreBackend.JSON,
    dsn: str | None = None,
    lance_index_threshold: int | None = None,
    ann_params: AnnParams | None = None,
    extractor_backend: ExtractorBackend = ExtractorBackend.TEXT,
    extractor_endpoint: str | None = None,
    extractor_timeout: float = 120.0,
    extractor_max_bytes: int | None = None,
    extractor_force_ocr: bool = False,
    extractor_ocr_language: str | None = None,
    sanitize: SanitizeConfig | None = None,
    chunker_strategy: ChunkStrategy = ChunkStrategy.WHITESPACE,
    chunker_recipe: str = "markdown",
    chunk_overlap: int = 0,
    chunker_tokenizer: str = "gpt2",
    chunker_semantic_model: str | None = None,
    chunker_enforce_max_tokens: bool = True,
    summary_provider: SummaryBackend = SummaryBackend.NONE,
    summary_model: str | None = None,
    summary_endpoint: str | None = None,
    summary_api_key: str | None = None,
    summary_timeout: float | None = None,
    summary_retries: int | None = None,
    summary_prompt: str | None = None,
    summary_max_input_chars: int | None = None,
) -> IndexServices:
    """Wire the production index adapters into an IndexServices container.

    Real and persistent: the extractor selected by ``extractor_backend`` (the
    embedded text reader by default; a doc-converter container otherwise), a
    token-bounded chunker, and the vector store selected by ``backend`` (JSON by
    default).

    ``embedding_provider`` selects the embedding: this factory defaults to the
    offline placeholder so direct construction stays light; the CLI passes the
    ``[embedding]`` config (fastembed by default). ``max_file_bytes`` and
    ``embedding_dim`` are tunable config values (None keeps the adapter defaults);
    the CLI threads them, ``backend``/``dsn``, and the ``extractor_*`` selection
    from config. The watcher is a stub.
    """
    from ..adapters.memory.index import InMemoryWatcher

    # The size cap is per concern: the embedded TEXT backend uses the text-reader
    # cap ([index].max_file_bytes); the HTTP doc converters use their own, larger
    # cap ([extractor].max_file_bytes). None keeps the adapter's built-in default.
    extractor_max = max_file_bytes if extractor_backend is ExtractorBackend.TEXT else extractor_max_bytes
    extractor = build_extractor(
        extractor_backend,
        endpoint=extractor_endpoint,
        max_bytes=extractor_max,
        timeout=extractor_timeout,
        force_ocr=extractor_force_ocr,
        ocr_language=extractor_ocr_language,
    )
    if sanitize is not None and sanitize.enabled:
        from ..adapters.sanitize import SanitizingExtractor

        extractor = SanitizingExtractor(extractor, sanitize)
    store = build_vector_store(
        backend, store_dir, dsn=dsn, lance_index_threshold=lance_index_threshold, ann_params=ann_params
    )
    return IndexServices(
        extract=extractor,
        chunk=build_chunker(
            chunker_strategy,
            recipe=chunker_recipe,
            overlap=chunk_overlap,
            tokenizer=chunker_tokenizer,
            semantic_model=chunker_semantic_model,
            enforce_max_tokens=chunker_enforce_max_tokens,
        ),
        embedding=_maybe_batch(
            build_embedding(
                embedding_provider,
                model=embedding_model,
                endpoint=embedding_endpoint,
                dim=embedding_dim,
                threads=embedding_threads,
                timeout=embedding_timeout,
                retries=embedding_retries,
                num_batch=embedding_num_batch,
                query_prefix=embedding_query_prefix,
                passage_prefix=embedding_passage_prefix,
                api_key=embedding_api_key,
            ),
            embedding_batch,
        ),
        store_reader=store,
        store_writer=store,
        watcher=InMemoryWatcher(),
        summarize=build_summarizer(
            summary_provider,
            model=summary_model,
            endpoint=summary_endpoint,
            api_key=summary_api_key,
            timeout=summary_timeout,
            retries=summary_retries,
            prompt=summary_prompt,
            max_input_chars=summary_max_input_chars,
        ),
    )


@dataclass(frozen=True, slots=True)
class DatasetServices:
    """Frozen container of the port implementations for ONE named dataset.

    The MCP server holds one of these per configured dataset and dispatches a
    tool call (search / reindex) to the matching set. ``partition`` is the
    resolved value (the dataset's own, or the store-wide default) recorded for
    reporting; ``read_only`` gates the reindex tool (the database GRANT is the
    real enforcer).
    """

    name: str
    backend: StoreBackend
    # Stable identity of the embedding (provider|model|endpoint) so a fan-out
    # search embeds the query once per distinct model across datasets.
    model_key: str
    store: VectorStore
    embedding: EmbeddingProvider
    extract: Extract
    chunk: ChunkText
    connector: SourceConnector
    collection: str
    partition: Partition
    read_only: bool
    writable: bool
    # The opt-in per-document summarizer for this dataset, or None when off.
    summarize: SummaryProvider | None = None
    # Serializes writes to this dataset's store across threads: the MCP write
    # tools (reindex/remember/forget) run in the FastMCP worker thread while the
    # live watcher's auto-reindex runs in the watch thread; both take this lock.
    # Reads (search) stay lock-free (eventual consistency is acceptable). Excluded
    # from eq/repr so it never affects dataclass comparison or logging.
    lock: threading.RLock = field(default_factory=threading.RLock, compare=False, repr=False)


def _default_dataset_store_dir() -> Path:
    """Default embedded-store directory when a dataset sets no ``store_dir``."""
    from pathlib import Path

    return Path.home() / ".semdex"


def build_dataset_services(
    dataset: DatasetConfig,
    *,
    default_partition: Partition,
    ann_params: AnnParams | None = None,
    summary: SummaryConfig | None = None,
    health: HealthConfig | None = None,
) -> DatasetServices:
    """Wire one dataset's store, embedding, extractor, chunker, connector, and summarizer.

    Reuses the same factories the CLI index path uses (``build_vector_store`` /
    ``build_embedding`` / ``build_chunker``), so a dataset behaves exactly like a
    CLI-indexed collection. Source roots and the store dir expand ``~``. The
    extractor is a ``RoutingExtractor`` (net routing model per file = subtree x
    filetype x scan-detection) built from the dataset's ``extractor`` default +
    ``routes`` + ``extractor_config``, with ``build_extractor`` injected as the
    per-backend factory; the chunker is the CLI default (markdown-aware recursive,
    degrading to whitespace without ``semdex[chunk]``).

    The opt-in summary tier is built only when the dataset sets ``summarize`` AND
    the global ``[summary]`` config (``summary``) selects a provider other than
    ``NONE``; the dataset's ``summary_model``/``summary_endpoint`` override that
    global model/endpoint. Otherwise ``summarize`` is ``None`` (tier off).

    ``health`` (the ``[health]`` config) tunes the server-backed providers only:
    it passes ``keep_alive`` through to the ollama providers, wraps an http-backed
    embedding/summary provider in the restart-on-exhaustion self-heal when
    ``restart_command`` is set, and warms each http provider at build time (default
    on). When ``health`` is ``None`` the behaviour is identical to before.
    """
    from pathlib import Path

    from ..adapters.discovery.filesystem import FilesystemConnector
    from ..adapters.extractor.router import RoutingExtractor

    keep_alive = health.keep_alive if health else None
    partition = dataset.partition or default_partition
    store_dir = Path(dataset.store_dir).expanduser() if dataset.store_dir else _default_dataset_store_dir()
    store = build_vector_store(dataset.backend, store_dir, dsn=dataset.dsn, ann_params=ann_params)
    embedding = build_embedding(
        dataset.embedding_provider,
        model=dataset.embedding_model,
        endpoint=dataset.embedding_endpoint,
        timeout=dataset.embedding_timeout,
        num_batch=dataset.embedding_num_batch,
        query_prefix=dataset.embedding_query_prefix,
        passage_prefix=dataset.embedding_passage_prefix,
        api_key=dataset.embedding_api_key,
        keep_alive=keep_alive,
    )
    embedding = _resilient_embedding(embedding, dataset, health)
    roots = [Path(source).expanduser() for source in dataset.sources]
    connector = FilesystemConnector(roots)
    extract = RoutingExtractor(
        roots=roots,
        default_backend=dataset.extractor,
        routes=dataset.routes,
        endpoints={entry.backend: entry for entry in dataset.extractor_config},
        scan_min_chars=dataset.scan_min_chars,
        factory=build_extractor,
    )
    model_key = f"{dataset.embedding_provider.value}|{dataset.embedding_model or ''}|{dataset.embedding_endpoint or ''}"
    summarize = _build_dataset_summarizer(dataset, summary, keep_alive)
    summarize = _resilient_summary(summarize, dataset, summary, health)
    _warmup_dataset_providers(dataset, embedding, summarize, health)
    return DatasetServices(
        name=dataset.name,
        backend=dataset.backend,
        model_key=model_key,
        store=store,
        embedding=embedding,
        extract=extract,
        chunk=build_chunker(ChunkStrategy.RECURSIVE),
        connector=connector,
        collection=dataset.collection,
        partition=partition,
        read_only=dataset.read_only,
        writable=dataset.writable,
        summarize=summarize,
    )


def _build_dataset_summarizer(
    dataset: DatasetConfig, summary: SummaryConfig | None, keep_alive: str | None = None
) -> SummaryProvider | None:
    """Build the dataset's summarizer, or None when the tier is off for it.

    Gated on the dataset opting in (``summarize``) AND a global ``[summary]``
    provider other than ``NONE``. The dataset's ``summary_model`` /
    ``summary_endpoint`` override the global model/endpoint; the provider,
    api_key, timeout, prompt, and clip come from the global ``[summary]`` config;
    ``keep_alive`` comes from ``[health]`` (ollama only).
    """
    if not dataset.summarize or summary is None or summary.provider is SummaryBackend.NONE:
        return None
    return build_summarizer(
        summary.provider,
        model=dataset.summary_model or summary.model,
        endpoint=dataset.summary_endpoint or summary.endpoint,
        api_key=summary.api_key,
        timeout=summary.timeout,
        retries=summary.retries,
        prompt=summary.prompt,
        max_input_chars=summary.max_input_chars,
        keep_alive=keep_alive,
    )


def _is_http_embedding(provider: EmbeddingBackend) -> bool:
    """Whether the embedding provider is a server-backed one that can be restarted."""
    return provider in (EmbeddingBackend.OLLAMA, EmbeddingBackend.OPENAI)


def _resilient_embedding(
    embedding: EmbeddingProvider, dataset: DatasetConfig, health: HealthConfig | None
) -> EmbeddingProvider:
    """Wrap an http-backed embedding in the restart-on-exhaustion self-heal, if enabled.

    Only ollama/openai providers are wrapped (fastembed/model2vec/sentence-
    transformers/placeholder run in-process, so there is nothing to restart). A
    ``None`` ``health`` or an unset ``restart_command`` leaves the provider as-is.
    """
    if health is None or not health.restart_command or not _is_http_embedding(dataset.embedding_provider):
        return embedding
    from ..adapters.health import HttpHealthProbe, probe_url_for, restart_hook_from_command
    from ..application.resilience import ResilientEmbedding

    hook = restart_hook_from_command(health.restart_command, health.restart_timeout)
    probe = HttpHealthProbe(probe_url_for(dataset.embedding_provider.value, dataset.embedding_endpoint))
    return ResilientEmbedding(embedding, restart=hook, probe=probe, recover_timeout=health.recover_timeout)


def _resilient_summary(
    summarize: SummaryProvider | None,
    dataset: DatasetConfig,
    summary: SummaryConfig | None,
    health: HealthConfig | None,
) -> SummaryProvider | None:
    """Wrap the summarizer in the restart-on-exhaustion self-heal, if enabled.

    Applies only when the tier is on (``summarize`` is not None, so the provider is
    the http-backed ollama/openai one) and ``restart_command`` is set.
    """
    if summarize is None or summary is None or health is None or not health.restart_command:
        return summarize
    from ..adapters.health import HttpHealthProbe, probe_url_for, restart_hook_from_command
    from ..application.resilience import ResilientSummary

    hook = restart_hook_from_command(health.restart_command, health.restart_timeout)
    endpoint = dataset.summary_endpoint or summary.endpoint
    probe = HttpHealthProbe(probe_url_for(summary.provider.value, endpoint))
    return ResilientSummary(summarize, restart=hook, probe=probe, recover_timeout=health.recover_timeout)


def _warmup_dataset_providers(
    dataset: DatasetConfig,
    embedding: EmbeddingProvider,
    summarize: SummaryProvider | None,
    health: HealthConfig | None,
) -> None:
    """Warm each http-backed provider so the model is resident before the first query.

    Best-effort and a no-op for in-process providers (nothing to warm), so the
    default (``health`` None) path is unchanged for a fastembed/placeholder dataset.
    """
    if health is not None and not health.warmup:
        return
    from ..adapters.health import warmup_embedding, warmup_summary

    if _is_http_embedding(dataset.embedding_provider):
        warmup_embedding(embedding)
    if summarize is not None:
        warmup_summary(summarize)


def build_watcher(mode: WatchMode = WatchMode.AUTO, *, debounce_ms: int = 400) -> Watcher:
    """Build the production filesystem watcher (watchfiles-backed).

    ``mode`` selects the strategy (``POLL`` forces polling for network mounts);
    ``debounce_ms`` batches a burst of changes into one reconcile. The watchfiles
    import is deferred to ``start()``, so constructing a watcher never requires
    ``semdex[watch]`` - only starting one does.
    """
    from ..adapters.watch import WatchfilesWatcher

    return WatchfilesWatcher(mode=mode, debounce_ms=debounce_ms)


def build_embedding(
    provider: EmbeddingBackend,
    *,
    model: str | None = None,
    endpoint: str | None = None,
    dim: int | None = None,
    threads: int | None = None,
    timeout: float | None = None,
    retries: int | None = None,
    api_key: str | None = None,
    keep_alive: str | None = None,
    num_batch: int | None = None,
    query_prefix: str = "",
    passage_prefix: str = "",
    allow_fallback: bool = True,
) -> EmbeddingProvider:
    """Select the embedding provider for the configured backend.

    ``FASTEMBED`` is the default real provider; because it is the default and its
    first run may need a download, it degrades gracefully to the placeholder (with
    a warning) if it cannot load. The explicitly-chosen providers (model2vec,
    ollama, openai, gemini, cohere, sentence-transformers) fail fast so a
    misconfiguration is visible. ``model`` ``None`` uses each provider's own default
    model; ``dim`` tunes the placeholder only (a real provider reports its model's
    dimension). ``timeout`` (seconds), ``retries``, and ``api_key`` apply to the
    HTTP providers (ollama, openai) and the cloud providers (gemini, cohere, which
    REQUIRE ``api_key``); ``None`` keeps each loader's own default (no auth for the
    local ones, 60 s, 4 attempts). ``keep_alive`` is the ollama VRAM-residency
    window (ollama only; None keeps ollama's default). ``num_batch`` is ollama's
    physical batch size in tokens (ollama only; None keeps ollama's default of
    2048, which silently truncates any longer input - see ``load_ollama_embedding``).
    ``query_prefix``/``passage_prefix`` are the instruction prefixes for models
    trained to receive one; they reach ollama, openai and sentence-transformers,
    which send raw text to a server that applies no template of its own. They are
    REJECTED for fastembed, model2vec, gemini and cohere, which apply their own
    query/passage handling: accepting a prefix there would either double-apply it
    or silently ignore it, and a knob that is silently ignored is worse than one
    that is absent. ``allow_fallback`` (default
    ``True``) governs only the fastembed graceful degradation: a bulk-embed or
    benchmark caller sets it ``False`` so a missing model raises instead of silently
    caching placeholder vectors.
    """
    _reject_unsupported_prefix(provider, query_prefix, passage_prefix)
    if provider is EmbeddingBackend.PLACEHOLDER:
        return _placeholder_embedding(dim)
    if provider is EmbeddingBackend.FASTEMBED:
        from ..adapters.embedding import load_fastembed_embedding
        from ..domain.errors import EmbeddingError

        try:
            return load_fastembed_embedding(model, threads=threads)
        except EmbeddingError as exc:
            if not allow_fallback:
                raise
            logger.warning("fastembed unavailable (%s); falling back to the placeholder embedding", exc)
            return _placeholder_embedding(dim)
    if provider is EmbeddingBackend.MODEL2VEC:
        from ..adapters.embedding import load_model2vec_embedding

        return load_model2vec_embedding(model)
    server_providers = (
        EmbeddingBackend.OLLAMA,
        EmbeddingBackend.OPENAI,
        EmbeddingBackend.GEMINI,
        EmbeddingBackend.COHERE,
    )
    if provider in server_providers:
        return _build_server_embedding(
            provider,
            model=model,
            endpoint=endpoint,
            timeout=timeout,
            retries=retries,
            api_key=api_key,
            keep_alive=keep_alive,
            num_batch=num_batch,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
        )
    from ..adapters.embedding import load_sentence_transformer_embedding

    st_kwargs: dict[str, Any] = {"query_prefix": query_prefix, "passage_prefix": passage_prefix}
    return (
        load_sentence_transformer_embedding(model, **st_kwargs)
        if model
        else load_sentence_transformer_embedding(**st_kwargs)
    )


# Providers that apply their own query/passage handling: fastembed calls the model's
# own query_embed, model2vec is a static model with no notion of one, and gemini and
# cohere pass a native task type to their API. A prefix here would double-apply or
# vanish, so it is refused rather than accepted and dropped.
_PREFIX_CAPABLE = (
    EmbeddingBackend.OLLAMA,
    EmbeddingBackend.OPENAI,
    EmbeddingBackend.SENTENCE_TRANSFORMERS,
)


def _reject_unsupported_prefix(provider: EmbeddingBackend, query_prefix: str, passage_prefix: str) -> None:
    """Fail loudly when a prefix is set for a provider that cannot honour it.

    The alternative is the defect this whole change exists to fix: a knob that is
    declared, documented and resolved, then never read - which reads as configured
    and behaves as absent.
    """
    if not (query_prefix or passage_prefix) or provider in _PREFIX_CAPABLE:
        return
    raise ConfigurationError(
        f"[embedding].query_prefix/passage_prefix cannot be applied by provider '{provider.value}', "
        f"which applies its own query/passage handling. Remove the prefix, or use one of: "
        f"{', '.join(p.value for p in _PREFIX_CAPABLE)}."
    )


def _build_server_embedding(
    provider: EmbeddingBackend,
    *,
    model: str | None,
    endpoint: str | None,
    timeout: float | None,
    retries: int | None,
    api_key: str | None,
    keep_alive: str | None,
    num_batch: int | None = None,
    query_prefix: str = "",
    passage_prefix: str = "",
) -> EmbeddingProvider:
    """Route to the http-backed (ollama/openai) or native-cloud (gemini/cohere) loader.

    The two families share the same ``endpoint``/``timeout``/``retries`` knobs but
    speak different wire protocols, so they build through separate helpers; this
    dispatch keeps ``build_embedding`` flat. ``keep_alive`` reaches only ollama.
    """
    if provider in (EmbeddingBackend.OLLAMA, EmbeddingBackend.OPENAI):
        return _build_http_embedding(
            provider,
            model=model,
            endpoint=endpoint,
            timeout=timeout,
            retries=retries,
            api_key=api_key,
            keep_alive=keep_alive,
            num_batch=num_batch,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
        )
    return _build_cloud_embedding(
        provider, model=model, endpoint=endpoint, timeout=timeout, retries=retries, api_key=api_key
    )


def _build_http_embedding(
    provider: EmbeddingBackend,
    *,
    model: str | None,
    endpoint: str | None,
    timeout: float | None,
    retries: int | None,
    api_key: str | None,
    keep_alive: str | None = None,
    num_batch: int | None = None,
    query_prefix: str = "",
    passage_prefix: str = "",
) -> EmbeddingProvider:
    """Load a server-backed embedding provider (ollama ``/api/embed`` or openai ``/v1``).

    Both share the ``endpoint``/``timeout``/``retries`` knobs; only openai uses
    ``api_key`` and only ollama uses ``keep_alive``. A ``None`` timeout/retries
    keeps each loader's own default (60 s, 4 attempts), so each is passed through
    only when set.
    """
    http_kwargs: dict[str, Any] = {"query_prefix": query_prefix, "passage_prefix": passage_prefix}
    if timeout is not None:
        http_kwargs["timeout"] = timeout
    if retries is not None:
        http_kwargs["retries"] = retries
    if provider is EmbeddingBackend.OLLAMA:
        from ..adapters.embedding import load_ollama_embedding

        if num_batch is not None:
            http_kwargs["num_batch"] = num_batch
        return load_ollama_embedding(model, endpoint=endpoint, keep_alive=keep_alive, **http_kwargs)
    from ..adapters.embedding import load_openai_embedding

    return load_openai_embedding(model, endpoint=endpoint, api_key=api_key, **http_kwargs)


def _build_cloud_embedding(
    provider: EmbeddingBackend,
    *,
    model: str | None,
    endpoint: str | None,
    timeout: float | None,
    retries: int | None,
    api_key: str | None,
) -> EmbeddingProvider:
    """Load a native cloud embedding provider (gemini batchEmbedContents / cohere /v2/embed).

    Both are paid managed APIs behind their own REST schema (NOT OpenAI-compatible,
    so they do not route through :func:`_build_http_embedding`) and both REQUIRE an
    ``api_key`` (env-only). They share the ``endpoint``/``timeout``/``retries``
    knobs and take no ``keep_alive`` (nothing to keep resident). A ``None``
    timeout/retries keeps each loader's own default (60 s, 4 attempts), so each is
    passed through only when set.
    """
    http_kwargs: dict[str, Any] = {}
    if timeout is not None:
        http_kwargs["timeout"] = timeout
    if retries is not None:
        http_kwargs["retries"] = retries
    if provider is EmbeddingBackend.GEMINI:
        from ..adapters.embedding import load_gemini_embedding

        return load_gemini_embedding(model, endpoint=endpoint, api_key=api_key, **http_kwargs)
    from ..adapters.embedding import load_cohere_embedding

    return load_cohere_embedding(model, endpoint=endpoint, api_key=api_key, **http_kwargs)


def _placeholder_embedding(dim: int | None) -> EmbeddingProvider:
    """The deterministic, dependency-free token-hash provider (dim tunable)."""
    from ..adapters.memory.index import InMemoryEmbeddingProvider

    return InMemoryEmbeddingProvider(dim=dim) if dim is not None else InMemoryEmbeddingProvider()


def build_summarizer(
    provider: SummaryBackend,
    *,
    model: str | None = None,
    endpoint: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
    retries: int | None = None,
    prompt: str | None = None,
    max_input_chars: int | None = None,
    keep_alive: str | None = None,
) -> SummaryProvider | None:
    """Select the LLM summarizer for the configured backend (the opt-in tier).

    ``NONE`` returns ``None`` - the tier is off, and every hit keeps
    ``summary=None``. ``OLLAMA`` and ``OPENAI`` are server-backed and share the
    ``endpoint``/``timeout``/``retries``/``prompt``/``max_input_chars`` knobs; only
    openai uses ``api_key`` and only ollama uses ``keep_alive``. ``model`` ``None``
    uses each provider's own default chat model. The adapter package is imported
    lazily so a NONE tier never needs ``semdex[summary]``.
    """
    if provider is SummaryBackend.NONE:
        return None
    return _build_http_summarizer(
        provider,
        model=model,
        endpoint=endpoint,
        api_key=api_key,
        timeout=timeout,
        retries=retries,
        prompt=prompt,
        max_input_chars=max_input_chars,
        keep_alive=keep_alive,
    )


def _build_http_summarizer(
    provider: SummaryBackend,
    *,
    model: str | None,
    endpoint: str | None,
    api_key: str | None,
    timeout: float | None,
    retries: int | None,
    prompt: str | None,
    max_input_chars: int | None,
    keep_alive: str | None = None,
) -> SummaryProvider:
    """Load a server-backed summarizer (ollama ``/api/chat`` or openai ``/v1``).

    Both share ``endpoint``/``timeout``/``retries``/``prompt``/``max_input_chars``;
    only openai uses ``api_key`` and only ollama uses ``keep_alive``. ``None``
    values keep each loader's own default (60 s timeout, 4 retry attempts, the
    built-in triage prompt, an 8000-char clip), so they are passed through only
    when set.
    """
    # Each loader has its own defaults for the unset optional knobs; pass a value
    # only when set so None never overrides a loader default with a wrong type.
    resolved_timeout = timeout if timeout is not None else 60.0
    resolved_prompt = prompt  # None -> the loader's built-in triage prompt
    resolved_clip = max_input_chars if max_input_chars is not None else 8000
    http_kwargs: dict[str, Any] = {}
    if retries is not None:
        http_kwargs["retries"] = retries
    if provider is SummaryBackend.OLLAMA:
        from ..adapters.summarizer import load_ollama_summarizer

        return load_ollama_summarizer(
            model,
            endpoint=endpoint,
            timeout=resolved_timeout,
            keep_alive=keep_alive,
            prompt=resolved_prompt,
            max_input_chars=resolved_clip,
            **http_kwargs,
        )
    from ..adapters.summarizer import load_openai_summarizer

    return load_openai_summarizer(
        model,
        endpoint=endpoint,
        timeout=resolved_timeout,
        api_key=api_key,
        prompt=resolved_prompt,
        max_input_chars=resolved_clip,
        **http_kwargs,
    )


class IndexServicesFactory(Protocol):
    """Builds an IndexServices container for a store directory and tunable config."""

    def __call__(
        self,
        store_dir: Path,
        *,
        embedding_provider: EmbeddingBackend = ...,
        embedding_model: str | None = ...,
        embedding_endpoint: str | None = ...,
        embedding_threads: int | None = ...,
        embedding_batch: int | None = ...,
        embedding_timeout: float | None = ...,
        embedding_retries: int | None = ...,
        embedding_num_batch: int | None = ...,
        embedding_query_prefix: str = ...,
        embedding_passage_prefix: str = ...,
        embedding_api_key: str | None = ...,
        max_file_bytes: int | None = ...,
        embedding_dim: int | None = ...,
        backend: StoreBackend = ...,
        dsn: str | None = ...,
        lance_index_threshold: int | None = ...,
        ann_params: AnnParams | None = ...,
        extractor_backend: ExtractorBackend = ...,
        extractor_endpoint: str | None = ...,
        extractor_timeout: float = ...,
        extractor_max_bytes: int | None = ...,
        extractor_force_ocr: bool = ...,
        extractor_ocr_language: str | None = ...,
        sanitize: SanitizeConfig | None = ...,
        chunker_strategy: ChunkStrategy = ...,
        chunker_recipe: str = ...,
        chunk_overlap: int = ...,
        chunker_tokenizer: str = ...,
        chunker_semantic_model: str | None = ...,
        chunker_enforce_max_tokens: bool = ...,
        summary_provider: SummaryBackend = ...,
        summary_model: str | None = ...,
        summary_endpoint: str | None = ...,
        summary_api_key: str | None = ...,
        summary_timeout: float | None = ...,
        summary_retries: int | None = ...,
        summary_prompt: str | None = ...,
        summary_max_input_chars: int | None = ...,
    ) -> IndexServices: ...


class DatasetServicesFactory(Protocol):
    """Builds a DatasetServices bundle for one dataset binding.

    Injected into the CLI (via :class:`Bootstrap`) so the ``serve`` command wires
    a dataset through the composition root without an adapter importing
    composition (which would break the layer contract).
    """

    def __call__(
        self,
        dataset: DatasetConfig,
        *,
        default_partition: Partition,
        ann_params: AnnParams | None = ...,
        summary: SummaryConfig | None = ...,
        health: HealthConfig | None = ...,
    ) -> DatasetServices: ...


class WatcherFactory(Protocol):
    """Builds a filesystem Watcher for the ``serve`` command's live-watch loop.

    Injected (via :class:`Bootstrap`) so ``serve`` starts the watcher through the
    composition root without an adapter importing composition.
    """

    def __call__(self, mode: WatchMode = ..., *, debounce_ms: int = ...) -> Watcher: ...


@dataclass(frozen=True, slots=True)
class Bootstrap:
    """Injected CLI bootstrap: the AppServices factory plus the index/dataset factories.

    Callable so it drops in wherever the plain ``services_factory`` was expected
    (``ctx.obj()`` still yields AppServices); the root callback additionally
    reads ``index_services_factory`` (index commands) and
    ``dataset_services_factory`` (the ``serve`` command) off it. Keeping it a
    callable preserves every existing services_factory injection site.
    """

    services_factory: Callable[[], AppServices]
    index_services_factory: IndexServicesFactory
    dataset_services_factory: DatasetServicesFactory = build_dataset_services
    watcher_factory: WatcherFactory = build_watcher

    def __call__(self) -> AppServices:
        return self.services_factory()


__all__ = [
    # Configuration
    "get_config",
    "get_default_config_path",
    "deploy_configuration",
    "display_config",
    # Email
    "send_email",
    "send_notification",
    "load_email_config_from_dict",
    # Logging
    "init_logging",
    # Composition
    "AppServices",
    "Bootstrap",
    "DatasetServices",
    "DatasetServicesFactory",
    "IndexServices",
    "IndexServicesFactory",
    "WatcherFactory",
    "build_chunker",
    "build_dataset_services",
    "build_embedding",
    "build_extractor",
    "build_index_production",
    "build_index_testing",
    "build_summarizer",
    "build_vector_store",
    "build_watcher",
    "build_production",
    "build_testing",
]
