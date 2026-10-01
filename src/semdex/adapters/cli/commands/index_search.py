"""Semantic index CLI commands: index and search.

Thin adapters: resolve the vector-store directory, build the index services via
the injected factory, and delegate to the application use cases. All indexing
and ranking logic lives in the use cases, not here.

Contents:
    * :func:`cli_index` - Index files/directories into a collection.
    * :func:`cli_search` - Search a collection for a query.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import lib_log_rich.runtime
import rich_click as click

from semdex.adapters.config.chunker import get_chunker_config
from semdex.adapters.config.embedding import get_embedding_config
from semdex.adapters.config.extractor import get_extractor_config
from semdex.adapters.config.index import get_index_config
from semdex.adapters.config.sanitize import get_sanitize_config
from semdex.adapters.config.summary import get_summary_config
from semdex.adapters.config.vectorstore import get_vector_store_config
from semdex.adapters.discovery import discover_sources
from semdex.application.use_cases import index_sources, search
from semdex.domain.errors import VectorStoreError
from semdex.domain.models import CompactionPolicy

from ..constants import CLICK_CONTEXT_SETTINGS
from ..context import CLIContext, get_cli_context
from ..exit_codes import ExitCode
from ..typed_click import argument, option

if TYPE_CHECKING:
    from semdex.adapters.config.index import IndexConfig
    from semdex.composition import IndexServices

    from ..context import IndexServicesFactory

logger = logging.getLogger(__name__)


def _default_store_dir() -> Path:
    """Per-tool store directory, outside any indexed tree."""
    return Path.home() / ".semdex"


def _require_index_factory(cli_ctx: CLIContext) -> IndexServicesFactory:
    """Return the injected index-services factory or fail with a clear message."""
    factory = cli_ctx.index_services_factory
    if factory is None:
        click.echo("Error: this build has no index backend wired.", err=True)
        raise SystemExit(ExitCode.CONFIG_ERROR)
    return factory


def _resolve(
    cli_ctx: CLIContext,
    *,
    store_dir: Path | None,
    collection: str | None,
    embedding_model: str | None,
) -> tuple[IndexConfig, Path, str, str | None]:
    """Resolve shared options as CLI flag > [index] config > built-in default."""
    idx = get_index_config(cli_ctx.config)
    resolved_store = store_dir or (Path(idx.store_dir) if idx.store_dir else _default_store_dir())
    return idx, resolved_store, collection or idx.collection, embedding_model or idx.embedding_model


def _build_services(
    cli_ctx: CLIContext,
    factory: IndexServicesFactory,
    *,
    store: Path,
    model: str | None,
    idx: IndexConfig,
) -> IndexServices:
    """Build index services, threading the [embedding], [chunker], [vector_store] and [extractor] config."""
    vs = get_vector_store_config(cli_ctx.config)
    ex = get_extractor_config(cli_ctx.config)
    em = get_embedding_config(cli_ctx.config)
    ch = get_chunker_config(cli_ctx.config)
    sn = get_sanitize_config(cli_ctx.config)
    sm = get_summary_config(cli_ctx.config)
    return factory(
        store,
        embedding_provider=em.provider,
        embedding_model=model if model is not None else em.model,
        embedding_endpoint=em.endpoint,
        embedding_threads=em.threads,
        embedding_batch=em.batch,
        embedding_timeout=em.timeout,
        embedding_retries=em.retries,
        embedding_num_batch=em.num_batch,
        embedding_query_prefix=em.query_prefix,
        embedding_passage_prefix=em.passage_prefix,
        embedding_api_key=em.api_key,
        max_file_bytes=idx.max_file_bytes,
        embedding_dim=idx.embedding_dim,
        backend=vs.backend,
        dsn=vs.dsn,
        lance_index_threshold=vs.lance_index_threshold,
        ann_params=vs.resolved_ann_params(vs.backend),
        extractor_backend=ex.backend,
        extractor_endpoint=ex.endpoint,
        extractor_timeout=ex.timeout,
        extractor_max_bytes=ex.max_file_bytes,
        extractor_force_ocr=ex.force_ocr,
        extractor_ocr_language=ex.ocr_language,
        sanitize=sn,
        chunker_strategy=ch.strategy,
        chunker_recipe=ch.recipe,
        chunk_overlap=ch.chunk_overlap,
        chunker_tokenizer=ch.tokenizer,
        chunker_semantic_model=ch.semantic_model,
        chunker_enforce_max_tokens=ch.enforce_max_tokens,
        summary_provider=sm.provider,
        summary_model=sm.model,
        summary_endpoint=sm.endpoint,
        summary_api_key=sm.api_key,
        summary_timeout=sm.timeout,
        summary_retries=sm.retries,
        summary_prompt=sm.prompt,
        summary_max_input_chars=sm.max_input_chars,
    )


def _snippet(text: str, max_chars: int) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= max_chars else collapsed[:max_chars] + "..."


@click.command("index", context_settings=CLICK_CONTEXT_SETTINGS)
@argument("paths", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@option("--collection", default=None, help="Target collection name (default: 'default' or [index] config)")
@option("--label", default=None, help="Free-form provenance label to tag indexed sources")
@option(
    "--store-dir",
    "store_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Vector store directory (default: ~/.semdex or [index] config)",
)
@option(
    "--embedding-model",
    "embedding_model",
    default=None,
    help="Semantic embedding model id (needs semdex[embed]); default: token placeholder (search must match).",
)
@click.pass_context
def cli_index(
    ctx: click.Context,
    *,
    paths: tuple[Path, ...],
    collection: str | None,
    label: str | None,
    store_dir: Path | None,
    embedding_model: str | None,
) -> None:
    """Index one or more files/directories into a collection.

    Discovers text/markdown files under each PATH, extracts and chunks them,
    embeds the chunks, and writes them to the persistent store. Reindexing the
    same sources replaces their chunks rather than duplicating them.
    """
    cli_ctx = get_cli_context(ctx)
    factory = _require_index_factory(cli_ctx)
    idx, store, target, model = _resolve(
        cli_ctx, store_dir=store_dir, collection=collection, embedding_model=embedding_model
    )
    resolved_label = label if label is not None else idx.default_label
    with lib_log_rich.runtime.bind(job_id="cli-index", extra={"command": "index", "collection": target}):
        logger.info("Indexing sources", extra={"collection": target, "store_dir": str(store)})
        services = _build_services(cli_ctx, factory, store=store, model=model, idx=idx)
        sources = discover_sources(
            paths, label=resolved_label, extensions=tuple(idx.extensions), hash_chunk_size=idx.hash_chunk_bytes
        )
        vs = get_vector_store_config(cli_ctx.config)
        report = index_sources(
            extract=services.extract,
            chunk=services.chunk,
            embedding=services.embedding,
            store=services.store_writer,
            collection=target,
            sources=sources,
            max_tokens=idx.max_tokens,
            compaction=CompactionPolicy(after_records=vs.compact_after_records, after_seconds=vs.compact_after_seconds),
            summarize=services.summarize,
        )
    click.echo(
        f"Indexed {report.sources_indexed} source(s), {report.chunks_indexed} chunk(s) into '{target}' at {store}."
    )


@click.command("search", context_settings=CLICK_CONTEXT_SETTINGS)
@argument("query")
@option("--collection", default=None, help="Collection to search (default: 'default' or [index] config)")
@option("--k", "top_k", type=int, default=None, help="Maximum number of hits (default: 5 or [index] config)")
@option(
    "--store-dir",
    "store_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Vector store directory (default: ~/.semdex or [index] config)",
)
@option(
    "--embedding-model",
    "embedding_model",
    default=None,
    help="Embedding model id used at index time (must match); default: token placeholder.",
)
@click.pass_context
def cli_search(
    ctx: click.Context,
    *,
    query: str,
    collection: str | None,
    top_k: int | None,
    store_dir: Path | None,
    embedding_model: str | None,
) -> None:
    """Search a collection and print ranked matches (score, path, snippet)."""
    cli_ctx = get_cli_context(ctx)
    factory = _require_index_factory(cli_ctx)
    idx, store, target, model = _resolve(
        cli_ctx, store_dir=store_dir, collection=collection, embedding_model=embedding_model
    )
    resolved_k = top_k if top_k is not None else idx.default_k
    with lib_log_rich.runtime.bind(job_id="cli-search", extra={"command": "search", "collection": target}):
        logger.info("Searching", extra={"collection": target, "store_dir": str(store)})
        services = _build_services(cli_ctx, factory, store=store, model=model, idx=idx)
        try:
            hits = search(
                embedding=services.embedding,
                store=services.store_reader,
                collection=target,
                query=query,
                k=resolved_k,
            )
        except VectorStoreError as exc:
            click.echo(f"Error: {exc}. Run 'semdex index' first.", err=True)
            raise SystemExit(ExitCode.GENERAL_ERROR) from exc
    if not hits:
        click.echo("No matches.")
        return
    for hit in hits:
        click.echo(f"{hit.score:.3f}  {hit.uri}  {_snippet(hit.chunk_text, idx.snippet_chars)}")


__all__ = ["cli_index", "cli_search"]
