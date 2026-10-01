"""`serve` CLI command: run the MCP server over stdio or Streamable HTTP.

A thin adapter: resolve ``[mcp]`` + ``[[dataset]]`` config (CLI flags override the
transport/host/port), build one :class:`DatasetServices` per dataset, assemble the
FastMCP server, and run it. All tool logic lives in the MCP server and the use
cases, not here. Each dataset's store is closed when the server stops.

Contents:
    * :func:`cli_serve` - Run the MCP server.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import lib_log_rich.runtime
import rich_click as click

from semdex.adapters.config.dataset import get_datasets
from semdex.adapters.config.embedding import get_embedding_config
from semdex.adapters.config.health import get_health_config
from semdex.adapters.config.mcp import get_mcp_config
from semdex.adapters.config.summary import get_summary_config
from semdex.adapters.config.vectorstore import get_vector_store_config
from semdex.adapters.config.watch import get_watch_config
from semdex.adapters.health import HealthLoop, HttpHealthProbe, probe_url_for, restart_hook_from_command
from semdex.adapters.mcp.auth import build_mcp_auth
from semdex.adapters.mcp.server import build_mcp_server
from semdex.application.use_cases.reconciling import reconcile
from semdex.domain.enums import EmbeddingBackend, McpTransport, SummaryBackend
from semdex.domain.errors import ConfigurationError, WatchError

from ..constants import CLICK_CONTEXT_SETTINGS
from ..context import get_cli_context
from ..exit_codes import ExitCode
from ..typed_click import option

if TYPE_CHECKING:
    from collections.abc import Callable

    from lib_layered_config import Config

    from semdex.adapters.config.health import HealthConfig
    from semdex.application.ports import HealthProbe, RestartHook, Watcher
    from semdex.composition import DatasetServices, DatasetServicesFactory
    from semdex.domain.models import ChangeEvent

    from ..context import CLIContext

logger = logging.getLogger(__name__)


def _require_dataset_factory(cli_ctx: CLIContext) -> DatasetServicesFactory:
    """Return the injected dataset-services factory or fail with a clear message.

    The factory comes from the composition-built Bootstrap; an adapter must not
    import composition directly (that would break the layer contract), so serve
    receives it through the CLI context.
    """
    factory = cli_ctx.dataset_services_factory
    if factory is None:
        click.echo("Error: this build has no dataset backend wired.", err=True)
        raise SystemExit(ExitCode.CONFIG_ERROR)
    return factory


@click.command("serve", context_settings=CLICK_CONTEXT_SETTINGS)
@option(
    "--transport",
    "transport",
    type=click.Choice([member.value for member in McpTransport]),
    default=None,
    help="Transport override (default: [mcp].transport, i.e. stdio).",
)
@option("--host", "host", default=None, help="HTTP bind host override (http only; default: [mcp].host).")
@option("--port", "port", type=int, default=None, help="HTTP bind port override (http only; default: [mcp].port).")
@click.pass_context
def cli_serve(
    ctx: click.Context,
    transport: str | None,
    host: str | None,
    port: int | None,
) -> None:
    """Run the semdex MCP server exposing list_datasets/search/reindex tools.

    stdio (the default) talks to a client that spawned this process; http serves
    Streamable HTTP on host:port for network clients. The datasets served come
    from the ``[[dataset]]`` config; auth and transport come from ``[mcp]``.
    """
    cli_ctx = get_cli_context(ctx)
    factory = _require_dataset_factory(cli_ctx)
    config = cli_ctx.config
    mcp_cfg = get_mcp_config(config)
    resolved_transport = McpTransport(transport) if transport is not None else mcp_cfg.transport
    resolved_host = host if host is not None else mcp_cfg.host
    resolved_port = port if port is not None else mcp_cfg.port

    try:
        vector_store = get_vector_store_config(config)
        summary = get_summary_config(config)
        health = get_health_config(config)
        services = [
            factory(
                dataset,
                default_partition=vector_store.default_partition,
                ann_params=vector_store.resolved_ann_params(dataset.backend, dataset.ann_recall),
                summary=summary,
                health=health,
            )
            for dataset in get_datasets(config)
        ]
        auth = build_mcp_auth(mcp_cfg, env=os.environ)
        server = build_mcp_server(
            services, auth=auth, rrf_k=mcp_cfg.rrf_k, mask_error_details=mcp_cfg.mask_error_details
        )
    except ConfigurationError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(ExitCode.CONFIG_ERROR) from exc

    watchers = _start_watchers(cli_ctx, config, services)
    health_loop = _start_health_loop(config, health)

    with lib_log_rich.runtime.bind(
        job_id="cli-serve",
        extra={"command": "serve", "transport": resolved_transport.value, "datasets": len(services)},
    ):
        logger.info(
            "Starting MCP server",
            extra={"transport": resolved_transport.value, "datasets": len(services), "watchers": len(watchers)},
        )
        try:
            if resolved_transport is McpTransport.HTTP:
                server.run(transport="http", host=resolved_host, port=resolved_port)
            else:
                server.run(transport="stdio")
        finally:
            if health_loop is not None:
                health_loop.stop()
            for watcher in watchers:
                watcher.stop()
            _close_stores(services)


def _start_watchers(cli_ctx: CLIContext, config: Config, services: list[DatasetServices]) -> list[Watcher]:
    """Start a live watcher per filesystem source dataset (auto-reindex on change).

    Watches a ``[[dataset]]`` that has ``sources`` and is neither ``writable`` (a
    client-written knowledge dataset) nor ``read_only`` (opts out of server-side
    reindex). Disabled entirely by ``[watch].enabled = false`` or when no watcher
    factory is wired. A watcher that cannot start (missing ``semdex[watch]``, or a
    root that does not exist) is skipped with a warning, not fatal.
    """
    watch_cfg = get_watch_config(config)
    watcher_factory = cli_ctx.watcher_factory
    if not watch_cfg.enabled or watcher_factory is None:
        return []
    watchers: list[Watcher] = []
    for dataset, svc in zip(get_datasets(config), services, strict=True):
        if not dataset.sources or dataset.writable or dataset.read_only:
            continue
        roots = [Path(source).expanduser() for source in dataset.sources]
        watcher = watcher_factory(watch_cfg.mode, debounce_ms=watch_cfg.debounce_ms)
        try:
            watcher.start(roots=roots, on_change=_reconcile_callback(svc))
        except WatchError as exc:
            logger.warning("watch disabled for dataset %s: %s", svc.name, exc)
            continue
        watchers.append(watcher)
        logger.info("Watching dataset %s (%d root(s))", svc.name, len(roots))
    return watchers


def _reconcile_callback(svc: DatasetServices) -> Callable[[ChangeEvent], None]:
    """Build the on-change callback that reconciles ``svc`` under its write lock."""

    def _on_change(_event: ChangeEvent) -> None:
        with svc.lock:
            try:
                reconcile(
                    connector=svc.connector,
                    extract=svc.extract,
                    chunk=svc.chunk,
                    embedding=svc.embedding,
                    store=svc.store,
                    collection=svc.collection,
                    summarize=svc.summarize,
                )
            except Exception:  # a bad file must not kill the watcher; log and keep watching
                logger.exception("auto-reindex failed for dataset %s", svc.name)

    return _on_change


def _start_health_loop(config: Config, health: HealthConfig) -> HealthLoop | None:
    """Start the opt-in active health-check loop, or return None when it is off.

    Runs only when ``[health].check_enabled``. Probes the configured (global)
    embedding and summary endpoints; with ``restart_command`` set each down
    endpoint is restarted, otherwise the loop only detects and logs (hook=None) and
    a warning notes it cannot restart. In-process embedding providers (fastembed
    etc.) expose no endpoint, so they are skipped.
    """
    if not health.check_enabled:
        return None
    if not health.restart_command:
        logger.warning("[health].check_enabled is set without a restart_command; the loop can detect but not restart")
    checks = _health_check_pairs(config, health)
    if not checks:
        logger.warning("[health].check_enabled is set but no server-backed embedding/summary endpoint is configured")
        return None
    loop = HealthLoop(checks, interval=health.check_interval, recover_timeout=health.recover_timeout)
    loop.start()
    logger.info("Started health-check loop over %d endpoint(s)", len(checks))
    return loop


def _health_check_pairs(config: Config, health: HealthConfig) -> list[tuple[HealthProbe, RestartHook | None]]:
    """Build the distinct (probe, hook) pairs for the configured http endpoints.

    Derives one probe per server-backed embedding/summary endpoint (deduplicated by
    URL, so a shared ollama host is probed once); every pair shares the one restart
    hook, or None when no ``restart_command`` is set.
    """
    hook = restart_hook_from_command(health.restart_command, health.restart_timeout)
    em = get_embedding_config(config)
    sm = get_summary_config(config)
    urls: dict[str, HealthProbe] = {}
    if em.provider in (EmbeddingBackend.OLLAMA, EmbeddingBackend.OPENAI):
        url = probe_url_for(em.provider.value, em.endpoint)
        urls.setdefault(url, HttpHealthProbe(url))
    if sm.provider in (SummaryBackend.OLLAMA, SummaryBackend.OPENAI):
        url = probe_url_for(sm.provider.value, sm.endpoint)
        urls.setdefault(url, HttpHealthProbe(url))
    return [(probe, hook) for probe in urls.values()]


def _close_stores(services: list[DatasetServices]) -> None:
    """Close each dataset's store when the server stops.

    ``close()`` is part of the VectorStore port: server backends close their DB
    connection, the sqlite store releases its file lock, embedded/json/in-memory
    stores are a no-op.
    """
    for svc in services:
        svc.store.close()


__all__ = ["cli_serve"]
