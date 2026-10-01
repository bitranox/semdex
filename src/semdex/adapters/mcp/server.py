"""Assemble the semdex MCP server: the tools over the search / reindex / write use cases.

The server is a delivery surface (a sibling of the CLI). It holds one
:class:`DatasetServices` per configured dataset, indexed by name, and exposes:

- ``list_datasets`` - the datasets this server serves (safe metadata only; never
  the dsn / secrets), including each dataset's ``writable`` flag.
- ``search`` - top-k semantic hits within one dataset.
- ``search_datasets`` - fan-out over several datasets, fused by Reciprocal Rank Fusion.
- ``reindex`` - reconcile one dataset to its source connector's current listing
  (rejected for a read-only or writable dataset).
- ``remember`` / ``forget`` - write / delete client-supplied knowledge in a
  ``writable`` dataset (rejected for a source-mirrored dataset).

It is source-type-blind: hits carry the opaque ``uri`` through unchanged.

``fastmcp`` is the optional ``mcp-server`` extra, imported lazily so importing this
module without the extra fails only when the server is actually built.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from ...application.use_cases.reconciling import reconcile
from ...application.use_cases.searching import DatasetSearchTarget, search, search_across_datasets
from ...application.use_cases.writing import forget as _forget_knowledge
from ...application.use_cases.writing import remember as _remember_knowledge
from .schemas import DatasetInfo, ForgetResult, FusedSearchHit, ReindexResult, RememberResult, SearchHit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fastmcp import FastMCP
    from fastmcp.server.auth import AuthProvider

    from ...composition import DatasetServices
    from ...domain.models import Hit

# Sentinel dataset name for the fan-out tool: omit / empty / this token selects all.
_ALL_KEYWORD = "all"


def build_mcp_server(
    datasets: Sequence[DatasetServices],
    *,
    auth: AuthProvider | None = None,
    name: str = "semdex",
    rrf_k: int = 60,
    mask_error_details: bool = True,
) -> FastMCP:
    """Build a FastMCP server exposing the datasets as list/search/reindex/fan-out tools.

    Args:
        datasets: The per-dataset service bundles to serve.
        auth: A FastMCP auth provider (from ``build_mcp_auth``) or ``None``.
        name: The server name advertised to clients.
        rrf_k: The Reciprocal Rank Fusion constant used by the ``search_datasets``
            fan-out tool (from ``[mcp].rrf_k``).
        mask_error_details: When ``True`` (the default), an unexpected exception
            raised inside a tool reaches the client as a generic error, never the
            original message/traceback; a deliberately raised ``ToolError`` (our
            own dataset-not-found / read-only checks) always reaches the client
            verbatim regardless of this flag - FastMCP's own distinction between
            "an error we chose to report" and "an unhandled exception". From
            ``[mcp].mask_error_details``.

    Returns:
        A configured ``FastMCP`` instance (not yet running).

    Raises:
        ConfigurationError: When ``fastmcp`` is not installed.
    """
    fastmcp_cls = _import_fastmcp()
    by_name = {svc.name: svc for svc in datasets}
    mcp = fastmcp_cls(name, auth=auth, mask_error_details=mask_error_details)

    def _dataset(dataset: str) -> DatasetServices:
        svc = by_name.get(dataset)
        if svc is None:
            known = ", ".join(sorted(by_name)) or "(none configured)"
            raise _tool_error(f"unknown dataset '{dataset}'; known datasets: {known}")
        return svc

    def _select(names: list[str] | None) -> list[DatasetServices]:
        # Omitted / empty / ["all"] (case-insensitive) selects every dataset;
        # otherwise a named subset (unknown name -> clear ToolError via _dataset).
        if not names or any(candidate.strip().lower() == _ALL_KEYWORD for candidate in names):
            return list(datasets)
        return [_dataset(candidate) for candidate in names]

    # The @mcp.tool decorator registers each function on `mcp`; the name is not
    # called directly afterwards, so pyright's reportUnusedFunction is a false
    # positive on these closures.
    @mcp.tool
    def list_datasets() -> list[DatasetInfo]:  # pyright: ignore[reportUnusedFunction]
        """List the datasets this server serves (name, backend, collection, partition, read_only, writable)."""
        return [
            DatasetInfo(
                name=svc.name,
                backend=svc.backend,
                collection=svc.collection,
                partition=svc.partition,
                read_only=svc.read_only,
                writable=svc.writable,
            )
            for svc in datasets
        ]

    @mcp.tool
    def search(dataset: str, query: str, k: int = 5) -> list[SearchHit]:  # pyright: ignore[reportUnusedFunction]
        """Return the top-k semantic hits for ``query`` within one dataset."""
        svc = _dataset(dataset)
        return [
            SearchHit(
                uri=hit.uri,
                ordinal=hit.ordinal,
                score=hit.score,
                text=hit.chunk_text,
                label=hit.label,
                collection=hit.collection,
                summary=hit.summary,
            )
            for hit in search_use_case(svc, query, k)
        ]

    @mcp.tool
    def reindex(dataset: str) -> ReindexResult:  # pyright: ignore[reportUnusedFunction]
        """Reconcile one dataset to its sources (prune vanished, re-index changed)."""
        svc = _dataset(dataset)
        if svc.read_only:
            raise _tool_error(f"dataset '{dataset}' is read-only; reindex is not offered")
        if svc.writable:
            raise _tool_error(f"dataset '{dataset}' is a writable knowledge dataset; use remember/forget, not reindex")
        # Serialize the store write against a concurrent live-watcher reconcile.
        with svc.lock:
            report = reconcile(
                connector=svc.connector,
                extract=svc.extract,
                chunk=svc.chunk,
                embedding=svc.embedding,
                store=svc.store,
                collection=svc.collection,
                summarize=svc.summarize,
            )
        return ReindexResult(
            dataset=svc.name,
            indexed=report.indexed,
            pruned=report.pruned,
            unchanged=report.unchanged,
        )

    @mcp.tool
    def search_datasets(  # pyright: ignore[reportUnusedFunction]
        query: str, k: int = 5, datasets: list[str] | None = None
    ) -> list[FusedSearchHit]:
        """Search several datasets at once and return ONE fused ranking.

        Use this to answer a question that may span more than one dataset (for
        example a shared corpus plus a department's own documents).

        Choosing datasets:
        - First call ``list_datasets`` to get the valid dataset names.
        - To search EVERYTHING this server serves: omit ``datasets`` or pass
          ["all"].
        - To search a SUBSET: pass their exact names, e.g. ["shared", "accounting"].
        - An unknown name returns an error that lists the valid names; retry with
          one of those.

        Ranking: results from different datasets are fused by RANK (Reciprocal Rank
        Fusion). Order results by the returned ``rrf_score``, NOT the per-dataset
        ``score`` - raw similarity scores are not comparable across datasets. Each
        row is tagged with its source ``dataset``. ``k`` is how many fused results
        to return.
        """
        targets = [
            DatasetSearchTarget(
                name=svc.name,
                model_key=svc.model_key,
                embedding=svc.embedding,
                store=svc.store,
                collection=svc.collection,
            )
            for svc in _select(datasets)
        ]
        fused = search_across_datasets(targets=targets, query=query, k=k, rrf_k=rrf_k)
        return [
            FusedSearchHit(
                uri=fh.hit.uri,
                ordinal=fh.hit.ordinal,
                score=fh.hit.score,
                rrf_score=fh.rrf_score,
                dataset=fh.dataset,
                text=fh.hit.chunk_text,
                label=fh.hit.label,
                collection=fh.hit.collection,
                summary=fh.hit.summary,
            )
            for fh in fused
        ]

    @mcp.tool
    def remember(  # pyright: ignore[reportUnusedFunction]
        dataset: str, text: str, title: str = "", entry_id: str | None = None
    ) -> RememberResult:
        """Write knowledge into a WRITABLE dataset and return its uri.

        Use this to save a note/fact the client wants remembered. Only datasets
        with ``writable: true`` (see ``list_datasets``) accept writes; a source
        dataset (filesystem / mail) returns an error.

        - ``text`` is the content; it is chunked and embedded like any document.
        - ``title`` is an optional label shown on search hits.
        - ``entry_id`` is an optional stable id: pass the same id again to UPDATE
          that entry (its old content is replaced); omit it to create a new entry
          with a generated id. The returned ``uri`` / ``id`` are what ``forget``
          takes.
        """
        svc = _dataset(dataset)
        if not svc.writable:
            raise _tool_error(f"dataset '{dataset}' is not writable; remember is only for knowledge datasets")
        resolved_id = entry_id or uuid.uuid4().hex
        uri = f"semdex://{svc.name}/{resolved_id}"
        with svc.lock:
            report = _remember_knowledge(
                chunk=svc.chunk,
                embedding=svc.embedding,
                store=svc.store,
                collection=svc.collection,
                uri=uri,
                text=text,
                label=title,
            )
        return RememberResult(uri=uri, id=resolved_id, chunks=report.chunks_indexed)

    @mcp.tool
    def forget(dataset: str, entry_id: str) -> ForgetResult:  # pyright: ignore[reportUnusedFunction]
        """Delete a knowledge entry (by its ``entry_id``) from a WRITABLE dataset.

        Idempotent: forgetting an unknown id is a no-op. Only writable datasets
        accept this; a source dataset returns an error.
        """
        svc = _dataset(dataset)
        if not svc.writable:
            raise _tool_error(f"dataset '{dataset}' is not writable; forget is only for knowledge datasets")
        uri = f"semdex://{svc.name}/{entry_id}"
        with svc.lock:
            _forget_knowledge(store=svc.store, collection=svc.collection, uri=uri)
        return ForgetResult(uri=uri, forgotten=True)

    return mcp


def search_use_case(svc: DatasetServices, query: str, k: int) -> list[Hit]:
    """Run the search use case for a dataset's services (named for readability)."""
    return search(embedding=svc.embedding, store=svc.store, collection=svc.collection, query=query, k=k)


def _tool_error(message: str) -> Exception:
    from fastmcp.exceptions import ToolError

    return ToolError(message)


def _import_fastmcp() -> type[FastMCP]:
    try:
        from fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        from ...domain.errors import ConfigurationError

        raise ConfigurationError("MCP server needs fastmcp; install semdex[mcp-server].") from exc
    return FastMCP


__all__ = ["build_mcp_server"]
