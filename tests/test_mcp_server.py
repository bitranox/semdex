from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from semdex.adapters.config.dataset import DatasetConfig
from semdex.adapters.mcp.server import build_mcp_server
from semdex.composition import DatasetServices, build_dataset_services
from semdex.domain.enums import EmbeddingBackend, Partition, StoreBackend


def _services(tmp_path: Path, *, read_only: bool = False) -> DatasetServices:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("banana cherry apple", encoding="utf-8")
    dataset = DatasetConfig(
        name="notes",
        backend=StoreBackend.JSON,
        store_dir=str(tmp_path / "store"),
        collection="notes",
        embedding_provider=EmbeddingBackend.PLACEHOLDER,
        sources=(str(tmp_path / "docs"),),
        read_only=read_only,
    )
    return build_dataset_services(dataset, default_partition=Partition.TABLE)


@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_list_datasets_and_reindex_then_search(tmp_path: Path) -> None:
    server = build_mcp_server([_services(tmp_path)])
    async with Client(server) as client:
        listed = await client.call_tool("list_datasets", {})
        assert any(d.name == "notes" for d in listed.data)
        await client.call_tool("reindex", {"dataset": "notes"})
        hits = await client.call_tool("search", {"dataset": "notes", "query": "banana apple", "k": 3})
        assert hits.data and hits.data[0].uri.endswith("a.md")


@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_reindex_rejects_read_only_dataset(tmp_path: Path) -> None:
    server = build_mcp_server([_services(tmp_path, read_only=True)])
    async with Client(server) as client:
        with pytest.raises(ToolError):  # FastMCP surfaces a tool error to the client
            await client.call_tool("reindex", {"dataset": "notes"})


def _services_named(tmp_path: Path, name: str, word: str) -> DatasetServices:
    docs = tmp_path / name
    docs.mkdir()
    (docs / "d.md").write_text(word, encoding="utf-8")
    dataset = DatasetConfig(
        name=name,
        backend=StoreBackend.JSON,
        store_dir=str(tmp_path / f"store-{name}"),
        collection=name,
        embedding_provider=EmbeddingBackend.PLACEHOLDER,
        sources=(str(docs),),
    )
    return build_dataset_services(dataset, default_partition=Partition.TABLE)


@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_search_datasets_fans_out_over_all(tmp_path: Path) -> None:
    shared = _services_named(tmp_path, "shared", "alpha bravo")
    acc = _services_named(tmp_path, "accounting", "charlie delta")
    server = build_mcp_server([shared, acc])
    async with Client(server) as client:
        await client.call_tool("reindex", {"dataset": "shared"})
        await client.call_tool("reindex", {"dataset": "accounting"})
        # omit datasets -> all; each fused row is tagged with its dataset
        res = await client.call_tool("search_datasets", {"query": "alpha charlie", "k": 5})
        datasets_seen = {row.dataset for row in res.data}
        assert datasets_seen == {"shared", "accounting"}
        assert all(row.rrf_score is not None and row.uri for row in res.data)


@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_search_datasets_subset_and_all_token(tmp_path: Path) -> None:
    shared = _services_named(tmp_path, "shared", "alpha bravo")
    acc = _services_named(tmp_path, "accounting", "charlie delta")
    server = build_mcp_server([shared, acc])
    async with Client(server) as client:
        await client.call_tool("reindex", {"dataset": "shared"})
        await client.call_tool("reindex", {"dataset": "accounting"})
        only = await client.call_tool("search_datasets", {"query": "alpha", "datasets": ["shared"]})
        assert {row.dataset for row in only.data} == {"shared"}
        allrows = await client.call_tool("search_datasets", {"query": "alpha", "datasets": ["all"]})
        assert {row.dataset for row in allrows.data} == {"shared", "accounting"}


@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_search_datasets_unknown_name_errors(tmp_path: Path) -> None:
    server = build_mcp_server([_services_named(tmp_path, "shared", "alpha")])
    async with Client(server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("search_datasets", {"query": "x", "datasets": ["nope"]})


def _writable_services(tmp_path: Path) -> DatasetServices:
    dataset = DatasetConfig(
        name="kb",
        backend=StoreBackend.JSON,
        store_dir=str(tmp_path / "kb"),
        collection="kb",
        embedding_provider=EmbeddingBackend.PLACEHOLDER,
        writable=True,
    )
    return build_dataset_services(dataset, default_partition=Partition.TABLE)


@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_remember_then_search_then_forget_via_mcp(tmp_path: Path) -> None:
    server = build_mcp_server([_writable_services(tmp_path)])
    async with Client(server) as client:
        listed = await client.call_tool("list_datasets", {})
        assert any(d.name == "kb" and d.writable is True for d in listed.data)
        written = await client.call_tool("remember", {"dataset": "kb", "text": "banana cherry apple", "title": "fruit"})
        entry_id = written.data.id
        assert written.data.uri == f"semdex://kb/{entry_id}"
        hits = await client.call_tool("search", {"dataset": "kb", "query": "banana apple", "k": 5})
        assert hits.data and hits.data[0].uri == f"semdex://kb/{entry_id}"
        await client.call_tool("forget", {"dataset": "kb", "entry_id": entry_id})
        gone = await client.call_tool("search", {"dataset": "kb", "query": "banana apple", "k": 5})
        assert gone.data == []


@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_remember_rejects_source_dataset(tmp_path: Path) -> None:
    server = build_mcp_server([_services(tmp_path)])  # a normal source dataset
    async with Client(server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("remember", {"dataset": "notes", "text": "x"})


@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_reindex_rejects_writable_dataset(tmp_path: Path) -> None:
    server = build_mcp_server([_writable_services(tmp_path)])
    async with Client(server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("reindex", {"dataset": "kb"})


@pytest.mark.os_agnostic
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mask_error_details", "secret_expected"),
    [
        pytest.param(True, False, id="masked"),
        pytest.param(False, True, id="unmasked"),
    ],
)
async def test_unexpected_error_is_masked_from_the_client(
    tmp_path: Path, *, mask_error_details: bool, secret_expected: bool
) -> None:
    """``mask_error_details`` gates whether an unexpected exception's text reaches the client.

    Parametrized over both settings rather than only the masked default: a test
    asserting only "the secret is absent" would also pass a tool that never ran at
    all (e.g. "Unknown tool"), proving nothing about masking. The ``False`` arm
    requires the secret to be PRESENT, which is only true if the tool actually ran
    and the knob actually reached FastMCP's constructor.
    """
    server = build_mcp_server([_services(tmp_path)], mask_error_details=mask_error_details)

    @server.tool
    def boom() -> None:
        raise RuntimeError("SECRET-INTERNAL-DETAIL")

    async with Client(server) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool("boom", {})
        assert ("SECRET-INTERNAL-DETAIL" in str(excinfo.value)) is secret_expected


@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_deliberate_tool_error_reaches_the_client_verbatim(tmp_path: Path) -> None:
    server = build_mcp_server([_services(tmp_path)])
    async with Client(server) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool("search", {"dataset": "does-not-exist", "query": "x"})
        assert "unknown dataset 'does-not-exist'" in str(excinfo.value)


@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_the_tool_surface_is_exactly_the_six_tools(tmp_path: Path) -> None:
    server = build_mcp_server([_services(tmp_path)])
    async with Client(server) as client:
        tools = await client.list_tools()
        names = {tool.name for tool in tools}
        assert names == {"list_datasets", "search", "reindex", "search_datasets", "remember", "forget"}
        required = {tool.name: tool.input_schema.get("required", []) for tool in tools}
        assert required == {
            "list_datasets": [],
            "search": ["dataset", "query"],
            "reindex": ["dataset"],
            "search_datasets": ["query"],
            "remember": ["dataset", "text"],
            "forget": ["dataset", "entry_id"],
        }
        output_schemas = {tool.name: tool.output_schema for tool in tools}
        assert all(output_schemas.values()), output_schemas
        assert _row_fields(output_schemas["list_datasets"]) >= {"name", "backend", "collection", "writable"}
        assert _row_fields(output_schemas["search"]) >= {"uri", "ordinal", "score", "text"}
        assert _row_fields(output_schemas["reindex"]) >= {"dataset", "indexed", "pruned", "unchanged"}
        assert _row_fields(output_schemas["search_datasets"]) >= {"uri", "rrf_score", "dataset"}
        assert _row_fields(output_schemas["remember"]) >= {"uri", "id", "chunks"}
        assert _row_fields(output_schemas["forget"]) >= {"uri", "forgotten"}
        # None of the tools should still describe an open bag of unnamed properties.
        for schema in output_schemas.values():
            assert schema is not None
            assert schema.get("additionalProperties") is not True, schema


def _row_fields(schema: dict[str, Any] | None) -> set[str]:
    """Return the named field set of a tool's output schema, unwrapping a list result."""
    assert schema is not None
    if schema.get("x-fastmcp-wrap-result"):
        result_schema: dict[str, Any] = schema["properties"]["result"]
        item_schema: dict[str, Any] = result_schema.get("items", result_schema)
        return set(item_schema.get("properties", {}))
    return set(schema.get("properties", {}))


def _descriptions(node: Any) -> list[str]:
    """Every ``description`` string anywhere in a JSON schema, ``$defs`` included."""
    if isinstance(node, dict):
        mapping = cast("dict[str, Any]", node)
        found = [value for key, value in mapping.items() if key == "description" and isinstance(value, str)]
        return found + [text for value in mapping.values() for text in _descriptions(value)]
    if isinstance(node, list):
        items = cast("list[Any]", node)
        return [text for item in items for text in _descriptions(item)]
    return []


@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_no_schema_description_carries_developer_docstring_prose(tmp_path: Path) -> None:
    """MCP clients read schema descriptions as plain one-line text.

    A domain enum's docstring is multi-paragraph RST written for developers (double-backtick
    markup, deployment advice); pydantic copies it into the schema's ``$defs`` unless the field
    overrides it, so it would reach every client verbatim.
    """
    server = build_mcp_server([_services(tmp_path)])
    async with Client(server) as client:
        tools = await client.list_tools()
    offending = {
        f"{tool.name}:{text[:40]!r}"
        for tool in tools
        for schema in (tool.input_schema, tool.output_schema)
        for text in _descriptions(schema)
        if "``" in text or "\n" in text
    }
    assert not offending, sorted(offending)
