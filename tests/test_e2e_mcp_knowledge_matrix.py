"""End-to-end MCP knowledge matrix: remember / search / forget across the full grid.

Exercises the writable-knowledge tools (`remember`, `search`, `search_datasets`,
`forget`) through the in-memory ``fastmcp.Client`` against a real ``build_mcp_server``,
over the cross product of:

- **embeddings**: ``placeholder`` (offline, CI) and the real providers ``fastembed`` /
  ``model2vec`` / ``sentence_transformers`` (download models), ``ollama`` (native
  ``/api/embed``), and ``openai`` (any OpenAI-compatible ``/v1/embeddings`` server -
  ollama ``/v1`` or llama.cpp), the servers ``local_only`` and skipped when absent. The
  ``ollama`` and ``openai`` request/response paths also run in CI against a mocked
  transport, so the wire contract is covered with no server. The native cloud providers
  ``gemini`` / ``cohere`` run ``local_only`` against their REAL vendor APIs when a
  test-only key is set (``SEMDEX_TEST_GEMINI_API_KEY`` / ``SEMDEX_TEST_COHERE_API_KEY`` in
  ``.env``), and skip otherwise (hosted-only, so the key is mandatory - no local fallback);
- **store backends**: ``json`` / ``sqlite_vec`` / ``lancedb`` (embedded, CI when the
  extra is present) and ``pgvector`` / ``mariadb`` (Docker ``integration`` with placeholder
  vectors, plus a ``local_only`` cell that writes a REAL embedding into the persistent
  px-semdex-test server via ``SEMDEX_TEST_PGVECTOR_DSN`` / ``SEMDEX_TEST_MARIADB_DSN``);
- **multiple knowledge bases on one server**: embedded via separate store dirs; server
  backends via BOTH many-collections-in-one-database (table partition) AND
  one-database-per-knowledge-base (database partition).

Cross-cutting ``local_only`` cells exercise the real rig end to end: the ``openai``
provider against ollama's own ``/v1`` (distinct from the llama.cpp ``/v1``), a
mixed-provider fan-out (one KB native ollama, one OpenAI ``/v1``, queried together), and
the source **reindex** path with a real ``qwen3-embedding:4b``.

Isolation is asserted (a write in one KB is invisible in another), fan-out is asserted
(``search_datasets`` returns every KB tagged with its source), and delete is asserted
(``forget`` removes only the targeted entry).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from fastmcp import Client

from semdex.adapters.config.dataset import DatasetConfig
from semdex.adapters.mcp.server import build_mcp_server
from semdex.composition import DatasetServices, build_dataset_services
from semdex.domain.enums import EmbeddingBackend, Partition, StoreBackend

if TYPE_CHECKING:
    from semdex.application.ports import EmbeddingProvider

# --- embedded-backend availability (skip a cell whose optional extra is absent) -------


def _importable(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


# ------------------------------- helpers ---------------------------------------------


def _writable_dataset(
    name: str,
    backend: StoreBackend,
    *,
    store_dir: str | None = None,
    dsn: str | None = None,
    collection: str,
    embedding: EmbeddingBackend = EmbeddingBackend.PLACEHOLDER,
    embedding_model: str | None = None,
    embedding_endpoint: str | None = None,
    embedding_api_key: str | None = None,
) -> DatasetConfig:
    """Build a writable KNOWLEDGE dataset (no sources; the client writes it)."""
    return DatasetConfig(
        name=name,
        backend=backend,
        store_dir=store_dir,
        dsn=dsn,
        collection=collection,
        embedding_provider=embedding,
        embedding_model=embedding_model,
        embedding_endpoint=embedding_endpoint,
        embedding_api_key=embedding_api_key,
        writable=True,
    )


def _services(dataset: DatasetConfig) -> DatasetServices:
    return build_dataset_services(dataset, default_partition=Partition.TABLE)


async def _remember(client: Client[Any], dataset: str, text: str, title: str = "") -> str:
    res = await client.call_tool("remember", {"dataset": dataset, "text": text, "title": title})
    return str(res.data.id)


async def _search_uris(client: Client[Any], dataset: str, query: str, k: int = 5) -> list[str]:
    res = await client.call_tool("search", {"dataset": dataset, "query": query, "k": k})
    return [row.uri for row in res.data]


async def _forget(client: Client[Any], dataset: str, entry_id: str) -> None:
    await client.call_tool("forget", {"dataset": dataset, "entry_id": entry_id})


async def _fanout(
    client: Client[Any], query: str, k: int = 10, datasets: list[str] | None = None
) -> list[tuple[str, str]]:
    payload: dict[str, object] = {"query": query, "k": k}
    if datasets is not None:
        payload["datasets"] = datasets
    res = await client.call_tool("search_datasets", payload)
    return [(row.dataset, row.uri) for row in res.data]


# ------------------------------- single-KB roundtrip ---------------------------------

_EMBEDDED_ROUNDTRIP = [
    pytest.param(EmbeddingBackend.PLACEHOLDER, StoreBackend.JSON, id="placeholder-json"),
    pytest.param(
        EmbeddingBackend.PLACEHOLDER,
        StoreBackend.SQLITE_VEC,
        marks=pytest.mark.skipif(not _importable("sqlite_vec"), reason="needs semdex[sqlite]"),
        id="placeholder-sqlite_vec",
    ),
    pytest.param(
        EmbeddingBackend.PLACEHOLDER,
        StoreBackend.LANCEDB,
        marks=pytest.mark.skipif(not _importable("lancedb"), reason="needs semdex[lance] (+ AVX2)"),
        id="placeholder-lancedb",
    ),
    pytest.param(EmbeddingBackend.FASTEMBED, StoreBackend.JSON, marks=pytest.mark.local_only, id="fastembed-json"),
    pytest.param(
        EmbeddingBackend.MODEL2VEC,
        StoreBackend.JSON,
        marks=[
            pytest.mark.local_only,
            pytest.mark.skipif(not _importable("model2vec"), reason="needs semdex[model2vec]"),
        ],
        id="model2vec-json",
    ),
]


@pytest.mark.os_agnostic
@pytest.mark.asyncio
@pytest.mark.parametrize(("embedding", "backend"), _EMBEDDED_ROUNDTRIP)
async def test_write_read_delete_roundtrip_embedded(
    tmp_path: Path, embedding: EmbeddingBackend, backend: StoreBackend
) -> None:
    """remember -> search finds it -> forget -> search no longer finds it (embedded backends)."""
    svc = _services(
        _writable_dataset("kb", backend, store_dir=str(tmp_path / "kb"), collection="kb", embedding=embedding)
    )
    server = build_mcp_server([svc])
    async with Client(server) as client:
        assert any(d.name == "kb" and d.writable is True for d in (await client.call_tool("list_datasets", {})).data)
        entry_id = await _remember(client, "kb", "banana cherry apple", title="fruit")
        uri = f"semdex://kb/{entry_id}"
        assert uri in await _search_uris(client, "kb", "banana apple")
        await _forget(client, "kb", entry_id)
        assert uri not in await _search_uris(client, "kb", "banana apple")


# ------------------------------- real embedding servers (ollama / sentence-transformers)


def _ollama_ready(model: str | None = None) -> tuple[str, str] | None:
    """Return (endpoint, model) if a reachable ollama server has the embed model, else None.

    ``model`` overrides the default (env ``SEMDEX_TEST_OLLAMA_EMBED_MODEL`` or
    ``nomic-embed-text``) so a cell can require a specific model - e.g. the reindex
    path pins ``qwen3-embedding:4b`` to prove a real high-dim model round-trips.
    """
    import json
    import os
    import urllib.request

    model = model or os.environ.get("SEMDEX_TEST_OLLAMA_EMBED_MODEL", "nomic-embed-text")
    for endpoint in (
        os.environ.get("SEMDEX_TEST_OLLAMA_URL"),
        os.environ.get("SEMDEX_BENCH_OLLAMA_URL"),
        "http://px-semdex-test-embeddings:11434",
    ):
        if not endpoint:
            continue
        try:
            with urllib.request.urlopen(f"{endpoint}/api/tags", timeout=3) as resp:  # noqa: S310 - trusted internal host
                names = [m.get("name", "") for m in json.load(resp).get("models", [])]
        except Exception:  # noqa: S112 - reachability probe; any failure means try the next endpoint
            continue
        if any(name == model or name.startswith(f"{model}:") for name in names):
            return endpoint, model
    return None


@pytest.mark.local_only
@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_write_read_delete_roundtrip_ollama(tmp_path: Path) -> None:
    """remember -> search -> forget with a real OLLAMA embedding server (json store)."""
    ready = _ollama_ready()
    if ready is None:
        # local_only: when this suite runs, the real infra is expected. A missing
        # server is a rig failure to surface loudly, not a silent skip.
        pytest.fail(
            "native-ollama e2e: no reachable ollama server with the embed model "
            "(default px-semdex-test-embeddings:11434; override SEMDEX_TEST_OLLAMA_URL / _EMBED_MODEL)"
        )
    endpoint, model = ready
    svc = _services(
        _writable_dataset(
            "kb",
            StoreBackend.JSON,
            store_dir=str(tmp_path / "kb"),
            collection="kb",
            embedding=EmbeddingBackend.OLLAMA,
            embedding_model=model,
            embedding_endpoint=endpoint,
        )
    )
    server = build_mcp_server([svc])
    async with Client(server) as client:
        entry_id = await _remember(client, "kb", "banana cherry apple", title="fruit")
        uri = f"semdex://kb/{entry_id}"
        assert uri in await _search_uris(client, "kb", "banana apple")
        await _forget(client, "kb", entry_id)
        assert uri not in await _search_uris(client, "kb", "banana apple")


def _mocked_ollama_provider(model: str, dim: int = 8) -> EmbeddingProvider:
    """A real ollama embedding adapter wired to a FAKE /api/embed (deterministic per-text vectors).

    Same mechanism as tests/test_embedding_ollama.py: an httpx.MockTransport stands in for the
    server, so the ollama request/response path is exercised in CI with no ollama running.
    """
    import hashlib
    import json

    import httpx

    from semdex.adapters.embedding import load_ollama_embedding

    def _vec(text: str) -> list[float]:
        return [byte / 255.0 for byte in hashlib.sha256(text.encode("utf-8")).digest()[:dim]]

    def handler(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        return httpx.Response(200, json={"embeddings": [_vec(text) for text in inputs]})

    return load_ollama_embedding(model, client=httpx.Client(transport=httpx.MockTransport(handler)))


@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_write_read_delete_roundtrip_ollama_mocked(tmp_path: Path) -> None:
    """Full remember/search/forget through the OLLAMA provider against a MOCKED /api/embed (CI, no server)."""
    import dataclasses

    # Build offline (placeholder), then swap in a real ollama adapter backed by the fake transport, so
    # the MCP flow drives the ollama /api/embed request+response path end to end without a live server.
    built = _services(_writable_dataset("kb", StoreBackend.JSON, store_dir=str(tmp_path / "kb"), collection="kb"))
    svc = dataclasses.replace(built, embedding=_mocked_ollama_provider("nomic-embed-text"))
    server = build_mcp_server([svc])
    async with Client(server) as client:
        entry_id = await _remember(client, "kb", "banana cherry apple", title="fruit")
        uri = f"semdex://kb/{entry_id}"
        assert uri in await _search_uris(client, "kb", "banana apple")
        await _forget(client, "kb", entry_id)
        assert uri not in await _search_uris(client, "kb", "banana apple")


# ------------------------------- OpenAI-compatible servers (ollama /v1 / llama.cpp) --


def _openai_ready() -> tuple[str, str, str | None] | None:
    """Return (endpoint, model, api_key) for a reachable OpenAI-compatible server, else None.

    Targets any server exposing GET ``/v1/models`` + POST ``/v1/embeddings``: a
    llama.cpp ``llama-server`` (``SEMDEX_TEST_LLAMACPP_URL``), an ollama ``/v1``, or
    hosted OpenAI (``SEMDEX_TEST_OPENAI_URL`` + ``_API_KEY``). ``endpoint`` is the
    base URL ending in ``/v1``. The model is the env override or the first id the
    server lists (llama.cpp serves one model under an arbitrary id).
    """
    import json
    import os
    import urllib.request

    api_key = os.environ.get("SEMDEX_TEST_OPENAI_API_KEY")
    override = os.environ.get("SEMDEX_TEST_OPENAI_EMBED_MODEL")
    for endpoint in (
        os.environ.get("SEMDEX_TEST_LLAMACPP_URL"),
        os.environ.get("SEMDEX_TEST_OPENAI_URL"),
        "http://px-semdex-test-embeddings:8080/v1",
    ):
        if not endpoint:
            continue
        req = urllib.request.Request(f"{endpoint.rstrip('/')}/models")  # noqa: S310 - trusted internal host
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        try:
            with urllib.request.urlopen(req, timeout=3) as resp:  # noqa: S310 - trusted internal host
                body: Any = json.load(resp)
            # OpenAI/ollama use {"data":[{"id":...}]}; llama.cpp uses {"models":[{"name":...}]}.
            listed: list[dict[str, Any]] = body.get("data") or body.get("models") or []
            served = [str(item.get("id") or item.get("name") or "") for item in listed]
        except Exception:  # noqa: S112 - reachability probe; any failure means try the next endpoint
            continue
        model = override or (served[0] if served else None)
        if model:
            return endpoint, model, api_key
    return None


@pytest.mark.local_only
@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_write_read_delete_roundtrip_openai(tmp_path: Path) -> None:
    """remember -> search -> forget against a real OpenAI-compatible /v1 server (ollama or llama.cpp)."""
    ready = _openai_ready()
    if ready is None:
        # local_only: a missing real server is a rig failure to surface, not a silent skip.
        pytest.fail(
            "openai e2e: no reachable OpenAI-compatible server "
            "(default px-semdex-test-embeddings:8080/v1; override SEMDEX_TEST_LLAMACPP_URL / SEMDEX_TEST_OPENAI_URL)"
        )
    endpoint, model, api_key = ready
    svc = _services(
        _writable_dataset(
            "kb",
            StoreBackend.JSON,
            store_dir=str(tmp_path / "kb"),
            collection="kb",
            embedding=EmbeddingBackend.OPENAI,
            embedding_model=model,
            embedding_endpoint=endpoint,
            embedding_api_key=api_key,
        )
    )
    server = build_mcp_server([svc])
    async with Client(server) as client:
        entry_id = await _remember(client, "kb", "banana cherry apple", title="fruit")
        uri = f"semdex://kb/{entry_id}"
        assert uri in await _search_uris(client, "kb", "banana apple")
        await _forget(client, "kb", entry_id)
        assert uri not in await _search_uris(client, "kb", "banana apple")


def _mocked_openai_provider(model: str, dim: int = 8) -> EmbeddingProvider:
    """A real openai embedding adapter wired to a FAKE /v1/embeddings (deterministic per-text vectors).

    Same mechanism as tests/test_embedding_openai.py: an httpx.MockTransport stands in for the
    server, so the openai request/response path is exercised in CI with no server running.
    """
    import hashlib
    import json

    import httpx

    from semdex.adapters.embedding import load_openai_embedding

    def _vec(text: str) -> list[float]:
        return [byte / 255.0 for byte in hashlib.sha256(text.encode("utf-8")).digest()[:dim]]

    def handler(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        data = [{"embedding": _vec(text), "index": i} for i, text in enumerate(inputs)]
        return httpx.Response(200, json={"data": data})

    return load_openai_embedding(model, client=httpx.Client(transport=httpx.MockTransport(handler)))


@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_write_read_delete_roundtrip_openai_mocked(tmp_path: Path) -> None:
    """Full remember/search/forget through the OPENAI provider against a MOCKED /v1/embeddings (CI, no server)."""
    import dataclasses

    # Build offline (placeholder), then swap in a real openai adapter backed by the fake transport, so
    # the MCP flow drives the openai /v1/embeddings request+response path end to end without a live server.
    built = _services(_writable_dataset("kb", StoreBackend.JSON, store_dir=str(tmp_path / "kb"), collection="kb"))
    svc = dataclasses.replace(built, embedding=_mocked_openai_provider("text-embedding-3-small"))
    server = build_mcp_server([svc])
    async with Client(server) as client:
        entry_id = await _remember(client, "kb", "banana cherry apple", title="fruit")
        uri = f"semdex://kb/{entry_id}"
        assert uri in await _search_uris(client, "kb", "banana apple")
        await _forget(client, "kb", entry_id)
        assert uri not in await _search_uris(client, "kb", "banana apple")


def _cloud_key(key_env: str, model_env: str, default_model: str) -> tuple[str, str] | None:
    """Return (model, api_key) for a native cloud embedder, or None when the test key is unset.

    The cloud embedders (gemini/cohere) are hosted-only: there is no keyless local
    fallback like the openai ``/v1`` path, so a TEST-ONLY key is mandatory. conftest
    loads the repo-root ``.env`` into the environment, so putting ``<key_env>`` there
    is enough to run these cells. Absent key -> the cell skips (a paid cloud key is
    optional and must never fail CI, unlike the always-on local rig).
    """
    import os

    api_key = os.environ.get(key_env)
    if not api_key:
        return None
    return os.environ.get(model_env, default_model), api_key


def _skip_if_rate_limited(exc: Exception, provider: EmbeddingBackend) -> None:
    """Skip (not fail) a cloud cell when the vendor returns 429 - a QUOTA condition, not a bug.

    A 429 / RESOURCE_EXHAUSTED means auth + the request were valid but the vendor
    would not serve it - a per-minute rate limit on a burst (this cell fires ~4
    embeds: probe + remember + two searches) OR depleted billing credits. Neither
    is a code defect, so it must not red ``make test``. Any OTHER error re-raises
    unchanged, so a real defect (4xx auth, schema mismatch) still fails loudly.
    """
    text = str(exc)
    if "429" in text or "RESOURCE_EXHAUSTED" in text or "rate limit" in text.lower():
        pytest.skip(
            f"{provider.value} e2e: 429 RESOURCE_EXHAUSTED (rate limit or depleted credits); check quota/billing"
        )


async def _cloud_roundtrip(tmp_path: Path, provider: EmbeddingBackend, model: str, api_key: str) -> None:
    """remember -> search -> forget through a cloud embedder, proving the task split end to end.

    ``remember`` embeds the stored entry as a passage (the document task type) and
    ``search`` embeds the query (the query task type); a hit that survives the
    round-trip proves BOTH asymmetric paths reach the real API correctly, and
    ``forget`` then removes only that entry. A vendor 429 (free-tier quota) skips
    rather than fails - see :func:`_skip_if_rate_limited`.
    """
    try:
        svc = _services(
            _writable_dataset(
                "kb",
                StoreBackend.JSON,
                store_dir=str(tmp_path / "kb"),
                collection="kb",
                embedding=provider,
                embedding_model=model,
                embedding_api_key=api_key,
            )
        )
        server = build_mcp_server([svc])
        async with Client(server) as client:
            entry_id = await _remember(client, "kb", "banana cherry apple", title="fruit")
            uri = f"semdex://kb/{entry_id}"
            assert uri in await _search_uris(client, "kb", "banana apple")
            await _forget(client, "kb", entry_id)
            assert uri not in await _search_uris(client, "kb", "banana apple")
    except Exception as exc:  # tolerate a vendor 429 (quota); real failures re-raise below
        _skip_if_rate_limited(exc, provider)
        raise


@pytest.mark.local_only
@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_write_read_delete_roundtrip_gemini(tmp_path: Path) -> None:
    """remember -> search -> forget against the REAL Google Gemini embedding API (test-only key)."""
    ready = _cloud_key("SEMDEX_TEST_GEMINI_API_KEY", "SEMDEX_TEST_GEMINI_MODEL", "gemini-embedding-001")
    if ready is None:
        pytest.skip("gemini e2e: set SEMDEX_TEST_GEMINI_API_KEY (test-only key) in .env to run")
    model, api_key = ready
    await _cloud_roundtrip(tmp_path, EmbeddingBackend.GEMINI, model, api_key)


@pytest.mark.local_only
@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_write_read_delete_roundtrip_cohere(tmp_path: Path) -> None:
    """remember -> search -> forget against the REAL Cohere embedding API (test-only key)."""
    ready = _cloud_key("SEMDEX_TEST_COHERE_API_KEY", "SEMDEX_TEST_COHERE_MODEL", "embed-v4.0")
    if ready is None:
        pytest.skip("cohere e2e: set SEMDEX_TEST_COHERE_API_KEY (test-only key) in .env to run")
    model, api_key = ready
    await _cloud_roundtrip(tmp_path, EmbeddingBackend.COHERE, model, api_key)


def _ollama_openai_ready() -> tuple[str, str] | None:
    """Return (``.../v1`` endpoint, model) for the ollama server's OpenAI-compat surface, else None.

    ollama exposes BOTH its native ``/api/embed`` and an OpenAI-compatible ``/v1``. This
    targets the latter, so the ``OPENAI`` provider is proven against ollama's ``/v1`` - a
    distinct server from the llama.cpp ``/v1`` that :func:`_openai_ready` defaults to.
    """
    ready = _ollama_ready()
    if ready is None:
        return None
    endpoint, model = ready
    return f"{endpoint.rstrip('/')}/v1", model


@pytest.mark.local_only
@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_write_read_delete_roundtrip_openai_via_ollama(tmp_path: Path) -> None:
    """OPENAI provider against ollama's OpenAI-compat /v1 (not llama.cpp): remember -> search -> forget."""
    ready = _ollama_openai_ready()
    if ready is None:
        # local_only: a missing real server is a rig failure to surface, not a silent skip.
        pytest.fail(
            "openai-vs-ollama e2e: no reachable ollama server with the embed model at its /v1 "
            "(default px-semdex-test-embeddings:11434; override SEMDEX_TEST_OLLAMA_URL / _EMBED_MODEL)"
        )
    endpoint, model = ready
    svc = _services(
        _writable_dataset(
            "kb",
            StoreBackend.JSON,
            store_dir=str(tmp_path / "kb"),
            collection="kb",
            embedding=EmbeddingBackend.OPENAI,
            embedding_model=model,
            embedding_endpoint=endpoint,
        )
    )
    server = build_mcp_server([svc])
    async with Client(server) as client:
        entry_id = await _remember(client, "kb", "banana cherry apple", title="fruit")
        uri = f"semdex://kb/{entry_id}"
        assert uri in await _search_uris(client, "kb", "banana apple")
        await _forget(client, "kb", entry_id)
        assert uri not in await _search_uris(client, "kb", "banana apple")


@pytest.mark.local_only
@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_write_read_delete_roundtrip_sentence_transformers(tmp_path: Path) -> None:
    """remember -> search -> forget with a real sentence-transformers model (json store)."""
    pytest.importorskip("sentence_transformers")
    svc = _services(
        _writable_dataset(
            "kb",
            StoreBackend.JSON,
            store_dir=str(tmp_path / "kb"),
            collection="kb",
            embedding=EmbeddingBackend.SENTENCE_TRANSFORMERS,
        )
    )
    server = build_mcp_server([svc])
    async with Client(server) as client:
        entry_id = await _remember(client, "kb", "banana cherry apple", title="fruit")
        uri = f"semdex://kb/{entry_id}"
        assert uri in await _search_uris(client, "kb", "banana apple")
        await _forget(client, "kb", entry_id)
        assert uri not in await _search_uris(client, "kb", "banana apple")


# ------------------------------- source dataset: reindex path (real qwen3) -----------


@pytest.mark.local_only
@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_reindex_source_dataset_with_qwen3(tmp_path: Path) -> None:
    """Filesystem SOURCE dataset (not a writable KB): reindex embeds real files, a later reindex reconciles a new one.

    Exercises the reconcile/reindex code path (SourceConnector -> extractor -> chunker ->
    real embedding -> store) with a real, high-dimensional model (qwen3-embedding:4b), which
    the writable remember/forget cells never touch.
    """
    ready = _ollama_ready("qwen3-embedding:4b")
    if ready is None:
        # local_only: the rig is expected to have qwen3 pulled; a miss is loud, not skipped.
        pytest.fail(
            "reindex+qwen3 e2e: ollama server lacks qwen3-embedding:4b "
            "(default px-semdex-test-embeddings:11434; pull it with `ollama pull qwen3-embedding:4b`)"
        )
    endpoint, model = ready
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "fruit.md").write_text("banana cherry apple", encoding="utf-8")
    dataset = DatasetConfig(
        name="notes",
        backend=StoreBackend.JSON,
        store_dir=str(tmp_path / "store"),
        collection="notes",
        embedding_provider=EmbeddingBackend.OLLAMA,
        embedding_model=model,
        embedding_endpoint=endpoint,
        sources=(str(docs),),
    )
    svc = build_dataset_services(dataset, default_partition=Partition.TABLE)
    server = build_mcp_server([svc])
    async with Client(server) as client:
        await client.call_tool("reindex", {"dataset": "notes"})
        assert any(uri.endswith("fruit.md") for uri in await _search_uris(client, "notes", "banana apple"))
        # add a file and reindex again: reconcile picks up the new source, prior one still indexed
        (docs / "veg.md").write_text("carrot potato onion", encoding="utf-8")
        await client.call_tool("reindex", {"dataset": "notes"})
        assert any(uri.endswith("veg.md") for uri in await _search_uris(client, "notes", "carrot onion"))
        assert any(uri.endswith("fruit.md") for uri in await _search_uris(client, "notes", "banana apple"))


# ------------------------------- summary tier (source dataset, mocked summarizer) ----


class _FakeSummarizer:
    """A deterministic in-test SummaryProvider (one summary per document, no server)."""

    @property
    def model_id(self) -> str:
        return "fake-summarizer"

    def summarize(self, text: str) -> str:
        return f"SUMMARY[{text.split(maxsplit=1)[0]}]" if text.split() else "SUMMARY[]"


async def _search_rows(client: Client[Any], dataset: str, query: str, k: int = 5) -> list[Any]:
    res = await client.call_tool("search", {"dataset": dataset, "query": query, "k": k})
    return list(res.data)


def _source_dataset(tmp_path: Path, *, name: str) -> DatasetConfig:
    """A filesystem SOURCE dataset (placeholder embedding, one file) for the summary cells."""
    docs = tmp_path / name
    docs.mkdir()
    (docs / "fruit.md").write_text("banana cherry apple", encoding="utf-8")
    return DatasetConfig(
        name=name,
        backend=StoreBackend.JSON,
        store_dir=str(tmp_path / f"{name}_store"),
        collection=name,
        embedding_provider=EmbeddingBackend.PLACEHOLDER,
        sources=(str(docs),),
    )


@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_reindex_attaches_summary_when_tier_on(tmp_path: Path) -> None:
    """Summary tier ON (mocked summarizer): reindex -> the search JSON carries a non-null summary."""
    import dataclasses

    built = build_dataset_services(_source_dataset(tmp_path, name="notes"), default_partition=Partition.TABLE)
    svc = dataclasses.replace(built, summarize=_FakeSummarizer())
    server = build_mcp_server([svc])
    async with Client(server) as client:
        await client.call_tool("reindex", {"dataset": "notes"})
        rows = await _search_rows(client, "notes", "banana apple")
        assert rows
        assert all(row.summary == "SUMMARY[banana]" for row in rows)


@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_reindex_leaves_summary_null_when_tier_off(tmp_path: Path) -> None:
    """Summary tier OFF (default): reindex -> the search JSON summary is null (backward compatible)."""
    svc = build_dataset_services(_source_dataset(tmp_path, name="notes"), default_partition=Partition.TABLE)
    server = build_mcp_server([svc])
    async with Client(server) as client:
        await client.call_tool("reindex", {"dataset": "notes"})
        rows = await _search_rows(client, "notes", "banana apple")
        assert rows
        assert all(row.summary is None for row in rows)


# ------------------------------- multi-KB (embedded, separate store dirs) -------------

_EMBEDDED_BACKENDS = [
    pytest.param(StoreBackend.JSON, id="json"),
    pytest.param(
        StoreBackend.SQLITE_VEC,
        marks=pytest.mark.skipif(not _importable("sqlite_vec"), reason="needs semdex[sqlite]"),
        id="sqlite_vec",
    ),
    pytest.param(
        StoreBackend.LANCEDB,
        marks=pytest.mark.skipif(not _importable("lancedb"), reason="needs semdex[lance] (+ AVX2)"),
        id="lancedb",
    ),
]


@pytest.mark.os_agnostic
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", _EMBEDDED_BACKENDS)
async def test_multi_knowledge_bases_isolated_and_fused_embedded(tmp_path: Path, backend: StoreBackend) -> None:
    """Two knowledge bases on one server (separate store dirs): isolation + fan-out + delete."""
    kb_a = _services(_writable_dataset("kb_a", backend, store_dir=str(tmp_path / "a"), collection="a"))
    kb_b = _services(_writable_dataset("kb_b", backend, store_dir=str(tmp_path / "b"), collection="b"))
    server = build_mcp_server([kb_a, kb_b])
    async with Client(server) as client:
        id_a = await _remember(client, "kb_a", "alpha alpha alpha", title="A")
        id_b = await _remember(client, "kb_b", "bravo bravo bravo", title="B")
        uri_a, uri_b = f"semdex://kb_a/{id_a}", f"semdex://kb_b/{id_b}"

        # isolation: kb_a's write is not visible when searching kb_b
        assert uri_a in await _search_uris(client, "kb_a", "alpha")
        assert uri_a not in await _search_uris(client, "kb_b", "alpha")

        # fan-out over both, each row tagged with its source dataset
        tagged = await _fanout(client, "alpha bravo")
        assert ("kb_a", uri_a) in tagged
        assert ("kb_b", uri_b) in tagged

        # delete in kb_a leaves kb_b untouched
        await _forget(client, "kb_a", id_a)
        assert uri_a not in await _search_uris(client, "kb_a", "alpha")
        assert uri_b in await _search_uris(client, "kb_b", "bravo")


# ------------------------------- mixed-provider fan-out (real servers) ----------------


@pytest.mark.local_only
@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_mixed_provider_fanout(tmp_path: Path) -> None:
    """Two KBs embedded by DIFFERENT real providers, fanned out in one search.

    kb_native uses the ollama native ``/api/embed``; kb_v1 uses the ``OPENAI`` provider
    against an OpenAI-compat ``/v1`` server. A single ``search_datasets`` must embed the
    query per-collection with that collection's own provider/model, so each KB's own hit
    surfaces - proving fan-out is not tied to one shared embedder.
    """
    native = _ollama_ready()
    v1 = _openai_ready()
    if native is None or v1 is None:
        # local_only: mixed fan-out needs BOTH stacks; a miss is a rig failure to surface.
        missing = ", ".join(
            name for name, ok in (("ollama-native", native is not None), ("openai-/v1", v1 is not None)) if not ok
        )
        pytest.fail(f"mixed-provider fan-out e2e: needs both real embed servers up; missing: {missing}")
    o_endpoint, o_model = native
    x_endpoint, x_model, x_api_key = v1
    kb_native = _services(
        _writable_dataset(
            "kb_native",
            StoreBackend.JSON,
            store_dir=str(tmp_path / "native"),
            collection="native",
            embedding=EmbeddingBackend.OLLAMA,
            embedding_model=o_model,
            embedding_endpoint=o_endpoint,
        )
    )
    kb_v1 = _services(
        _writable_dataset(
            "kb_v1",
            StoreBackend.JSON,
            store_dir=str(tmp_path / "v1"),
            collection="v1",
            embedding=EmbeddingBackend.OPENAI,
            embedding_model=x_model,
            embedding_endpoint=x_endpoint,
            embedding_api_key=x_api_key,
        )
    )
    server = build_mcp_server([kb_native, kb_v1])
    async with Client(server) as client:
        id_a = await _remember(client, "kb_native", "alpha alpha alpha", title="A")
        id_b = await _remember(client, "kb_v1", "bravo bravo bravo", title="B")
        uri_a, uri_b = f"semdex://kb_native/{id_a}", f"semdex://kb_v1/{id_b}"

        # each KB searched with its own provider; the other KB's write is invisible (isolation)
        assert uri_a in await _search_uris(client, "kb_native", "alpha")
        assert uri_a not in await _search_uris(client, "kb_v1", "alpha")

        # one fan-out embeds the query per-collection with the right provider; both hits surface
        tagged = await _fanout(client, "alpha bravo")
        assert ("kb_native", uri_a) in tagged
        assert ("kb_v1", uri_b) in tagged


# ------------------------------- server backends (Docker) ----------------------------


def _pg_ready(port: int) -> bool:
    import psycopg

    try:
        psycopg.connect(
            f"host=127.0.0.1 port={port} user=postgres password=semdex dbname=postgres", connect_timeout=2
        ).close()
    except Exception:
        return False
    return True


def _pg_conninfo(port: int, dbname: str) -> str:
    return f"host=127.0.0.1 port={port} user=postgres password=semdex dbname={dbname}"


def _pg_create_db(port: int, dbname: str) -> None:
    import psycopg

    conn: Any = psycopg.connect(_pg_conninfo(port, "postgres"), autocommit=True)
    try:
        conn.execute(f'CREATE DATABASE "{dbname}"')
    finally:
        conn.close()


def _maria_ready(port: int) -> bool:
    import pymysql

    try:
        pymysql.connect(host="127.0.0.1", port=port, user="root", password="semdex", connect_timeout=2).close()
    except Exception:
        return False
    return True


def _maria_dsn(port: int, dbname: str) -> str:
    return f"mysql://root:semdex@127.0.0.1:{port}/{dbname}"


def _maria_create_db(port: int, dbname: str) -> None:
    import pymysql

    conn = pymysql.connect(host="127.0.0.1", port=port, user="root", password="semdex")
    try:
        conn.cursor().execute(f"CREATE DATABASE IF NOT EXISTS `{dbname}`")
        conn.commit()
    finally:
        conn.close()


@pytest.fixture(scope="module")
def pg_port(service_container: Callable[..., int]) -> int:
    pytest.importorskip("psycopg")
    pytest.importorskip("pgvector")
    return service_container(
        image="pgvector/pgvector:pg16", container_port=5432, env={"POSTGRES_PASSWORD": "semdex"}, ready=_pg_ready
    )


@pytest.fixture(scope="module")
def maria_port(service_container: Callable[..., int]) -> int:
    pytest.importorskip("pymysql")
    return service_container(
        image="mariadb:11.8", container_port=3306, env={"MARIADB_ROOT_PASSWORD": "semdex"}, ready=_maria_ready
    )


def _fresh(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.mark.integration
@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_write_read_delete_roundtrip_pgvector(pg_port: int) -> None:
    """Single knowledge base on pgvector: remember -> search -> forget."""
    db = _fresh("kb")
    _pg_create_db(pg_port, db)
    svc = _services(_writable_dataset("kb", StoreBackend.PGVECTOR, dsn=_pg_conninfo(pg_port, db), collection="kb"))
    server = build_mcp_server([svc])
    async with Client(server) as client:
        entry_id = await _remember(client, "kb", "banana cherry apple")
        uri = f"semdex://kb/{entry_id}"
        assert uri in await _search_uris(client, "kb", "banana apple")
        await _forget(client, "kb", entry_id)
        assert uri not in await _search_uris(client, "kb", "banana apple")


@pytest.mark.integration
@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_multi_kb_pgvector_separate_databases(pg_port: int) -> None:
    """Two knowledge bases, one database each on the same pg server (database partition)."""
    db_a, db_b = _fresh("kba"), _fresh("kbb")
    _pg_create_db(pg_port, db_a)
    _pg_create_db(pg_port, db_b)
    kb_a = _services(
        _writable_dataset("kb_a", StoreBackend.PGVECTOR, dsn=_pg_conninfo(pg_port, db_a), collection="mem")
    )
    kb_b = _services(
        _writable_dataset("kb_b", StoreBackend.PGVECTOR, dsn=_pg_conninfo(pg_port, db_b), collection="mem")
    )
    await _assert_isolated_and_fused(kb_a, kb_b)


@pytest.mark.integration
@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_multi_kb_pgvector_shared_database_many_collections(pg_port: int) -> None:
    """Two knowledge bases sharing one database, one collection each (table partition)."""
    db = _fresh("shared")
    _pg_create_db(pg_port, db)
    dsn = _pg_conninfo(pg_port, db)
    kb_a = _services(_writable_dataset("kb_a", StoreBackend.PGVECTOR, dsn=dsn, collection="coll_a"))
    kb_b = _services(_writable_dataset("kb_b", StoreBackend.PGVECTOR, dsn=dsn, collection="coll_b"))
    await _assert_isolated_and_fused(kb_a, kb_b)


@pytest.mark.integration
@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_write_read_delete_roundtrip_mariadb(maria_port: int) -> None:
    """Single knowledge base on mariadb: remember -> search -> forget."""
    db = _fresh("kb")
    _maria_create_db(maria_port, db)
    svc = _services(_writable_dataset("kb", StoreBackend.MARIADB, dsn=_maria_dsn(maria_port, db), collection="kb"))
    server = build_mcp_server([svc])
    async with Client(server) as client:
        entry_id = await _remember(client, "kb", "banana cherry apple")
        uri = f"semdex://kb/{entry_id}"
        assert uri in await _search_uris(client, "kb", "banana apple")
        await _forget(client, "kb", entry_id)
        assert uri not in await _search_uris(client, "kb", "banana apple")


@pytest.mark.integration
@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_multi_kb_mariadb_separate_databases(maria_port: int) -> None:
    """Two knowledge bases, one database each on the same mariadb server (database partition)."""
    db_a, db_b = _fresh("kba"), _fresh("kbb")
    _maria_create_db(maria_port, db_a)
    _maria_create_db(maria_port, db_b)
    kb_a = _services(
        _writable_dataset("kb_a", StoreBackend.MARIADB, dsn=_maria_dsn(maria_port, db_a), collection="mem")
    )
    kb_b = _services(
        _writable_dataset("kb_b", StoreBackend.MARIADB, dsn=_maria_dsn(maria_port, db_b), collection="mem")
    )
    await _assert_isolated_and_fused(kb_a, kb_b)


@pytest.mark.integration
@pytest.mark.os_agnostic
@pytest.mark.asyncio
async def test_multi_kb_mariadb_shared_database_many_collections(maria_port: int) -> None:
    """Two knowledge bases sharing one database, one collection each (table partition)."""
    db = _fresh("shared")
    _maria_create_db(maria_port, db)
    dsn = _maria_dsn(maria_port, db)
    kb_a = _services(_writable_dataset("kb_a", StoreBackend.MARIADB, dsn=dsn, collection="coll_a"))
    kb_b = _services(_writable_dataset("kb_b", StoreBackend.MARIADB, dsn=dsn, collection="coll_b"))
    await _assert_isolated_and_fused(kb_a, kb_b)


async def _assert_isolated_and_fused(kb_a: DatasetServices, kb_b: DatasetServices) -> None:
    """Shared assertions for a two-knowledge-base server: isolation + fan-out + scoped delete."""
    server = build_mcp_server([kb_a, kb_b])
    async with Client(server) as client:
        id_a = await _remember(client, "kb_a", "alpha alpha alpha", title="A")
        id_b = await _remember(client, "kb_b", "bravo bravo bravo", title="B")
        uri_a, uri_b = f"semdex://kb_a/{id_a}", f"semdex://kb_b/{id_b}"

        assert uri_a in await _search_uris(client, "kb_a", "alpha")
        assert uri_a not in await _search_uris(client, "kb_b", "alpha")  # isolation

        tagged = await _fanout(client, "alpha bravo")
        assert ("kb_a", uri_a) in tagged
        assert ("kb_b", uri_b) in tagged

        await _forget(client, "kb_a", id_a)
        assert uri_a not in await _search_uris(client, "kb_a", "alpha")
        assert uri_b in await _search_uris(client, "kb_b", "bravo")  # delete is scoped to kb_a


# ------------------------------- real embedding into a persistent server store --------
#
# The Docker pgvector/mariadb cells above use PLACEHOLDER vectors; the ollama/openai cells
# use a JSON store. This crosses them: a REAL embedding written into a REAL server store
# (the persistent pgvector/mariadb on px-semdex-test), proving real float vectors round-trip
# through the DB's vector column + index via the MCP tools. local_only: the store is reached
# by an env DSN that carries a password (never hardcoded/defaulted), fail-not-skip when the
# embed server is absent but the driver + DSN are present.


def _real_embedding() -> tuple[EmbeddingBackend, str, str, str | None] | None:
    """Return (provider, model, endpoint, api_key) for whichever real embed server is up, else None."""
    native = _ollama_ready()
    if native is not None:
        endpoint, model = native
        return EmbeddingBackend.OLLAMA, model, endpoint, None
    openai = _openai_ready()
    if openai is not None:
        endpoint, model, api_key = openai
        return EmbeddingBackend.OPENAI, model, endpoint, api_key
    return None


def _pg_conninfo_swap_db(admin: str, dbname: str) -> str:
    """Return the libpq conninfo `admin` with its dbname replaced by `dbname`."""
    tokens = [tok for tok in admin.split() if not tok.startswith("dbname=")]
    return " ".join([*tokens, f"dbname={dbname}"])


def _pg_fresh_db(admin: str) -> str:
    """Create a throwaway database on the pg server named by `admin`; return its name."""
    import psycopg

    dbname = _fresh("e2e")
    conn: Any = psycopg.connect(_pg_conninfo_swap_db(admin, "postgres"), autocommit=True)
    try:
        conn.execute(f'CREATE DATABASE "{dbname}"')
    finally:
        conn.close()
    return dbname


def _pg_drop_db(admin: str, dbname: str) -> None:
    import psycopg

    conn: Any = psycopg.connect(_pg_conninfo_swap_db(admin, "postgres"), autocommit=True)
    try:
        conn.execute(f'DROP DATABASE IF EXISTS "{dbname}" WITH (FORCE)')  # FORCE: evict lingering conns (pg13+)
    finally:
        conn.close()


def _maria_url_swap_db(url: str, dbname: str) -> str:
    """Return the mysql:// `url` (which must end in /<db>) with its database replaced by `dbname`."""
    base, _, _old = url.rpartition("/")
    return f"{base}/{dbname}"


def _maria_admin_connect(url: str) -> Any:
    from urllib.parse import urlparse

    import pymysql

    parsed = urlparse(url)
    return pymysql.connect(
        host=parsed.hostname or "127.0.0.1",
        port=parsed.port or 3306,
        user=parsed.username or "root",
        password=parsed.password or "",
    )


def _maria_fresh_db(url: str) -> str:
    dbname = _fresh("e2e")
    conn = _maria_admin_connect(url)
    try:
        conn.cursor().execute(f"CREATE DATABASE `{dbname}`")
        conn.commit()
    finally:
        conn.close()
    return dbname


def _maria_drop_db(url: str, dbname: str) -> None:
    conn = _maria_admin_connect(url)
    try:
        conn.cursor().execute(f"DROP DATABASE IF EXISTS `{dbname}`")
        conn.commit()
    finally:
        conn.close()


_SERVER_STORE = [
    pytest.param(
        StoreBackend.PGVECTOR,
        "SEMDEX_TEST_PGVECTOR_DSN",
        marks=pytest.mark.skipif(not (_importable("psycopg") and _importable("pgvector")), reason="needs semdex[pg]"),
        id="pgvector",
    ),
    pytest.param(
        StoreBackend.MARIADB,
        "SEMDEX_TEST_MARIADB_DSN",
        marks=pytest.mark.skipif(not _importable("pymysql"), reason="needs semdex[mariadb]"),
        id="mariadb",
    ),
]


@pytest.mark.local_only
@pytest.mark.os_agnostic
@pytest.mark.asyncio
@pytest.mark.parametrize(("backend", "dsn_env"), _SERVER_STORE)
async def test_real_embedding_into_server_store(backend: StoreBackend, dsn_env: str) -> None:
    """A real ollama/openai embedding written into a persistent pgvector/mariadb store: remember -> search -> forget."""
    import os

    embed = _real_embedding()
    if embed is None:
        # local_only: a missing real embed server is a rig failure to surface, not a silent skip.
        pytest.fail(
            "real-embed x server-store e2e: no reachable ollama/openai embed server "
            "(default px-semdex-test-embeddings:11434 / px-semdex-test-embeddings:8080/v1)"
        )
    admin = os.environ.get(dsn_env)
    if not admin:
        # SKIP, unlike the embed-server check above: that server is shared rig infrastructure whose
        # absence is a real failure, whereas a DSN is per-developer configuration carrying a password
        # that cannot be committed. Failing on it made `make test` (which runs local_only) permanently
        # red on every box without these servers, which is how three unrelated broken assertions in
        # the perf suite went unnoticed. Set the env var to run the e2e for real.
        pytest.skip(
            f"real-embed x {backend.value} e2e: set {dsn_env} to the px-semdex-test {backend.value} server DSN "
            "(carries a password - load from the environment, never commit it)"
        )
    provider, model, endpoint, api_key = embed
    if backend is StoreBackend.PGVECTOR:
        dbname = _pg_fresh_db(admin)
        dsn = _pg_conninfo_swap_db(admin, dbname)
    else:
        dbname = _maria_fresh_db(admin)
        dsn = _maria_url_swap_db(admin, dbname)
    try:
        svc = _services(
            _writable_dataset(
                "kb",
                backend,
                dsn=dsn,
                collection="kb",
                embedding=provider,
                embedding_model=model,
                embedding_endpoint=endpoint,
                embedding_api_key=api_key,
            )
        )
        server = build_mcp_server([svc])
        async with Client(server) as client:
            entry_id = await _remember(client, "kb", "banana cherry apple", title="fruit")
            uri = f"semdex://kb/{entry_id}"
            assert uri in await _search_uris(client, "kb", "banana apple")
            await _forget(client, "kb", entry_id)
            assert uri not in await _search_uris(client, "kb", "banana apple")
    finally:
        if backend is StoreBackend.PGVECTOR:
            _pg_drop_db(admin, dbname)
        else:
            _maria_drop_db(admin, dbname)
