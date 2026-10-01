"""Performance sweeps for the embedding-provider and chunker plugins.

Companion to ``test_perf_tunables.py`` (which sweeps the ``[index]`` tunables),
per the CLAUDE.md "extend perf tests per plugin" rule. Same discipline:
benchmarks that assert only robust invariants (correctness preserved, monotonic
relationships) and print time/size, never raw timing thresholds. ``local_only``:
they build real chunkers and download embedding models, so run via ``make ti`` /
``pytest -m local_only -s``, not CI ``make test``.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from semdex.adapters.discovery import discover_sources
from semdex.adapters.discovery.location import to_uri
from semdex.application.use_cases import index_sources, search
from semdex.composition import build_chunker, build_index_production
from semdex.domain.enums import ChunkStrategy, EmbeddingBackend
from semdex.domain.models import ExtractedDocument, SourceRef

pytestmark = [pytest.mark.local_only, pytest.mark.os_agnostic]

_VOCAB = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa")
_CHUNK_STRATEGIES = (
    ChunkStrategy.WHITESPACE,
    ChunkStrategy.RECURSIVE,
    ChunkStrategy.MARKDOWN,
    ChunkStrategy.FAST,
)


def _make_corpus(root: Path, *, n_files: int, words_per_file: int) -> Path:
    corpus = root / "corpus"
    corpus.mkdir()
    for i in range(n_files):
        words = [_VOCAB[(i + j) % len(_VOCAB)] for j in range(words_per_file)]
        (corpus / f"doc{i:03d}.md").write_text("# Heading\n\n" + " ".join(words), encoding="utf-8")
    return corpus


def _prose_doc() -> ExtractedDocument:
    source = SourceRef(uri="/x/doc.md", label="l", content_hash="h", mtime=1.0)
    text = "# Title\n\n" + ("Alpha beta gamma delta epsilon zeta. " * 30)
    return ExtractedDocument(source=source, text=text)


def test_perf_chunker_max_tokens() -> None:
    """Recursive chunk count is non-increasing as max_tokens grows; report time."""
    pytest.importorskip("chonkie")
    doc = _prose_doc()
    chunker = build_chunker(ChunkStrategy.RECURSIVE, recipe="")  # recipe="" stays offline
    rows: list[tuple[int, int, float]] = []
    for max_tokens in (32, 64, 128, 256, 512):
        start = time.perf_counter()
        chunks = chunker(doc, max_tokens=max_tokens)
        rows.append((max_tokens, len(chunks), time.perf_counter() - start))

    counts = [count for _, count, _ in rows]
    assert counts == sorted(counts, reverse=True), counts  # bigger chunks -> fewer chunks

    print("\nrecursive max_tokens sweep:  tokens | chunks | seconds")
    for max_tokens, count, seconds in rows:
        print(f"  {max_tokens:5d} | {count:6d} | {seconds:.4f}")


def test_perf_chunker_overlap() -> None:
    """More overlap means more total tokens across chunks (repeated context)."""
    pytest.importorskip("chonkie")
    doc = _prose_doc()
    rows: list[tuple[int, int]] = []
    for overlap in (0, 8, 16, 32):
        chunks = build_chunker(ChunkStrategy.RECURSIVE, recipe="", overlap=overlap)(doc, max_tokens=64)
        rows.append((overlap, sum(chunk.token_count for chunk in chunks)))

    totals = [total for _, total in rows]
    assert totals == sorted(totals), totals  # overlap only adds context, never removes

    print("\nrecursive overlap sweep (chunk_size=64):  overlap | total_tokens")
    for overlap, total in rows:
        print(f"  {overlap:7d} | {total:6d}")


def test_perf_chunker_strategies(tmp_path: Path) -> None:
    """Every strategy preserves the obvious match; report chunk count + index time."""
    pytest.importorskip("chonkie")
    pytest.importorskip("semantic_text_splitter")
    corpus = _make_corpus(tmp_path, n_files=16, words_per_file=300)
    target = corpus / "target.md"
    target.write_text("# T\n\n" + "quantum entanglement physics " * 40, encoding="utf-8")
    sources = discover_sources([corpus])
    rows: list[tuple[str, int, float]] = []
    for strategy in _CHUNK_STRATEGIES:
        services = build_index_production(
            tmp_path / f"store_{strategy.value}",
            chunker_strategy=strategy,
            chunker_recipe="",
            embedding_provider=EmbeddingBackend.PLACEHOLDER,
            embedding_dim=64,
        )
        start = time.perf_counter()
        report = index_sources(
            extract=services.extract,
            chunk=services.chunk,
            embedding=services.embedding,
            store=services.store_writer,
            collection="c",
            sources=sources,
            max_tokens=128,
        )
        elapsed = time.perf_counter() - start
        hits = search(
            embedding=services.embedding, store=services.store_reader, collection="c", query="quantum entanglement", k=3
        )
        assert hits[0].uri == to_uri(target), f"{strategy.value} lost the obvious match"
        rows.append((strategy.value, report.chunks_indexed, elapsed))

    print("\nchunker strategy sweep (placeholder embed):  strategy | chunks | index_seconds")
    for name, chunks, seconds in rows:
        print(f"  {name:10s} | {chunks:6d} | {seconds:.3f}")


def test_perf_embedding_providers(tmp_path: Path) -> None:
    """Real providers win a paraphrase query the lexical placeholder cannot; report dim/time."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "space.md").write_text("The rocket engine burns fuel to reach orbit.", encoding="utf-8")
    (corpus / "cooking.md").write_text("Simmer the sauce with garlic and basil.", encoding="utf-8")
    target = corpus / "plants.md"
    target.write_text("Photosynthesis converts sunlight into chemical energy in green leaves.", encoding="utf-8")
    sources = discover_sources([corpus])
    query = "how do plants make food from light"

    rows: list[tuple[str, int, bool, float]] = []
    for provider in (EmbeddingBackend.PLACEHOLDER, EmbeddingBackend.FASTEMBED, EmbeddingBackend.MODEL2VEC):
        if provider is EmbeddingBackend.MODEL2VEC:
            pytest.importorskip("model2vec")
        services = build_index_production(
            tmp_path / f"store_{provider.value}", embedding_provider=provider, embedding_dim=64
        )
        index_sources(
            extract=services.extract,
            chunk=services.chunk,
            embedding=services.embedding,
            store=services.store_writer,
            collection="c",
            sources=sources,
            max_tokens=256,
        )
        start = time.perf_counter()
        hits = search(embedding=services.embedding, store=services.store_reader, collection="c", query=query, k=3)
        won_target = hits[0].uri == to_uri(target)
        rows.append((provider.value, services.embedding.dim, won_target, time.perf_counter() - start))

    won = {name: hit for name, _, hit, _ in rows}
    assert won["fastembed"], "fastembed should win the paraphrase query"  # the semantic default earns its keep

    print("\nembedding provider sweep (paraphrase query):  provider | dim | top1_is_target | search_seconds")
    for name, dim, hit, seconds in rows:
        print(f"  {name:11s} | {dim:4d} | {hit!s:5s} | {seconds:.4f}")
