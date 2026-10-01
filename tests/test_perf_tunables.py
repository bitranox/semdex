"""Performance sweeps validating the tunable [index] config defaults.

These are benchmarks, not fast unit tests: they build indices across a range of
each tunable and report time / size, asserting only robust, deterministic
invariants (correctness preserved across the sweep, monotonic relationships) -
never raw timing thresholds, which would flake. Marked ``local_only`` so they run
on demand (``make ti`` / ``pytest -m local_only -s``), not in the CI ``make test``.

Purpose per CLAUDE.md "Configurable values": confirm the built-in defaults
(max_tokens=256, embedding_dim=256, hash_chunk_bytes=65536) are sensible choices.
``max_file_bytes`` is a correctness guard (reject oversize files), not a
perf-swept value, so it is not swept here.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pytest

from semdex.adapters.discovery import discover_sources
from semdex.adapters.discovery.location import to_uri
from semdex.application.use_cases import index_sources, search
from semdex.composition import build_index_production

pytestmark = [pytest.mark.local_only, pytest.mark.os_agnostic]

_VOCAB = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa")


def _make_corpus(root: Path, *, n_files: int, words_per_file: int) -> Path:
    """Create a deterministic corpus of markdown files under a fresh directory."""
    corpus = root / "corpus"
    corpus.mkdir()
    for i in range(n_files):
        words = [_VOCAB[(i + j) % len(_VOCAB)] for j in range(words_per_file)]
        (corpus / f"doc{i:03d}.md").write_text(" ".join(words), encoding="utf-8")
    return corpus


def test_perf_sweep_max_tokens(tmp_path: Path) -> None:
    """Chunk count is non-increasing as max_tokens grows; report index time."""
    corpus = _make_corpus(tmp_path, n_files=24, words_per_file=400)
    sources = discover_sources([corpus])
    rows: list[tuple[int, int, float]] = []
    for max_tokens in (64, 128, 256, 512, 1024):
        services = build_index_production(tmp_path / f"mt_{max_tokens}", embedding_dim=64)
        start = time.perf_counter()
        report = index_sources(
            extract=services.extract,
            chunk=services.chunk,
            embedding=services.embedding,
            store=services.store_writer,
            collection="c",
            sources=sources,
            max_tokens=max_tokens,
        )
        rows.append((max_tokens, report.chunks_indexed, time.perf_counter() - start))

    chunk_counts = [chunks for _, chunks, _ in rows]
    assert chunk_counts == sorted(chunk_counts, reverse=True), chunk_counts  # bigger chunks -> fewer chunks

    print("\nmax_tokens sweep (default 256):  tokens | chunks | index_seconds")
    for max_tokens, chunks, seconds in rows:
        print(f"  {max_tokens:5d} | {chunks:6d} | {seconds:.3f}")


def test_perf_sweep_embedding_dim(tmp_path: Path) -> None:
    """Store size grows with dim; the obvious match stays top-1 across dims."""
    corpus = _make_corpus(tmp_path, n_files=24, words_per_file=400)
    target = corpus / "target.md"
    target.write_text("quantum entanglement physics " * 40, encoding="utf-8")
    sources = discover_sources([corpus])
    rows: list[tuple[int, int, float]] = []
    for dim in (16, 32, 64, 128, 256, 512):
        store_dir = tmp_path / f"dim_{dim}"
        services = build_index_production(store_dir, embedding_dim=dim)
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
        hits = search(
            embedding=services.embedding, store=services.store_reader, collection="c", query="quantum entanglement", k=3
        )
        elapsed = time.perf_counter() - start
        assert hits[0].uri == to_uri(target), f"dim={dim} lost the obvious match"
        rows.append((dim, (store_dir / "store.json").stat().st_size, elapsed))

    sizes = [size for _, size, _ in rows]
    assert sizes == sorted(sizes), sizes  # vector store grows with dimension

    print("\nembedding_dim sweep (default 256):  dim | store_bytes | search_seconds")
    for dim, size, seconds in rows:
        print(f"  {dim:4d} | {size:9d} | {seconds:.4f}")


def test_perf_sweep_hash_chunk_bytes(tmp_path: Path) -> None:
    """hash_chunk_bytes changes only speed, never the hash; report timing."""
    big = tmp_path / "big.txt"
    big.write_bytes(("lorem ipsum dolor sit amet " * 200_000).encode("utf-8"))  # ~5 MB
    expected = hashlib.sha256(big.read_bytes()).hexdigest()
    rows: list[tuple[int, float]] = []
    hashes: set[str] = set()
    for chunk in (4096, 16384, 65536, 262144, 1048576):
        start = time.perf_counter()
        source = discover_sources([big], hash_chunk_size=chunk)[0]
        rows.append((chunk, time.perf_counter() - start))
        hashes.add(source.content_hash)

    assert hashes == {expected}  # chunk size is a pure perf knob; result is identical

    print("\nhash_chunk_bytes sweep (default 65536):  chunk_bytes | hash_seconds")
    for chunk, seconds in rows:
        print(f"  {chunk:8d} | {seconds:.4f}")
