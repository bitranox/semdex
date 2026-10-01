#!/usr/bin/env python
# pyright: basic
"""Surgically repair oversized chunks in the pre-embed cache - WITHOUT re-embedding everything.

Before the [chunker].enforce_max_tokens size-guard, the semantic/late chunk sets could hold
oversized chunks (a minified-JS/web-scrape blob as one un-splittable "sentence", up to ~43090
tokens vs a 512 target); the embedder truncated those at embed time and dropped the overflow.
This fixes the EXISTING cache cheaply:

  1. re-chunk ONLY the oversized chunks of a set (split them at token boundaries, losslessly),
     keeping every other chunk exactly as it was;
  2. for each embedder cell of that set, build a new vectors.npy that COPIES the unchanged chunks'
     existing vectors and embeds ONLY the new split pieces (a few thousand, not ~300k).

Cost: ~minutes (GPU) to ~15 min (fastembed CPU) per cell, never the days a full re-embed takes.
`restart2`'s slow vectors are reused (they ARE the "unchanged" vectors that get copied).

RUN ORDER: AFTER the running sweeps finish and the size-guard is active (else a later chunk pass
recreates giants). Env:

  CACHE_ROOT                cache dir (default /embeddings)
  SEMDEX_REPAIR_MAX_TOKENS  force ONE cap across every set. Unset by default, and normally left
                            unset: each set is held to its own max_tokens + overlap, read from
                            its meta.json. A flat cap is what left a t256 set repaired to 512.
  SEMDEX_REPAIR_TOKENIZER   tokenizer for the offset split (default gpt2)
  SEMDEX_REPAIR_ONLY        comma list of chunk-set dir names to repair (default: all oversized)
  SEMDEX_BENCH_OLLAMA_URL / SEMDEX_BENCH_OPENAI_URL  endpoints for ollama/openai embedder cells
  SEMDEX_REPAIR_EMBED_BATCH pieces per embed request (default 128; one request per cell exceeds
                            the adapter's fixed 60s timeout on a large set)
  SEMDEX_REPAIR_NUM_BATCH   ollama options.num_batch (default 4096; below the served context it
                            truncates long inputs silently or answers 400, which is never retried)
  MODE=dry_run (default, report only) | apply  (apply backs up each file to *.pre-repair)

RESUME: a cell is skipped when it already has a `vectors.npy.pre-repair` sibling, and the set's
`chunks.parquet` is rewritten only after every cell succeeds. So a crashed run is resumable by
re-running: finished cells are skipped, unfinished ones are rebuilt. If a run crashed under the
OLD ordering (parquet written first), its set is no longer selectable - restore its
`chunks.parquet` from `chunks.parquet.pre-repair` and re-run.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

OffsetsOf = Callable[[str], list[tuple[int, int]]]


def plan_repair(
    chunks: list[tuple[str, int]], *, max_tokens: int, offsets_of: OffsetsOf
) -> tuple[list[tuple[str, int]], list[int]]:
    """Return (new_chunks, carry): the repaired chunk list and, per new chunk, the OLD index whose
    vector to COPY (or -1 when the chunk is a fresh split piece that must be re-embedded).

    Non-oversized chunks pass through unchanged (carry = their old index); an oversized chunk is
    replaced by its lossless <=max_tokens split pieces (each carry = -1).
    """
    from semdex.adapters.chunker._base import split_oversized

    new_chunks: list[tuple[str, int]] = []
    carry: list[int] = []
    for old_index, (text, count) in enumerate(chunks):
        pieces = split_oversized(text, count, max_tokens=max_tokens, offsets_of=offsets_of)
        if pieces == [(text, count)]:
            new_chunks.append((text, count))
            carry.append(old_index)
        else:
            for piece in pieces:
                new_chunks.append(piece)
                carry.append(-1)
    return new_chunks, carry


def _offsets_of(tokenizer: str) -> OffsetsOf:
    """The token-offset function the size-guard uses (reused from the chonkie adapter)."""
    from semdex.adapters.chunker.chonkie import ChonkieChunker
    from semdex.domain.enums import ChunkStrategy

    return ChonkieChunker(ChunkStrategy.SEMANTIC, tokenizer=tokenizer)._offsets_of


def refresh_meta_count(chunk_dir: Path, rows: int) -> None:
    """Bring the set's recorded chunk count back in line with its parquet.

    The repair changes how many chunks a set holds, so leaving meta.json alone makes every
    consumer that trusts it disagree with the data by exactly the number of pieces the repair
    created. That is not cosmetic: a count read from meta rather than from the parquet is what a
    catalogue, a corpus-size table or a progress estimate quotes.
    """
    meta_path = chunk_dir / "meta.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text())
    if meta.get("count") == rows:
        return
    meta["count"] = rows
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))


def backup_path(path: Path) -> Path:
    """A free ``*.pre-repair`` name for ``path``, never an existing one.

    ``Path.rename`` overwrites silently on POSIX, so a second repair pass would have renamed the
    already-repaired file over the ORIGINAL backup and destroyed the only way back to the
    pre-repair data. Successive passes get .pre-repair, .pre-repair-2, .pre-repair-3.
    """
    candidate = path.with_name(path.name + ".pre-repair")
    generation = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.pre-repair-{generation}")
        generation += 1
    return candidate


def _marker_path(chunk_dir: Path) -> Path:
    return chunk_dir / ".repaired.json"


def cells_done_at(chunk_dir: Path, cap: int) -> set[str]:
    """Cell names already rebuilt for THIS cap, from the set's repair marker.

    Resume used to key off "does a vectors.npy.pre-repair exist", which is true for any earlier
    pass at any cap. A second pass at a tighter cap therefore skipped every previously repaired
    cell while still rewriting chunks.parquet, leaving the vectors at the OLD row count and the
    set permanently misaligned. Keying on the cap makes a re-repair at a new cap rebuild the cells
    it must, while a genuine crash-resume at the same cap still skips what it finished.
    """
    marker = _marker_path(chunk_dir)
    if not marker.exists():
        return set()
    state = json.loads(marker.read_text())
    return {name for name, done_cap in state.get("cells", {}).items() if int(done_cap) == cap}


def record_cell_done(chunk_dir: Path, cell_name: str, cap: int) -> None:
    marker = _marker_path(chunk_dir)
    state = json.loads(marker.read_text()) if marker.exists() else {"cells": {}}
    state.setdefault("cells", {})[cell_name] = cap
    marker.write_text(json.dumps(state, indent=2, sort_keys=True))


def cap_for(chunk_dir: Path, override: int | None) -> int:
    """The cap this chunk set must be held to: its OWN max_tokens plus its overlap.

    Reading one cap from the environment and applying it to every set is what left oversized
    chunks behind: a t256 set repaired against a flat 512 cap is still over ITS cap by design.
    Overlap is added because it is prepended context that deliberately overshoots max_tokens, so
    the true ceiling for a set built with overlap is max_tokens + overlap.
    """
    if override is not None:
        return override
    meta = json.loads((chunk_dir / "meta.json").read_text())
    return int(meta["max_tokens"]) + int(meta.get("overlap", 0) or 0)


def _oversized_sets(cache_root: Path, override: int | None) -> list[tuple[Path, int]]:
    """Every chunk set holding a chunk above ITS OWN cap, with that cap."""
    import pyarrow.parquet as pq

    hits: list[tuple[Path, int]] = []
    for meta in sorted((cache_root / "chunks").glob("*/meta.json")):
        parquet = meta.parent / "chunks.parquet"
        if not parquet.exists():
            continue
        cap = cap_for(meta.parent, override)
        tc = pq.read_table(parquet, columns=["token_count"]).column("token_count").to_numpy()
        if (tc > cap).any():
            hits.append((meta.parent, cap))
    return hits


def _embedder_for(label: str):
    """Map a cell's dir-safe embedder label back to (backend, model_id) via preembed's registry."""
    from preembed_vectors import _EMBED_MODELS, _dirsafe

    reverse = {_dirsafe(key): value for key, value in _EMBED_MODELS.items()}
    return reverse.get(label)


def _repair_set(chunk_dir: Path, cache_root: Path, max_tokens: int, tokenizer: str) -> None:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from semdex.adapters.chunker._base import split_oversized
    from semdex.composition import build_embedding
    from semdex.domain.enums import EmbeddingBackend

    table = pq.read_table(chunk_dir / "chunks.parquet")
    texts = table.column("text").to_pylist()
    counts = table.column("token_count").to_pylist()
    uris = table.column("source_uri").to_pylist()
    offsets = _offsets_of(tokenizer)
    new_chunks, carry = plan_repair(list(zip(texts, counts, strict=True)), max_tokens=max_tokens, offsets_of=offsets)
    if all(c != -1 for c in carry):
        return  # nothing split (all fell back to unsplittable) - leave the set untouched

    # per-new-chunk provenance: a giant's pieces inherit its source uri (ordinals re-numbered below)
    new_uris: list[str] = []
    for old_index, (text, count) in enumerate(zip(texts, counts, strict=True)):
        pieces = split_oversized(text, count, max_tokens=max_tokens, offsets_of=offsets)
        new_uris.extend([uris[old_index]] * len(pieces))

    # Rebuild each embedder cell FIRST, then rewrite chunks.parquet (see the ordering note in
    # main's docstring): the parquet is what makes a set selectable, so writing it last means a
    # crash mid-cells leaves the set still selectable and the run resumable.
    done = cells_done_at(chunk_dir, max_tokens)
    for cell in sorted((cache_root / "vectors").glob(f"{chunk_dir.name}__*")):
        label = cell.name.split("__", 2)[2]
        spec = _embedder_for(label)
        old_npy = cell / "vectors.npy"
        if cell.name in done:
            # Rebuilt already AT THIS CAP, so its vectors.npy is the new one and carrying old
            # indices into it would corrupt the cell. Cell-level resume within a pass.
            print(f"    skip {cell.name}: already repaired at cap {max_tokens}", flush=True)
            continue
        if spec is None or not old_npy.exists():
            print(f"    skip {cell.name}: no embedder for {label!r} or no vectors.npy", flush=True)
            continue
        backend, model_id = spec
        endpoint = None
        num_batch = None
        keep_alive = None
        timeout = None
        if backend is EmbeddingBackend.OLLAMA:
            endpoint = os.environ.get("SEMDEX_BENCH_OLLAMA_URL")
            # ollama's PHYSICAL batch defaults to 2048 tokens, below the 4096-token context it
            # serves these models with. Left unset, a long input is silently re-run truncated, and
            # a many-input request answers 400 - a client error retry_http never retries, which
            # kills an unattended pass outright.
            num_batch = int(os.environ.get("SEMDEX_REPAIR_NUM_BATCH", "4096"))
            # A set holds several ollama cells in a row (bge-m3, then qwen3 4b, then 8b). ollama's
            # default 5-minute keep_alive leaves the PREVIOUS model resident while the next loads,
            # and 4b (~5.6G) + 8b (~7.8G) plus the e5 shim does not fit the 16G card: the load
            # stalls or falls back to CPU, every request hits the timeout, and the retries are
            # exhausted. A short keep_alive still spans a cell (requests are seconds apart) but
            # evicts between cells.
            keep_alive = os.environ.get("SEMDEX_REPAIR_KEEP_ALIVE", "30s")
            # The adapter's 60s default does not cover a cold multi-GB model load, which is what
            # the FIRST request of each ollama cell pays for.
            timeout = float(os.environ.get("SEMDEX_REPAIR_TIMEOUT", "600"))
        elif backend is EmbeddingBackend.OPENAI:
            endpoint = os.environ.get("SEMDEX_BENCH_OPENAI_URL")
            timeout = float(os.environ.get("SEMDEX_REPAIR_TIMEOUT", "600"))
        embedding = build_embedding(
            backend,
            model=model_id,
            endpoint=endpoint,
            num_batch=num_batch,
            keep_alive=keep_alive,
            timeout=timeout,
            allow_fallback=False,
        )
        old = np.load(old_npy, mmap_mode="r")
        new = np.empty((len(new_chunks), embedding.dim), dtype="float32")
        to_embed_idx = [j for j, c in enumerate(carry) if c == -1]
        for j, c in enumerate(carry):
            if c != -1:
                new[j] = old[c]
        # Embed in sub-batches: one embed_passages call per cell sends every piece in a single
        # HTTP request, which on a large set exceeds the adapter's fixed 60s timeout.
        batch = int(os.environ.get("SEMDEX_REPAIR_EMBED_BATCH", "128"))
        for start in range(0, len(to_embed_idx), batch):
            window = to_embed_idx[start : start + batch]
            vecs = embedding.embed_passages([new_chunks[j][0] for j in window])
            for j, vec in zip(window, vecs, strict=True):
                new[j] = np.asarray(vec, dtype="float32")
        old_npy.rename(backup_path(old_npy))
        np.save(old_npy, new)
        record_cell_done(chunk_dir, cell.name, max_tokens)
        print(
            f"    {cell.name}: {len(to_embed_idx)} embedded, {len(new_chunks) - len(to_embed_idx)} copied", flush=True
        )

    # write the new chunks.parquet last (backup first)
    parquet_path = chunk_dir / "chunks.parquet"
    parquet_path.rename(backup_path(parquet_path))
    schema = pa.schema(
        [("text", pa.string()), ("source_uri", pa.string()), ("ordinal", pa.int32()), ("token_count", pa.int32())]
    )
    pq.write_table(
        pa.table(
            {
                "text": [t for t, _c in new_chunks],
                "source_uri": new_uris,
                "ordinal": list(range(len(new_chunks))),
                "token_count": [c for _t, c in new_chunks],
            },
            schema=schema,
        ),
        chunk_dir / "chunks.parquet",
    )
    refresh_meta_count(chunk_dir, len(new_chunks))


def main() -> int:
    cache_root = Path(os.environ.get("CACHE_ROOT", "/embeddings"))
    # Unset by default: each set is held to its own cap. The override exists only to force one
    # cap across every set, which is what the first repair pass did and why it left work behind.
    raw_override = os.environ.get("SEMDEX_REPAIR_MAX_TOKENS")
    override = int(raw_override) if raw_override else None
    tokenizer = os.environ.get("SEMDEX_REPAIR_TOKENIZER", "gpt2")
    sets = _oversized_sets(cache_root, override)
    # Repairing rewrites a cache that has no other copy, so it must be possible to prove the whole
    # path on ONE small set before committing to all of them.
    only = [name for name in os.environ.get("SEMDEX_REPAIR_ONLY", "").split(",") if name]
    if only:
        sets = [(d, cap) for d, cap in sets if d.name in only]
        print(f"SEMDEX_REPAIR_ONLY -> {len(sets)} of the oversized set(s) selected", flush=True)
    scope = f"a flat cap of {override}" if override else "each set's own max_tokens + overlap"
    print(f"=== {len(sets)} chunk set(s) above {scope} ===", flush=True)
    for chunk_dir, cap in sets:
        cells = list((cache_root / "vectors").glob(f"{chunk_dir.name}__*"))
        print(f"  {chunk_dir.name}: cap {cap}, {len(cells)} vector cell(s)", flush=True)
    if os.environ.get("MODE") != "apply":
        print("\nMODE=apply to repair (backs up each file to *.pre-repair; embeds only the split pieces).")
        return 0
    for chunk_dir, cap in sets:
        print(f"repairing {chunk_dir.name} at cap {cap} ...", flush=True)
        _repair_set(chunk_dir, cache_root, max_tokens=cap, tokenizer=tokenizer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
