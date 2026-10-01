#!/usr/bin/env python
# pyright: basic
# Benchmark harness on numpy (no strict stubs); strict mode would only add reportUnknown* noise.
"""Exact top-k cosine retrieval over a memory-mapped vector cell, in bounded memory.

The obvious implementation loads the cell, normalises it, and runs one matrix-vector product per
query. It is exact and it is what this replaces, but it has two defects that only show up at
scale. It materialises the whole cell twice (``np.asarray`` on a memmap copies even when the
dtype already matches, then normalising allocates a second copy), which is 33 GB for a
1M x 4096 cell. And it re-reads the entire matrix once per query, so an 800-query run over a
2.9 GB cell moves 2.3 TB through memory.

This streams instead. One block of rows at a time is read, normalised, and multiplied against
ALL queries in a single gemm, and a running top-k per query is merged forward. Peak memory is
about twice the block budget regardless of how large the cell is, and the matrix is read once
per run rather than once per query.

Exactness is preserved: the same dot products, to within float32 gemm association. The ordering
is stricter than before - ties break on ascending row index, so the result no longer depends on
numpy's unstable quicksort, and re-running a cell returns the identical ranking. That matters
because duplicate chunk text is real in MIRACL and msmarco, so exact score ties do occur.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["normalize_rows", "topk_stream"]

# 256 MiB per block. Two of these (the row block and its score block) plus the query matrix is
# the whole footprint, so the default keeps a run comfortably under 1 GB on every cached cell.
_DEFAULT_BLOCK_BYTES = 256 * 1024 * 1024
_EPS = 1e-12


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise each row in place-ish, guarding the zero vector.

    Clipping rather than masking keeps a zero row at zero instead of producing NaN, so one
    degenerate chunk cannot poison a whole cell's ranking.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, _EPS, None)


def _block_rows(dim: int, n_queries: int, block_bytes: int) -> int:
    """Rows per block, bounding BOTH the row block and the score block it produces."""
    per_row = max(dim, n_queries) * 4  # float32; the score block is (rows x n_queries)
    return max(1, block_bytes // max(per_row, 1))


def _merge(
    best_scores: np.ndarray,
    best_index: np.ndarray,
    block_scores: np.ndarray,
    block_index: np.ndarray,
    fetch: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the best ``fetch`` of the running set and one block's candidates, per query."""
    scores = np.concatenate([best_scores, block_scores], axis=1)
    index = np.concatenate([best_index, block_index], axis=1)
    if scores.shape[1] <= fetch:
        return scores, index
    keep = np.argpartition(-scores, fetch - 1, axis=1)[:, :fetch]
    rows = np.arange(scores.shape[0])[:, None]
    return scores[rows, keep], index[rows, keep]


def _block_candidates(block: np.ndarray, offset: int, fetch: int) -> tuple[np.ndarray, np.ndarray]:
    """Per query, this block's best ``fetch`` scores and their GLOBAL row indices."""
    n_queries, n_rows = block.shape
    if n_rows <= fetch:
        return block, np.broadcast_to(np.arange(offset, offset + n_rows), (n_queries, n_rows)).copy()
    keep = np.argpartition(-block, fetch - 1, axis=1)[:, :fetch]
    rows = np.arange(n_queries)[:, None]
    return block[rows, keep], keep + offset


def topk_stream(
    vectors_path: Path | np.ndarray,
    queries: np.ndarray,
    *,
    fetch: int,
    block_bytes: int = _DEFAULT_BLOCK_BYTES,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-``fetch`` cosine matches for every query, streamed over the cell.

    Args:
        vectors_path: ``vectors.npy`` holding ``(n_chunks, dim)`` float32 passage vectors, or an
            in-memory array of the same shape. The array form exists so the store benchmark can
            compute exact ground truth for a row subset it already holds, using this same tested
            code rather than a second hand-rolled top-k.
        queries: ``(n_queries, dim)`` query vectors. Normalised here, so callers need not.
        fetch: how many candidates to keep per query. Over-fetch beyond the reporting k, because
            chunks are deduplicated to documents afterwards and a document can own many chunks.
        block_bytes: memory budget per streamed block.

    Returns:
        ``(index, score)``, both ``(n_queries, min(fetch, n_chunks))``, each row ordered by
        descending score with ties broken on ascending row index.

    Raises:
        ValueError: if the query dimension does not match the cell's.
    """
    # never np.asarray(...) on the memmap: that copies the whole cell
    mm = vectors_path if isinstance(vectors_path, np.ndarray) else np.load(vectors_path, mmap_mode="r")
    n_chunks, dim = int(mm.shape[0]), int(mm.shape[1])
    query_matrix = normalize_rows(np.asarray(queries, dtype=np.float32))
    if query_matrix.shape[1] != dim:
        raise ValueError(f"query dim {query_matrix.shape[1]} does not match cell dim {dim}")

    fetch = min(fetch, n_chunks)
    n_queries = query_matrix.shape[0]
    step = _block_rows(dim, n_queries, block_bytes)
    best_scores = np.empty((n_queries, 0), dtype=np.float32)
    best_index = np.empty((n_queries, 0), dtype=np.int64)

    for start in range(0, n_chunks, step):
        stop = min(start + step, n_chunks)
        block = normalize_rows(np.array(mm[start:stop], dtype=np.float32))  # copies the SLICE only
        scores = (block @ query_matrix.T).T  # (n_queries, block_rows), one gemm for all queries
        block_scores, block_index = _block_candidates(scores, start, fetch)
        best_scores, best_index = _merge(best_scores, best_index, block_scores, block_index, fetch)

    order = np.lexsort((best_index, -best_scores), axis=1)  # deterministic: score desc, then index asc
    rows = np.arange(n_queries)[:, None]
    return best_index[rows, order], best_scores[rows, order]
