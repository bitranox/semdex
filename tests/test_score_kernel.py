"""Equivalence and bounding for scripts/_score_kernel.py.

The kernel replaces a load-the-whole-cell-and-loop-per-query implementation whose results are
already published, so the bar is that it returns the SAME ranking, not merely a plausible one.
Every test here compares against a naive reference written inline: if both were refactored
together the test would prove nothing, so the reference stays deliberately dumb and separate.

On the memory claim: tracemalloc is NOT used. It observes Python's allocators, and numpy arrays
come from numpy's own, so a tracemalloc ceiling here passes whether or not the kernel streams -
a well-formed assertion that asserts nothing. Instead the block size is asserted directly as the
pure function it is, the equivalence tests run on a cell many times the block budget so the
streaming path is genuinely exercised, and the process-level peak RSS is measured for real on the
largest cached cell and recorded in tests/benchmarks/raw/README.md.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "_score_kernel.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("_score_kernel", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def kernel() -> Any:
    return _load()


def _reference(
    vectors: npt.NDArray[np.float32], queries: npt.NDArray[np.float32], fetch: int
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float32]]:
    """The implementation being replaced: normalise everything, one query at a time, argsort.

    Fully dtype-annotated rather than using a bare np.ndarray: unparameterised, pyright strict
    reads it as ndarray[Any, dtype[Any]] and every downstream expression becomes partially
    unknown, which is a real loss of checking and not just linter noise.
    """
    norms: npt.NDArray[np.float32] = np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None).astype(
        np.float32
    )
    normalized: npt.NDArray[np.float32] = vectors / norms
    index_rows: list[npt.NDArray[np.intp]] = []
    score_rows: list[npt.NDArray[np.float32]] = []
    for row in range(queries.shape[0]):
        query: npt.NDArray[np.float32] = queries[row]
        unit: npt.NDArray[np.float32] = query / max(float(np.linalg.norm(query)), 1e-12)
        sims: npt.NDArray[np.float32] = normalized @ unit
        # A STABLE argsort on the negated scores is exactly "descending score, ties by ascending
        # index", which is the kernel's tie-break, and it says so more directly than lexsort.
        order: npt.NDArray[np.intp] = np.argsort(-sims, kind="stable")[:fetch]
        index_rows.append(order)
        score_rows.append(sims[order])
    return np.asarray(index_rows), np.asarray(score_rows)


def _write_cell(tmp_path: Path, vectors: npt.NDArray[np.float32]) -> Path:
    path = tmp_path / "vectors.npy"
    np.save(path, vectors.astype(np.float32))
    return path


@pytest.mark.parametrize("block_bytes", [1 << 30, 1 << 14, 1 << 11])
@pytest.mark.parametrize("n_queries", [1, 20])
def test_matches_the_reference_at_every_block_size(
    kernel: Any, tmp_path: Path, block_bytes: int, n_queries: int
) -> None:
    """Same ranking and same scores however many blocks the cell is cut into."""
    rng = np.random.default_rng(20260806)
    vectors = rng.standard_normal((5000, 64), dtype=np.float32)
    queries = rng.standard_normal((n_queries, 64), dtype=np.float32)
    path = _write_cell(tmp_path, vectors)

    index, score = kernel.topk_stream(path, queries, fetch=25, block_bytes=block_bytes)
    ref_index, ref_score = _reference(vectors, queries, 25)

    np.testing.assert_array_equal(index, ref_index)
    np.testing.assert_allclose(score, ref_score, atol=1e-6)


def test_ties_break_on_ascending_index_whatever_the_block_size(kernel: Any, tmp_path: Path) -> None:
    """Duplicate chunk text is real in MIRACL and msmarco, so exact score ties must be stable.

    Without a defined tie-break the order comes from an unstable quicksort and changes with the
    block boundaries, so two runs of the same cell can disagree on which duplicate ranks first.
    """
    rng = np.random.default_rng(7)
    unique = rng.standard_normal((40, 16), dtype=np.float32)
    vectors = np.repeat(unique, 5, axis=0)  # every vector appears 5 times, so every score ties
    queries = rng.standard_normal((3, 16), dtype=np.float32)
    path = _write_cell(tmp_path, vectors)

    runs = [kernel.topk_stream(path, queries, fetch=20, block_bytes=b)[0] for b in (1 << 30, 1 << 12, 1 << 9)]
    for other in runs[1:]:
        np.testing.assert_array_equal(runs[0], other)

    scores = kernel.topk_stream(path, queries, fetch=20, block_bytes=1 << 9)[1]
    for row_index, row_score in zip(runs[0], scores, strict=True):
        tied = row_index[np.isclose(row_score, row_score[0])]
        assert list(tied) == sorted(tied), "tied candidates must be ordered by ascending row index"


def test_block_rows_bounds_both_the_row_block_and_the_score_block(kernel: Any) -> None:
    """The pure bound, asserted directly rather than inferred from a memory reading."""
    budget = 256 * 1024 * 1024
    for dim, n_queries in ((4096, 1000), (64, 100000), (768, 1)):
        rows = kernel._block_rows(dim, n_queries, budget)
        assert rows >= 1
        assert rows * dim * 4 <= budget, "row block exceeds the budget"
        assert rows * n_queries * 4 <= budget, "score block exceeds the budget"


def test_a_cell_larger_than_the_block_budget_is_still_exact(kernel: Any, tmp_path: Path) -> None:
    """Many blocks, and the merge must not lose a candidate that appeared in an early one."""
    rng = np.random.default_rng(99)
    vectors = rng.standard_normal((3000, 32), dtype=np.float32)
    # Plant the best match for query 0 in the FIRST block so a merge that forgets earlier blocks
    # would drop it. A random cell can pass that bug by luck; this one cannot.
    queries = rng.standard_normal((2, 32), dtype=np.float32)
    vectors[3] = queries[0] * 10.0
    path = _write_cell(tmp_path, vectors)

    index, _ = kernel.topk_stream(path, queries, fetch=10, block_bytes=1 << 10)
    assert index[0][0] == 3
    ref_index, _ = _reference(vectors, queries, 10)
    np.testing.assert_array_equal(index, ref_index)


def test_fetch_larger_than_the_cell_returns_every_row(kernel: Any, tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    vectors = rng.standard_normal((12, 8), dtype=np.float32)
    path = _write_cell(tmp_path, vectors)

    index, score = kernel.topk_stream(path, queries=rng.standard_normal((2, 8), dtype=np.float32), fetch=500)

    assert index.shape == (2, 12)
    assert score.shape == (2, 12)
    for row in index:
        assert sorted(row) == list(range(12))


def test_a_dimension_mismatch_fails_loudly(kernel: Any, tmp_path: Path) -> None:
    """A silent shape error here would surface as an opaque gemm failure far from the cause."""
    path = _write_cell(tmp_path, np.ones((10, 16), dtype=np.float32))

    with pytest.raises(ValueError, match="does not match cell dim"):
        kernel.topk_stream(path, np.ones((2, 8), dtype=np.float32), fetch=5)


def test_a_zero_vector_does_not_poison_the_ranking(kernel: Any, tmp_path: Path) -> None:
    """One degenerate chunk must score 0, not NaN, or it takes the whole cell's ordering with it."""
    rng = np.random.default_rng(11)
    vectors = rng.standard_normal((50, 8), dtype=np.float32)
    vectors[7] = 0.0
    path = _write_cell(tmp_path, vectors)

    _, score = kernel.topk_stream(path, rng.standard_normal((2, 8), dtype=np.float32), fetch=50)

    assert not np.isnan(score).any()


def test_an_in_memory_array_gives_the_same_answer_as_the_file(kernel: Any, tmp_path: Path) -> None:
    """The store benchmark computes ground truth from a row subset it already holds in memory.

    That path must not be a second, untested top-k implementation, so it reuses this kernel and
    this test pins the two forms together.
    """
    rng = np.random.default_rng(21)
    vectors = rng.standard_normal((2000, 24), dtype=np.float32)
    queries = rng.standard_normal((5, 24), dtype=np.float32)
    path = _write_cell(tmp_path, vectors)

    from_file = kernel.topk_stream(path, queries, fetch=15, block_bytes=1 << 12)
    from_array = kernel.topk_stream(vectors, queries, fetch=15, block_bytes=1 << 12)

    np.testing.assert_array_equal(from_file[0], from_array[0])
    np.testing.assert_allclose(from_file[1], from_array[1], atol=1e-6)
