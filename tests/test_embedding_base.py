"""Unit tests for the to_vector/to_vectors normalization helpers.

``CallableEmbedding`` delegation is already covered by
tests/test_embedding_factory.py::test_callable_embedding_delegates; this file
covers only the row-normalization helpers in adapters/embedding/_base.py.
"""

from __future__ import annotations

import pytest

from semdex.adapters.embedding._base import to_vector, to_vectors

pytestmark = pytest.mark.os_agnostic


class _FakeArray:
    """A minimal stand-in for a numpy row: exposes only ``.tolist()``."""

    def tolist(self) -> list[float]:
        return [1.5, 2.5]


def test_to_vector_converts_plain_sequence_to_float_tuple() -> None:
    """A plain sequence of ints is normalized to a tuple of floats."""
    assert to_vector([1, 2, 3]) == (1.0, 2.0, 3.0)


def test_to_vector_converts_array_like_via_tolist() -> None:
    """An object exposing .tolist() (numpy row) is normalized the same way."""
    assert to_vector(_FakeArray()) == (1.5, 2.5)


def test_to_vectors_maps_over_rows() -> None:
    """to_vectors applies to_vector to every row, preserving order."""
    assert to_vectors([[1, 2], [3, 4]]) == [(1.0, 2.0), (3.0, 4.0)]
