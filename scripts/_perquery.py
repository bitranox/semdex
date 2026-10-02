#!/usr/bin/env python
# pyright: basic
# Benchmark harness on numpy (no strict stubs); strict mode would only add reportUnknown* noise.
"""Write one cell's per-query score arrays, refusing to replace a file of a different kind.

Every scorer persists ``<dir>/<cell>.npz`` and references it from its score row by hash. Three of
them name a cell ``<corpus>__<profile>__<embedder>``, so two scorers that share a directory share
filenames, and the second run replaces the first one's arrays with no error anywhere: the
product-k scorer once did exactly that to the dense arrays of every cell it scored, and the
exporter only found out when it tried to read them. A directory per scorer is the first defence;
this check is what holds when somebody points two scorers at one directory again.

The kind of a file is its set of array names. Rewriting a file of the same kind is a re-score and
is allowed; anything else is refused before a byte is written. The write goes to a temporary file
that replaces the target in one rename, so an interrupted run never leaves a truncated archive
under a name the exporter trusts.
"""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

__all__ = ["write_per_query"]


def write_per_query(root: Path, cell: str, arrays: Mapping[str, np.ndarray]) -> str:
    """Write ``<root>/<cell>.npz`` from ``arrays`` and return the file's sha256.

    Args:
        root: The per-query directory of the calling scorer.
        cell: The cell name; the file is ``<cell>.npz``.
        arrays: Array name to array. The names are the file's kind.

    Returns:
        The sha256 hex digest of the written file, for the score row to reference.

    Raises:
        ValueError: ``<cell>.npz`` already exists with a different set of array names, or cannot
            be read as an npz archive. The existing file is left untouched.

    Examples:
        >>> import tempfile
        >>> root = Path(tempfile.mkdtemp())
        >>> len(write_per_query(root, "c", {"qids": np.asarray(["q1"]), "ndcg": np.zeros(1)}))
        64
        >>> write_per_query(root, "c", {"qids": np.asarray(["q1"]), "ks": np.zeros(1)})  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: c.npz holds a different kind of per-query file: it has ['ndcg', 'qids'], ...
    """
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{cell}.npz"
    _refuse_a_different_kind(path, set(arrays))
    tmp = path.with_name(f"{path.name}.tmp")
    # Written member by member, the way np.savez does it, because unpacking a mapping into savez
    # would bind any member called "allow_pickle" to that parameter instead of writing it.
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, array in arrays.items():
            with archive.open(f"{name}.npy", "w", force_zip64=True) as member:
                np.lib.format.write_array(member, np.asanyarray(array), allow_pickle=False)
    tmp.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refuse_a_different_kind(path: Path, names: set[str]) -> None:
    if not path.exists():
        return
    try:
        with np.load(path, allow_pickle=False) as existing:
            have = set(existing.files)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError(f"{path.name} exists but is not a readable npz archive ({exc}); not replacing it") from exc
    if have != names:
        raise ValueError(
            f"{path.name} holds a different kind of per-query file: it has {sorted(have)}, this write has "
            f"{sorted(names)}; two scorers are writing to one directory ({path.parent}). Point them at "
            "separate directories; the existing file is untouched"
        )
