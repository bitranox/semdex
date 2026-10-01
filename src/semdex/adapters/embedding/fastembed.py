"""fastembed embedding provider (ONNX, no torch; the default provider).

Wraps a fastembed ``TextEmbedding`` model, using its built-in query/passage
asymmetry (``query_embed`` / ``passage_embed``). The model is a light ONNX model
(``BAAI/bge-small-en-v1.5`` by default), offline after the first download.
fastembed is a core dependency; the composition root still falls back to the
placeholder if a model cannot be loaded (e.g. offline first run).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from ...domain.errors import EmbeddingError
from ._base import CallableEmbedding, to_vector, to_vectors

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...application.ports import EmbeddingProvider
    from ...domain.models import Vector

# Light, good-quality default (384-dim); overridable via [embedding].model.
_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def resolve_cache_dir(cache_dir: str | None) -> str:
    """Resolve where fastembed stores its downloaded models, preferring a PERSISTENT dir.

    fastembed's own default cache lives under the system temp dir, which a reboot wipes;
    an offline load then fails and the composition root silently degrades to the
    placeholder, caching junk vectors. So the adapter resolves its own default:
    an explicit ``cache_dir`` wins, else ``FASTEMBED_CACHE_PATH`` (the env override
    fastembed itself honors), else the per-user cache dir - all persistent across reboots.
    """
    if cache_dir:
        return cache_dir
    env = os.environ.get("FASTEMBED_CACHE_PATH")
    if env:
        return env
    from platformdirs import user_cache_dir

    return str(Path(user_cache_dir("semdex")) / "fastembed")


def load_fastembed_embedding(
    model_id: str | None = None, *, threads: int | None = None, cache_dir: str | None = None
) -> EmbeddingProvider:
    """Load a fastembed model and wrap it as an EmbeddingProvider.

    ``threads`` sets onnxruntime's thread count; ``None`` uses all CPUs. Passing an
    explicit count is fastembed's recommended way to avoid the noisy
    ``pthread_setaffinity_np`` warnings onnxruntime emits in cgroup-restricted
    environments (containers). ``cache_dir`` overrides where models are cached; when
    ``None`` a persistent per-user dir is used (see :func:`resolve_cache_dir`) rather
    than fastembed's reboot-wiped temp default. Raises
    :class:`~semdex.domain.errors.EmbeddingError` if fastembed is missing or the model
    cannot be loaded.
    """
    resolved = model_id or _DEFAULT_MODEL
    resolved_threads = threads if threads is not None else os.cpu_count()
    resolved_cache = resolve_cache_dir(cache_dir)
    try:
        from fastembed import TextEmbedding  # type: ignore  # optional-at-runtime; see module docstring
    except ImportError as exc:  # pragma: no cover - fastembed is a core dep
        raise EmbeddingError("fastembed is not installed; install semdex[fastembed]") from exc
    try:
        model = TextEmbedding(model_name=resolved, threads=resolved_threads, cache_dir=resolved_cache)  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
    except Exception as exc:
        raise EmbeddingError(f"fastembed could not load model '{resolved}': {exc}") from exc

    def passages_fn(texts: Sequence[str]) -> list[Vector]:
        return to_vectors(model.passage_embed(list(texts)))  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]

    def query_fn(text: str) -> Vector:
        return to_vector(next(iter(model.query_embed([text]))))  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]

    dim = int(getattr(model, "embedding_size", 0)) or len(query_fn("dimension probe"))
    return CallableEmbedding(model_id=resolved, dim=dim, passages_fn=passages_fn, query_fn=query_fn)


__all__ = [
    "load_fastembed_embedding",
    "resolve_cache_dir",
]
