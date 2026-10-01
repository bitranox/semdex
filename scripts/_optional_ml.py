"""Typed access to the heavy ML packages the benchmark scripts load at run time.

torch and sentence-transformers arrive with the ``embed`` extra, which CI's ``.[dev]`` install
leaves out on purpose: torch alone is several gigabytes per job. A static ``import torch`` is then
unresolvable for pyright in CI. The scripts therefore reach both packages through this module,
which imports them by name when called and states, as Protocols, the small surface the scripts
use. Pyright checks every call site against those Protocols with or without the extra installed.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Sequence
from typing import Protocol, cast

__all__ = ["CrossEncoderModel", "load_cross_encoder", "set_torch_threads"]


class _Torch(Protocol):
    def set_num_threads(self, num: int, /) -> None: ...


class CrossEncoderModel(Protocol):
    """The part of ``sentence_transformers.CrossEncoder`` the rerank sweep uses."""

    def predict(
        self, sentences: Sequence[tuple[str, str]], *, batch_size: int, show_progress_bar: bool
    ) -> Iterable[float]: ...


class _CrossEncoderFactory(Protocol):
    def __call__(self, model_name: str, *, device: str, max_length: int) -> CrossEncoderModel: ...


class _SentenceTransformers(Protocol):
    CrossEncoder: _CrossEncoderFactory


def set_torch_threads(threads: int) -> None:
    """Cap torch's CPU thread pool.

    Args:
        threads: Number of threads torch may use for intra-op parallelism.

    Raises:
        ModuleNotFoundError: torch is not installed (the ``embed`` extra is absent).
    """
    torch = cast("_Torch", importlib.import_module("torch"))
    torch.set_num_threads(threads)


def load_cross_encoder(name: str, *, max_length: int) -> CrossEncoderModel:
    """Load a sentence-transformers cross-encoder on the CPU.

    Args:
        name: Hugging Face model id, for example ``BAAI/bge-reranker-base``.
        max_length: Token limit for each (query, passage) pair.

    Returns:
        The loaded model.

    Raises:
        ModuleNotFoundError: sentence-transformers is not installed (the ``embed`` extra is absent).
    """
    module = cast("_SentenceTransformers", importlib.import_module("sentence_transformers"))
    return module.CrossEncoder(name, device="cpu", max_length=max_length)
