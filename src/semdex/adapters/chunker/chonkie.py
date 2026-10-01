"""chonkie chunker adapter: recursive (default), semantic (SDPM), and late.

``RECURSIVE`` uses ``RecursiveChunker`` with the markdown recipe (token-accurate,
no embedding spend) plus ``OverlapRefinery`` when overlap>0; the recipe falls
back to chonkie's default rules if it cannot be fetched (offline). ``SEMANTIC``
uses ``SemanticChunker`` with a skip window (the SDPM double-pass merge) and
``LATE`` uses ``LateChunker``; both need ``semdex[chunk-semantic]`` and only beat
recursive with a real embedding provider. The underlying chunker is built once
and rebuilt only if ``max_tokens`` changes.
"""

from __future__ import annotations

import importlib
import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

from ...domain.enums import ChunkStrategy
from ...domain.errors import ChunkingError
from ._base import chunks_from_pairs, split_oversized

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from ...domain.models import Chunk, ExtractedDocument

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 256


def _whitespace_offsets(text: str) -> list[tuple[int, int]]:
    """Fallback token offsets (one per whitespace word) when the tokenizer is a bare string."""
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for word in text.split(" "):
        start = text.index(word, cursor) if word else cursor
        offsets.append((start, start + len(word)))
        cursor = start + len(word)
    return offsets


# Tokenizer names chonkie tokenizes locally (no model, no network) - pass them through as-is.
_CHONKIE_BUILTIN_TOKENIZERS = frozenset({"character", "word", "byte", "row"})
# Bare HuggingFace ids whose local cache lives under the canonical org-qualified repo, so the
# bare id is not found offline. Map to the cached repo id. Extend as more tokenizers are used.
_TOKENIZER_ALIASES = {"gpt2": "openai-community/gpt2"}


@lru_cache(maxsize=16)
def resolve_tokenizer(name: str) -> Any:
    """Resolve a tokenizer NAME to a concrete ``tokenizers.Tokenizer`` object once (cached).

    Handing chonkie a tokenizer *string* makes it resolve through its ``tokie`` backend, whose
    ``from_pretrained`` issues a blocking hub request that ignores ``HF_HUB_OFFLINE`` and stalls
    for minutes on a slow HuggingFace CDN (it only bites the overlap path, ``OverlapRefinery``,
    so it is invisible until overlap>0). Passing a pre-built ``Tokenizer`` object skips that path
    entirely. Builtin names (``character``/``word``/...) tokenize locally, so they pass through;
    an unresolvable name also falls back to the string so chonkie can try its own resolution.

    Upstream issue: https://github.com/feyninc/chonkie/issues/631
    """
    if name in _CHONKIE_BUILTIN_TOKENIZERS:
        return name
    try:
        tokenizers: Any = importlib.import_module("tokenizers")
    except ImportError:
        return name
    for repo_id in (name, _TOKENIZER_ALIASES.get(name)):
        if not repo_id:
            continue
        try:
            return tokenizers.Tokenizer.from_pretrained(repo_id)
        except Exception as exc:  # any load failure (offline miss, bad id) falls back to the string
            logger.debug("tokenizer %r not loadable as %r (%s)", name, repo_id, exc)
    return name


class ChonkieChunker:
    """Chunk a document with chonkie's recursive/semantic/late chunkers."""

    def __init__(  # noqa: PLR0913 - config-holding constructor; each arg is a distinct chunk setting
        self,
        strategy: ChunkStrategy,
        *,
        recipe: str = "markdown",
        overlap: int = 0,
        tokenizer: str = "gpt2",
        semantic_model: str | Any | None = None,
        enforce_max_tokens: bool = True,
    ) -> None:
        self._strategy = strategy
        self._recipe = recipe
        self._overlap = overlap
        self._tokenizer = tokenizer
        # Lossless size-guard: split any chunk over max_tokens into token windows. The semantic/late
        # strategies group at sentence granularity, so an un-splittable "sentence" (a minified-JS blob)
        # can overflow far past the target; without this the embedder truncates it and drops the rest.
        self._enforce_max_tokens = enforce_max_tokens
        # The embedding model SemanticChunker/LateChunker cut boundaries on. None keeps chonkie's
        # own default (potion-base-32M), which is English-distilled - set an explicit multilingual
        # model for non-English corpora, where the default finds boundaries on text it cannot read.
        # A str is resolved by chonkie locally (model2vec/sentence-transformers); a pre-built chonkie
        # BaseEmbeddings object is passed through as-is (e.g. an endpoint-backed breakpoint model).
        self._semantic_model = semantic_model
        self._cached: tuple[int, Any] | None = None

    def __call__(self, document: ExtractedDocument, *, max_tokens: int = _DEFAULT_MAX_TOKENS) -> list[Chunk]:
        chunker = self._resolve(max_tokens)
        raw = chunker(document.text)
        if self._strategy is ChunkStrategy.RECURSIVE and self._overlap > 0:
            raw = self._overlapped(raw)
        pairs: Iterable[tuple[str, int]] = ((chunk.text, chunk.token_count) for chunk in raw)
        if self._strategy is ChunkStrategy.LATE:
            # chonkie's LateChunker reports token_count 1 for every chunk of a long document (a
            # 30,000-character court decision: 60 of 60 chunks) while counting a short one
            # correctly. A count of 1 satisfies any cap, so the size-guard below would pass an
            # oversized late chunk through to be clipped by the embedder. The count is therefore
            # never taken from chonkie for late chunks: it is re-measured in the configured
            # tokenizer, the unit recursive chunks are counted in.
            pairs = ((chunk.text, len(self._offsets_of(chunk.text))) for chunk in raw)
        # Only the embedding-driven strategies overflow (sentence-granularity grouping of an
        # un-splittable blob). recursive caps already, and its overlap intentionally pushes a chunk
        # slightly over max_tokens - guarding it would wrongly re-split those overlap chunks.
        if self._enforce_max_tokens and self._strategy in (ChunkStrategy.SEMANTIC, ChunkStrategy.LATE):
            pairs = self._guard(pairs, max_tokens)
        return chunks_from_pairs(pairs, source=document.source)

    def _guard(self, pairs: Iterable[tuple[str, int]], max_tokens: int) -> Iterator[tuple[str, int]]:
        """Split any oversized (text, token_count) pair into <=max_tokens pieces, losslessly."""
        for text, count in pairs:
            yield from split_oversized(text, count, max_tokens=max_tokens, offsets_of=self._offsets_of)

    def _offsets_of(self, text: str) -> list[tuple[int, int]]:
        """Per-token (start, end) char offsets from the resolved tokenizer; whitespace fallback.

        ``resolve_tokenizer`` returns either a bare string (builtin/unresolvable name) or a
        ``tokenizers.Tokenizer`` object; the latter's ``encode(text).offsets`` is read at this
        optional-dependency boundary via ``getattr`` and coerced to a concrete ``list[tuple[int, int]]``.
        """
        tok: Any = resolve_tokenizer(self._tokenizer)
        offsets: list[tuple[int, int]] = []
        if callable(getattr(tok, "encode", None)):
            try:
                # boundary read of the optional-dep Tokenizer's Encoding.offsets
                offsets = cast("list[tuple[int, int]]", tok.encode(text).offsets)
            except Exception:
                # a bare-string tokenizer, or an encode quirk of whichever tokenizers
                # build is installed: leave offsets empty and fall back to whitespace.
                offsets = []
        return list(offsets) if offsets else _whitespace_offsets(text)

    def _resolve(self, max_tokens: int) -> Any:
        if self._cached is not None and self._cached[0] == max_tokens:
            return self._cached[1]
        chunker = self._build(max_tokens)
        self._cached = (max_tokens, chunker)
        return chunker

    def _build(self, max_tokens: int) -> Any:
        try:
            import chonkie
        except ImportError as exc:
            raise ChunkingError("chonkie is not installed; install semdex[chunk]") from exc
        if self._strategy is ChunkStrategy.RECURSIVE:
            return self._build_recursive(chonkie, max_tokens)
        # Pass embedding_model only when set (explicit calls, not a **spread, so the typed
        # chonkie kwargs stay checked), so an unset model keeps chonkie's own default.
        model = self._semantic_model
        if self._strategy is ChunkStrategy.SEMANTIC:
            # A non-zero skip window makes SemanticChunker do the SDPM double-pass merge.
            if model:
                return chonkie.SemanticChunker(chunk_size=max_tokens, skip_window=1, embedding_model=model)
            return chonkie.SemanticChunker(chunk_size=max_tokens, skip_window=1)
        if model:
            return chonkie.LateChunker(chunk_size=max_tokens, embedding_model=model)
        return chonkie.LateChunker(chunk_size=max_tokens)

    def _build_recursive(self, chonkie: Any, max_tokens: int) -> Any:
        tokenizer = resolve_tokenizer(self._tokenizer)
        if self._recipe:
            try:
                return chonkie.RecursiveChunker.from_recipe(
                    self._recipe, lang="en", tokenizer=tokenizer, chunk_size=max_tokens
                )
            except Exception as exc:
                logger.warning(
                    "chonkie recipe %r unavailable (%s); using the default recursive rules", self._recipe, exc
                )
        return chonkie.RecursiveChunker(tokenizer=tokenizer, chunk_size=max_tokens)

    def _overlapped(self, raw: Any) -> Any:
        import chonkie

        return chonkie.OverlapRefinery(tokenizer=resolve_tokenizer(self._tokenizer), context_size=self._overlap)(raw)


# Static conformance assertion -- pyright verifies ChonkieChunker satisfies ChunkText.
if TYPE_CHECKING:
    from ...application.ports import ChunkText

    _assert_chunk: ChunkText = ChonkieChunker(ChunkStrategy.RECURSIVE)


__all__ = [
    "ChonkieChunker",
    "resolve_tokenizer",
]
