"""Best-effort startup warm-up for the server-backed providers.

At ``serve`` startup semdex issues one throwaway call per http-backed provider so
the model is resident on the GPU before the first real query pays the cold-load
cost. Warm-up is BEST-EFFORT: any failure (server not up yet, model still
pulling) is logged and swallowed, never fatal - the provider's own retry/self-heal
path covers a real outage on the first live call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...application.ports import EmbeddingProvider, SummaryProvider

logger = logging.getLogger(__name__)

# The dummy text each warm-up call sends; short so a warm-up is cheap.
_WARMUP_TEXT = "warmup"


def warmup_embedding(provider: EmbeddingProvider) -> bool:
    """Issue one throwaway ``embed_query`` so the model loads; ``True`` on success.

    Any exception is logged and swallowed (warm-up must never fail startup).
    """
    try:
        provider.embed_query(_WARMUP_TEXT)
    except Exception as exc:
        # Broad by design: warm-up is best-effort; any failure must never be fatal.
        logger.warning("embedding warm-up failed (%s); continuing", exc)
        return False
    return True


def warmup_summary(provider: SummaryProvider) -> bool:
    """Issue one throwaway ``summarize`` so the chat model loads; ``True`` on success.

    Any exception is logged and swallowed (warm-up must never fail startup).
    """
    try:
        provider.summarize(_WARMUP_TEXT)
    except Exception as exc:
        # Broad by design: warm-up is best-effort; any failure must never be fatal.
        logger.warning("summary warm-up failed (%s); continuing", exc)
        return False
    return True


__all__ = [
    "warmup_embedding",
    "warmup_summary",
]
