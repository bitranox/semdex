"""Content-sanitizer configuration parsed from the ``[sanitize]`` section.

Opt-in (``enabled=false`` by default). When on, the composition wraps the extractor in a
``SanitizingExtractor`` that detects machine-content spans (minified JS/CSS, base64, dense blobs)
and - per ``mode`` - flags them (``dry_run``, text unchanged) or strips them before chunking.
All thresholds are config, not magic numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from lib_layered_config import Config


class SanitizeConfig(BaseModel):
    """Validated, immutable content-sanitizer settings.

    Example:
        >>> SanitizeConfig().enabled
        False
        >>> SanitizeConfig().mode
        'dry_run'
    """

    model_config = ConfigDict(frozen=True)

    # Opt-in. Off by default: never remove content until a retrieval (nDCG) check justifies it.
    enabled: bool = False
    # dry_run = detect + log only (text unchanged); strip = also excise the spans.
    mode: Literal["dry_run", "strip"] = "dry_run"
    # Detection granularity: text is scored in windows of this many characters.
    window_chars: int = Field(default=400, gt=0)
    # Only spans at least this long are acted on (a short code snippet in prose is left alone).
    min_span_chars: int = Field(default=400, gt=0)
    # A window is junk-like when its whitespace ratio is below this (minified code/base64).
    whitespace_ratio_min: float = Field(default=0.08, ge=0, le=1)
    # ...or its symbol (non-alnum, non-space) ratio is above this (code punctuation).
    symbol_ratio_max: float = Field(default=0.35, ge=0, le=1)
    # ...or its mean whitespace-token length is above this (base64 / minified: huge runs).
    word_length_max: float = Field(default=15.0, gt=0)
    # How many of the three signals must trip before a window counts as machine content.
    min_detectors: int = Field(default=2, ge=1, le=3)


def get_sanitize_config(config: Config) -> SanitizeConfig:
    """Parse the ``[sanitize]`` section into a SanitizeConfig (disabled when absent).

    Example:
        >>> from lib_layered_config import Config
        >>> get_sanitize_config(Config({}, {})).enabled
        False
    """
    return SanitizeConfig.model_validate(config.get("sanitize", {}))


__all__ = [
    "SanitizeConfig",
    "get_sanitize_config",
]
