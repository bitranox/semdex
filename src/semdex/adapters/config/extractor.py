"""Extractor configuration model parsed from the ``[extractor]`` section.

Selects which file-to-text/markdown extractor the index uses. The composition
root turns this into a concrete adapter behind the ``Extract`` port. The four
doc converters (markitdown / xberg / docling / mineru) run as their own
Docker container and are reached over HTTP, so ``endpoint`` names that service.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from ...domain.enums import ExtractorBackend

if TYPE_CHECKING:
    from lib_layered_config import Config

# Tunable "assumed" defaults for the HTTP extractor backends, exposed as config
# keys (not hardcoded in a function) so they can be tuned per deployment. See
# CLAUDE.md "Configurable values".
_DEFAULT_TIMEOUT_SECONDS = 120.0
_DEFAULT_MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MiB: doc files run larger than notes


class ExtractorConfig(BaseModel):
    """Validated, immutable extractor backend selection.

    Example:
        >>> ExtractorConfig().backend.value
        'text'
        >>> ExtractorConfig().timeout
        120.0
    """

    model_config = ConfigDict(frozen=True)

    backend: ExtractorBackend = ExtractorBackend.TEXT
    # Base URL of the extractor's container (e.g. "http://127.0.0.1:5001"); the
    # embedded TEXT backend ignores it, the HTTP backends require it.
    endpoint: str | None = None
    timeout: float = Field(default=_DEFAULT_TIMEOUT_SECONDS, gt=0)
    # Upper bound on the file the HTTP backends will upload; an oversized file is
    # rejected (against stat) before it is read, so the read stays bounded.
    max_file_bytes: int = Field(default=_DEFAULT_MAX_FILE_BYTES, gt=0)
    # OCR hints for the xberg backend (ignored by the others). force_ocr runs OCR
    # even on a searchable PDF; ocr_language is the Tesseract language (e.g. "eng").
    # Both default off, matching xberg's own default (OCR only when it decides the
    # input needs it, e.g. a scanned image).
    force_ocr: bool = False
    ocr_language: str | None = None


def get_extractor_config(config: Config) -> ExtractorConfig:
    """Parse the ``[extractor]`` section into an ExtractorConfig.

    Falls back to the embedded TEXT backend when the section is absent.

    Example:
        >>> from lib_layered_config import Config
        >>> get_extractor_config(Config({}, {})).backend.value
        'text'
    """
    return ExtractorConfig.model_validate(config.get("extractor", {}))


__all__ = [
    "ExtractorConfig",
    "get_extractor_config",
]
