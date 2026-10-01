"""Index configuration model parsed from the ``[index]`` config section.

Boundary model: the layered config is external input, so it is validated into a
typed, frozen Pydantic model here. The CLI resolves each value as
``CLI flag > this config > built-in default``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from lib_layered_config import Config

# Built-in defaults for the tunable "assumed" values. Each is exposed as a config
# key (not hardcoded in a function) so it can be tuned per deployment and, later,
# validated with performance tests. See CLAUDE.md "Configurable values".
_DEFAULT_MAX_TOKENS = 256
_DEFAULT_MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MiB
_DEFAULT_EMBEDDING_DIM = 256
_DEFAULT_SNIPPET_CHARS = 80
_DEFAULT_HASH_CHUNK_BYTES = 65536
_DEFAULT_EXTENSIONS = (".md", ".txt")


class IndexConfig(BaseModel):
    """Validated, immutable defaults for the index/search commands.

    Example:
        >>> IndexConfig().collection
        'default'
        >>> IndexConfig().max_tokens
        256
    """

    model_config = ConfigDict(frozen=True)

    store_dir: str | None = None
    collection: str = "default"
    embedding_model: str | None = None
    default_label: str = ""
    default_k: int = Field(default=5, gt=0)
    # Tunable assumptions (config-driven, not hardcoded):
    max_tokens: int = Field(default=_DEFAULT_MAX_TOKENS, gt=0)
    max_file_bytes: int = Field(default=_DEFAULT_MAX_FILE_BYTES, gt=0)
    embedding_dim: int = Field(default=_DEFAULT_EMBEDDING_DIM, gt=0)
    snippet_chars: int = Field(default=_DEFAULT_SNIPPET_CHARS, gt=0)
    hash_chunk_bytes: int = Field(default=_DEFAULT_HASH_CHUNK_BYTES, gt=0)
    extensions: list[str] = Field(default_factory=lambda: list(_DEFAULT_EXTENSIONS))


def get_index_config(config: Config) -> IndexConfig:
    """Parse the ``[index]`` section of the merged config into an IndexConfig.

    Falls back to built-in defaults when the section (or a key) is absent.

    Example:
        >>> from lib_layered_config import Config
        >>> get_index_config(Config({}, {})).collection
        'default'
    """
    return IndexConfig.model_validate(config.get("index", {}))


__all__ = [
    "IndexConfig",
    "get_index_config",
]
