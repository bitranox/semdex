"""Application layer - use cases and port definitions.

Contains use cases that orchestrate domain logic and port protocols that
define the interfaces for adapter implementations.

Contents:
    * :mod:`.ports` - Protocol definitions adapters implement
    * :mod:`.use_cases` - Workflows orchestrating domain logic through the ports
"""

from __future__ import annotations

from .ports import (
    ChunkText,
    DeployConfiguration,
    DisplayConfig,
    EmbeddingProvider,
    Extract,
    GetConfig,
    GetDefaultConfigPath,
    InitLogging,
    LoadEmailConfigFromDict,
    SendEmail,
    SendNotification,
    VectorStore,
    VectorStoreReader,
    VectorStoreWriter,
    Watcher,
)
from .use_cases import IndexReport, index_sources, search

__all__ = [
    "ChunkText",
    "DeployConfiguration",
    "DisplayConfig",
    "EmbeddingProvider",
    "Extract",
    "GetConfig",
    "GetDefaultConfigPath",
    "IndexReport",
    "InitLogging",
    "LoadEmailConfigFromDict",
    "SendEmail",
    "SendNotification",
    "VectorStore",
    "VectorStoreReader",
    "VectorStoreWriter",
    "Watcher",
    "index_sources",
    "search",
]
