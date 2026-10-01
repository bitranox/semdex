"""Domain layer - pure business logic with no I/O or framework dependencies.

Contains entities, value objects, and domain services that form the core
business logic of the application.

Contents:
    * :mod:`.enums` - Domain enumerations (OutputFormat, DeployTarget, WatchMode, ...)
    * :mod:`.models` - Domain value objects (Chunk, Hit, Collection, ...)
    * :mod:`.errors` - Domain exception types
"""

from __future__ import annotations

from .enums import ChangeKind, DeployTarget, OutputFormat, StoreBackend, WatchMode
from .errors import (
    CollectionModelMismatchError,
    ConfigurationError,
    DeliveryError,
    EmbeddingError,
    ExtractionError,
    InvalidCollectionError,
    InvalidRecipientError,
    VectorStoreError,
    WatchError,
)
from .models import ChangeEvent, Chunk, Collection, ExtractedDocument, Hit, SourceRef, Vector

__all__ = [
    # Enums
    "ChangeKind",
    "DeployTarget",
    "OutputFormat",
    "StoreBackend",
    "WatchMode",
    # Value objects
    "ChangeEvent",
    "Chunk",
    "Collection",
    "ExtractedDocument",
    "Hit",
    "SourceRef",
    "Vector",
    # Errors
    "CollectionModelMismatchError",
    "ConfigurationError",
    "DeliveryError",
    "EmbeddingError",
    "ExtractionError",
    "InvalidCollectionError",
    "InvalidRecipientError",
    "VectorStoreError",
    "WatchError",
]
