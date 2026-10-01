"""Domain-specific exceptions for typed error handling at boundaries."""

from __future__ import annotations


class ConfigurationError(Exception):
    """Missing, invalid, or incomplete configuration.

    Raised when required configuration values are absent, malformed, or
    logically inconsistent. Typically caught at CLI boundaries to provide
    user-friendly error messages.

    Example:
        >>> from semdex.domain.errors import ConfigurationError
        >>> err = ConfigurationError("No SMTP hosts configured")
        >>> str(err)
        'No SMTP hosts configured'
    """


class DeliveryError(Exception):
    """Email/notification delivery failed at SMTP level.

    Raised when all configured SMTP hosts fail to accept the message.
    Contains details about the delivery failure for logging and user feedback.

    Example:
        >>> from semdex.domain.errors import DeliveryError
        >>> err = DeliveryError("Connection refused by smtp.example.com:587")
        >>> str(err)
        'Connection refused by smtp.example.com:587'
    """


class InvalidRecipientError(ValueError):
    """Email address validation failure.

    Raised when a recipient address fails RFC 5321/5322 validation.
    Inherits from ValueError so existing ``except ValueError`` handlers
    continue to catch it during the migration period.

    Example:
        >>> from semdex.domain.errors import InvalidRecipientError
        >>> err = InvalidRecipientError("Invalid email: not-an-email")
        >>> str(err)
        'Invalid email: not-an-email'
        >>> isinstance(err, ValueError)
        True
    """


class ExtractionError(Exception):
    """A source file could not be converted to text/markdown.

    Raised by an :class:`~semdex.application.ports.Extract` adapter when a
    file's format is unsupported or its content cannot be decoded.

    Example:
        >>> from semdex.domain.errors import ExtractionError
        >>> str(ExtractionError("unsupported format: .xyz"))
        'unsupported format: .xyz'
    """


class ChunkingError(Exception):
    """A chunker adapter failed to split a document.

    Raised by a :class:`~semdex.application.ports.ChunkText` adapter when its
    library is missing or a document cannot be chunked.

    Example:
        >>> from semdex.domain.errors import ChunkingError
        >>> str(ChunkingError("chonkie is not installed"))
        'chonkie is not installed'
    """


class EmbeddingError(Exception):
    """An embedding provider failed to produce vectors.

    Raised when the model cannot be loaded or a text batch cannot be embedded.
    """


class SummaryError(Exception):
    """An LLM summarizer failed to produce a summary.

    Raised when the summarizer's chat server is unreachable, returns an
    unexpected response, or its optional HTTP dependency is missing.
    """


class VectorStoreError(Exception):
    """A vector store operation failed.

    Base class for vector store failures so boundaries can catch the whole
    family with a single ``except VectorStoreError``.
    """


class CollectionModelMismatchError(VectorStoreError):
    """A query or upsert used a model/dim that does not match the collection.

    Vectors from different models occupy different spaces and different dims;
    mixing them silently corrupts a similarity search, so it is rejected.

    Example:
        >>> from semdex.domain.errors import CollectionModelMismatchError, VectorStoreError
        >>> issubclass(CollectionModelMismatchError, VectorStoreError)
        True
    """


class WatchError(Exception):
    """The filesystem watcher failed to start or observe a root."""


class InvalidCollectionError(ValueError):
    """A collection was constructed with invalid fields.

    Raised for an empty name/model_id or a non-positive dimension. Inherits
    from ValueError so existing ``except ValueError`` handlers still catch it.

    Example:
        >>> from semdex.domain.errors import InvalidCollectionError
        >>> isinstance(InvalidCollectionError("dim must be positive"), ValueError)
        True
    """


__all__ = [
    "ChunkingError",
    "CollectionModelMismatchError",
    "ConfigurationError",
    "DeliveryError",
    "EmbeddingError",
    "ExtractionError",
    "InvalidCollectionError",
    "InvalidRecipientError",
    "SummaryError",
    "VectorStoreError",
    "WatchError",
]
