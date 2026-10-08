"""Map a filesystem path to and from the opaque ``file://`` URI a source carries.

A :class:`~semdex.domain.models.SourceRef` identifies its location with an opaque
``uri`` string so the store and search layers stay source-type-blind: a file is
``file:///abs/path``, an email will be ``imap://mailbox/uid``. This module owns
the filesystem (``file``) scheme end of that mapping - the discovery connector
produces the URI, the extractors resolve it back to a path to read the bytes.
Pure string/path manipulation (no I/O), so it stays in the adapters layer.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

# typeshed routes the Windows url2pathname through nturl2path and marks it deprecated for every
# Python version, but nturl2path is deprecated only from 3.14, where urllib.request.url2pathname
# no longer uses it. Remove the ignore once the floor reaches 3.14 (the stub then picks that one).
from urllib.request import url2pathname  # pyright: ignore[reportDeprecated]

__all__ = ["from_uri", "to_uri"]


def to_uri(path: Path) -> str:
    """Return the ``file://`` URI for *path*, made absolute and normalised lexically.

    An absolute, ``..``-free path makes the URI a stable identity independent of the
    caller's working directory. Symlinks are deliberately NOT followed: a document is
    named by the path it was discovered under, because the extractors derive the upload
    name and MIME type from the URI, and a content-addressed store (a HuggingFace cache,
    git-annex, nix) links ``report.pdf`` to an extension-less blob that no extractor can
    type. :meth:`Path.as_uri` percent-encodes spaces and non-ASCII names.
    """
    # Path.resolve() would follow the link; pathlib has no lexical normaliser, so absolute()
    # (no link following) plus os.path.normpath (collapses ``..`` as text) is the equivalent.
    return Path(os.path.normpath(path.absolute())).as_uri()


def from_uri(uri: str) -> Path:
    """Return the filesystem path named by a ``file://`` *uri* (inverse of :func:`to_uri`).

    Args:
        uri: A ``file:`` URI, as produced by :func:`to_uri`.

    Returns:
        The path the URI names.

    Raises:
        ValueError: *uri* is not a ``file:`` URI. A bare path is refused too: a POSIX
            one would happen to parse, but a Windows one loses its drive (``urlparse``
            reads ``C:`` as the scheme), so accepting either hides the mistake on
            every platform but one.

    Examples:
        >>> from_uri("file:///srv/notes/a%20b.md").name
        'a b.md'
    """
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"not a file: URI: {uri!r}")
    return Path(url2pathname(parsed.path))  # pyright: ignore[reportDeprecated] - see the import
