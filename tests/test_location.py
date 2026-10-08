"""File-scheme source-location helpers: Path <-> file:// URI round-trip.

The filesystem source connector maps a real path to the opaque ``uri`` a
``SourceRef`` carries (and the store keys on); the extractors map it back to a
path to read. These must round-trip, including paths with spaces/unicode.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from semdex.adapters.discovery.location import from_uri, to_uri


def test_file_uri_round_trips_through_path(tmp_path: Path) -> None:
    """to_uri then from_uri returns the original path, spaces and all."""
    path = tmp_path / "a b.md"
    path.write_text("x", encoding="utf-8")

    uri = to_uri(path)

    assert uri.startswith("file://")
    assert from_uri(uri) == path


def test_to_uri_is_absolute_and_percent_encodes(tmp_path: Path) -> None:
    """A space becomes %20 so the URI is well-formed; the scheme is file://."""
    uri = to_uri(tmp_path / "a b.md")

    assert uri.startswith("file:///")
    assert "%20" in uri
    assert " " not in uri


def test_from_uri_decodes_unicode(tmp_path: Path) -> None:
    """A non-ASCII name survives the round trip."""
    path = tmp_path / "café.md"

    assert from_uri(to_uri(path)) == path


def _symlink_or_skip(link: Path, target: Path) -> None:
    """Create *link* -> *target*, skipping where the platform refuses symlinks (Windows without the privilege)."""
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable here: {exc}")


def test_to_uri_keeps_a_symlinks_own_name(tmp_path: Path) -> None:
    """A symlinked document is named by the link, not by the file it points at.

    Discovery selects a file by its own suffix, and the extractors derive the upload name
    and MIME type from the URI. A content-addressed store (a HuggingFace cache, git-annex,
    nix) links ``report.pdf`` to an extension-less blob, so following the link would hand
    the extractor ``abc123`` as ``application/octet-stream`` and show the blob in hits.
    """
    store = tmp_path / "store"
    store.mkdir()
    blob = store / "abc123"
    blob.write_text("x", encoding="utf-8")
    link = tmp_path / "report.pdf"
    _symlink_or_skip(link, blob)

    uri = to_uri(link)

    assert uri.endswith("/report.pdf")
    assert from_uri(uri) == link


def test_to_uri_normalises_dot_dot_lexically(tmp_path: Path) -> None:
    """Two spellings of one path give one URI, so identity does not depend on how it was named."""
    (tmp_path / "sub").mkdir()

    assert to_uri(tmp_path / "sub" / ".." / "a.md") == to_uri(tmp_path / "a.md")


@pytest.mark.parametrize(
    "not_a_file_uri",
    ["/tmp/a.md", "C:\\Users\\x\\a.md", "a.md", "semdex://knowledge/abc", "https://example.com/a.md"],
)
def test_from_uri_refuses_anything_but_a_file_uri(not_a_file_uri: str) -> None:
    """A bare path or another scheme is refused instead of read as a path.

    A bare POSIX path happens to survive urlparse, so a caller passing one works
    on Linux and loses the drive on Windows (urlparse reads "C:" as a scheme).
    Refusing it on every platform makes that mistake fail where it is made.
    """
    with pytest.raises(ValueError, match="file:"):
        from_uri(not_a_file_uri)
