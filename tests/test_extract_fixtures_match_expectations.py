"""The extraction fixtures must actually contain the phrases they are scored on.

The extractor matrix scores each backend by how many expected phrases it recovers, but it needs
Docker and is `integration`-marked, so a broken EXPECTATION shows up only in a lane that rarely
runs. It happened: a whole-tree `ruff format` rewrote the Python block inside `sample.md` from
single to double quotes, `expected.toml` kept the old form, and md coverage silently dropped to 75
percent for every extractor - including `text`, which reads the file verbatim and therefore cannot
lose anything. It sat unnoticed from 2026-07-26 because CI's integration lane was blocked.

These tests are the cheap half of that check: for the formats read verbatim, every expected phrase
must be present in the fixture bytes. No Docker, no extractor, so they run in `make test`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import rtoml

pytestmark = pytest.mark.os_agnostic

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "extract"
# The formats the embedded `text` extractor supports: it returns the file's own characters, so
# its phrase recall is a property of the fixture alone and must be exactly 1.0.
_VERBATIM = ("sample.md", "sample.txt")


def _expected() -> dict[str, dict[str, list[str]]]:
    return rtoml.load(_FIXTURES / "expected.toml")["fixtures"]


def _normalize(text: str) -> str:
    """Same normalization the matrix scores with, so this test and that one agree."""
    return re.sub(r"\s+", " ", text).strip().lower()


@pytest.mark.parametrize("fixture_name", _VERBATIM)
def test_a_verbatim_fixture_contains_every_phrase_it_is_scored_on(fixture_name: str) -> None:
    """A verbatim read cannot lose content, so anything missing here is a stale expectation."""
    text = _normalize((_FIXTURES / fixture_name).read_text(encoding="utf-8"))

    missing = [phrase for phrase in _expected()[fixture_name]["phrases"] if _normalize(phrase) not in text]

    assert not missing, (
        f"{fixture_name} does not contain {missing}. The fixture or expected.toml drifted; a "
        f"verbatim extractor would score below 1.0 through no fault of its own."
    )


def test_every_fixture_named_in_the_expectations_exists() -> None:
    """An expectation for a deleted fixture would silently never be checked by anything."""
    missing = [name for name in _expected() if not (_FIXTURES / name).exists()]

    assert not missing, f"expected.toml names fixtures that do not exist: {missing}"


def test_the_formatter_is_kept_away_from_the_fixtures() -> None:
    """The fixtures are measurement input; a formatter rewriting them changes the measurement.

    Asserted against pyproject rather than by running ruff, because the failure mode is a config
    edit that drops the exclusion, not ruff changing its behaviour. `force-exclude` matters
    separately: without it ruff reformats a path named explicitly on the command line even when
    it is excluded, and the gate does name paths.
    """
    config = rtoml.load(Path(__file__).resolve().parents[1] / "pyproject.toml")

    ruff = config["tool"]["ruff"]
    assert "tests/fixtures" in ruff.get("exclude", []), "tests/fixtures must stay out of ruff's reach"
    assert ruff.get("force-exclude") is True, "without force-exclude the exclusion does not hold for named paths"
