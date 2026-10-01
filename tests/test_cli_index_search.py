"""End-to-end CLI tests for the index and search commands."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from lib_layered_config import Config

from semdex.adapters import cli as cli_mod
from semdex.composition import AppServices, Bootstrap, build_index_production, build_production

# Pin the offline placeholder embedding + whitespace chunker so these end-to-end
# tests are deterministic and never download a model or fetch a chunking recipe
# (the real defaults are fastembed + markdown-aware recursive).
_OFFLINE = {"embedding": {"provider": "placeholder"}, "chunker": {"strategy": "whitespace"}}


@pytest.fixture
def bootstrap() -> Bootstrap:
    """Production bootstrap pinned to the offline placeholder embedding + whitespace chunker."""
    return _bootstrap_with_config(Config(_OFFLINE, {}))


def _bootstrap_with_config(config: Config) -> Bootstrap:
    """Bootstrap whose root callback receives a fixed Config (with an [index] section)."""

    def _get_config(
        *, profile: str | None = None, start_dir: str | None = None, dotenv_path: str | None = None
    ) -> Config:
        return config

    services = dataclasses.replace(build_production(), get_config=_get_config)

    def _services_factory() -> AppServices:
        return services

    return Bootstrap(services_factory=_services_factory, index_services_factory=build_index_production)


@pytest.mark.os_agnostic
def test_index_then_search_persists_across_invocations(
    tmp_path: Path, cli_runner: CliRunner, bootstrap: Bootstrap
) -> None:
    """`index` writes a durable store that a later `search` invocation reads."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "fruit.md").write_text("banana cherry apple", encoding="utf-8")
    (corpus / "car.md").write_text("engine piston valve", encoding="utf-8")
    store = tmp_path / "store"

    indexed = cli_runner.invoke(cli_mod.cli, ["index", str(corpus), "--store-dir", str(store)], obj=bootstrap)
    assert indexed.exit_code == 0, indexed.output
    assert "2 source(s)" in indexed.output

    found = cli_runner.invoke(cli_mod.cli, ["search", "apple banana", "--store-dir", str(store)], obj=bootstrap)
    assert found.exit_code == 0, found.output
    assert "fruit.md" in found.output


@pytest.mark.os_agnostic
def test_search_on_missing_collection_errors_with_hint(
    tmp_path: Path, cli_runner: CliRunner, bootstrap: Bootstrap
) -> None:
    """Searching a never-indexed collection fails with a run-index hint."""
    result = cli_runner.invoke(
        cli_mod.cli, ["search", "anything", "--store-dir", str(tmp_path / "empty")], obj=bootstrap
    )
    assert result.exit_code != 0
    assert "index" in result.output.lower()


@pytest.mark.os_agnostic
def test_index_and_search_are_registered(cli_runner: CliRunner, bootstrap: Bootstrap) -> None:
    """Both commands appear in the CLI help."""
    result = cli_runner.invoke(cli_mod.cli, ["--help"], obj=bootstrap)
    assert result.exit_code == 0
    assert "index" in result.output
    assert "search" in result.output


@pytest.mark.os_agnostic
def test_index_and_search_honor_index_config(
    tmp_path: Path,
    cli_runner: CliRunner,
    config_factory: Callable[[dict[str, Any]], Config],
) -> None:
    """With no flags, store_dir and collection come from the [index] config."""
    store = tmp_path / "store"
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("alpha beta gamma", encoding="utf-8")
    config = config_factory({"index": {"store_dir": str(store), "collection": "mem"}, **_OFFLINE})
    boot = _bootstrap_with_config(config)

    indexed = cli_runner.invoke(cli_mod.cli, ["index", str(corpus)], obj=boot)
    assert indexed.exit_code == 0, indexed.output
    assert "'mem'" in indexed.output
    assert str(store) in indexed.output

    found = cli_runner.invoke(cli_mod.cli, ["search", "alpha beta"], obj=boot)
    assert found.exit_code == 0, found.output
    assert "note.md" in found.output
