"""CLI test for the serve command: it resolves config + flags and runs the server.

Does not actually block on a real server - ``build_mcp_server`` is monkeypatched
to a recorder whose ``.run(...)`` captures the transport/host/port it is called
with, so the test asserts the flag-over-config resolution without a socket.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner
from lib_layered_config import Config

from semdex.adapters import cli as cli_mod
from semdex.adapters.cli.commands import serve as serve_mod
from semdex.composition import Bootstrap, build_index_production, build_production


class _Recorder:
    """Stand-in FastMCP server: records the run() call instead of serving."""

    def __init__(self) -> None:
        self.run_kwargs: dict[str, Any] | None = None

    def run(self, **kwargs: Any) -> None:
        self.run_kwargs = kwargs


def _returns(recorder: _Recorder) -> Callable[..., _Recorder]:
    """A typed build_mcp_server stand-in that ignores its args and yields the recorder."""

    def _build(*_args: object, **_kwargs: object) -> _Recorder:
        return recorder

    return _build


def _bootstrap_with_config(config: Config) -> Bootstrap:
    def _get_config(
        *, profile: str | None = None, start_dir: str | None = None, dotenv_path: str | None = None
    ) -> Config:
        return config

    services = dataclasses.replace(build_production(), get_config=_get_config)
    return Bootstrap(services_factory=lambda: services, index_services_factory=build_index_production)


@pytest.mark.os_agnostic
def test_serve_resolves_http_with_port_override(
    tmp_path: Path, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = Config(
        {
            "mcp": {"transport": "http", "host": "127.0.0.1", "port": 8080},
            "dataset": [
                {
                    "name": "notes",
                    "backend": "json",
                    "store_dir": str(tmp_path / "store"),
                    "collection": "notes",
                    "embedding_provider": "placeholder",
                    "sources": [str(tmp_path / "docs")],
                }
            ],
        },
        {},
    )
    recorder = _Recorder()
    monkeypatch.setattr(serve_mod, "build_mcp_server", _returns(recorder))

    result = cli_runner.invoke(cli_mod.cli, ["serve", "--port", "9001"], obj=_bootstrap_with_config(config))

    assert result.exit_code == 0, result.output
    assert recorder.run_kwargs == {"transport": "http", "host": "127.0.0.1", "port": 9001}


@pytest.mark.os_agnostic
def test_serve_defaults_to_stdio(tmp_path: Path, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    config = Config({"dataset": []}, {})
    recorder = _Recorder()
    monkeypatch.setattr(serve_mod, "build_mcp_server", _returns(recorder))

    result = cli_runner.invoke(cli_mod.cli, ["serve"], obj=_bootstrap_with_config(config))

    assert result.exit_code == 0, result.output
    assert recorder.run_kwargs == {"transport": "stdio"}


@pytest.mark.parametrize("mask_error_details", [True, False])
@pytest.mark.os_agnostic
def test_mask_error_details_reaches_build_mcp_server(
    tmp_path: Path, cli_runner: CliRunner, monkeypatch: pytest.MonkeyPatch, *, mask_error_details: bool
) -> None:
    """``[mcp].mask_error_details`` must reach ``build_mcp_server``, not just validate.

    Substitutes at the real production seam ``serve_mod.build_mcp_server`` (the
    same import the command calls through), never an internal semdex detail: a
    config field nothing forwards would still parse and show up in ``semdex
    config`` while changing no runtime behaviour, exactly the ``num_batch`` defect
    ``test_cli_embedding_knobs_reach_factory.py`` was written to catch for the
    embedding knobs.
    """
    config = Config({"mcp": {"mask_error_details": mask_error_details}, "dataset": []}, {})
    recorder = _Recorder()
    captured: dict[str, Any] = {}

    def _build(*_args: object, **kwargs: object) -> _Recorder:
        captured.update(kwargs)
        return recorder

    monkeypatch.setattr(serve_mod, "build_mcp_server", _build)

    result = cli_runner.invoke(cli_mod.cli, ["serve"], obj=_bootstrap_with_config(config))

    assert result.exit_code == 0, result.output
    assert captured["mask_error_details"] is mask_error_details
