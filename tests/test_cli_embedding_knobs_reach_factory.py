"""Every [embedding] knob the CLI resolves must actually reach the index factory.

A config key nothing forwards is a dead knob: it validates, it shows up in ``semdex config``,
and it changes nothing at all. That is exactly what happened to ``num_batch`` - the field and its
documentation existed while the CLI never passed it on - so this pins the whole set through the
real command path rather than through the private helper that does the threading.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from click.testing import CliRunner
from lib_layered_config import Config

from semdex.adapters import cli as cli_mod
from semdex.composition import Bootstrap, IndexServices, build_index_production, build_production

if TYPE_CHECKING:
    from collections.abc import Callable

# Knobs the [embedding] section resolves, and the factory keyword each must arrive as.
_EXPECTED: dict[str, object] = {
    "embedding_timeout": 120.0,
    "embedding_retries": 2,
    "embedding_num_batch": 4096,
    "embedding_batch": 64,
}

_CONFIG = {
    "embedding": {
        # placeholder keeps the run offline; the knobs below are threaded regardless of provider.
        "provider": "placeholder",
        "timeout": 120.0,
        "retries": 2,
        "num_batch": 4096,
        "batch": 64,
    },
    "chunker": {"strategy": "whitespace"},
}


def _bootstrap(factory: Callable[..., IndexServices]) -> Bootstrap:
    """Production bootstrap with a fixed Config and a recording index factory."""

    def _get_config(
        *, profile: str | None = None, start_dir: str | None = None, dotenv_path: str | None = None
    ) -> Config:
        return Config(_CONFIG, {})

    services = dataclasses.replace(build_production(), get_config=_get_config)
    return Bootstrap(services_factory=lambda: services, index_services_factory=factory)


@pytest.mark.os_agnostic
def test_every_embedding_knob_reaches_the_index_factory(tmp_path: Path, cli_runner: CliRunner) -> None:
    captured: dict[str, Any] = {}

    def recording_factory(store_dir: Path, **kwargs: Any) -> IndexServices:
        captured.update(kwargs)
        return build_index_production(store_dir, **kwargs)

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("banana cherry apple", encoding="utf-8")

    result = cli_runner.invoke(
        cli_mod.cli,
        ["index", str(corpus), "--store-dir", str(tmp_path / "store")],
        obj=_bootstrap(recording_factory),
    )

    assert result.exit_code == 0, result.output
    for keyword, expected in _EXPECTED.items():
        assert captured[keyword] == expected, f"{keyword} never reached the factory"
