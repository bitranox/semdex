"""Console script entry point with production wiring.

This module provides the entry point for console scripts (pip-installed commands).
It wires production services from the composition layer before invoking the CLI.

System Role:
    Sits at package level (outside adapters) to properly wire composition into
    the adapters layer without violating clean architecture layer constraints.
"""

from __future__ import annotations

from .adapters.cli.main import main as cli_main
from .composition import Bootstrap, build_dataset_services, build_index_production, build_production


def main() -> int:
    """Console script entry point with production services wired.

    Injects a Bootstrap carrying the AppServices factory plus the index- and
    dataset-services factories, so config/logging/email, index/search, and the
    serve command are all served from the composition root.

    Returns:
        Exit code from CLI execution.
    """
    bootstrap = Bootstrap(
        services_factory=build_production,
        index_services_factory=build_index_production,
        dataset_services_factory=build_dataset_services,
    )
    return cli_main(services_factory=bootstrap)


__all__ = ["main"]
