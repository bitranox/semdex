"""Every third-party module the package imports at run time is a declared dependency.

A module that is only installed because another dependency pulls it in works until that
dependency drops it or caps its version, and then the package breaks for a reason its own
metadata never mentioned. Imports under ``if TYPE_CHECKING:`` are not counted: they never
run in an installed package.

semdex reaches its optional backends (vector stores, embedders, extractors, the MCP server)
through imports inside the adapter that needs them, so such a module may come from an
optional extra instead of ``[project].dependencies``. The ``dev`` extra does not count: it is
never installed for a user. An import that runs when its module is imported (not inside a
function or a ``try``) must be a core dependency, or importing that module fails without the
extra.
"""

from __future__ import annotations

import ast
import re
import sys
from functools import cache
from importlib.metadata import packages_distributions
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import rtoml

if TYPE_CHECKING:
    from collections.abc import Mapping

_ROOT = Path(__file__).parent.parent
_PACKAGE = "semdex"
_SOURCE = _ROOT / "src" / _PACKAGE

#: Distributions that ship no import package of their own and install the one that provides
#: the module: fastmcp 4 is a meta-package over fastmcp-slim, which provides ``fastmcp``.
_META_PACKAGES = {"fastmcp-slim": "fastmcp"}


def _normalise(name: str) -> str:
    """PEP 503 name normalisation, so ``rich_click`` and ``rich-click`` compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _names(requirements: list[str]) -> set[str]:
    matches = (re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement) for requirement in requirements)
    return {_normalise(match.group(0)) for match in matches if match}


def _project() -> dict[str, Any]:
    return rtoml.load(_ROOT / "pyproject.toml")["project"]


def _core_distributions() -> set[str]:
    requirements: list[str] = _project()["dependencies"]
    return _names(requirements)


def _extra_distributions() -> set[str]:
    """Every distribution a user-facing extra installs; ``dev`` is the developer's own."""
    extras: dict[str, list[str]] = _project()["optional-dependencies"]
    return {name for extra, requirements in extras.items() if extra != "dev" for name in _names(requirements)}


def _is_type_checking_block(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _imported_module(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name.split(".")[0] for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        return [node.module.split(".")[0]]
    return []


def _runtime_imports(tree: ast.AST) -> dict[str, bool]:
    """Top-level module names imported outside ``if TYPE_CHECKING:`` blocks.

    Returns:
        Each module, and whether one of its imports runs when the file itself is imported
        (that is, not inside a function, a class body's method or a ``try``).
    """
    found: dict[str, bool] = {}
    pending: list[tuple[ast.AST, bool]] = [(tree, True)]
    while pending:
        node, at_import = pending.pop()
        if _is_type_checking_block(node):
            continue
        for module in _imported_module(node):
            found[module] = found.get(module, False) or at_import
        deferred = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.Try))
        pending.extend((child, at_import and not deferred) for child in ast.iter_child_nodes(node))
    return found


def _third_party_runtime_imports() -> dict[str, dict[str, bool]]:
    """Each third-party top-level module the package imports at run time, by importer and timing."""
    importers: dict[str, dict[str, bool]] = {}
    for path in sorted(_SOURCE.rglob("*.py")):
        for module, at_import in _runtime_imports(ast.parse(path.read_text(encoding="utf-8"))).items():
            if module in sys.stdlib_module_names or module == _PACKAGE:
                continue
            importers.setdefault(module, {})[str(path.relative_to(_ROOT))] = at_import
    return importers


@cache
def _installed_providers() -> Mapping[str, list[str]]:
    """Import package to distributions, read once: building it walks every installed package."""
    return packages_distributions()


def _providers(module: str) -> set[str]:
    providers = {_normalise(dist) for dist in _installed_providers().get(module, [module])}
    return providers | {_META_PACKAGES[name] for name in providers if name in _META_PACKAGES}


@pytest.mark.os_agnostic
def test_the_scan_sees_the_package_s_own_third_party_imports() -> None:
    """Control: the scan must find imports it is known to contain, or an empty result proves nothing."""
    imports = _third_party_runtime_imports()

    assert {"pydantic", "lib_layered_config", "rich_click", "httpx"} <= set(imports)
    assert any(imports["pydantic"].values())
    assert not any(imports["httpx"].values())


@pytest.mark.os_agnostic
def test_every_runtime_import_is_a_declared_dependency() -> None:
    declared = _core_distributions() | _extra_distributions()

    undeclared = {
        module: sorted(importers)
        for module, importers in _third_party_runtime_imports().items()
        if not _providers(module) & declared
    }

    assert undeclared == {}, f"imported at run time but in no dependency list or user extra: {undeclared}"


@pytest.mark.os_agnostic
def test_every_import_that_runs_on_module_import_is_a_core_dependency() -> None:
    core = _core_distributions()

    not_core = {
        module: sorted(path for path, at_import in importers.items() if at_import)
        for module, importers in _third_party_runtime_imports().items()
        if any(importers.values()) and not _providers(module) & core
    }

    assert not_core == {}, f"imported when its module loads but not in [project].dependencies: {not_core}"
