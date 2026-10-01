"""Module entry point for ``python -m semdex``.

Delegates to the same production wiring as the console script (``entry.main``)
so ``python -m semdex`` and the installed ``semdex`` command behave identically -
in particular both serve the index/search commands (the ``Bootstrap`` carries the
index services factory, not just the AppServices factory).
"""

from __future__ import annotations

from .entry import main

if __name__ == "__main__":
    raise SystemExit(main())
