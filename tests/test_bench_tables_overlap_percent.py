"""Overlap is printed in tokens AND as a percent of the chunk cap.

Industry guidance quotes overlap in percent (10-25 percent of a 512 or 1024 token chunk), while the
cache keys and the chunker count tokens. A reader comparing a published verdict with that guidance
needs both, so every profile label and every overlap level carries the percent of its own cap.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gen_bench_tables.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("gen_bench_tables_overlap_percent", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator() -> Any:
    return _load()


@pytest.mark.parametrize(
    ("overlap", "cap", "percent"),
    [(0, 512, 0), (26, 256, 10), (38, 256, 15), (51, 512, 10), (77, 512, 15), (128, 512, 25), (205, 1024, 20)],
)
def test_percent_is_of_the_cap(generator: Any, overlap: int, cap: int, percent: int) -> None:
    assert generator._overlap_percent(overlap, cap) == percent


def test_profile_label_carries_tokens_and_percent(generator: Any) -> None:
    axes = {"strategy": "recursive", "max_tokens": 512, "overlap_tokens": 102}
    assert generator._display_profile(axes) == "recursive cap512 ov102tok (20%)"


def test_overlap_free_strategy_label_is_unchanged(generator: Any) -> None:
    axes = {"strategy": "semantic", "max_tokens": 256, "overlap_tokens": 0}
    assert generator._display_profile(axes) == "semantic hint256 breakpoint-default(potion-base-32M)"


def _overlap_effect(from_level: int, to_level: int, cap: int) -> dict[str, Any]:
    return {
        "axis": "overlap_tokens",
        "corpus": "mldr_en_8k_slice",
        "embedding": "model2vec:potion-base-8M",
        "held_fixed": {"strategy": "recursive", "max_tokens": cap, "breakpoint_model": None},
        "from_level": from_level,
        "to_level": to_level,
        "mean_delta": 0.001,
        "ci_lo": -0.001,
        "ci_hi": 0.003,
        "wins": 3,
        "losses": 2,
        "resolved": False,
    }


def test_overlap_levels_in_a_knob_row_carry_their_percent(generator: Any) -> None:
    rows = generator._knob_rows([_overlap_effect(128, 0, 512)], "overlap_tokens", None)
    assert rows[0][3] == "0 (0%) to 128 (25%)"


def test_other_axes_keep_their_plain_levels(generator: Any) -> None:
    effect = {
        **_overlap_effect(256, 512, 512),
        "axis": "max_tokens",
        "held_fixed": {"strategy": "recursive", "overlap_tokens": 0},
    }
    rows = generator._knob_rows([effect], "max_tokens", None)
    assert rows[0][3] == "256 to 512"
