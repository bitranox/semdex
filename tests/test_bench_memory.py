"""Memory measurement helpers for the store and embedder footprint benchmark.

Every defect this module can have produces a plausible number rather than an error, which is why
it is tested at all for a benchmark harness:

* peak RSS never falls, so two cells in one process report the same high-water mark and the run
  looks ordered and consistent while being wrong;
* the interpreter's own several MB, left in, is a constant added to every row that flatters the
  heavy ones by comparison;
* a cell that dies has to skip loudly, because a silently dropped backend reads as one that was
  never asked for.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_MODULE = Path(__file__).resolve().parents[1] / "scripts" / "_bench_memory.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("_bench_memory", _MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mem() -> Any:
    return _load()


# --- reading the numbers ------------------------------------------------------------------------


def test_peak_is_reported_in_megabytes_not_kilobytes(mem: Any) -> None:
    """/proc reports kB and macOS's getrusage reports bytes; a missed conversion is a 1024x error.

    A running interpreter holds a few MB and nowhere near a few GB, so a wrong unit lands far
    outside this band rather than looking merely surprising.
    """
    if not mem.can_measure():
        pytest.skip("no /proc and no resource module (Windows): nothing to read a peak from")
    peak = mem.peak_rss_mb()

    assert 1.0 < peak < 4000.0


_STATUS = """\
Name:\tpython3
VmPeak:\t 4194304 kB
VmSize:\t 2097152 kB
VmHWM:\t  1048576 kB
VmRSS:\t   524288 kB
Threads:\t1
"""


def test_peak_reads_the_high_water_field_and_current_reads_the_present_one(mem: Any, tmp_path: Path) -> None:
    """Which field each one reads IS the contract, and swapping them is undetectable downstream.

    Reading VmHWM for both keeps "current <= peak" true forever, so the serving number silently
    becomes the ingest high-water mark and every store looks as heavy as its worst moment.

    Driven from a status file with known values rather than from a real allocation: proving it
    that way needs the allocator to hand pages back to the kernel, which is not a guarantee, and
    the version of this test that did so flaked once under memory pressure.
    """
    status = tmp_path / "status"
    status.write_text(_STATUS)

    assert mem.peak_rss_mb(source=status) == 1024.0  # 1048576 kB
    assert mem.current_rss_mb(source=status) == 512.0  # 524288 kB


def test_a_missing_status_file_does_not_crash_the_run(mem: Any, tmp_path: Path) -> None:
    """No /proc on macOS or Windows; the benchmark should degrade, not abort."""
    assert mem.peak_rss_mb(source=tmp_path / "absent") >= 0.0


def test_a_platform_with_neither_proc_nor_resource_reads_zero_and_says_it_cannot_measure(
    mem: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows has no /proc and no resource module: the helper degrades to 0.0, and can_measure
    says so, which is what lets the driver refuse to publish a table of zeros."""
    monkeypatch.setattr(mem.sys, "platform", "win32")

    assert mem.peak_rss_mb(source=tmp_path / "absent") == 0.0
    assert mem.can_measure(source=tmp_path / "absent") is False


def test_the_field_match_is_anchored_so_a_prefix_cannot_win(mem: Any, tmp_path: Path) -> None:
    """`VmRSS` is a prefix of nothing today, but `VmPeak`/`VmSize` sit in the same file and a
    loose `in` test would let the first line containing the name answer for it."""
    status = tmp_path / "status"
    status.write_text("VmRSSanity:\t 999 kB\nVmRSS:\t 4096 kB\n")

    assert mem.current_rss_mb(source=status) == 4.0


# --- the interpreter's own cost -----------------------------------------------------------------


def test_the_baseline_is_a_real_interpreter_measurement(mem: Any) -> None:
    """Measured, not assumed: it moves with the Python build and is a large share of small rows."""
    if not mem.can_measure():
        pytest.skip("no /proc and no resource module (Windows): nothing to read a peak from")
    baseline = mem.interpreter_baseline_mb()

    assert 1.0 < baseline < 200.0


def test_the_baseline_is_subtracted_from_every_row(mem: Any) -> None:
    """Left in, it is a constant added to all of them - which compresses the ratio between the
    lightest store and the heaviest and makes them look closer than they are."""
    row = mem.Measurement(peak_mb=100.0, resident_mb=80.0, baseline_mb=12.0, extra={}).as_row()

    assert row["peak_mb"] == 88.0
    assert row["resident_mb"] == 68.0


def test_the_gross_figure_survives_alongside_the_net_one(mem: Any) -> None:
    """A reader sizing a container needs the total the process actually occupied, not only the
    part attributed to the store."""
    row = mem.Measurement(peak_mb=100.0, resident_mb=80.0, baseline_mb=12.0, extra={}).as_row()

    assert row["peak_mb_gross"] == 100.0
    assert row["baseline_mb"] == 12.0


def test_a_row_below_the_baseline_clamps_to_zero_rather_than_going_negative(mem: Any) -> None:
    """Scheduling noise can put a trivial cell a hair under the baseline. A negative megabyte
    would be published as a fact."""
    row = mem.Measurement(peak_mb=10.0, resident_mb=9.0, baseline_mb=12.0, extra={}).as_row()

    assert row["peak_mb"] == 0.0
    assert row["resident_mb"] == 0.0


def test_extra_fields_reach_the_row(mem: Any) -> None:
    row = mem.Measurement(peak_mb=50.0, resident_mb=40.0, baseline_mb=10.0, extra={"backend": "json"}).as_row()

    assert row["backend"] == "json"


# --- running a cell -----------------------------------------------------------------------------


def test_a_cell_runs_in_its_own_interpreter_and_returns_its_json(mem: Any, tmp_path: Path) -> None:
    """The whole design: each cell is a separate process, so no peak is inherited from the last."""
    script = tmp_path / "cell.py"
    script.write_text('import os\nprint("{\\"ok\\": " + os.environ["CELL_VALUE"] + "}")\n')

    assert mem.run_cell(script, {"CELL_VALUE": "42"}) == {"ok": 42}


def test_a_cell_that_fails_after_printing_a_result_is_still_a_skip(mem: Any, tmp_path: Path) -> None:
    """The exit code decides, not whether some JSON appeared.

    A cell that printed a measurement and then died - out of memory during the query phase, say -
    printed a number for a run that did not finish. Testing only with a cell that prints NOTHING
    cannot tell this apart from the no-JSON path, and that gap let a mutation deleting the
    exit-code check survive.
    """
    script = tmp_path / "boom.py"
    script.write_text('import sys\nprint(\'{"peak_mb": 99}\')\nsys.stderr.write("died\\n")\nsys.exit(1)\n')

    assert mem.run_cell(script, {}) is None


def test_a_cell_that_dies_before_printing_anything_is_a_skip(mem: Any, tmp_path: Path) -> None:
    """A backend whose optional dependency is missing must not discard the cells already paid for."""
    script = tmp_path / "silent-boom.py"
    script.write_text("import sys\nsys.stderr.write('no such backend\\n')\nsys.exit(1)\n")

    assert mem.run_cell(script, {}) is None


def test_a_cell_that_prints_no_json_is_a_skip_not_a_crash(mem: Any, tmp_path: Path) -> None:
    """Exit 0 with no result is still no result; parsing it would raise inside the sweep."""
    script = tmp_path / "quiet.py"
    script.write_text("print('warming up')\n")

    assert mem.run_cell(script, {}) is None


def test_the_last_json_line_wins_over_earlier_chatter(mem: Any, tmp_path: Path) -> None:
    """Loaders print progress, and some of it is JSON. The measurement is what the cell prints
    last, so a chatty dependency cannot be mistaken for the result."""
    script = tmp_path / "chatty.py"
    script.write_text('print(\'{"progress": 1}\')\nprint("loading")\nprint(\'{"peak_mb": 7}\')\n')

    assert mem.run_cell(script, {}) == {"peak_mb": 7}


# --- extrapolating a container capacity ---------------------------------------------------------


def _cells(backend: str, pairs: list[tuple[int, float]]) -> list[dict[str, Any]]:
    return [{"backend": backend, "rows": rows, "resident_mb": mb} for rows, mb in pairs]


@pytest.fixture(scope="module")
def tables() -> Any:
    spec = importlib.util.spec_from_file_location(
        "gen_bench_tables", Path(__file__).resolve().parents[1] / "scripts" / "gen_bench_tables.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a_flat_store_gets_no_invented_capacity(tables: Any) -> None:
    """The defect this guard exists for.

    sqlite_vec drifted 0.7 MB across 99,000 rows - noise - and dividing 8 GB by it produced
    "about 1,152,205,428 chunks", a number with ten significant digits and no measurement behind
    any of them.
    """
    verdict = tables._rows_per_gb(_cells("sqlite_vec", [(1_000, 45.1), (100_000, 45.8)]))["sqlite_vec"]

    assert "flat" in verdict
    assert "chunks servable" not in verdict.lower()
    assert "1,1" not in verdict


def test_a_store_whose_footprint_falls_is_not_extrapolated(tables: Any) -> None:
    """lancedb builds its ANN index at 100,000 rows and gets LIGHTER crossing it, so the ladder
    straddles two regimes and a single line describes neither."""
    verdict = tables._rows_per_gb(_cells("lancedb", [(1_000, 166.1), (50_000, 320.0), (100_000, 164.0)]))["lancedb"]

    assert "not extrapolated" in verdict


def test_a_genuinely_growing_store_does_get_a_number(tables: Any) -> None:
    """The guard must not refuse everything; json grows by 30x across the ladder."""
    verdict = tables._rows_per_gb(_cells("json", [(1_000, 62.3), (100_000, 1894.5)]))["json"]

    assert "about" in verdict and "chunks" in verdict


def test_the_capacity_lands_in_the_right_order_of_magnitude(tables: Any) -> None:
    """18.5 kB per chunk into 8 GB is a few hundred thousand, not tens or millions."""
    verdict = tables._rows_per_gb(_cells("json", [(1_000, 62.3), (100_000, 1894.5)]))["json"]
    capacity = int(verdict.split("about ")[1].split(" ")[0].replace(",", ""))

    assert 200_000 < capacity < 800_000


def test_a_single_measured_scale_yields_no_verdict(tables: Any) -> None:
    """One point cannot establish a slope, and a default slope would be fiction."""
    assert tables._rows_per_gb(_cells("json", [(1_000, 62.3)])) == {}
