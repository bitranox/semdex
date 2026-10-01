"""Energy arithmetic behind the cost table.

Every function here fails by producing a believable number rather than an error, and the published
cost figures are divisions by these results:

* a wrapped RAPL counter reads negative and becomes an energy saving;
* a power gauge summed as though it were a counter is off by whatever the sampling caught;
* a marginal figure taken against an idle baseline measured once, before the run, absorbs whatever
  a neighbouring container was doing at that moment;
* and a workload quieter than the idle swing it was extracted from yields three decimal places of
  nothing at all.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_MODULE = Path(__file__).resolve().parents[1] / "scripts" / "_bench_energy.py"

pytestmark = pytest.mark.os_agnostic


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("_bench_energy", _MODULE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def energy() -> Any:
    return _load()


# --- the wrapping counter -----------------------------------------------------------------------


def test_a_normal_interval_is_the_plain_difference(energy: Any) -> None:
    assert energy.counter_delta_uj(1_000, 3_500, max_range_uj=262_143_328_850) == 2_500


def test_a_wrapped_counter_does_not_report_negative_energy(energy: Any) -> None:
    """RAPL wraps at about 262 kJ, roughly an hour of idle on this box.

    Uncorrected the delta is negative, and a negative joule count divided by a document count is
    published as a machine that generated power while embedding.
    """
    max_range = 262_143_328_850

    delta = energy.counter_delta_uj(max_range - 1_000, 4_000, max_range_uj=max_range)

    assert delta == 5_000


def test_a_zero_range_is_rejected_rather_than_silently_uncorrectable(energy: Any) -> None:
    """Without a range a wrap cannot be corrected, so the caller has to be told, not handed a
    number that is right except across a boundary it will eventually cross."""
    with pytest.raises(ValueError, match="max_energy_range_uj"):
        energy.counter_delta_uj(500, 100, max_range_uj=0)


# --- the integrated gauge -----------------------------------------------------------------------


def test_a_constant_load_integrates_to_power_times_time(energy: Any) -> None:
    samples = [(0.0, 100.0), (1.0, 100.0), (2.0, 100.0)]

    assert energy.integrate_power_w(samples) == pytest.approx(200.0)


def test_a_ramp_is_integrated_trapezoidally_not_by_an_endpoint(energy: Any) -> None:
    """A GPU ramping 0 to 200 W over a second averages 100 W.

    Holding either endpoint flat gives 0 or 200 - a factor of two on a rising load, which is
    exactly the shape of every embedding run's first second.
    """
    assert energy.integrate_power_w([(0.0, 0.0), (1.0, 200.0)]) == pytest.approx(100.0)


def test_irregular_sample_spacing_is_weighted_by_its_own_interval(energy: Any) -> None:
    """HTTP-polled probes do not sample evenly, and treating every gap as equal would weight a
    slow poll the same as a fast one."""
    samples = [(0.0, 100.0), (1.0, 100.0), (11.0, 100.0)]

    assert energy.integrate_power_w(samples) == pytest.approx(1100.0)


def test_too_few_samples_is_zero_rather_than_a_guess(energy: Any) -> None:
    """One reading times an assumed duration is a gauge being treated as a counter."""
    assert energy.integrate_power_w([(0.0, 250.0)]) == 0.0
    assert energy.integrate_power_w([]) == 0.0


# --- separating the work from the machine --------------------------------------------------------


def _result(energy: Any, **kwargs: Any) -> Any:
    base = {"total_j": 6000.0, "seconds": 60.0, "idle_w": 75.0, "idle_spread_w": 5.0, "units": 1000}
    return energy.EnergyResult(**{**base, **kwargs})


def test_marginal_energy_subtracts_the_idle_machine(energy: Any) -> None:
    """100 W average for 60 s against a 75 W idle box: the WORK is the 25 W, not the 100."""
    result = _result(energy)

    assert result.average_w == pytest.approx(100.0)
    assert result.marginal_j == pytest.approx(1500.0)


def test_both_denominators_are_available(energy: Any) -> None:
    """Marginal answers "what does one more document cost"; total answers "what does running this
    service cost" when the machine exists only for it. They differ by 4x here, so publishing one
    without saying which it is would be the whole answer."""
    result = _result(energy)

    assert result.marginal_j_per_unit == pytest.approx(1.5)
    assert result.total_j_per_unit == pytest.approx(6.0)


def test_a_workload_below_idle_clamps_instead_of_reporting_a_saving(energy: Any) -> None:
    """The idle estimate drifts on a shared box. A negative marginal is that drift, and printed
    as-is it says embedding the corpus saved electricity."""
    result = _result(energy, total_j=4000.0)

    assert result.marginal_j == 0.0


def test_a_workload_quieter_than_the_idle_swing_is_not_resolved(energy: Any) -> None:
    """The check that stops the arithmetic being mistaken for a measurement.

    Idle here swings 70 to 97 W. A workload adding 10 W cannot be told apart from that, however
    many decimals the subtraction yields.
    """
    result = _result(energy, total_j=6600.0, idle_spread_w=27.0)  # 35 W marginal against 27 W swing

    assert result.signal_to_noise == pytest.approx(35.0 / 27.0, rel=1e-3)
    assert not result.resolved


def test_a_workload_well_clear_of_the_noise_is_resolved(energy: Any) -> None:
    result = _result(energy, total_j=12000.0, idle_spread_w=5.0)  # 125 W marginal against 5 W

    assert result.resolved


def test_a_zero_length_run_does_not_divide_by_zero(energy: Any) -> None:
    result = _result(energy, seconds=0.0)

    assert result.average_w == 0.0
    assert result.marginal_j == 0.0


# --- the idle baseline ---------------------------------------------------------------------------


def test_the_baseline_is_the_median_not_the_mean(energy: Any) -> None:
    """One neighbour's burst inside a baseline window drags a mean up, which then over-subtracts
    and understates the workload it was meant to isolate."""
    watts, _spread = energy.idle_estimate([74.0, 75.0, 76.0, 300.0])

    assert watts == pytest.approx(75.5)


def test_the_spread_is_the_full_range(energy: Any) -> None:
    """It answers how far one window can land from the middle, which is what bounds resolution."""
    _watts, spread = energy.idle_estimate([70.2, 96.7, 77.4])

    assert spread == pytest.approx(26.5)


def test_no_baseline_samples_yield_zeros_rather_than_an_exception(energy: Any) -> None:
    assert energy.idle_estimate([]) == (0.0, 0.0)


# --- money ----------------------------------------------------------------------------------------


def test_cost_converts_joules_to_kwh_before_pricing(energy: Any) -> None:
    """3.6 MJ is one kWh. Pricing joules directly is a factor of 3.6 million."""
    assert energy.cost_per_million(3.6, tariff_eur_per_kwh=0.30) == pytest.approx(0.30)


def test_the_tariff_is_a_parameter_and_scales_the_result(energy: Any) -> None:
    """It is the assumption the whole money column rests on, so it must be visible and variable
    rather than a constant folded into the arithmetic."""
    cheap = energy.cost_per_million(3.6, tariff_eur_per_kwh=0.10)
    dear = energy.cost_per_million(3.6, tariff_eur_per_kwh=0.40)

    assert dear == pytest.approx(cheap * 4)


# --- sizing the measurement window ---------------------------------------------------------------


def test_a_fast_provider_repeats_the_corpus_to_fill_the_window(energy: Any) -> None:
    """The placeholder embeds ~8,000 docs/s, so a 2,000-document corpus is a quarter-second pass.

    Without repetition the window is 250 ms and the figure is arithmetic on baseline drift - the
    same defect the throughput benchmark had at 0.01 seconds.
    """
    per_pass, passes = energy.window_plan(8000.0, corpus_size=2000, min_seconds=15.0)

    assert per_pass == 2000
    assert passes == 60
    assert per_pass * passes / 8000.0 >= 15.0


def test_a_slow_provider_truncates_the_corpus_instead_of_running_for_minutes(energy: Any) -> None:
    """fastembed manages single digits per second on this CPU. One pass of 2,000 documents is
    eleven minutes, and three windows plus idle would be an hour for one row."""
    per_pass, passes = energy.window_plan(5.0, corpus_size=2000, min_seconds=15.0)

    assert per_pass == 75
    assert passes == 1


def test_the_window_is_filled_rather_than_approached(energy: Any) -> None:
    """Ceiling, not floor: a floor leaves every window a fraction short of the floor duration that
    was chosen precisely because shorter windows are not measurements."""
    _per_pass, passes = energy.window_plan(1500.0, corpus_size=1000, min_seconds=1.0)

    assert passes == 2


def test_an_unmeasurable_rate_falls_back_to_one_pass_of_everything(energy: Any) -> None:
    """A probe too fast to time yields no rate; running the corpus once is the honest default."""
    assert energy.window_plan(0.0, corpus_size=500, min_seconds=15.0) == (500, 1)


def test_a_rate_so_slow_the_target_rounds_to_nothing_still_measures_one_document(energy: Any) -> None:
    assert energy.window_plan(0.001, corpus_size=500, min_seconds=1.0) == (1, 1)
