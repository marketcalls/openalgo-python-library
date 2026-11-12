"""The tables of a Performance: drawdown episodes and the worst of them, losing
streaks, the benchmark comparison; and the sign of every drawdown figure.

2 June 2025 is a Monday; equity closes at 15:30 IST and opens at 100 rupees.
"""

import math

import numpy as np
import pandas as pd
import pytest

from openalgo.backtest import metrics
from openalgo.backtest.errors import ConfigError

IST = "Asia/Kolkata"


def at(day: str, clock: str = "15:30") -> pd.Timestamp:
    return pd.Timestamp(f"{day} {clock}", tz=IST)


def closes(*values: float, start: str = "2025-06-02") -> pd.Series:
    days = pd.bdate_range(start, periods=len(values), tz=IST)
    return pd.Series(values, index=days + pd.Timedelta(hours=15, minutes=30))


# Two episodes: from the 110 peak (Tue) to the 88 trough (Wed), recovered at 121 (Fri);
# then from 121 down to 115 (Mon), never recovered.
CURVE = closes(105.0, 110.0, 88.0, 100.0, 121.0, 115.0)


def analyse(equity: pd.Series, **kwargs) -> metrics.Performance:
    return metrics.analyse(equity, exchanges="NSE", opening_wealth=100.0, **kwargs)


def test_every_drawdown_episode_is_listed_in_the_order_it_starts():
    episodes = analyse(CURVE).drawdown_episodes
    assert list(episodes["start"]) == [at("2025-06-03"), at("2025-06-06")]
    assert list(episodes["trough"]) == [at("2025-06-04"), at("2025-06-09")]
    assert list(episodes["recovered"]) == [True, False]
    assert episodes["depth"].to_numpy() == pytest.approx([88 / 110 - 1, 115 / 121 - 1])
    # Peak to recovery (Tue to Fri), and peak to the last point (Fri to Mon).
    assert episodes["days"].to_numpy() == pytest.approx([3.0, 3.0])
    assert episodes["days_to_trough"].to_numpy() == pytest.approx([1.0, 3.0])
    assert episodes.loc[0, "days_to_recover"] == pytest.approx(2.0)
    assert pd.isna(episodes.loc[1, "end"]) and math.isnan(episodes.loc[1, "days_to_recover"])


def test_drawdown_details_rank_the_deepest_first():
    details = analyse(CURVE).drawdown_details(top=1)
    assert list(details.index) == [1]
    assert details.loc[1, "depth"] == pytest.approx(-0.2)
    assert len(analyse(CURVE).drawdown_details(top=10)) == 2


def test_a_fall_from_the_opening_starts_at_the_start_of_the_run():
    details = analyse(closes(95.0, 101.0)).drawdown_details()
    assert details.loc[1, "start"] == at("2025-06-02")
    assert details.loc[1, "depth"] == pytest.approx(-0.05)


@pytest.mark.parametrize("top", [0, -1, 1.5, True])
def test_a_count_that_is_not_a_positive_whole_number_is_refused(top):
    with pytest.raises(ConfigError, match="top"):
        analyse(CURVE).drawdown_details(top=top)


def test_a_curve_that_never_falls_has_no_drawdown_detail():
    assert analyse(closes(101.0, 102.0)).drawdown_details().empty


def test_losing_session_streaks_are_counted_by_length():
    # Returns +5%, then -2%, -1% (a streak of 2), +3%, -1% (a streak of 1).
    streaks = analyse(closes(105.0, 102.9, 101.871, 104.927, 103.878)).losing_streaks
    assert list(streaks.index) == [1, 2]
    assert list(streaks["sessions"]) == [1, 1]
    assert list(streaks["trades"]) == [0, 0]


def benchmark_run() -> metrics.Performance:
    """Sessions 27 and 30 June, 1 and 2 July 2025; the benchmark from Thursday 26 June."""
    equity = closes(102.0, 104.04, 101.9592, 102.978792, start="2025-06-27")
    index = [at("2025-06-26"), *equity.index]
    bench = pd.Series([1000.0, 1010.0, 1000.0, 1010.0, 1020.0], index=index)
    return analyse(equity, benchmark=bench)


def test_the_benchmark_comparison_compounds_both_per_year_and_per_month():
    p = benchmark_run()
    monthly = p.benchmark_comparison("month")
    # June: strategy +2%, +2%; benchmark +1%, -0.990099%.
    assert monthly["strategy"].to_numpy() == pytest.approx([1.02 * 1.02 - 1, 0.98 * 1.01 - 1])
    assert monthly["benchmark"].to_numpy() == pytest.approx([0.0, 1020 / 1000 - 1])
    assert list(monthly["beat"]) == [True, False]
    yearly = p.benchmark_comparison()
    assert list(yearly.index) == [2025]
    assert yearly.loc[2025, "excess"] == pytest.approx(102.978792 / 100 - 1 - 0.02)


def test_a_period_compares_the_benchmark_from_its_first_value_in_the_period():
    equity = closes(102.0, 104.04)
    bench = pd.Series([1000.0, 1010.0], index=equity.index)
    comparison = analyse(equity, benchmark=bench).benchmark_comparison()
    # The opening has no benchmark value, so the first session has no benchmark return
    # and the year runs from the first session's 1000 to 1010, as the run's total does.
    assert comparison.loc[2025, "benchmark"] == pytest.approx(1010 / 1000 - 1)
    assert bool(comparison.loc[2025, "beat"]) is True


def test_a_comparison_needs_a_benchmark_and_a_known_period():
    with pytest.raises(ConfigError, match="benchmark"):
        analyse(CURVE).benchmark_comparison()
    with pytest.raises(ConfigError, match="period"):
        benchmark_run().benchmark_comparison("week")


# ----- the sign of a drawdown ---------------------------------------------------------


def test_every_drawdown_figure_is_zero_or_negative():
    p = analyse(CURVE)
    m = p.metrics
    assert m["max_drawdown"] == pytest.approx(-0.2)
    # Cumulative P&L fell from +10 (110) to -12 (88): 22 rupees.
    assert m["max_drawdown_amount"] == pytest.approx(-22.0)
    assert m["conditional_drawdown"] < 0
    assert (p.drawdown.to_numpy() <= 0).all()
    assert (p.drawdown_episodes["depth"] < 0).all()
    assert (p.horizons["max_drawdown"] <= 0).all()
    study = metrics.monte_carlo(p, simulations=50, seed=1)
    assert (study.runs["max_drawdown"] <= 0).all()


def test_durations_and_the_ulcer_index_are_zero_or_positive():
    p = analyse(CURVE)
    assert p.metrics["ulcer_index"] > 0
    assert p.metrics["max_drawdown_days"] > 0
    assert (p.drawdown_episodes["days"] >= 0).all()
    assert np.isfinite(p.metrics["recovery_factor"])
    # Net profit 15 over a drawdown of 22 rupees.
    assert p.metrics["recovery_factor"] == pytest.approx(15 / 22)
