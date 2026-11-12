"""The opening wealth: the capital before the first equity point.

`metrics.analyse(equity, ..., opening_wealth=capital)` reads every point as a mark after
the opening, so the P&L of the first point, the return of its session and a fall below
the capital all count. Without an opening wealth the first point is the starting equity,
exactly as before.

Expected values are worked out by hand from the definitions in
`openalgo.backtest.metrics`. 2 June 2025 is a Monday; every weekday of June 2025 is an
NSE session, and so are Friday 31 January and Monday 3 February 2025.
"""

import math
import statistics

import pandas as pd
import pytest

from openalgo.backtest import metrics
from openalgo.backtest.errors import ConfigError

IST = "Asia/Kolkata"


def at(day: str, clock: str = "15:30") -> pd.Timestamp:
    return pd.Timestamp(f"{day} {clock}", tz=IST)


def curve(*points: tuple[pd.Timestamp, float]) -> pd.Series:
    return pd.Series([value for _, value in points], index=[when for when, _ in points])


def closes(*values: float) -> pd.Series:
    """One 15:30 close per weekday from Monday 2 June 2025."""
    days = pd.bdate_range("2025-06-02", periods=len(values), tz=IST)
    return pd.Series(values, index=days + pd.Timedelta(hours=15, minutes=30))


def test_a_session_that_gains_one_percent_on_the_opening_wealth():
    equity = curve((at("2025-06-02", "09:16"), 100.5), (at("2025-06-02"), 101.0))
    result = metrics.analyse(equity, exchanges="NSE", opening_wealth=100.0)
    # 101 / 100 - 1: the 0.5 earned by the first point counts.
    assert result.metrics["total_return"] == pytest.approx(0.01)
    assert list(result.session_returns.index.strftime("%Y-%m-%d")) == ["2025-06-02"]
    assert result.session_returns.to_numpy() == pytest.approx([0.01])


def test_the_first_point_of_a_daily_series_has_its_session_return():
    result = metrics.analyse(closes(102.0, 99.96, 104.958), exchanges="NSE", opening_wealth=100.0)
    # 102 / 100 - 1, 99.96 / 102 - 1 and 104.958 / 99.96 - 1.
    assert result.session_returns.to_numpy() == pytest.approx([0.02, -0.02, 0.05])
    assert result.metrics["sessions"] == 3
    assert result.metrics["total_return"] == pytest.approx(1.02 * 0.98 * 1.05 - 1)


def test_volatility_counts_the_return_of_the_first_session():
    m = metrics.analyse(
        closes(102.0, 99.96, 104.958), exchanges="NSE", opening_wealth=100.0
    ).metrics
    expected = statistics.stdev([0.02, -0.02, 0.05]) * math.sqrt(m["sessions_per_year"])
    assert m["volatility"] == pytest.approx(expected)


def test_the_opening_wealth_is_the_start_equity_and_profit_is_measured_from_it():
    m = metrics.analyse(closes(104.0, 103.0), exchanges="NSE", opening_wealth=100.0).metrics
    assert (m["start_equity"], m["end_equity"], m["net_profit"]) == (100.0, 103.0, 3.0)
    assert m["start"] == at("2025-06-02")


def test_one_point_after_the_opening_wealth_is_enough():
    m = metrics.analyse(closes(101.0), exchanges="NSE", opening_wealth=100.0).metrics
    assert m["sessions"] == 1
    assert m["total_return"] == pytest.approx(0.01)


def test_a_fall_below_the_opening_wealth_is_a_drawdown_from_it():
    result = metrics.analyse(closes(95.0, 98.0, 105.0), exchanges="NSE", opening_wealth=100.0)
    m = result.metrics
    # 95 / 100 - 1 and 98 / 100 - 1; 105 is above the opening.
    assert result.drawdown.to_numpy() == pytest.approx([-0.05, -0.02, 0.0])
    assert m["max_drawdown"] == pytest.approx(-0.05)
    assert m["max_drawdown_amount"] == pytest.approx(-5.0)
    # The peak is the opening, timed at the start; recovered at the Wednesday close.
    assert m["max_drawdown_peak"] == m["start"] == at("2025-06-02")
    assert m["max_drawdown_trough"] == at("2025-06-02")
    assert m["max_drawdown_recovery"] == at("2025-06-04")
    assert m["max_drawdown_days"] == pytest.approx(2.0)
    assert m["longest_drawdown_start"] == at("2025-06-02")


def test_monthly_and_yearly_returns_include_the_first_session():
    equity = curve((at("2025-01-31"), 110.0), (at("2025-02-03"), 99.0))
    result = metrics.analyse(equity, exchanges="NSE", opening_wealth=100.0)
    assert result.monthly_returns.loc[2025, 1] == pytest.approx(0.10)
    assert result.monthly_returns.loc[2025, 2] == pytest.approx(-0.10)
    assert result.yearly_returns.loc[2025] == pytest.approx(1.1 * 0.9 - 1)


def test_equity_at_zero_at_the_first_point_ruins_the_account():
    m = metrics.analyse(closes(0.0, 10.0), exchanges="NSE", opening_wealth=100.0).metrics
    assert m["ruined"] is True and m["ruin_time"] == at("2025-06-02")
    assert (m["total_return"], m["max_drawdown"], m["sessions"]) == (-1.0, -1.0, 1)


def test_with_an_opening_wealth_the_benchmark_base_is_its_last_value_before_the_first_point():
    bench = curve(
        (at("2025-05-30"), 1000.0),
        (at("2025-06-02"), 1010.0),
        (at("2025-06-03"), 999.9),
        (at("2025-06-04"), 1049.895),
    )
    result = metrics.analyse(
        closes(102.0, 99.96, 104.958), exchanges="NSE", opening_wealth=100.0, benchmark=bench
    )
    m = result.metrics
    # Benchmark returns from the Friday 30 May close: 0.01, -0.01 and 0.05.
    strategy, index = [0.02, -0.02, 0.05], [0.01, -0.01, 0.05]
    assert m["benchmark_sessions"] == 3
    assert m["benchmark_total_return"] == pytest.approx(1049.895 / 1000.0 - 1)
    assert m["beta"] == pytest.approx(
        statistics.covariance(strategy, index) / statistics.variance(index)
    )
    # Scaled so that the base (as of the opening, which has no time) is the opening wealth.
    assert list(result.benchmark_equity.index) == [at(f"2025-06-0{day}") for day in (2, 3, 4)]
    assert result.benchmark_equity.to_numpy() == pytest.approx([101.0, 99.99, 104.9895])


@pytest.mark.parametrize("value", [0.0, -100.0, math.nan, math.inf, "100", True])
def test_an_opening_wealth_that_is_not_a_positive_number_is_refused(value):
    with pytest.raises(ConfigError, match="opening_wealth"):
        metrics.analyse(closes(101.0, 102.0), exchanges="NSE", opening_wealth=value)


def test_without_an_opening_wealth_the_first_point_is_the_starting_equity():
    result = metrics.analyse(closes(95.0, 98.0, 105.0), exchanges="NSE")
    m = result.metrics
    # Today's reading: 95 is the start, so its session has no return and no drawdown.
    assert result.session_returns.to_numpy() == pytest.approx([98 / 95 - 1, 105 / 98 - 1])
    assert (m["sessions"], m["start_equity"], m["net_profit"]) == (2, 95.0, 10.0)
    assert m["total_return"] == pytest.approx(105 / 95 - 1)
    assert (m["max_drawdown"], m["max_drawdown_peak"]) == (0.0, None)
    assert result.drawdown.to_numpy() == pytest.approx([0.0, 0.0, 0.0])


def test_an_opening_wealth_of_none_is_the_same_as_none_given():
    equity = curve(
        (at("2025-06-02", "09:15"), 100.0),
        (at("2025-06-02"), 110.0),
        (at("2025-06-03"), 99.0),
        (at("2025-06-04"), 121.0),
    )
    omitted = metrics.analyse(equity, exchanges="NSE")
    explicit = metrics.analyse(equity, exchanges="NSE", opening_wealth=None)
    assert omitted.metrics["total_return"] == pytest.approx(0.21)
    assert omitted.metrics["max_drawdown"] == pytest.approx(-0.10)
    pd.testing.assert_series_equal(omitted.session_returns, explicit.session_returns)
    pd.testing.assert_series_equal(omitted.drawdown, explicit.drawdown)
    assert omitted.metrics.keys() == explicit.metrics.keys()
    for key, value in omitted.metrics.items():
        other = explicit.metrics[key]
        both_nan = isinstance(value, float) and math.isnan(value) and math.isnan(other)
        assert value == other or both_nan, key
