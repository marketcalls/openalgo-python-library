"""Rolling statistics over a window of sessions, and the horizon summary.

Expected rolling values are the definitions of the whole-run figures applied to each
window with the standard library; horizons are counted on the calendar by hand.
"""

import math
import statistics

import pandas as pd
import pytest

from openalgo.backtest import metrics
from openalgo.backtest.errors import ConfigError

IST = "Asia/Kolkata"
RETURNS = [0.01, -0.01, 0.02, 0.0, -0.03, 0.015]


def closes(returns: list[float], start: str = "2025-06-02") -> pd.Series:
    values, level = [], 100.0
    for r in returns:
        level *= 1 + r
        values.append(level)
    days = pd.bdate_range(start, periods=len(returns), tz=IST)
    return pd.Series(values, index=days + pd.Timedelta(hours=15, minutes=30))


def analyse(returns: list[float], **kwargs) -> metrics.Performance:
    return metrics.analyse(closes(returns), exchanges="NSE", opening_wealth=100.0, **kwargs)


def test_each_rolling_value_is_the_statistic_of_its_window():
    p = analyse(RETURNS, rolling_window=3)
    rolling, n_year = p.rolling, p.metrics["sessions_per_year"]
    assert list(rolling.columns) == ["volatility", "sharpe", "sortino", "win_rate"]
    assert rolling.iloc[:2].isna().all().all()
    window = RETURNS[1:4]
    assert rolling["volatility"].iloc[3] == pytest.approx(
        statistics.stdev(window) * math.sqrt(n_year)
    )
    assert rolling["sharpe"].iloc[3] == pytest.approx(
        statistics.mean(window) / statistics.stdev(window) * math.sqrt(n_year)
    )
    downside = math.sqrt(sum(min(r, 0.0) ** 2 for r in window) / 3)
    assert rolling["sortino"].iloc[3] == pytest.approx(
        statistics.mean(window) / downside * math.sqrt(n_year)
    )
    # -0.01, 0.02 and a flat session: one win of two decided.
    assert rolling["win_rate"].iloc[3] == pytest.approx(0.5)
    assert p.metrics["rolling_window_sessions"] == 3


def test_rolling_beta_regresses_each_window_on_the_benchmark():
    equity = closes(RETURNS)
    levels, level = [1000.0], 1000.0
    for r in RETURNS:
        level *= 1 + r / 2
        levels.append(level)
    index = [equity.index[0] - pd.Timedelta(days=3), *equity.index]
    p = metrics.analyse(
        equity,
        exchanges="NSE",
        opening_wealth=100.0,
        benchmark=pd.Series(levels, index=index),
        rolling_window=3,
    )
    # Every strategy return is twice the benchmark's.
    assert p.rolling["beta"].iloc[2:].to_numpy() == pytest.approx([2.0] * 4)


def test_the_default_window_is_half_a_year_of_sessions():
    p = analyse(RETURNS)
    assert p.metrics["rolling_window_sessions"] == round(p.metrics["sessions_per_year"] / 2)
    # Six sessions never fill it.
    assert p.rolling["sharpe"].isna().all()


@pytest.mark.parametrize("window", [1, 0, 2.5, "20", True])
def test_a_window_that_is_not_two_or_more_sessions_is_refused(window):
    with pytest.raises(ConfigError, match="rolling_window"):
        analyse(RETURNS, rolling_window=window)


# ----- horizons ---------------------------------------------------------------------


def long_run() -> metrics.Performance:
    """0.1 percent every weekday from Monday 3 June 2024 to Tuesday 30 September 2025."""
    days = pd.bdate_range("2024-06-03", "2025-09-30")
    return metrics.analyse(
        closes([0.001] * len(days), start="2024-06-03"),
        exchanges="NSE",
        opening_wealth=100.0,
    )


def test_horizons_the_run_covers_are_listed_with_since_start():
    horizons = long_run().horizons
    # From 3 June 2024: one year back from 30 September 2025 is covered, three are not.
    assert list(horizons.index) == ["1M", "3M", "6M", "1Y", "Since start"]


def test_a_horizon_compounds_the_sessions_after_its_start_date():
    horizons = long_run().horizons
    # One month back from 30 September 2025 is 30 August, a Saturday: the sessions are
    # the weekdays from 1 to 30 September.
    sessions = len(pd.bdate_range("2025-09-01", "2025-09-30"))
    row = horizons.loc["1M"]
    assert row["sessions"] == sessions
    assert str(row["from"]) == "2025-08-30"
    assert row["total_return"] == pytest.approx(1.001**sessions - 1)
    assert row["max_drawdown"] == 0.0


def test_since_start_is_the_whole_run():
    p = long_run()
    row = p.horizons.loc["Since start"]
    assert row["total_return"] == pytest.approx(p.metrics["total_return"])
    assert row["cagr"] == pytest.approx(p.metrics["cagr"])
    assert row["sessions"] == p.metrics["sessions"]


def test_a_horizon_cagr_is_annualised_over_its_calendar_days():
    row = long_run().horizons.loc["1Y"]
    # 30 September 2024 to 30 September 2025 is 365 days.
    assert row["cagr"] == pytest.approx((1 + row["total_return"]) ** (365.25 / 365) - 1)


def test_a_short_run_has_only_since_start():
    assert list(analyse(RETURNS).horizons.index) == ["Since start"]
