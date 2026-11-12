"""Volatility, Sharpe, Sortino, the per-session risk-free rate, the sessions-per-year
rule and the benchmark-relative figures.

The calendars here are built by hand so that sessions per year can be counted: 2025
and 2026 each have 261 weekdays (each starts on a weekday and has 52 weeks and one
day), and a calendar year has 365 days. Expected ratios are the published formulas
applied with the standard library, never the code under test.
"""

import math
import statistics
import warnings

import pandas as pd
import pytest

from openalgo.backtest import metrics
from openalgo.backtest.calendar import Calendar, CalendarWarning
from openalgo.backtest.errors import CalendarError, ConfigError

IST = "Asia/Kolkata"


def at(day: str, clock: str = "15:30") -> pd.Timestamp:
    return pd.Timestamp(f"{day} {clock}", tz=IST)


def holiday(day: str, closed: list[str]) -> dict:
    return {
        "date": day,
        "description": "test holiday",
        "holiday_type": "TRADING_HOLIDAY",
        "closed_exchanges": closed,
        "open_exchanges": [],
    }


def calendar_with(year: int, rows: list[dict], exchanges: list[str]) -> Calendar:
    cal = Calendar(builtin=False)
    cal.add_holidays({"data": rows}, year=year, verified=True, exchanges=exchanges)
    return cal


@pytest.fixture
def nse_2025() -> Calendar:
    """NSE 2025 with one holiday (Friday 15 August): 260 sessions, sourced."""
    return calendar_with(2025, [holiday("2025-08-15", ["NSE"])], ["NSE"])


NSE_2025_SESSIONS_PER_YEAR = 260 * 365.25 / 365


def growth_curve(returns: list[float], start: str = "2025-06-02") -> pd.Series:
    """Equity 100 at 09:15 on `start`, then one close per weekday compounding `returns`."""
    closes = pd.bdate_range(start, periods=len(returns), tz=IST) + pd.Timedelta(
        hours=15, minutes=30
    )
    values = [100.0]
    for r in returns:
        values.append(values[-1] * (1 + r))
    index = [at(start, "09:15"), *closes]
    return pd.Series(values, index=index)


RETURNS = [0.01, -0.02, 0.015, 0.005, -0.01]


def test_sessions_per_year_counts_the_whole_calendar_year_of_the_run(nse_2025):
    m = metrics.analyse(growth_curve(RETURNS), exchanges="NSE", calendar=nse_2025).metrics
    assert m["sessions_per_year"] == pytest.approx(NSE_2025_SESSIONS_PER_YEAR)
    assert (m["annualisation_years"], m["annualisation_coverage"]) == ("2025", "sourced")
    assert m["annualisation_template_years"] == ""


def test_a_run_across_two_years_counts_both_years():
    cal = calendar_with(2025, [holiday("2025-08-15", ["NSE"])], ["NSE"])
    cal.add_holidays(
        {"data": [holiday("2026-01-26", ["NSE"])]}, year=2026, verified=True, exchanges=["NSE"]
    )
    equity = growth_curve(RETURNS, start="2025-12-29")
    m = metrics.analyse(equity, exchanges="NSE", calendar=cal).metrics
    # 260 sessions in each year over 730 calendar days.
    assert m["sessions_per_year"] == pytest.approx(520 * 365.25 / 730)
    assert m["annualisation_years"] == "2025 to 2026"


def test_the_union_calendar_of_the_traded_exchanges_is_used():
    # NSE is shut on 15 August and MCX trades: the union has all 261 weekdays.
    cal = calendar_with(2025, [holiday("2025-08-15", ["NSE"])], ["NSE", "MCX"])
    m = metrics.analyse(growth_curve(RETURNS), exchanges=["NSE", "MCX"], calendar=cal).metrics
    assert m["sessions_per_year"] == pytest.approx(261 * 365.25 / 365)


def test_a_year_without_a_holiday_list_is_reported_and_warned():
    with pytest.warns(CalendarWarning, match="NSE 2025"):
        result = metrics.analyse(
            growth_curve(RETURNS), exchanges="NSE", calendar=Calendar(builtin=False)
        )
    m = result.metrics
    assert (m["annualisation_coverage"], m["annualisation_template_years"]) == (
        "template",
        "NSE 2025",
    )
    assert m["sessions_per_year"] == pytest.approx(261 * 365.25 / 365)


def test_volatility_sharpe_and_sortino_follow_their_definitions(nse_2025):
    n_per_year = NSE_2025_SESSIONS_PER_YEAR
    rate = 0.065
    rf = (1 + rate) ** (1 / n_per_year) - 1
    excess = [r - rf for r in RETURNS]
    downside = math.sqrt(sum(min(x, 0.0) ** 2 for x in excess) / len(excess))
    m = metrics.analyse(
        growth_curve(RETURNS), exchanges="NSE", risk_free_rate=rate, calendar=nse_2025
    ).metrics
    assert m["risk_free_per_session"] == pytest.approx(rf)
    assert m["volatility"] == pytest.approx(statistics.stdev(RETURNS) * math.sqrt(n_per_year))
    expected_sharpe = statistics.mean(excess) / statistics.stdev(excess) * math.sqrt(n_per_year)
    assert m["sharpe"] == pytest.approx(expected_sharpe)
    expected_sortino = statistics.mean(excess) / downside * math.sqrt(n_per_year)
    assert m["sortino"] == pytest.approx(expected_sortino)


def test_the_annual_risk_free_rate_is_never_subtracted_per_session(nse_2025):
    m = metrics.analyse(
        growth_curve(RETURNS), exchanges="NSE", risk_free_rate=0.065, calendar=nse_2025
    ).metrics
    per_bar_annual = [r - 0.065 for r in RETURNS]
    wrong = statistics.mean(per_bar_annual) / statistics.stdev(per_bar_annual)
    wrong *= math.sqrt(NSE_2025_SESSIONS_PER_YEAR)
    assert abs(m["sharpe"] - wrong) > 50


def test_a_risk_free_rate_written_in_percent_is_refused(nse_2025):
    with pytest.raises(ConfigError, match="0.065 is 6.5"):
        metrics.analyse(
            growth_curve(RETURNS), exchanges="NSE", risk_free_rate=6.5, calendar=nse_2025
        )


def test_an_unknown_exchange_is_a_calendar_error():
    with pytest.raises(CalendarError):
        metrics.analyse(growth_curve(RETURNS), exchanges="NASDAQ")


def test_no_exchange_is_refused():
    with pytest.raises(ConfigError, match="exchanges"):
        metrics.analyse(growth_curve(RETURNS), exchanges=[])


# ----- benchmark --------------------------------------------------------------------

BENCHMARK_RETURNS = [0.01, -0.02, 0.03]


def benchmark_levels(returns: list[float], index: pd.Index) -> pd.Series:
    levels = [1000.0]
    for r in returns:
        levels.append(levels[-1] * (1 + r))
    return pd.Series(levels, index=index)


def test_twice_the_benchmark_has_beta_two_correlation_one_and_alpha_the_rate(nse_2025):
    strategy = growth_curve([2 * r for r in BENCHMARK_RETURNS])
    bench = benchmark_levels(BENCHMARK_RETURNS, strategy.index)
    m = metrics.analyse(
        strategy, exchanges="NSE", benchmark=bench, risk_free_rate=0.065, calendar=nse_2025
    ).metrics
    rf = (1.065) ** (1 / NSE_2025_SESSIONS_PER_YEAR) - 1
    assert m["beta"] == pytest.approx(2.0)
    assert m["correlation"] == pytest.approx(1.0)
    # mean(2b - rf) - 2 * mean(b - rf) = rf per session.
    assert m["alpha"] == pytest.approx(rf * NSE_2025_SESSIONS_PER_YEAR)
    assert m["benchmark_sessions"] == 3
    assert m["benchmark_total_return"] == pytest.approx(1.01 * 0.98 * 1.03 - 1)


def test_the_benchmark_is_scaled_to_the_starting_equity(nse_2025):
    strategy = growth_curve(BENCHMARK_RETURNS)
    bench = benchmark_levels(BENCHMARK_RETURNS, strategy.index)
    result = metrics.analyse(strategy, exchanges="NSE", benchmark=bench, calendar=nse_2025)
    assert result.benchmark_equity.to_numpy() == pytest.approx(strategy.to_numpy())


def test_a_benchmark_without_a_value_at_the_start_begins_at_its_first_close(nse_2025):
    strategy = growth_curve(BENCHMARK_RETURNS)
    # Daily closes only: none at the 09:15 start.
    bench = benchmark_levels(BENCHMARK_RETURNS[1:], strategy.index[1:])
    m = metrics.analyse(strategy, exchanges="NSE", benchmark=bench, calendar=nse_2025).metrics
    assert m["benchmark_sessions"] == 2
    assert m["benchmark_total_return"] == pytest.approx(0.98 * 1.03 - 1)


def test_a_benchmark_is_never_back_filled(nse_2025):
    strategy = growth_curve([0.01, 0.01, 0.01])
    # No benchmark close on Tuesday: Tuesday's level is Monday's, never Wednesday's.
    index = strategy.index[[0, 1, 3]]
    bench = pd.Series([1000.0, 1010.0, 1050.0], index=index)
    result = metrics.analyse(strategy, exchanges="NSE", benchmark=bench, calendar=nse_2025)
    scaled = result.benchmark_equity.to_numpy()
    assert scaled[2] == pytest.approx(scaled[1])


def test_benchmark_returns_must_be_levels_not_returns(nse_2025):
    strategy = growth_curve(BENCHMARK_RETURNS)
    returns = pd.Series([0.0, 0.01, -0.02, 0.03], index=strategy.index)
    with pytest.raises(ConfigError, match="levels"):
        metrics.analyse(strategy, exchanges="NSE", benchmark=returns, calendar=nse_2025)


def test_benchmark_keys_are_absent_without_a_benchmark(nse_2025):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = metrics.analyse(growth_curve(RETURNS), exchanges="NSE", calendar=nse_2025).metrics
    assert "beta" not in m and "alpha" not in m
