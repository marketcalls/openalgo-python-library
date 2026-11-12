"""Session returns, total return, CAGR, cash flows and the edge cases of an equity curve.

Expected values are worked out by hand from the definitions in
`openalgo.backtest.metrics`. 2 June 2025 is a Monday; the default calendar has a
sourced NSE holiday list for 2025 and 2026, and 2 to 6 June 2025 are all sessions.
"""

import math

import pandas as pd
import pytest

from openalgo.backtest import metrics
from openalgo.backtest.errors import ConfigError

IST = "Asia/Kolkata"


def at(day: str, clock: str = "15:30") -> pd.Timestamp:
    return pd.Timestamp(f"{day} {clock}", tz=IST)


def curve(*points: tuple[pd.Timestamp, float]) -> pd.Series:
    return pd.Series([value for _, value in points], index=[when for when, _ in points])


def june_curve(*values: float) -> pd.Series:
    """The start at 09:15 on Monday 2 June, then one close per weekday from 2 June."""
    closes = [at(f"2025-06-{day:02d}") for day in (2, 3, 4, 5, 6, 9, 10, 11, 12, 13)]
    return curve((at("2025-06-02", "09:15"), values[0]), *zip(closes, values[1:], strict=False))


def test_each_ist_session_date_gives_one_return_from_its_last_point():
    equity = curve(
        (at("2025-06-02", "09:15"), 100.0),
        (at("2025-06-02", "11:00"), 104.0),
        (at("2025-06-02"), 110.0),
        (at("2025-06-03"), 99.0),
    )
    returns = metrics.analyse(equity, exchanges="NSE").session_returns
    assert list(returns.index.strftime("%Y-%m-%d")) == ["2025-06-02", "2025-06-03"]
    # 110 / 100 - 1 and 99 / 110 - 1.
    assert returns.to_numpy() == pytest.approx([0.10, -0.10])


def test_session_returns_are_indexed_by_naive_ist_dates():
    returns = metrics.analyse(june_curve(100.0, 101.0, 102.0), exchanges="NSE").session_returns
    assert returns.index.tz is None
    assert returns.index.name == "date"


def test_a_point_after_ist_midnight_belongs_to_the_next_session_date():
    # Given in UTC: 22:30 UTC on 2 June is 04:00 IST on 3 June.
    equity = pd.Series(
        [100.0, 101.0, 102.0],
        index=pd.DatetimeIndex(
            ["2025-06-02 03:45", "2025-06-02 10:00", "2025-06-02 22:30"], tz="UTC"
        ),
    )
    returns = metrics.analyse(equity, exchanges="MCX").session_returns
    assert list(returns.index.strftime("%Y-%m-%d")) == ["2025-06-02", "2025-06-03"]


def test_int64_utc_nanoseconds_are_accepted_as_timestamps():
    equity = june_curve(100.0, 110.0)
    nanos = equity.index.tz_convert("UTC").as_unit("ns").asi8
    result = metrics.analyse(pd.Series(equity.to_numpy(), index=nanos), exchanges="NSE")
    assert result.metrics["start"] == at("2025-06-02", "09:15")
    assert result.metrics["total_return"] == pytest.approx(0.10)


@pytest.mark.parametrize("unit", ["s", "ms", "us"])
def test_integer_timestamps_in_a_coarser_unit_are_refused(unit):
    equity = june_curve(100.0, 110.0)
    coarse = equity.index.tz_convert("UTC").as_unit(unit).asi8
    with pytest.raises(ConfigError, match="nanoseconds"):
        metrics.analyse(pd.Series(equity.to_numpy(), index=coarse), exchanges="NSE")


def test_total_return_is_time_weighted_and_a_deposit_is_not_a_return():
    equity = june_curve(100.0, 110.0, 160.0, 168.0)
    deposit = pd.Series([50.0], index=[at("2025-06-03")])
    m = metrics.analyse(equity, exchanges="NSE", cash_flows=deposit).metrics
    # Session returns 0.10, (160 - 50) / 110 - 1 = 0 and 168 / 160 - 1 = 0.05.
    assert m["total_return"] == pytest.approx(1.10 * 1.00 * 1.05 - 1)
    assert m["net_cash_flows"] == 50.0
    assert m["net_profit"] == pytest.approx(168 - 100 - 50)


def test_cagr_compounds_over_elapsed_calendar_time():
    # Exactly one year of 365.25 days from 00:00 IST on 1 January 2025.
    start = at("2025-01-01", "00:00")
    equity = curve(
        (start, 100.0), (at("2025-06-02"), 104.0), (start + pd.Timedelta(days=365.25), 110.0)
    )
    m = metrics.analyse(equity, exchanges="NSE").metrics
    assert m["elapsed_days"] == pytest.approx(365.25)
    assert m["cagr"] == pytest.approx(0.10)


def test_half_a_year_at_ten_percent_annualises_to_twenty_one_percent():
    start = at("2025-01-01", "00:00")
    end = start + pd.Timedelta(days=365.25 / 2)
    equity = curve((start, 100.0), (at("2025-03-03"), 103.0), (end, 110.0))
    assert metrics.analyse(equity, exchanges="NSE").metrics["cagr"] == pytest.approx(0.21)


def test_cagr_is_not_annualised_below_four_weeks():
    m = metrics.analyse(june_curve(100.0, 101.0, 102.0, 103.0), exchanges="NSE").metrics
    assert m["elapsed_days"] < 28
    assert math.isnan(m["cagr"]) and math.isnan(m["calmar"])
    assert m["total_return"] == pytest.approx(0.03)


def test_a_flat_curve_has_no_return_no_volatility_and_no_ratio():
    days = pd.bdate_range("2025-06-02", "2025-07-31", tz=IST) + pd.Timedelta(hours=15, minutes=30)
    equity = pd.Series(100.0, index=days)
    m = metrics.analyse(equity, exchanges="NSE").metrics
    assert (m["total_return"], m["cagr"], m["volatility"], m["max_drawdown"]) == (0, 0, 0, 0)
    assert math.isnan(m["sharpe"]) and math.isnan(m["sortino"]) and math.isnan(m["calmar"])


def test_equity_that_reaches_zero_ruins_the_account():
    m = metrics.analyse(june_curve(100.0, 50.0, 0.0, 20.0), exchanges="NSE").metrics
    assert m["ruined"] is True
    assert m["ruin_time"] == at("2025-06-03")
    assert (m["total_return"], m["cagr"], m["max_drawdown"]) == (-1.0, -1.0, -1.0)
    # Returns stop at the ruin session: -0.5 then -1.
    assert m["sessions"] == 2


def test_after_ruin_the_rupee_figures_stay_true():
    m = metrics.analyse(june_curve(100.0, 50.0, 0.0, 20.0), exchanges="NSE").metrics
    assert m["end_equity"] == 20.0
    assert m["net_profit"] == -80.0


def test_a_single_session_has_a_return_but_no_volatility():
    equity = curve((at("2025-06-02", "09:15"), 100.0), (at("2025-06-02"), 101.0))
    m = metrics.analyse(equity, exchanges="NSE").metrics
    assert m["sessions"] == 1
    assert m["total_return"] == pytest.approx(0.01)
    assert math.isnan(m["volatility"]) and math.isnan(m["sharpe"]) and math.isnan(m["cagr"])


def test_missing_sessions_are_counted_and_folded_into_the_next_return():
    # No point on Wednesday 4 June: Thursday's return spans two sessions.
    equity = curve(
        (at("2025-06-02", "09:15"), 100.0),
        (at("2025-06-02"), 100.0),
        (at("2025-06-03"), 100.0),
        (at("2025-06-05"), 121.0),
        (at("2025-06-06"), 121.0),
    )
    result = metrics.analyse(equity, exchanges="NSE")
    assert result.metrics["missing_sessions"] == 1
    assert result.session_returns.loc["2025-06-05"] == pytest.approx(0.21)


def test_start_and_end_figures_are_reported_in_ist():
    m = metrics.analyse(june_curve(100.0, 110.0, 99.0), exchanges="NSE").metrics
    assert m["start"] == at("2025-06-02", "09:15")
    assert m["end"] == at("2025-06-03")
    assert (m["start_equity"], m["end_equity"], m["net_profit"]) == (100.0, 99.0, -1.0)
