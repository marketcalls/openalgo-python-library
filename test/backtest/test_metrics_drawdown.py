"""Maximum and longest drawdown, their dates and durations, the rupee drawdown and Calmar.

2 June 2025 is a Monday; every weekday of June 2025 is an NSE session.
"""

import math

import pandas as pd
import pytest

from openalgo.backtest import metrics

IST = "Asia/Kolkata"


def at(day: str, clock: str = "15:30") -> pd.Timestamp:
    return pd.Timestamp(f"{day} {clock}", tz=IST)


def june_curve(*values: float) -> pd.Series:
    """The start at 09:15 on 2 June, then the closes of the weekdays from 2 June."""
    closes = pd.bdate_range("2025-06-02", periods=len(values) - 1, tz=IST)
    index = [at("2025-06-02", "09:15"), *(closes + pd.Timedelta(hours=15, minutes=30))]
    return pd.Series(values, index=index)


def test_maximum_drawdown_with_its_dates_and_duration():
    m = metrics.analyse(june_curve(100.0, 110.0, 99.0, 121.0), exchanges="NSE").metrics
    # 99 / 110 - 1.
    assert m["max_drawdown"] == pytest.approx(-0.10)
    assert m["max_drawdown_peak"] == at("2025-06-02")
    assert m["max_drawdown_trough"] == at("2025-06-03")
    assert m["max_drawdown_recovery"] == at("2025-06-04")
    assert m["max_drawdown_days"] == pytest.approx(2.0)
    assert m["max_drawdown_amount"] == pytest.approx(11.0)


def test_the_drawdown_series_is_the_fall_from_the_running_peak_at_every_point():
    equity = pd.Series(
        [100.0, 110.0, 88.0, 110.0],
        index=[
            at("2025-06-02", "09:15"),
            at("2025-06-02", "10:00"),
            at("2025-06-02", "12:00"),
            at("2025-06-02"),
        ],
    )
    result = metrics.analyse(equity, exchanges="NSE")
    # The intraday dip to 88 counts although the session closed at its peak.
    assert result.drawdown.to_numpy() == pytest.approx([0.0, 0.0, 88 / 110 - 1, 0.0])
    assert result.metrics["max_drawdown"] == pytest.approx(-0.2)


def test_the_longest_drawdown_is_told_apart_from_the_deepest():
    m = metrics.analyse(
        june_curve(100.0, 100.0, 80.0, 100.0, 95.0, 96.0, 97.0, 99.0), exchanges="NSE"
    ).metrics
    # Deepest: peak Mon 2 June close, trough Tue, recovered Wed (2 days).
    assert m["max_drawdown"] == pytest.approx(-0.2)
    assert m["max_drawdown_days"] == pytest.approx(2.0)
    # Longest: from Wed 4 June to the last close, Tue 10 June, not recovered (6 days).
    assert m["longest_drawdown_start"] == at("2025-06-04")
    assert m["longest_drawdown_end"] == at("2025-06-10")
    assert m["longest_drawdown_days"] == pytest.approx(6.0)
    assert m["longest_drawdown_recovered"] is False


def test_an_unrecovered_maximum_drawdown_runs_to_the_end():
    m = metrics.analyse(june_curve(100.0, 120.0, 90.0, 96.0), exchanges="NSE").metrics
    assert m["max_drawdown_recovery"] is None
    # From the Monday close to the Wednesday close.
    assert m["max_drawdown_days"] == pytest.approx(2.0)


def test_a_curve_that_never_falls_has_no_drawdown_dates():
    m = metrics.analyse(june_curve(100.0, 101.0, 102.0), exchanges="NSE").metrics
    assert (m["max_drawdown"], m["max_drawdown_days"], m["longest_drawdown_days"]) == (0, 0, 0)
    assert m["max_drawdown_peak"] is None and m["longest_drawdown_recovered"] is None


def test_a_withdrawal_is_not_a_drawdown():
    equity = june_curve(100.0, 110.0, 60.0, 60.0)
    withdrawal = pd.Series([-50.0], index=[at("2025-06-03")])
    m = metrics.analyse(equity, exchanges="NSE", cash_flows=withdrawal).metrics
    assert m["max_drawdown"] == 0.0
    assert m["max_drawdown_amount"] == 0.0


def test_the_rupee_drawdown_excludes_cash_flows():
    equity = june_curve(100.0, 110.0, 150.0, 140.0)
    deposit = pd.Series([50.0], index=[at("2025-06-03")])
    m = metrics.analyse(equity, exchanges="NSE", cash_flows=deposit).metrics
    # Cumulative P&L: 0, 10, 0 (150 - 100 - 50), -10: a fall of 20 rupees from its peak.
    assert m["max_drawdown_amount"] == pytest.approx(20.0)


def test_calmar_is_cagr_over_the_depth_of_the_maximum_drawdown():
    start = at("2025-01-01", "00:00")
    equity = pd.Series(
        [100.0, 120.0, 90.0, 110.0],
        index=[start, at("2025-03-03"), at("2025-06-02"), start + pd.Timedelta(days=365.25)],
    )
    m = metrics.analyse(equity, exchanges="NSE").metrics
    # One year at 10 percent; the deepest fall is 90 / 120 - 1 = -25 percent.
    assert m["cagr"] == pytest.approx(0.10)
    assert m["calmar"] == pytest.approx(0.10 / 0.25)


def test_calmar_is_undefined_without_cagr():
    m = metrics.analyse(june_curve(100.0, 90.0, 95.0), exchanges="NSE").metrics
    assert math.isnan(m["calmar"])
