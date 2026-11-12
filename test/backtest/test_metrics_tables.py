"""Monthly and yearly return tables, compounded from session returns by IST date."""

import math

import pandas as pd
import pytest

from openalgo.backtest import metrics

IST = "Asia/Kolkata"


def at(day: str, clock: str = "15:30") -> pd.Timestamp:
    return pd.Timestamp(f"{day} {clock}", tz=IST)


def series(points: list[tuple[str, float]]) -> pd.Series:
    return pd.Series([v for _, v in points], index=[at(d) for d, _ in points])


def test_months_compound_their_sessions_and_years_compound_their_months():
    # January sessions +10 percent twice, February -50 percent.
    equity = series(
        [("2025-01-01", 100.0), ("2025-01-02", 110.0), ("2025-01-03", 121.0), ("2025-02-03", 60.5)]
    )
    result = metrics.analyse(equity, exchanges="NSE")
    table = result.monthly_returns
    assert list(table.columns) == list(range(1, 13))
    assert table.loc[2025, 1] == pytest.approx(1.1 * 1.1 - 1)
    assert table.loc[2025, 2] == pytest.approx(-0.5)
    assert all(math.isnan(table.loc[2025, month]) for month in range(3, 13))
    assert result.yearly_returns.loc[2025] == pytest.approx(1.21 * 0.5 - 1)


def test_a_run_over_two_years_has_one_row_per_year():
    equity = series([("2025-12-30", 100.0), ("2025-12-31", 102.0), ("2026-01-02", 99.96)])
    result = metrics.analyse(equity, exchanges="NSE")
    assert list(result.monthly_returns.index) == [2025, 2026]
    assert result.yearly_returns.loc[2025] == pytest.approx(0.02)
    assert result.yearly_returns.loc[2026] == pytest.approx(99.96 / 102 - 1)


def test_the_month_of_a_session_is_its_ist_date():
    # 30 June 22:00 UTC is Tuesday 1 July 03:30 IST: a July session.
    equity = pd.Series(
        [100.0, 101.0, 103.02],
        index=pd.DatetimeIndex(
            ["2025-06-27 10:00", "2025-06-30 10:00", "2025-06-30 22:00"], tz="UTC"
        ),
    )
    table = metrics.analyse(equity, exchanges="MCX").monthly_returns
    assert table.loc[2025, 6] == pytest.approx(0.01)
    assert table.loc[2025, 7] == pytest.approx(0.02)
