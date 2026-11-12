"""`Result.performance` measures a run from its capital.

The capital is the opening wealth of the run, before its first equity point (the close
of the first slice). So the P&L made in the first slice, the return of the first session
and a fall below the capital all count in the metrics.

Own data, no charges and no slippage, SBIN on NSE. Intraday: Tuesday 1 July 2025, one
1-minute bar a minute from 09:15 IST to 15:29; the first bar opens and closes at the
prices each test gives and every later bar opens and closes at one price. Daily: one bar
on each of 1, 2 and 3 July 2025. A timer at 09:15 buys at the first open of the run, so
the first slice holds the position from its open to its close.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

import openalgo.backtest as bt

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
DAY = dt.date(2025, 7, 1)
SESSION_MINUTES = 375


def at(hour: int, minute: int, day: dt.date = DAY) -> dt.datetime:
    return dt.datetime(day.year, day.month, day.day, hour, minute, tzinfo=IST)


def bars(stamps: list[dt.datetime], opens: list[float], closes: list[float]) -> pd.DataFrame:
    """Bars with the given opens and closes, a rupee of range around both."""
    first, last = np.asarray(opens), np.asarray(closes)
    columns = {
        "open": first,
        "high": np.maximum(first, last) + 0.5,
        "low": np.minimum(first, last) - 0.5,
        "close": last,
        "volume": np.full(len(first), 1_000.0),
    }
    return pd.DataFrame(columns, index=pd.DatetimeIndex(stamps, name="timestamp"))


def session(first_open: float, first_close: float, later: float) -> pd.DataFrame:
    """The intraday session of the module docstring."""
    stamps = [at(9, 15) + dt.timedelta(minutes=k) for k in range(SESSION_MINUTES)]
    rest = [later] * (SESSION_MINUTES - 1)
    return bars(stamps, [first_open, *rest], [first_close, *rest])


class BuyHundredAtTheFirstOpen(bt.Strategy):
    """Buys 100 shares at the first open of the run and holds them."""

    symbol, exchange, interval = "SBIN", "NSE", "15m"
    product = "CNC"

    def init(self):
        self.at("09:15", self.enter)

    def enter(self):
        if self.position.is_flat:
            self.buy(qty=100)


class DailyBuyThousandAtTheFirstOpen(BuyHundredAtTheFirstOpen):
    """Buys 1,000 shares at the open of the first session and holds them."""

    interval = "D"

    def enter(self):
        if self.position.is_flat:
            self.buy(qty=1_000)


def run(strategy: type, data: pd.DataFrame, capital: float) -> bt.Result:
    return bt.backtest(strategy, data=data, capital=capital, costs=None, slippage=0)


def test_a_one_session_intraday_run_that_gains_one_percent():
    # 100 shares bought at 100.00, marked at 105.00 by the first slice and 110.00 after.
    result = run(BuyHundredAtTheFirstOpen, session(100.0, 105.0, 110.0), capital=100_000)
    m = result.metrics
    # 100 x 10.00 = 1,000 rupees on 100,000.
    assert m["total_return"] == pytest.approx(0.01)
    assert (m["start_equity"], m["net_profit"]) == (100_000.0, pytest.approx(1_000.0))
    returns = result.performance.session_returns
    assert list(returns.index.date) == [DAY]
    assert returns.to_numpy() == pytest.approx([0.01])


def test_a_run_whose_only_pnl_is_in_the_first_slice_reports_it():
    # Bought at 100.00 and marked at 101.00 from the close of the first slice on.
    result = run(BuyHundredAtTheFirstOpen, session(100.0, 101.0, 101.0), capital=1_000_000)
    m = result.metrics
    assert m["net_profit"] == pytest.approx(100.0)
    assert m["total_return"] == pytest.approx(100 / 1_000_000)


def test_a_fall_below_the_capital_in_the_first_slice_is_a_drawdown_from_it():
    # Bought at 100.00, marked at 95.00 by the first slice, back at 100.00 after it.
    result = run(BuyHundredAtTheFirstOpen, session(100.0, 95.0, 100.0), capital=1_000_000)
    m = result.metrics
    # 999,500 / 1,000,000 - 1, from the capital at the start (the 09:15 open of the
    # first slice) to the 09:17 recovery. A fall is negative, in rupees too.
    assert m["max_drawdown"] == pytest.approx(-500 / 1_000_000)
    assert m["max_drawdown_amount"] == pytest.approx(-500.0)
    assert m["max_drawdown_peak"] == m["start"] == at(9, 15)
    assert m["max_drawdown_recovery"] == at(9, 17)
    assert m["max_drawdown_days"] == pytest.approx(2 / (24 * 60))


def test_an_intraday_run_starts_at_the_open_of_its_first_slice():
    result = run(BuyHundredAtTheFirstOpen, session(100.0, 105.0, 110.0), capital=100_000)
    m = result.metrics
    # The first one-minute slice opens at 09:15; the last closes at 15:30.
    assert m["start"] == at(9, 15)
    assert m["elapsed_days"] == pytest.approx(375 / (24 * 60))


def test_result_returns_measure_the_first_slice_from_the_capital():
    result = run(BuyHundredAtTheFirstOpen, session(100.0, 105.0, 110.0), capital=100_000)
    returns = result.returns
    # 100 shares marked from 100.00 to 105.00 in the first slice (500 rupees on the
    # 100,000 capital), then to 110.00 in the second (500 more on 100,500).
    assert returns.iloc[0] == pytest.approx(500 / 100_000)
    assert returns.iloc[1] == pytest.approx(500 / 100_500)
    assert (returns.iloc[2:] == 0.0).all()


def test_a_daily_run_starts_at_the_open_of_its_first_session():
    days = [DAY, dt.date(2025, 7, 2), dt.date(2025, 7, 3)]
    data = bars([at(9, 15, day) for day in days], [100.0, 102.0, 99.96], [102.0, 99.96, 104.95])
    result = run(DailyBuyThousandAtTheFirstOpen, data, capital=200_000)
    m = result.metrics
    # The daily slice is the whole NSE session, from its 09:15 open to 15:30.
    assert m["start"] == at(9, 15)
    assert m["elapsed_days"] == pytest.approx(2 + 6.25 / 24)
    expected = [202_000 / 200_000 - 1, 199_960 / 202_000 - 1, 204_950 / 199_960 - 1]
    assert result.returns.to_numpy() == pytest.approx(expected)


def test_a_three_session_daily_run_has_the_return_of_each_session():
    days = [DAY, dt.date(2025, 7, 2), dt.date(2025, 7, 3)]
    data = bars([at(9, 15, day) for day in days], [100.0, 102.0, 99.96], [102.0, 99.96, 104.95])
    result = run(DailyBuyThousandAtTheFirstOpen, data, capital=200_000)
    # 1,000 shares bought at 100.00 leave 100,000 in cash: equity 202,000, 199,960 and
    # 204,950 at the three closes, on 200,000 of capital.
    expected = [202_000 / 200_000 - 1, 199_960 / 202_000 - 1, 204_950 / 199_960 - 1]
    returns = result.performance.session_returns
    assert list(returns.index.date) == days
    assert returns.to_numpy() == pytest.approx(expected)
    assert result.metrics["total_return"] == pytest.approx(204_950 / 200_000 - 1)
    assert result.performance.monthly_returns.loc[2025, 7] == pytest.approx(204_950 / 200_000 - 1)
