"""Bars and strategies shared by the strategy-surface tests.

The strategies live in this module, not in a test file, because `bt.optimize` runs them
in spawned worker processes, which import them by module and name.

Bars: one NSE session on 2025-07-01 from 09:15 IST, 1-minute bars. In `minute_bars` the
bar opening k minutes after 09:15 opens at 500.00 + 0.10 k, its high is the open plus
0.50, its low the open minus 0.50 and its close the open plus 0.05, so prices can be
read off the inputs. `wave_bars` adds a sine of amplitude 3 rupees and a period of 63
minutes, so moving averages cross several times a day.
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

import openalgo.backtest as bt
from openalgo import ta

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
DAY = dt.date(2025, 7, 1)
SESSION_MINUTES = 375
CAPITAL = 1_000_000


def at(hour: int, minute: int, day: dt.date = DAY) -> dt.datetime:
    """An IST time on the test day."""
    return dt.datetime(day.year, day.month, day.day, hour, minute, tzinfo=IST)


def _frame(opens: np.ndarray, first: dt.datetime, step: dt.timedelta) -> pd.DataFrame:
    index = pd.DatetimeIndex([first + step * k for k in range(len(opens))], name="timestamp")
    columns = {
        "open": opens,
        "high": opens + 0.5,
        "low": opens - 0.5,
        "close": opens + 0.05,
        "volume": np.full(len(opens), 1_000.0),
    }
    return pd.DataFrame(columns, index=index)


def minute_bars(count: int = SESSION_MINUTES, skip: tuple[int, ...] = ()) -> pd.DataFrame:
    """Rising 1-minute bars from 09:15, without the minutes in `skip`."""
    frame = _frame(500.0 + 0.10 * np.arange(count), at(9, 15), dt.timedelta(minutes=1))
    return frame.drop(frame.index[list(skip)])


def wave_bars(count: int = SESSION_MINUTES) -> pd.DataFrame:
    """1-minute bars that rise slowly and swing by 3 rupees."""
    k = np.arange(count)
    opens = np.round((500.0 + 0.02 * k + 3.0 * np.sin(2 * np.pi * k / 63)) / 0.05) * 0.05
    return _frame(opens, at(9, 15), dt.timedelta(minutes=1))


def daily_bars(days: list[dt.date]) -> pd.DataFrame:
    """One bar per session date, stamped 09:15 IST, opening at 100 + its number."""
    frame = _frame(100.0 + np.arange(len(days)), at(9, 15), dt.timedelta(days=1))
    frame.index = pd.DatetimeIndex([at(9, 15, day) for day in days], name="timestamp")
    return frame


def run(strategy: type, data: pd.DataFrame | None = None, **settings) -> bt.Result:
    """A backtest on own data with no charges and no slippage."""
    arguments = {
        "data": minute_bars() if data is None else data,
        "capital": CAPITAL,
        "costs": None,
        "slippage": 0,
        **settings,
    }
    return bt.backtest(strategy, **arguments)


class EmaCross(bt.Strategy):
    """Long while the fast EMA is above the slow one, on 5-minute bars."""

    symbol, exchange, interval = "SBIN", "NSE", "5m"
    product = "CNC"
    fast = bt.param(3, (3, 5, 8))
    slow = bt.param(10, (10, 15, 20))

    def init(self):
        self.fast_line = self.I(ta.ema, self.close, self.fast)
        self.slow_line = self.I(ta.ema, self.close, self.slow)

    def on_bar(self, bar):
        if self.position.is_flat and self.fast_line.crossed_above(self.slow_line):
            self.buy(qty=10)
        elif self.position.is_long and self.fast_line.crossed_below(self.slow_line):
            self.exit()


class EmaCrossConditions(bt.Strategy):
    """`EmaCross` written as conditions (14-strategy-api 4.1): the same decisions, made
    in Rust."""

    symbol, exchange, interval = "SBIN", "NSE", "5m"
    product = "CNC"
    fast = bt.param(3, (3, 5, 8))
    slow = bt.param(10, (10, 15, 20))

    def init(self):
        fast_line = self.I(ta.ema, self.close, self.fast)
        slow_line = self.I(ta.ema, self.close, self.slow)
        self.entry_when(fast_line.crossed_above(slow_line), qty=10)
        self.exit_when(fast_line.crossed_below(slow_line))


class EmaCrossIntraday(EmaCross):
    """EmaCross on MIS (intraday product)."""

    product = "MIS"


def final_equity(result: bt.Result) -> float:
    """A custom objective: the last equity point, rupees."""
    return float(result.equity["equity_rupees"].iloc[-1])


def process_id(result: bt.Result) -> float:
    """An objective that says which process ran the combination."""
    return float(os.getpid())


__all__ = [
    "CAPITAL",
    "DAY",
    "IST",
    "EmaCross",
    "EmaCrossConditions",
    "EmaCrossIntraday",
    "at",
    "daily_bars",
    "final_equity",
    "minute_bars",
    "process_id",
    "run",
    "wave_bars",
]
