"""Shared inputs of the condition-strategy tests: 5-minute SBIN bars on the built-in NSE
calendar, the run settings, and the comparison of two results table by table.

`walk_bars` builds a session from tick moves: bar k opens at the previous close and
closes `moves[k]` ticks (0.05 rupees) away, with a high and low half a rupee outside.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd

import openalgo.backtest as bt

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
TICK = 0.05
BAR_MINUTES = 5
# 09:15 to 15:30 IST in 5-minute bars.
BARS_PER_SESSION = 75
SESSIONS = (dt.date(2025, 7, 1), dt.date(2025, 7, 2))
CAPITAL = 1_000_000


def bar_open(day: dt.date, k: int) -> dt.datetime:
    """Open of 5-minute bar k of a session (k = 0 opens at 09:15 IST)."""
    start = dt.datetime(day.year, day.month, day.day, 9, 15, tzinfo=IST)
    return start + dt.timedelta(minutes=BAR_MINUTES * k)


def bars_from_closes(opens: list[float], closes: list[float], times: list) -> pd.DataFrame:
    open_, close = np.array(opens), np.array(closes)
    columns = {
        "open": open_,
        "high": np.maximum(open_, close) + 0.5,
        "low": np.minimum(open_, close) - 0.5,
        "close": close,
        "volume": np.full(len(times), 1_000.0),
    }
    return pd.DataFrame(columns, index=pd.DatetimeIndex(times, name="timestamp"))


def walk_bars(moves: list[int], missing: set[int] = frozenset(), sessions=SESSIONS) -> pd.DataFrame:
    """Bars of whole sessions from tick moves (one per bar, cycled); the bars whose
    position is in `missing` are left out (SPEC-07)."""
    times, opens, closes = [], [], []
    price = 500.0
    count = len(sessions) * BARS_PER_SESSION
    for position in range(count):
        day, k = sessions[position // BARS_PER_SESSION], position % BARS_PER_SESSION
        close = round((price + moves[position % len(moves)] * TICK) / TICK) * TICK
        if position not in missing:
            times.append(bar_open(day, k))
            opens.append(round(price, 2))
            closes.append(round(close, 2))
        price = close
    return bars_from_closes(opens, closes, times)


def run(strategy: type, data: pd.DataFrame, **settings: Any) -> Any:
    """`bt.backtest` with no charges and no slippage over the data's sessions."""
    arguments = {"data": data, "capital": CAPITAL, "costs": None, "slippage": 0, **settings}
    return bt.backtest(strategy, **arguments)


def assert_same_result(first: Any, second: Any) -> None:
    """Every table of two results is equal (fills, orders, trades, positions, equity,
    events, stale marks, costs, contract notes), and so is the gross P&L."""
    tables = first.to_dataframes()
    for name, table in second.to_dataframes().items():
        pd.testing.assert_frame_equal(tables[name], table, obj=name)
    assert first.gross_pnl == second.gross_pnl


__all__ = [
    "BARS_PER_SESSION",
    "CAPITAL",
    "IST",
    "SESSIONS",
    "assert_same_result",
    "bar_open",
    "bars_from_closes",
    "run",
    "walk_bars",
]
