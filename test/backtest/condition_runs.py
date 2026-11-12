"""Shared inputs of the condition-strategy tests: 5-minute SBIN bars on the built-in NSE
calendar, the run settings, the two forms of one condition strategy, and the exact
comparison of two results table by table.

`walk_bars` builds a session from tick moves: bar k opens at the previous close and
closes `moves[k]` ticks (0.05 rupees) away, with a high and low half a rupee outside.

`conditions` declares four condition arrays with entry_when, exit_when, short_when and
cover_when (run by the Rust signal driver); `literal_on_bar` is the `on_bar` of
14-strategy-api 4.1 written out for the same arrays (run by the Python event driver).
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
# What the strategies of these tests trade unless a test names another instrument.
SBIN_5M = {"symbol": "SBIN", "exchange": "NSE", "interval": "5m"}
# The four conditions, in the order of 14-strategy-api 4.1.
CONDITION_NAMES = ("entry", "exit", "short", "cover")


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


def literal_on_bar(
    arrays: dict, pyramiding: int, order_args: dict, attributes: dict = SBIN_5M
) -> type:
    """The `on_bar` of 14-strategy-api 4.1 written out for `arrays` (one boolean per
    strategy bar under each of CONDITION_NAMES). A strategy bar with no base bar is not
    delivered, so the bar's position in the arrays is the number of bars seen."""
    entry, exit_, short, cover = (arrays[name] for name in CONDITION_NAMES)

    class Literal(bt.Strategy):
        def on_bar(self, bar):
            i = len(self.close) - 1
            position = self.position
            flat = position.is_flat and not self._ctx.has_pending_entry(self.symbol)
            if flat:
                if entry[i]:
                    self.buy(**order_args)
                elif short[i]:
                    self.sell(**order_args)
            elif position.is_long:
                if exit_[i]:
                    self.exit()
                elif entry[i] and position.adds < pyramiding:
                    self.buy(**order_args)
            elif position.is_short:
                if cover[i]:
                    self.exit()
                elif short[i] and position.adds < pyramiding:
                    self.sell(**order_args)

    for name, value in {**attributes, "pyramiding": pyramiding}.items():
        setattr(Literal, name, value)
    return Literal


def conditions(arrays: dict, pyramiding: int, order_args: dict, attributes: dict = SBIN_5M) -> type:
    """The same strategy declared with entry_when, exit_when, short_when, cover_when."""

    class Conditions(bt.Strategy):
        def init(self):
            self.entry_when(arrays["entry"], **order_args)
            self.exit_when(arrays["exit"])
            self.short_when(arrays["short"], **order_args)
            self.cover_when(arrays["cover"])

    for name, value in {**attributes, "pyramiding": pyramiding}.items():
        setattr(Conditions, name, value)
    return Conditions


def assert_same_result(first: Any, second: Any) -> None:
    """Every table of two results is equal bit for bit (fills, orders, trades,
    positions, equity, events, stale marks, costs, contract notes, orders a sizer
    skipped), and so is the gross P&L: floats are compared exactly, not within a
    tolerance."""
    tables = first.to_dataframes()
    for name, table in second.to_dataframes().items():
        pd.testing.assert_frame_equal(tables[name], table, obj=name, check_exact=True)
    pd.testing.assert_frame_equal(first.skipped, second.skipped, obj="skipped", check_exact=True)
    assert first.gross_pnl == second.gross_pnl


__all__ = [
    "BARS_PER_SESSION",
    "CAPITAL",
    "CONDITION_NAMES",
    "IST",
    "SBIN_5M",
    "SESSIONS",
    "assert_same_result",
    "bar_open",
    "bars_from_closes",
    "conditions",
    "literal_on_bar",
    "run",
    "walk_bars",
]
