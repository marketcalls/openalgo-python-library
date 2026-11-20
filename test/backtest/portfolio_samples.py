"""Bars and strategies of the portfolio-run tests (`bt.backtest` given a list).

Bars: NSE sessions of July 2025 from 09:15 IST, built in the process (nothing contacts a
server). `flat_bars(price)` is one session of 1-minute bars whose open, high, low and close
are all `price`; `stepped_bars(before, after, minute)` is priced `before` up to the bar
opening `minute` minutes after 09:15 and `after` from it on; `daily_bars` (from
`strategy_samples`) is one bar per session opening at 100 + its number.

Strategies act on a script, so their orders can be worked out by hand:

- `Scripted`: a 1-minute SBIN strategy that, flat and with no working order, places a
  MARKET order of `qty` units on `side` at its strategy-bar closes after skipping the first
  `wait` of them, at most `entries` times. Decided at the close of the bar opening at
  09:15 + k, it fills at the open of the next bar (09:16 + k).
- `DailyScript`: a daily SBIN strategy (CNC, the daily default) that buys `qty` units at the
  close of its bar number `buy_on` (1 is the first) and exits at the close of `sell_on` (0
  never); each fills at the next session's open.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from strategy_samples import DAY, SESSION_MINUTES, at

import openalgo.backtest as bt

#: The sessions of the daily runs: Tuesday 1 to Friday 4 July 2025.
DAYS = [dt.date(2025, 7, day) for day in (1, 2, 3, 4)]


def flat_bars(price: float, count: int = SESSION_MINUTES, day: dt.date = DAY) -> pd.DataFrame:
    """1-minute bars from 09:15 IST on `day` whose open, high, low and close are `price`."""
    return stepped_bars(price, price, count, count, day)


def stepped_bars(
    before: float, after: float, minute: int, count: int = SESSION_MINUTES, day: dt.date = DAY
) -> pd.DataFrame:
    """1-minute bars from 09:15 IST on `day`, every price `before` up to the bar opening
    `minute` minutes after 09:15 and `after` from that bar on."""
    prices = np.where(np.arange(count) < minute, before, after).astype(float)
    index = pd.DatetimeIndex(
        [at(9, 15, day) + dt.timedelta(minutes=k) for k in range(count)], name="timestamp"
    )
    columns = {"open": prices, "high": prices, "low": prices, "close": prices}
    return pd.DataFrame({**columns, "volume": np.full(count, 1_000.0)}, index=index)


class Scripted(bt.Strategy):
    """See the module docstring."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"
    qty = 100
    side = "BUY"
    wait = 0
    entries = 1

    def on_bar(self, bar):
        seen = self.state.get("bars", 0)
        self.state["bars"] = seen + 1
        placed = self.state.get("entries", 0)
        if seen < self.wait or placed >= self.entries:
            return
        if self.position.is_flat and not self.orders:
            self.state["entries"] = placed + 1
            order = self.buy if self.side == "BUY" else self.sell
            order(qty=self.qty)


class DailyScript(bt.Strategy):
    """See the module docstring."""

    symbol, exchange, interval = "SBIN", "NSE", "D"
    qty = 10
    buy_on = 1
    sell_on = 0

    def on_bar(self, bar):
        number = self.state.get("bars", 0) + 1
        self.state["bars"] = number
        if number == self.buy_on:
            self.buy(qty=self.qty)
        elif number == self.sell_on:
            self.exit()


def no_costs(**settings) -> dict:
    """Run arguments with no charges and no slippage, updated by `settings`."""
    return {"costs": None, "slippage": 0, **settings}


__all__ = ["DAYS", "DailyScript", "Scripted", "flat_bars", "no_costs", "stepped_bars"]
