"""Runs and trade lists shared by the robustness tests.

Trade lists: `trade_performance(pnl)` is the `Performance` of a hand-built trade list on
NSE sessions from 2025-01-01: trade k is entered at 09:30 IST and exits at 15:00 IST on
session `sessions_apart * k`, with net P&L `pnl[k]` (no charges), and the equity at
each session close is the capital plus the P&L closed so far. So trade k's entry session
is session `sessions_apart * k`, and the exit order is the list's order. `check_of` runs
one check on such a list as a report runs it.

Bars: `bars(steps)` is one NSE daily bar per session from 2023-01-02, opening at 500
plus the running sum of `steps`, with the high and low 1 rupee either side of the open
and the close at the open. `Flip` buys 10 shares when flat and exits when long, deciding
at each bar's close and filling at the next open, so its trade k is bought at the open
of bar 2k + 1 and sold at the open of bar 2k + 2: its net P&L is 10 x steps[2k + 1].
`steps_for(trade_moves)` builds steps whose traded moves are `trade_moves`, repeated,
with the move between two trades taking back the traded one so the price stays near 500.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd

import openalgo.backtest as bt
from openalgo.backtest import metrics
from openalgo.backtest._robust._assess import run_check
from openalgo.backtest._robust._registry import BY_NAME
from openalgo.backtest._robust._request import build_request
from openalgo.backtest._robust._trades import trade_run

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
CAPITAL = 100_000.0
QUANTITY = 10
FIRST_PRICE = 500.0


def nse_sessions(start: str, count: int) -> list[dt.date]:
    """The first `count` NSE sessions from `start` of the built-in calendar."""
    end = (pd.Timestamp(start) + pd.Timedelta(days=2 * count + 30)).strftime("%Y-%m-%d")
    days = list(bt.calendar.default_calendar().sessions("NSE", start, end)["date"])
    return days[:count]


def _at(day: dt.date, hour: int, minute: int) -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.combine(day, dt.time(hour, minute), tzinfo=IST))


def trade_performance(
    pnl: list[float] | np.ndarray,
    *,
    capital: float = CAPITAL,
    sessions_apart: int = 1,
    extra_sessions: int = 0,
) -> metrics.Performance:
    """The Performance of the hand-built trade list of the module docstring, with
    `extra_sessions` sessions without a trade after the last one."""
    pnl = np.asarray(pnl, dtype=float)
    count = sessions_apart * len(pnl) + extra_sessions
    days = nse_sessions("2025-01-01", count)
    traded = [days[sessions_apart * k] for k in range(len(pnl))]
    trades = pd.DataFrame(
        {
            "entry_time": [_at(day, 9, 30) for day in traded],
            "exit_time": [_at(day, 15, 0) for day in traded],
            "entry_price": 100.0,
            "exit_price": 100.0,
            "quantity": 1.0,
            "side": "BUY",
            "gross_pnl": pnl,
            "costs": 0.0,
            "net_pnl": pnl,
        }
    )
    closed = np.zeros(count)
    for k, value in enumerate(pnl):
        closed[sessions_apart * k] = value
    equity = pd.Series(
        capital + np.cumsum(closed), index=pd.DatetimeIndex([_at(day, 15, 30) for day in days])
    )
    return metrics.analyse(equity, exchanges="NSE", trades=trades, opening_wealth=capital)


def check_of(
    performance: metrics.Performance,
    name: str,
    *,
    runs: int = 1_000,
    seed: int = 0,
    objective: Any = "sharpe",
    **settings: Any,
):
    """The `CheckResult` of check `name` on `performance`, with `settings` as its dict:
    the check as a report runs it, on a hand-built trade list with no strategy."""
    request = build_request(
        checks=[name],
        objective=objective,
        runs=runs,
        seed=seed,
        workers=None,
        others=None,
        check_settings={name: settings},
        ranges={},
    )
    return run_check(BY_NAME[name], trade_run(performance), request)


def steps_for(trade_moves: list[float], trades: int) -> np.ndarray:
    """Open-to-open steps whose traded moves (module docstring) are `trade_moves`
    repeated over `trades` trades."""
    moves = np.resize(np.asarray(trade_moves, dtype=float), trades)
    steps = np.zeros(2 * trades + 2)
    steps[1::2][:trades] = moves
    steps[2::2][:trades] = -moves
    return steps


def bars(steps: np.ndarray) -> pd.DataFrame:
    """The daily bars of the module docstring, one per step."""
    days = nse_sessions("2023-01-02", len(steps))
    opens = FIRST_PRICE + np.concatenate(([0.0], np.cumsum(steps)[:-1]))
    index = pd.DatetimeIndex(
        [dt.datetime.combine(day, dt.time(9, 15), tzinfo=IST) for day in days], name="timestamp"
    )
    columns = {
        "open": opens,
        "high": opens + 1.0,
        "low": opens - 1.0,
        "close": opens,
        "volume": np.full(len(steps), 1_000.0),
    }
    return pd.DataFrame(columns, index=index)


class Flip(bt.Strategy):
    """Buys 10 shares when flat and exits when long (module docstring)."""

    symbol, exchange, interval = "SBIN", "NSE", "D"

    def on_bar(self, bar):
        if self.position.is_flat:
            self.buy(qty=QUANTITY)
        else:
            self.exit()


#: Run arguments with no charges and no slippage, so a trade's net P&L is its move.
FREE = {"capital": CAPITAL, "costs": None, "slippage": 0}
#: Traded moves of a planted edge: 7 of 10 win, the mean is 1.6 rupees a share.
EDGE_MOVES = [3.0, 2.0, 4.0, -1.0, 3.0, 2.0, -2.0, 5.0, 1.0, -1.0]
#: Traded moves of a strategy fitted to noise: wins and losses of 2 rupees, alternating,
#: so the trades sum to zero.
NOISE_MOVES = [2.0, -2.0]


def edge_run(trades: int = 60, **settings) -> bt.Result:
    """`Flip` on bars with a planted edge."""
    return bt.backtest(Flip, data=bars(steps_for(EDGE_MOVES, trades)), **FREE, **settings)


def noise_run(trades: int = 60, **settings) -> bt.Result:
    """`Flip` on bars whose traded moves are noise."""
    return bt.backtest(Flip, data=bars(steps_for(NOISE_MOVES, trades)), **FREE, **settings)
