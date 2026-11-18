"""Runs and trade lists shared by the robustness tests.

Trade lists: `trade_performance(pnl)` is the `Performance` of a hand-built trade list on
NSE sessions from 2025-01-01: trade k is the j-th of session `sessions_apart * (k //
per_session)`, j = k % per_session, entered at 09:30 IST plus j minutes and exiting at
15:00 IST less `per_session - 1 - j` minutes, with net P&L `pnl[k]` (no charges), and
the equity at each session close is the capital plus the P&L closed so far. With one
trade a session (the default) trade k is entered on session `sessions_apart * k`, and the
exit order is always the list's order. `check_of` runs one check on such a list as a
report runs it, and `blocks_of` gives what its section draws.

Bars: `bars(steps)` is one NSE daily bar per session from 2023-01-02, opening at 500
plus the running sum of `steps`, with the high and low 1 rupee either side of the open
and the close at the open. `Flip` buys 10 shares when flat and exits when long, deciding
at each bar's close and filling at the next open, so its trade k is bought at the open
of bar 2k + 1 and sold at the open of bar 2k + 2: its net P&L is 10 x steps[2k + 1]. It
buys only with no working order, which on time it never has at a decision, so that with
its entries held (the delayed check) it enters late rather than twice.
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
from openalgo.backtest._robust._assess import Checked, run_check
from openalgo.backtest._robust._registry import BY_NAME
from openalgo.backtest._robust._request import build_request
from openalgo.backtest._robust._trades import trade_run
from openalgo.backtest._robust._verdict import CheckResult

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
    per_session: int = 1,
) -> metrics.Performance:
    """The Performance of the hand-built trade list of the module docstring, with
    `extra_sessions` sessions without a trade after the last one."""
    pnl = np.asarray(pnl, dtype=float)
    session_of = sessions_apart * (np.arange(len(pnl)) // per_session)
    within = np.arange(len(pnl)) % per_session
    count = sessions_apart * int(np.ceil(len(pnl) / per_session)) + extra_sessions
    days = nse_sessions("2025-01-01", count)
    traded = [days[k] for k in session_of]
    minute = pd.Timedelta(minutes=1)
    trades = pd.DataFrame(
        {
            "entry_time": [
                _at(day, 9, 30) + j * minute for day, j in zip(traded, within, strict=True)
            ],
            "exit_time": [
                _at(day, 15, 0) - (per_session - 1 - j) * minute
                for day, j in zip(traded, within, strict=True)
            ],
            "entry_price": 100.0,
            "exit_price": 100.0,
            "quantity": 1.0,
            "side": "BUY",
            "gross_pnl": pnl,
            "costs": 0.0,
            "net_pnl": pnl,
        }
    )
    closed = np.bincount(session_of, weights=pnl, minlength=count)
    equity = pd.Series(
        capital + np.cumsum(closed), index=pd.DatetimeIndex([_at(day, 15, 30) for day in days])
    )
    return metrics.analyse(equity, exchanges="NSE", trades=trades, opening_wealth=capital)


def checked(
    performance: metrics.Performance,
    name: str,
    *,
    runs: int = 1_000,
    seed: int = 0,
    objective: Any = "sharpe",
    **settings: Any,
) -> Checked:
    """Check `name` on `performance`, with `settings` as its dict, as a report runs it
    on a hand-built trade list with no strategy: its result and what its section
    draws."""
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


def check_of(performance: metrics.Performance, name: str, **options: Any) -> CheckResult:
    """The `CheckResult` of `checked`."""
    return checked(performance, name, **options).result


def blocks_of(performance: metrics.Performance, name: str, **options: Any) -> tuple:
    """What the section of `checked` draws: rows of blocks."""
    return checked(performance, name, **options).blocks


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
    return opens_bars(FIRST_PRICE + np.concatenate(([0.0], np.cumsum(steps)[:-1])))


def opens_bars(opens: list[float] | np.ndarray) -> pd.DataFrame:
    """Daily bars as `bars` makes them (one per NSE session from 2023-01-02, the high and
    low 1 rupee either side of the open, the close at the open), from their opens."""
    opens = np.asarray(opens, dtype=float)
    days = nse_sessions("2023-01-02", len(opens))
    index = pd.DatetimeIndex(
        [dt.datetime.combine(day, dt.time(9, 15), tzinfo=IST) for day in days], name="timestamp"
    )
    columns = {
        "open": opens,
        "high": opens + 1.0,
        "low": opens - 1.0,
        "close": opens,
        "volume": np.full(len(opens), 1_000.0),
    }
    return pd.DataFrame(columns, index=index)


class Flip(bt.Strategy):
    """Buys 10 shares when flat with no working order and exits when long (module
    docstring)."""

    symbol, exchange, interval = "SBIN", "NSE", "D"

    def on_bar(self, bar):
        if self.position.is_flat:
            if not self.orders:
                self.buy(qty=QUANTITY)
        else:
            self.exit()


#: Run arguments with no charges and no slippage, so a trade's net P&L is its move.
FREE = {"capital": CAPITAL, "costs": None, "slippage": 0}
#: Traded moves of a planted edge: 7 of 10 win, the mean is 1.6 rupees a share.
EDGE_MOVES = [3.0, 2.0, 4.0, -1.0, 3.0, 2.0, -2.0, 5.0, 1.0, -1.0]
#: Traded moves of a strategy with no edge: wins and losses of 2 rupees, alternating, so
#: the trades sum to zero.
NO_EDGE_MOVES = [2.0, -2.0]
#: Traded moves of a small edge buried in noise: wins of 2.5 rupees and losses of 2,
#: alternating, so a trade makes 0.25 rupees a share on average against a spread of 2.25.
SMALL_EDGE_MOVES = [2.5, -2.0]


def edge_run(trades: int = 60, **settings) -> bt.Result:
    """`Flip` on bars with a planted edge."""
    return bt.backtest(Flip, data=bars(steps_for(EDGE_MOVES, trades)), **FREE, **settings)


def no_edge_run(trades: int = 60, **settings) -> bt.Result:
    """`Flip` on bars whose traded moves sum to zero."""
    return bt.backtest(Flip, data=bars(steps_for(NO_EDGE_MOVES, trades)), **FREE, **settings)


def small_edge_run(trades: int = 60, **settings) -> bt.Result:
    """`Flip` on bars whose traded moves hold a small edge in much noise."""
    data = bars(steps_for(SMALL_EDGE_MOVES, trades))
    return bt.backtest(Flip, data=data, **FREE, **settings)


class OddCloseFlip(bt.Strategy):
    """Buys 10 shares when flat and the bar closed on an odd number of 5-paisa ticks (a
    close ending in 5 paisa), and exits the bar after: a rule fitted to the last digit of
    the price, which means nothing."""

    symbol, exchange, interval = "SBIN", "NSE", "D"

    def on_bar(self, bar):
        if self.position.is_flat and round(bar.close / 0.05) % 2 == 1:
            self.buy(qty=QUANTITY)
        elif self.position.is_long:
            self.exit()


#: Opens of a block of three daily bars that plants a gain on a close ending in 5 paisa:
#: the first closes at 500.05 (odd ticks, so OddCloseFlip buys at the next open, 500.10),
#: and the third opens at 503.10, where it sells: 3 rupees a share. The next block opens
#: at 500.05 again.
ODD_CLOSE_BLOCK = [500.05, 500.10, 503.10]


def odd_close_run(blocks: int = 40, **settings) -> bt.Result:
    """`OddCloseFlip` on `blocks` blocks of `ODD_CLOSE_BLOCK`: one trade of +30 rupees
    each."""
    data = opens_bars(ODD_CLOSE_BLOCK * blocks)
    return bt.backtest(OddCloseFlip, data=data, **FREE, **settings)


def rising_run(trades: int = 60, **settings) -> bt.Result:
    """`Flip` on bars that rise 1 rupee every session: every trade makes 10 rupees, from
    the drift alone."""
    return bt.backtest(Flip, data=bars(np.ones(2 * trades + 2)), **FREE, **settings)


class Weekday(bt.Strategy):
    """Buys 10 shares at the close of a session on weekday `day` (0 Monday to 4 Friday)
    when flat with no working order, and exits at the next session's close: its trade of
    session d buys at the open of session d + 1 and sells at the open of session d + 2.
    It decides by the date, so a run of any window of the sessions trades the same
    sessions."""

    symbol, exchange, interval = "SBIN", "NSE", "D"
    day = 0

    def on_bar(self, bar):
        if self.position.is_flat:
            if bar.time.weekday() == self.day and not self.orders:
                self.buy(qty=QUANTITY)
        else:
            self.exit()


#: The move a weekday's trade makes when its weekday is favoured, and when it is not,
#: rupees a share.
FAVOURED_MOVE, OTHER_MOVE = 3.0, -1.0


def weekday_bars(count: int, favoured) -> pd.DataFrame:
    """`count` daily bars (`opens_bars`) whose move from the open of session s - 1 to the
    open of session s is `FAVOURED_MOVE` when session s - 2 falls on one of the weekdays
    `favoured(s)` gives, else `OTHER_MOVE`: a `Weekday` trade of session d makes 10 x
    the move into session d + 2."""
    days = nse_sessions("2023-01-02", count)
    steps = np.zeros(count)
    for s in range(2, count):
        steps[s] = FAVOURED_MOVE if days[s - 2].weekday() in favoured(s) else OTHER_MOVE
    return opens_bars(FIRST_PRICE + np.cumsum(steps))


def weekday_trades(count: int, day: int, first: int = 0, last: int | None = None) -> list[int]:
    """The sessions, positions from `first` to `last` of the first `count` NSE sessions
    from 2023-01-02, at whose close `Weekday` with `day` buys in a run of those sessions:
    each one on that weekday that leaves two sessions after it in the run (to buy and to
    sell), as long as the trade before it has sold."""
    days = nse_sessions("2023-01-02", count)
    last = count - 1 if last is None else last
    bought, free_from = [], first
    for d in range(first, last - 1):
        if d >= free_from and days[d].weekday() == day:
            bought.append(d)
            free_from = d + 2
    return bought


#: Settings that keep the bar checks of a full report small in a test: one process, 20
#: runs of each check and 10 candidates of each stress generator.
SMALL_BARS = {
    "workers": 1,
    "vs_random": {"runs": 20},
    "permutation": {"runs": 20},
    "noise": {"runs": 20},
    "stress": {"noise_candidates": 10, "permutation_candidates": 10},
}
