"""Runs shared by the benchmark tests (07-research-and-reporting 2.4; `_benchmark`).

The strategies live in this module, not in a test file, because `bt.optimize` runs them
in spawned worker processes, which import them by module and name.

The share: SBIN on NSE, daily bars on the five sessions 2026-11-02 to 2026-11-06. It
closes at 1,000, 1,010, 1,030, 1,020 and 1,040, and each session opens at the previous
close (the first at 1,000). `Hold` buys `qty` shares (100) at the first close, filled at
the second session's open (1,000.00), and holds them; with no charges and no slippage
the equity at the session closes is 1,00,000, then 1,00,000 + qty x (close - 1,000).

The index: NIFTY on NSE_INDEX closes at 20,000 on 2026-10-30, the session before the
run, then moves by BENCH_RETURNS on the five sessions of the run.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd
from data_fixtures.fakes import DAY_SECONDS, FakeClient

import openalgo.backtest as bt

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
OPENING_DAY = dt.date(2026, 10, 30)
DAYS = [dt.date(2026, 11, day) for day in (2, 3, 4, 5, 6)]
CLOSES = [1_000.0, 1_010.0, 1_030.0, 1_020.0, 1_040.0]
OPENS = [1_000.0, *CLOSES[:-1]]
ENTRY_PRICE = 1_000.0
CAPITAL = 100_000
QTY = 100
QTY_RANGE = (25, 50, 100)
OPENING_LEVEL = 20_000.0
BENCH_RETURNS = [0.01, 0.02, -0.01, 0.005, 0.01]
# The present of every run: after the last session has closed.
NOW = "2026-11-10"


def compounded(start: float, returns: list[float]) -> list[float]:
    """`start` and the levels it compounds to by `returns`."""
    levels = [start]
    for change in returns:
        levels.append(levels[-1] * (1 + change))
    return levels


NIFTY_LEVELS = compounded(OPENING_LEVEL, BENCH_RETURNS)


def session_close(day: dt.date) -> pd.Timestamp:
    """15:30 IST on `day`, the NSE session close."""
    return pd.Timestamp(dt.datetime(day.year, day.month, day.day, 15, 30, tzinfo=IST))


def nifty_closes() -> pd.Series:
    """The index closes indexed by naive session dates, the opening day first."""
    return pd.Series(NIFTY_LEVELS, index=pd.DatetimeIndex([OPENING_DAY, *DAYS]))


def _daily_frame(days: list[dt.date], opens: list[float], closes: list[float]) -> pd.DataFrame:
    """Own daily bars stamped at IST midnight of each session date."""
    index = pd.DatetimeIndex(
        [pd.Timestamp(day).tz_localize("Asia/Kolkata") for day in days], name="timestamp"
    )
    columns = {
        "open": opens,
        "high": [max(pair) + 5.0 for pair in zip(opens, closes, strict=True)],
        "low": [min(pair) - 5.0 for pair in zip(opens, closes, strict=True)],
        "close": closes,
        "volume": [1_000.0] * len(days),
    }
    return pd.DataFrame(columns, index=index)


def share_bars() -> pd.DataFrame:
    """The share's own daily bars."""
    return _daily_frame(DAYS, OPENS, CLOSES)


def index_bars() -> pd.DataFrame:
    """The index's own daily bars, the opening day first; each opens at its close."""
    return _daily_frame([OPENING_DAY, *DAYS], NIFTY_LEVELS, NIFTY_LEVELS)


def _historify_rows(days: list[dt.date], opens: list[float], closes: list[float]) -> list:
    """Daily rows as Historify serves them: stamped 00:00 UTC of the IST session date."""
    rows = []
    for day, open_, close in zip(days, opens, closes, strict=True):
        rows.append(
            {
                "timestamp": (day - dt.date(1970, 1, 1)).days * DAY_SECONDS,
                "open": open_,
                "high": max(open_, close) + 5.0,
                "low": min(open_, close) - 5.0,
                "close": close,
                "volume": 1_000,
                "oi": 0,
            }
        )
    return rows


def fake_client(with_index: bool = True) -> FakeClient:
    """A Historify that holds the share and, with `with_index`, the index."""
    held = {("SBIN", "NSE", "D"): _historify_rows(DAYS, OPENS, CLOSES)}
    if with_index:
        days = [OPENING_DAY, *DAYS]
        held[("NIFTY", "NSE_INDEX", "D")] = _historify_rows(days, NIFTY_LEVELS, NIFTY_LEVELS)
    return FakeClient(db=held)


def index_requests(client: FakeClient) -> list[tuple[str, str, str, str]]:
    """(symbol, exchange, first date, last date) of every history request on an index
    exchange."""
    return [
        (p["symbol"], p["exchange"], p["start_date"], p["end_date"])
        for endpoint, p in client.calls
        if endpoint == "history" and p["exchange"].endswith("_INDEX")
    ]


class Hold(bt.Strategy):
    """Buys `qty` shares at the first close and holds them."""

    symbol, exchange, interval = "SBIN", "NSE", "D"
    # 100 shares at 1,000.00 take the whole capital, so the swept values are smaller.
    qty = bt.param(QTY, QTY_RANGE)

    def on_bar(self, bar: Any) -> None:
        if len(self.close) == 1:
            self.buy(qty=self.qty)


def own_run(**settings: Any) -> bt.Result:
    """`Hold` on own data, with no charges and no slippage."""
    arguments = {
        "data": share_bars(),
        "capital": CAPITAL,
        "costs": None,
        "slippage": 0,
        "now": NOW,
        **settings,
    }
    return bt.backtest(Hold, **arguments)


def client_run(client: FakeClient, **settings: Any) -> bt.Result:
    """`Hold` on bars from the fake Historify, with no charges and no slippage."""
    arguments = {
        "source": "db",
        "start": DAYS[0],
        "end": DAYS[-1],
        "capital": CAPITAL,
        "costs": None,
        "slippage": 0,
        "now": NOW,
        **settings,
    }
    return bt.backtest(Hold, client, **arguments)


__all__ = [
    "BENCH_RETURNS",
    "CAPITAL",
    "CLOSES",
    "DAYS",
    "ENTRY_PRICE",
    "IST",
    "NIFTY_LEVELS",
    "NOW",
    "OPENING_DAY",
    "QTY",
    "QTY_RANGE",
    "Hold",
    "client_run",
    "compounded",
    "fake_client",
    "index_bars",
    "index_requests",
    "nifty_closes",
    "own_run",
    "session_close",
    "share_bars",
]
