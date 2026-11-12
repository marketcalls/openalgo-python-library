"""Runs shared by the accounting tests (charges, slippage, margin, sizing, reports).

Intraday: one NSE share on 2026-11-04, 1-minute bars over the whole session (09:15 to
15:30 IST). Bars opening before 09:45 open at 1,000.00 and later bars at 1,100.00; each
bar's high and low are 1.00 either side of its open and it closes at its open. The
`round_trip` strategy (15-minute) buys at the 09:15 bar's close, so its entry fills at
the 09:30 open, 1,000.00 before slippage, and exits at the 09:30 bar's close, filling at
the 09:45 open, 1,100.00 before slippage.

Daily: the same share on 2026-11-02, 03 and 04 with daily bars opening at 1,000.00,
1,000.00 and 1,100.00 (high and low 10.00 either side, close at the open). A daily
strategy that buys at the first close fills at the second session's open and exits at
the third session's open.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd

import openalgo.backtest as bt
from openalgo.backtest.reference import Instrument

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
DAY = dt.date(2026, 11, 4)
DAILY_DAYS = (dt.date(2026, 11, 2), dt.date(2026, 11, 3), dt.date(2026, 11, 4))
SESSION_MINUTES = 375
CAPITAL = 1_000_000
ENTRY_PRICE = 1_000.0
EXIT_PRICE = 1_100.0
# Minutes after 09:15 at which the price steps from the entry to the exit price.
STEP_MINUTE = 30


def at(hour: int, minute: int, day: dt.date = DAY) -> dt.datetime:
    """An IST time on `day`."""
    return dt.datetime(day.year, day.month, day.day, hour, minute, tzinfo=IST)


def session_bars(day: dt.date = DAY, minutes: int = SESSION_MINUTES) -> pd.DataFrame:
    """The intraday bars of the module docstring, `minutes` of them from 09:15 (385 for
    an NFO session that closes at 15:40)."""
    opens = np.where(np.arange(minutes) < STEP_MINUTE, ENTRY_PRICE, EXIT_PRICE)
    index = pd.DatetimeIndex(
        [at(9, 15, day) + dt.timedelta(minutes=int(k)) for k in range(minutes)],
        name="timestamp",
    )
    columns = {
        "open": opens,
        "high": opens + 1.0,
        "low": opens - 1.0,
        "close": opens,
        "volume": np.full(minutes, 1_000.0),
    }
    return pd.DataFrame(columns, index=index)


def daily_bars() -> pd.DataFrame:
    """The daily bars of the module docstring."""
    opens = np.array([ENTRY_PRICE, ENTRY_PRICE, EXIT_PRICE])
    index = pd.DatetimeIndex([at(9, 15, day) for day in DAILY_DAYS], name="timestamp")
    columns = {
        "open": opens,
        "high": opens + 10.0,
        "low": opens - 10.0,
        "close": opens,
        "volume": np.full(3, 1_000.0),
    }
    return pd.DataFrame(columns, index=index)


def round_trip_strategy(
    interval: str = "15m", symbol: str = "SBIN", exchange: str = "NSE", **entry: Any
) -> type[bt.Strategy]:
    """A strategy that buys with `entry` (the arguments of `buy`) at its first bar close
    and exits the whole position at its second."""

    def on_bar(self: bt.Strategy, bar: Any) -> None:
        if len(self.close) == 1:
            self.buy(**entry)
        elif len(self.close) == 2 and not self.position.is_flat:
            self.exit()

    attributes = {"symbol": symbol, "exchange": exchange, "interval": interval, "on_bar": on_bar}
    return type("RoundTrip", (bt.Strategy,), attributes)


def round_trip(entry: dict[str, Any] | None = None, **settings: Any) -> Any:
    """The intraday round trip with `entry` and the run arguments `settings` (MIS on a
    1-minute base by default; costs and slippage at their defaults unless given)."""
    arguments = {"data": session_bars(), "capital": CAPITAL, "base": "1m", "product": "MIS"}
    arguments.update(settings)
    return bt.backtest(round_trip_strategy(**(entry or {})), **arguments)


NIFTY_FUTURE = Instrument(
    symbol="NIFTY24NOV26FUT",
    exchange="NFO",
    kind="future",
    lot_size=75,
    tick_size=0.05,
    freeze_qty=1_800,
    expiry=dt.date(2026, 11, 24),
    underlying="NIFTY",
    settlement="cash",
)


def future_round_trip(
    entry: dict[str, Any] | None = None, instrument: Instrument = NIFTY_FUTURE, **settings: Any
) -> Any:
    """The intraday round trip on an NFO future (an NFO session to 15:40), NRML."""
    arguments = {
        "data": session_bars(minutes=385),
        "capital": CAPITAL,
        "base": "1m",
        "product": "NRML",
        "instrument": instrument,
    }
    arguments.update(settings)
    strategy = round_trip_strategy("15m", instrument.symbol, "NFO", **(entry or {}))
    return bt.backtest(strategy, **arguments)


def daily_round_trip(entry: dict[str, Any] | None = None, **settings: Any) -> Any:
    """The daily round trip on delivery (CNC)."""
    arguments = {"data": daily_bars(), "capital": CAPITAL, "slippage": 0}
    arguments.update(settings)
    return bt.backtest(round_trip_strategy("D", **(entry or {"qty": 400})), **arguments)
