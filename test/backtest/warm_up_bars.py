"""Bars for the warm-up tests: seeded walks over the sessions of the default NSE calendar,
built here so that every expected value can be computed from them by hand (no server).

`walk` gives daily bars: each close is the previous one times e^(drift + vol z), z a
standard normal draw, from Rs 100.00; the open is the previous close, the high and low
0.5 percent either side; volume 1,000 units. `intraday` gives bars of `minutes` minutes
from 09:15 to 15:30 IST with the same rules (moves of 0.2 percent, high and low 0.1
percent either side).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import openalgo.backtest as bt

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
CALENDAR = bt.calendar.default_calendar()
SETTINGS = {"capital": 1_000_000, "costs": None, "slippage": 0}
SESSION_MINUTES = 375


def sessions(first: str, last: str) -> list[dt.date]:
    """The NSE sessions from `first` to `last`, both included."""
    return list(CALENDAR.sessions("NSE", first, last)["date"])


def _bars(stamps: list[dt.datetime], steps: np.ndarray, spread: float) -> pd.DataFrame:
    closes = 100.0 * np.exp(np.cumsum(steps))
    opens = np.concatenate(([100.0], closes[:-1]))
    frame = {
        "open": opens,
        "high": np.maximum(opens, closes) * (1 + spread),
        "low": np.minimum(opens, closes) * (1 - spread),
        "close": closes,
        "volume": np.full(len(stamps), 1_000.0),
    }
    return pd.DataFrame(frame, index=pd.DatetimeIndex(stamps, name="timestamp"))


def walk(
    first: str, last: str, seed: int = 7, drift: float = 0.0, vol: float = 0.01
) -> pd.DataFrame:
    """Daily bars of the module docstring over the sessions from `first` to `last`."""
    days = sessions(first, last)
    steps = np.random.default_rng(seed).normal(drift, vol, len(days))
    stamps = [dt.datetime.combine(day, dt.time(9, 15), tzinfo=IST) for day in days]
    return _bars(stamps, steps, 0.005)


def intraday(first: str, last: str, minutes: int = 5, seed: int = 9) -> pd.DataFrame:
    """Bars of `minutes` minutes of the module docstring over the sessions from `first`
    to `last`."""
    stamps = [
        dt.datetime.combine(day, dt.time(9, 15), tzinfo=IST) + dt.timedelta(minutes=k)
        for day in sessions(first, last)
        for k in range(0, SESSION_MINUTES, minutes)
    ]
    steps = np.random.default_rng(seed).normal(0.0, 0.002, len(stamps))
    return _bars(stamps, steps, 0.001)
