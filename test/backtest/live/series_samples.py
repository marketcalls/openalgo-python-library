"""Bars, a kernel session and a backtest shared by the live series' tests
(`test_series*.py`).

The bars: 1-minute NSE bars of the twelve sessions from 2025-06-16 to 2025-07-01, a
seeded random walk from 500.00 (each bar opens at the last close, moves by a normal step
of 0.40 rupees rounded to the paisa, high and low 0.30 beyond open and close, volume
1,000 plus the bar's number in its session). The live day is the last session unless a
test names another; the sessions before it are the history. Nothing is fitted to the
code: the expected values come from a backtest of the same bars, or from `openalgo.ta`
on the expected closes.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

import openalgo.backtest as bt
from openalgo.backtest._calendar_convert import native_attr, to_day
from openalgo.backtest._causality_checks import IndicatorChecks
from openalgo.backtest._context import RunData
from openalgo.backtest._live._types import LiveBar
from openalgo.backtest._live.series import LiveRunContext, LiveSeries
from openalgo.backtest.calendar import default_calendar
from openalgo.backtest.reference import Instrument

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
LIVE_DAY = dt.date(2025, 7, 1)
DAYS = tuple(default_calendar().sessions("NSE", "2025-06-16", LIVE_DAY)["date"])
MINUTE_NS = 60 * 10**9
FIELDS = ("open", "high", "low", "close", "volume")
NO_BARS = (np.array([], dtype=np.int64), *(np.array([], dtype=np.float64) for _ in range(5)))
SBIN = Instrument("SBIN", "NSE", "equity", tick_size=0.05)
# The kernel's form of SBIN (`Engine.session`).
SBIN_NATIVE = {
    "symbol": "SBIN",
    "exchange": "NSE",
    "kind": "EQ",
    "lot_size": 1,
    "tick_size": 0.05,
    "multiplier": 1.0,
    "freeze_qty": None,
    "last_trading_day": None,
}
SETTINGS = {"capital": 1_000_000, "costs": None, "slippage": 0}


def minute_bars(days: tuple[dt.date, ...] = DAYS, seed: int = 7) -> pd.DataFrame:
    """The bars of the module docstring in the bars layout, oldest first."""
    rng = np.random.default_rng(seed)
    rows, price = [], 500.0
    for day in days:
        for number, (ts_open, ts_close) in enumerate(LiveSeries("SBIN", "NSE", day).slices):
            open_ = price
            close = round(open_ + rng.normal(0.0, 0.4), 2)
            high, low = max(open_, close) + 0.3, min(open_, close) - 0.3
            rows.append((ts_open, ts_close, open_, high, low, close, 1_000.0 + number, np.nan))
            price = close
    columns = ("ts_open", "ts_close", *FIELDS, "oi")
    return pd.DataFrame(rows, columns=columns)


def on_day(bars: pd.DataFrame, day: dt.date) -> pd.DataFrame:
    """The bars of one IST session date."""
    dates = pd.to_datetime(bars["ts_open"], utc=True).dt.tz_convert(IST).dt.date
    return bars[dates == day].reset_index(drop=True)


def before(bars: pd.DataFrame, day: dt.date) -> pd.DataFrame:
    """The bars of the sessions before `day`."""
    dates = pd.to_datetime(bars["ts_open"], utc=True).dt.tz_convert(IST).dt.date
    return bars[dates < day].reset_index(drop=True)


def user_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """The bars as a user hands them to `bt.backtest`: indexed by IST bar open."""
    index = pd.DatetimeIndex(
        pd.to_datetime(bars["ts_open"], utc=True).dt.tz_convert(IST), name="timestamp"
    )
    return pd.DataFrame({name: bars[name].to_numpy() for name in FIELDS}, index=index)


def live_bar(row: Any, source: str = "ticks") -> LiveBar:
    """A row of the bars layout as a live bar of `source` ("ticks" or "candle")."""
    return LiveBar(
        symbol="SBIN",
        exchange="NSE",
        ts_open_ns=int(row.ts_open),
        ts_close_ns=int(row.ts_close),
        open=float(row.open),
        high=float(row.high),
        low=float(row.low),
        close=float(row.close),
        volume=float(row.volume),
        ticks=0 if source == "candle" else 1,
        source=source,
    )


def backtest_views(
    strategy: type, bars: pd.DataFrame, read: Callable, day: dt.date = LIVE_DAY
) -> list[tuple]:
    """(decision instant, bar, what `read(strategy)` returns) at every `on_bar` of a
    backtest of `strategy` on the live `day`, over the bars up to its end."""
    seen: list[tuple] = []

    class Recording(strategy):
        def on_bar(self, bar):
            seen.append((self._ctx.decided_at, tuple(bar), read(self)))

    data = user_frame(pd.concat((before(bars, day), on_day(bars, day)), ignore_index=True))
    bt.backtest(Recording, data=data, start=day, end=day, **SETTINGS)
    return seen


class LiveRun:
    """A strategy in a live run context on a kernel session of the live `day`, driven as
    the session loop drives it (16-session-design 2.5 B and C): each closed slice's bar
    goes to the kernel and to the series, the slice is released, and the kernel's pauses
    reveal the bars and call `on_bar`. No server is involved."""

    def __init__(
        self,
        strategy: type,
        history: pd.DataFrame,
        interval_minutes: int,
        day: dt.date = LIVE_DAY,
    ) -> None:
        self.series = LiveSeries("SBIN", "NSE", day, history=history)
        config = {
            "first_day": to_day(day),
            "last_day": to_day(day),
            "base_nanos": MINUTE_NS,
            "strategy_interval": interval_minutes * MINUTE_NS,
            "fill": "next_open",
            "quantity_mode": "venue",
            "capital": float(SETTINGS["capital"]),
            "costs": None,
            "slippage": ("zero", 0.0, 0),
            "risk": {
                "risk_basis": "gross",
                "after_risk_exit": "lock_session",
                "round_levels": True,
                "strict_config": True,
            },
            "deliver_events": False,
        }
        calendar = default_calendar()
        self.engine = native_attr("Engine").session(calendar._cal, config, [SBIN_NATIVE], [NO_BARS])
        run = RunData(
            exchange="NSE",
            interval=strategy.interval,
            base="1m",
            base_bars={},
            calendar=calendar,
            checks=IndicatorChecks("live-series-test").for_run(),
        )
        self.context = LiveRunContext(self.engine, [SBIN], "MIS", None, run, {"SBIN": self.series})
        self.strategy = strategy()
        self.strategy._ctx = self.context
        self.strategy.init()
        self._released = 0

    def close_slice(self, bar: LiveBar) -> None:
        """The next slice closed with `bar`: kernel and series take it, the slice is
        released, and the pauses are driven until the kernel waits."""
        assert self.engine.advance()[0] == "waiting"
        prices = (bar.open, bar.high, bar.low, bar.close, bar.volume)
        self.engine.add_bar(0, (bar.ts_open_ns, *prices))
        self.series.take(bar)
        self._released += 1
        self.engine.release(self._released)
        self.drive()

    def drive(self) -> None:
        """Run the kernel's pauses until it waits or the day is over, as the backtest's
        driver does."""
        while True:
            kind, decided_at, rows, _events, timers = self.engine.advance()
            if kind == "waiting":
                return
            revealed = self.context.reveal(decided_at, rows)
            if kind == "finished":
                return
            if kind == "timers":
                self.context.timers.fire(timers)
            elif revealed:
                self.strategy.on_bar(next(iter(revealed.values())))


__all__ = [
    "DAYS",
    "LIVE_DAY",
    "MINUTE_NS",
    "SBIN",
    "LiveRun",
    "backtest_views",
    "before",
    "live_bar",
    "minute_bars",
    "on_day",
    "user_frame",
]
