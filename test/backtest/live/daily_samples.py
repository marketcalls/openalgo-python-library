"""Daily and weekly strategies on the fake OpenAlgo, shared by the pre-open decision's tests
(`test_session_daily*.py`; design 20-positional-live section 8, the lead's ruling N5b-1).

The market is `session_samples`': SBIN on NSE, the live day Monday 2026-11-16. The broker
serves 1-minute candles of the four sessions before it (Monday 2026-11-09 to Friday
2026-11-13, a week whose Tuesday is a holiday; `plan_samples.candles`' seeded walk from
500.00) and its daily candles made from them (`fake_openalgo.routes_data`), unless a test
names another day or other sessions. The session starts before the open, at 09:05 IST,
unless a test says otherwise; SBIN trades 10 units
every 10 s from the open at `PRICE` until 15:02. The strategies record each bar `on_bar`
is given, so a test reads what was decided. Expected values come from the candles and
the calendar, never from the session's code.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from plan_samples import DAY, candles
from session_samples import ist, rig

import openalgo.backtest as bt
from openalgo.backtest.calendar import default_calendar

#: The four sessions before the live day, oldest first.
SESSIONS = (
    dt.date(2026, 11, 9),
    dt.date(2026, 11, 11),
    dt.date(2026, 11, 12),
    dt.date(2026, 11, 13),
)
#: The live day's open and the start before it.
OPEN = ist(9, 15)
START = (9, 5, 0)
#: SBIN's price from the open, rupees.
PRICE = 500.0
#: The units each strategy buys.
UNITS = 10


def close_of(day: dt.date) -> int:
    """NSE's close of `day`, UTC nanoseconds (the calendar's)."""
    return default_calendar().session("NSE", day)[1]


def open_of(day: dt.date) -> int:
    """NSE's open of `day`, UTC nanoseconds (the calendar's)."""
    return default_calendar().session("NSE", day)[0]


def daily_candle(day: dt.date) -> tuple[float, float, float, float, float]:
    """(open, high, low, close, volume) of SBIN's daily candle of `day`, made from its
    1-minute candles as the broker makes it."""
    rows = [row for row in candles(SESSIONS, today_until=None) if _ist_day(row[0]) == day]
    return (
        rows[0][1],
        max(row[2] for row in rows),
        min(row[3] for row in rows),
        rows[-1][4],
        sum(row[5] for row in rows),
    )


def _ist_day(ts_ns: int) -> dt.date:
    return dt.datetime.fromtimestamp(ts_ns / 1e9 + 19_800, tz=dt.UTC).date()


class Recorded(bt.Strategy):
    """A daily CNC strategy on SBIN that buys `UNITS` at its first decision and records the
    open and close (UTC nanoseconds) of each bar `on_bar` is given, and how many closes of
    its interval it saw then, in the class's `decided` (cleared by `daily_world`)."""

    symbol, exchange, interval, product = "SBIN", "NSE", "D", "CNC"
    decided: list[tuple[int, int, int]] = []
    rules: dict[str, Any] = {}

    def on_bar(self, bar):
        type(self).decided.append((bar.ts_open, bar.ts_close, len(self.close)))
        if not self.position.qty and not self.orders:
            self.buy(UNITS, **type(self).rules)


class WeeklyRecorded(Recorded):
    """Recorded on weekly bars."""

    interval = "W"
    decided: list[tuple[int, int, int]] = []


def daily_world(
    make_fake_openalgo: Any,
    monkeypatch: Any,
    tmp_path: Any,
    *,
    start: tuple[int, int, int] = START,
    analyzer: bool = False,
    day: dt.date = DAY,
    sessions: tuple[dt.date, ...] = SESSIONS,
) -> Any:
    """A rig started at the IST time `start` of `day` (module docstring), the broker serving
    the 1-minute candles of `sessions` only (and the daily candles made from them), in the
    sandbox when `analyzer`; each strategy's record cleared."""
    for strategy in (Recorded, WeeklyRecorded):
        strategy.decided.clear()
    world = rig(
        make_fake_openalgo,
        monkeypatch,
        tmp_path,
        start=start,
        trades_until=None,
        analyzer=analyzer,
        day=day,
        rows=candles(sessions, today_until=None),
    )
    world.script.price = lambda now_ns: PRICE
    world.script.trade()
    first = max(open_of(day), ist(*start, day=day) + 10 * 10**9)
    world.script.trades(first, ist(15, 2, day=day))
    return world


__all__ = [
    "OPEN",
    "PRICE",
    "SESSIONS",
    "START",
    "UNITS",
    "Recorded",
    "WeeklyRecorded",
    "close_of",
    "daily_candle",
    "daily_world",
    "open_of",
]
