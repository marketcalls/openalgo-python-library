"""The fake OpenAlgo's manual clock, and the IST wall-clock texts the server writes.

Every time in the fake is UTC nanoseconds read from one `ManualClock`: the broker's order
and trade times, the ticks' exchange times, the candles, the HTTP `Date` header. Hand
the same clock's `now` and `sleep` to the live layer, so the test decides when time
passes and nothing waits for real.
"""

from __future__ import annotations

import datetime as dt

from openalgo.backtest._calendar_convert import IST
from openalgo.backtest._layout import NANOS_PER_SECOND

#: Nanoseconds in a millisecond: the feed stamps its ticks in epoch milliseconds.
NANOS_PER_MILLI = 1_000_000
#: The wall-clock text of the server's books (sandbox/order_manager.py, get_orderbook).
BOOK_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


class ManualClock:
    """A clock that moves only when told. `now()` is UTC nanoseconds; `sleep(seconds)`
    moves the clock on by that much and returns at once, and keeps what was asked in
    `slept` (seconds, in order)."""

    def __init__(self, start_ns: int) -> None:
        if not isinstance(start_ns, int) or start_ns <= 0:
            raise ValueError(f"start_ns is UTC nanoseconds after 1970, not {start_ns!r}")
        self._now_ns = start_ns
        self.slept: list[float] = []

    def now(self) -> int:
        """The present, UTC nanoseconds."""
        return self._now_ns

    def advance(self, seconds: float) -> int:
        """Move the clock on by `seconds` (zero or more) and return the new present, UTC
        nanoseconds. Raises ValueError for a negative wait: the clock never goes back."""
        if seconds < 0:
            raise ValueError(f"the clock only moves forward, not by {seconds} seconds")
        self._now_ns += round(seconds * NANOS_PER_SECOND)
        return self._now_ns

    def sleep(self, seconds: float) -> None:
        """Stand in for a wait of `seconds`: the clock moves on by that much at once."""
        self.slept.append(seconds)
        self.advance(seconds)


def ist_ns(year: int, month: int, day: int, hour: int, minute: int, second: int = 0) -> int:
    """UTC nanoseconds of an IST wall-clock time."""
    moment = dt.datetime(year, month, day, hour, minute, second, tzinfo=IST)
    return int(moment.timestamp()) * NANOS_PER_SECOND


def ist_moment(ts_ns: int) -> dt.datetime:
    """An instant (UTC nanoseconds) as an IST date and time, to the whole second."""
    return dt.datetime.fromtimestamp(ts_ns // NANOS_PER_SECOND, tz=IST)


def book_time(ts_ns: int) -> str:
    """An instant as the IST text of the server's order and trade books."""
    return ist_moment(ts_ns).strftime(BOOK_TIME_FORMAT)


def epoch_ms(ts_ns: int) -> int:
    """An instant as the epoch milliseconds of the feed's `timestamp` and `ltt`."""
    return ts_ns // NANOS_PER_MILLI


__all__ = [
    "BOOK_TIME_FORMAT",
    "NANOS_PER_MILLI",
    "ManualClock",
    "book_time",
    "epoch_ms",
    "ist_moment",
    "ist_ns",
]
