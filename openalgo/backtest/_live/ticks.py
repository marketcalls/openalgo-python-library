"""The tick adapter: one market-data message of the SDK's feed as the shared `Tick`.

`tick_from_feed` reads the message a data callback of `FeedAPI` receives
(`subscribe_ltp`, `subscribe_quote`, `subscribe_depth`):

    {"type": "market_data", "symbol": "SBIN", "exchange": "NSE", "mode": 2,
     "data": {"ltp": 812.4, "volume": 1904211, "ltt": 1794815400000, ...}}

The data of mode 1 (LTP) holds `ltp`, `timestamp` and, when the server sent one, `ltt`;
mode 2 (quote) holds every quote field, 0 for a field the server left out; mode 3
(depth) holds `ltp`, `timestamp` and `depth` ({"buy": [...], "sell": [...]}, best level
first) and, as the SDK passes it on, neither a volume nor a last trade time. The server's
own frame has the same envelope, so it reads the same way.

The last trade time
-------------------

`ltt` is whatever the broker's adapter in the OpenAlgo server put there, and the adapters
differ: most send epoch milliseconds, several epoch seconds (some as a string of digits),
some fall back to the server's clock in milliseconds, some feeds count seconds from
1980-01-01 instead of 1970, and a few send a clock time or a date as text. The SDK's
quote data has 0 for a feed that sent none. One documented rule reads them all, and
never guesses:

1. Missing: no `ltt`, None, "" or 0 gives no event time (`Tick.ts_event_ns` None).
2. A number, or a string of decimal digits, is read by its magnitude: the plausible
   instants from 2000 to 2100 cover less than a factor of 5 in every unit, and the units
   (seconds, milliseconds, microseconds, nanoseconds) are a factor of 1000 apart, so at
   most one unit puts a value in range, and that is its unit.
3. The instant must also lie near its receipt: at most `MAX_EVENT_LEAD_NS` after it and
   `MAX_EVENT_AGE_NS` before it.

Anything else (a negative number, text that is no number, a value in no unit's range or
outside the window) is implausible: the tick keeps its price and gets no event time, and
the tally counts it (`TickTally.event_times_rejected`).

Prices are rupees, volume units, times UTC nanoseconds.
"""

from __future__ import annotations

import datetime as dt
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._layout import NANOS_PER_SECOND
from ._types import Tick

#: The kind of message a data callback receives for a tick.
MARKET_DATA_TYPE = "market_data"
#: The feed's mode numbers: LTP, quote and depth.
LTP_MODE, QUOTE_MODE, DEPTH_MODE = 1, 2, 3

#: The earliest instant a last trade time can stand for, UTC seconds: 2000-01-01, the
#: bound the live bars put on an event time too.
EARLIEST_EVENT_S = int(dt.datetime(2000, 1, 1, tzinfo=dt.UTC).timestamp())
#: The first instant past the latest a last trade time can stand for, UTC seconds:
#: 2100-01-01.
LATEST_EVENT_S = int(dt.datetime(2100, 1, 1, tzinfo=dt.UTC).timestamp())
#: How far a last trade time may lead its receipt: far beyond any clock error a session
#: runs with (the preflight allows 2 seconds), and short of the 5 h 30 min by which an
#: IST clock time counted as UTC would lead.
MAX_EVENT_LEAD_NS = 3_600 * NANOS_PER_SECOND
#: How far a last trade time may trail its receipt: an instrument can go untraded for
#: months (an illiquid bond or small company), not for more than a year; a count of
#: seconds from 1980-01-01 read from 1970 trails by ten years.
MAX_EVENT_AGE_NS = 366 * 86_400 * NANOS_PER_SECOND

# The units a last trade time is counted in, as (counts per second, nanoseconds per
# count), from the coarsest. Rule 2 of the module docstring picks one by magnitude.
_UNITS = (
    (1, NANOS_PER_SECOND),
    (1_000, 1_000_000),
    (1_000_000, 1_000),
    (1_000_000_000, 1),
)


@dataclass
class TickTally:
    """What `tick_from_feed` could not use, counted for a summary. Not thread-safe: the
    one loop that adapts the messages owns it.

    `not_ticks`: messages that are no market data of an instrument (an acknowledgement,
    an error, an order update). `unusable`: market data whose price is not a positive
    number, or whose volume, bid or ask is present but not usable; such a message makes
    no tick. `event_times_rejected`: ticks kept without the implausible last trade time
    they carried.
    """

    not_ticks: int = 0
    unusable: int = 0
    event_times_rejected: int = 0

    @property
    def dropped(self) -> int:
        """Messages that made no tick: `not_ticks` plus `unusable`."""
        return self.not_ticks + self.unusable


class _Unusable(Exception):
    """A present value that cannot be used (private: `tick_from_feed` counts it)."""


def tick_from_feed(
    message: Mapping[str, Any], *, recv_ns: int, tally: TickTally | None = None
) -> Tick | None:
    """The `Tick` of one market-data message of the SDK's feed, or None.

    `message` is what a data callback of `FeedAPI` receives (see the module docstring);
    `recv_ns` is when the loop took it off its queue, UTC nanoseconds, and becomes
    `Tick.ts_recv_ns`. `ts_event_ns` is the last trade time by the rule of the module
    docstring; `volume` the day's cumulative volume when the data holds one, None when
    it holds none, and 0 when it holds 0 (the SDK fills a quote volume the server left
    out with 0; the live bars apply their own rule to it); `bid` and `ask` the best
    levels of the book in depth mode, None for an empty side or a level priced 0.

    Returns None for a message that is no market data of an instrument and for one
    whose LTP is not a positive finite number or whose volume, bid or ask is present
    but not a usable number (a number too large to convert among them); `tally`, when
    given, counts them and every implausible last trade time. Never raises for the
    content of a message.
    """
    tally = TickTally() if tally is None else tally
    parts = _market_data(message)
    if parts is None:
        tally.not_ticks += 1
        return None
    symbol, exchange, mode, data = parts
    try:
        ltp = _price(data.get("ltp"))
        if ltp is None:
            raise _Unusable
        volume = _volume(data.get("volume"))
        bid, ask = _best_levels(data.get("depth")) if mode == DEPTH_MODE else (None, None)
    except _Unusable:
        tally.unusable += 1
        return None
    event_ns, implausible = _event_time(data.get("ltt"), recv_ns)
    tally.event_times_rejected += implausible
    return Tick(symbol, exchange, ltp, recv_ns, event_ns, volume, bid, ask)


def _market_data(message: Any) -> tuple[str, str, int, Mapping[str, Any]] | None:
    """(symbol, exchange, mode, data) of a market-data message, None for any other."""
    if not isinstance(message, Mapping) or message.get("type") != MARKET_DATA_TYPE:
        return None
    symbol, exchange = message.get("symbol"), message.get("exchange")
    mode, data = message.get("mode"), message.get("data")
    named = (
        isinstance(symbol, str) and bool(symbol) and isinstance(exchange, str) and bool(exchange)
    )
    known_mode = not isinstance(mode, bool) and mode in (LTP_MODE, QUOTE_MODE, DEPTH_MODE)
    if not (named and known_mode and isinstance(data, Mapping)):
        return None
    return symbol, exchange, mode, data


def _real(value: Any) -> float | None:
    """`value` as a finite float when it is a real number (never a bool), else None: also
    for one too large for a float."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _price(value: Any) -> float | None:
    """A positive price; None for a missing one (None or 0). Raises _Unusable for a
    present value that is no finite number of 0 or more."""
    if value is None:
        return None
    number = _real(value)
    if number is None or number < 0:
        raise _Unusable
    return None if number == 0 else number


def _volume(value: Any) -> float | None:
    """The day's cumulative volume, None when absent. Raises _Unusable for a present
    value that is not a finite number of 0 or more."""
    if value is None:
        return None
    number = _real(value)
    if number is None or number < 0:
        raise _Unusable
    return number


def _best_levels(depth: Any) -> tuple[float | None, float | None]:
    """(bid, ask): the price of the first level of each side of the book."""
    if depth is None:
        return None, None
    if not isinstance(depth, Mapping):
        raise _Unusable
    return _best(depth.get("buy")), _best(depth.get("sell"))


def _best(levels: Any) -> float | None:
    if levels is None or (isinstance(levels, list) and not levels):
        return None
    if not isinstance(levels, list) or not isinstance(levels[0], Mapping):
        raise _Unusable
    return _price(levels[0].get("price"))


def _count(value: Any) -> int | float | None:
    """A last trade time as a number: an int, a finite float or a string of decimal
    digits; None for anything else, a string of more digits than Python reads into an
    int among them."""
    if isinstance(value, str):
        if not (value.isascii() and value.isdecimal()):
            return None
        try:
            return int(value)
        except ValueError:
            return None
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    return _real(value)


def _event_time(value: Any, recv_ns: int) -> tuple[int | None, bool]:
    """(the last trade time in UTC nanoseconds or None, whether it was implausible), by
    the rule of the module docstring."""
    if value is None or value == "":
        return None, False
    count = _count(value)
    if count == 0:
        return None, False
    if count is None or count < 0:
        return None, True
    for per_second, nanos_per_count in _UNITS:
        if EARLIEST_EVENT_S * per_second <= count < LATEST_EVENT_S * per_second:
            # The whole counts in exact integers and only the fraction in floating point,
            # so 1794803399.25 seconds is 1794803399250000000 ns to the nanosecond.
            whole = math.floor(count)
            event_ns = whole * nanos_per_count + round((count - whole) * nanos_per_count)
            near = recv_ns - MAX_EVENT_AGE_NS <= event_ns <= recv_ns + MAX_EVENT_LEAD_NS
            return (event_ns, False) if near else (None, True)
    return None, True


__all__ = [
    "DEPTH_MODE",
    "EARLIEST_EVENT_S",
    "LATEST_EVENT_S",
    "LTP_MODE",
    "MARKET_DATA_TYPE",
    "MAX_EVENT_AGE_NS",
    "MAX_EVENT_LEAD_NS",
    "QUOTE_MODE",
    "TickTally",
    "tick_from_feed",
]
