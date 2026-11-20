"""Broker candles as live bars, and the check of tick-built bars against them
(08-live-trading section 5, rule 2; D27). Part of `bars`, which re-exports it.

The broker's 1-minute candle of a slice verifies the bar built from that slice's ticks:
every difference goes to the session's drift log. The strategy keeps deciding on the bar
built from ticks (lead decision 2026-11-21, 2.4.0 item P2-1): a candle is the bar only of
a slice that made no tick bar. Times are UTC nanoseconds, prices rupees, volume units.
"""

from __future__ import annotations

import datetime as dt
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .._calendar_convert import IST
from .._layout import NANOS_PER_SECOND
from ..calendar import Calendar, default_calendar
from ..errors import ConfigError, DataError, TimestampError
from ._types import LiveBar

#: `LiveBar.source` of a bar built from WebSocket ticks.
TICKS_SOURCE = "ticks"
#: `LiveBar.source` of a broker candle.
CANDLE_SOURCE = "candle"

_PRICE_FIELDS = ("open", "high", "low", "close")
# An intraday interval is shorter than one day, as the calendar's grid requires.
_ONE_DAY_NS = 24 * 3600 * NANOS_PER_SECOND
# The instants a candle may open at, UTC nanoseconds: from 2000 to 2100. Outside them a
# number is not the time of a candle (milliseconds, a wrong unit, or a row position).
_FIRST_OPEN_NS = int(dt.datetime(2000, 1, 1, tzinfo=dt.UTC).timestamp()) * NANOS_PER_SECOND
_END_OPEN_NS = int(dt.datetime(2100, 1, 1, tzinfo=dt.UTC).timestamp()) * NANOS_PER_SECOND

Row = Mapping[str, Any] | pd.Series


def check_interval(interval_ns: int) -> int:
    """A bar interval in nanoseconds as an int. Raises ConfigError, whatever the type of
    the value, unless it is a positive whole number of seconds shorter than one day (the
    calendar builds its intraday grid from such intervals)."""
    whole = isinstance(interval_ns, numbers.Integral) and not isinstance(interval_ns, bool)
    if not (whole and 0 < interval_ns < _ONE_DAY_NS and interval_ns % NANOS_PER_SECOND == 0):
        raise ConfigError(
            "interval_ns is the bar interval in nanoseconds and must be a positive whole "
            "number of seconds shorter than a day, for example 60_000_000_000 for one "
            f"minute; got {interval_ns!r}"
        )
    return int(interval_ns)


def _aware_ns(stamp: dt.datetime, what: str) -> int:
    if stamp.tzinfo is None:
        raise TimestampError(
            f"{what}: the timestamp {stamp} has no time zone, and a zone is never guessed: "
            "pass the row as client.history returns it (IST-aware)"
        )
    return int(pd.Timestamp(stamp).value)


def _epoch_ns(value: Any, scale: int, unit: str, what: str) -> int:
    """`value` counted in `unit` (`scale` nanoseconds each) as UTC nanoseconds, when it
    is a whole number of them from 2000 to 2100."""
    number = isinstance(value, numbers.Real) and not isinstance(value, bool)
    whole = number and math.isfinite(value) and value == int(value)
    if whole and _FIRST_OPEN_NS <= int(value) * scale < _END_OPEN_NS:
        return int(value) * scale
    raise TimestampError(
        f"{what}: expected whole {unit} from 2000 to 2100 for the candle's open, got "
        f"{value!r} (a value in milliseconds is refused)"
    )


def _candle_open_ns(row: Row, what: str) -> int:
    """The candle's open, UTC nanoseconds, read only from an explicit time: a
    "timestamp" field, a "ts_open" field, or a name that is an aware timestamp."""
    if "timestamp" in row:
        stamp = row["timestamp"]
        if isinstance(stamp, dt.datetime):
            return _aware_ns(stamp, what)
        return _epoch_ns(stamp, NANOS_PER_SECOND, "epoch seconds", what)
    if "ts_open" in row:
        return _epoch_ns(row["ts_open"], 1, "UTC nanoseconds", what)
    name = getattr(row, "name", None)
    if isinstance(name, dt.datetime):
        return _aware_ns(name, what)
    raise TimestampError(
        f"{what}: the row has no timestamp or ts_open field and is not named by a "
        f"timestamp (its name is {name!r}); a row position is never read as a time: pass "
        "the timestamp"
    )


def _price(row: Row, name: str, what: str) -> float:
    if name not in row:
        raise DataError(f"{what} has no {name!r} field")
    value = row[name]
    usable = isinstance(value, numbers.Real) and not isinstance(value, bool)
    if not (usable and math.isfinite(value) and value > 0):
        raise DataError(f"{what}: {name} must be a positive price in rupees, got {value!r}")
    return float(value)


def _candle_volume(row: Row, what: str) -> float:
    """The candle's volume; 0.0 when the row carries none (no field, None or NaN)."""
    value = row.get("volume")
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    usable = isinstance(value, numbers.Real) and not isinstance(value, bool)
    if not (usable and math.isfinite(value) and value >= 0):
        raise DataError(f"{what}: volume must be a number of units, got {value!r}")
    return float(value)


def _candle_close_ns(
    exchange: str, ts_open: int, interval_ns: int, calendar: Calendar, what: str
) -> int:
    """The candle's close: its open plus the interval, cut at its session's close."""
    day = pd.Timestamp(ts_open, tz="UTC").tz_convert(IST).date()
    session = calendar.session(exchange, day)
    if session is None or not session[0] - interval_ns < ts_open < session[1]:
        raise DataError(
            f"{what} opens at {pd.Timestamp(ts_open, tz='UTC').tz_convert(IST)}, outside the "
            f"{exchange} session of {day}: ask for candles of the session only"
        )
    return min(ts_open + interval_ns, session[1])


def candle_bar(
    symbol: str,
    exchange: str,
    row: Row,
    interval_ns: int,
    *,
    calendar: Calendar | None = None,
) -> LiveBar:
    """A broker candle as a live bar with `source` "candle" and `ticks` 0.

    The candle's open is read only from an explicit time: a "timestamp" field (an aware
    datetime, or whole epoch seconds from 2000 to 2100 as in the server's payload), a
    "ts_open" field (UTC nanoseconds, the backtest's bars layout), or the row's name
    when it is an aware timestamp (a row of the frame `client.history` returns). The
    candle closes `interval_ns` (nanoseconds) after its open, cut at the session close
    of `calendar` (the package default when None), as the session's last slice is.
    Prices are rupees; the volume is 0.0 when the row has none.

    Raises ConfigError for a bad interval; TimestampError for a row without an explicit
    time (a row position is never one), a timestamp without a time zone, or a number
    that is not whole seconds or nanoseconds from 2000 to 2100; CalendarError for an
    unknown exchange; DataError for a candle outside its day's session, a missing or
    unusable price or volume, or a high and low that do not contain the open and close.
    """
    interval_ns = check_interval(interval_ns)
    calendar = default_calendar() if calendar is None else calendar
    what = f"the {exchange}:{symbol} candle"
    ts_open = _candle_open_ns(row, what)
    ts_close = _candle_close_ns(exchange, ts_open, interval_ns, calendar, what)
    open_, high, low, close = (_price(row, name, what) for name in _PRICE_FIELDS)
    if not (low <= min(open_, close) and high >= max(open_, close)):
        raise DataError(
            f"{what} is not a valid bar: low {low} and high {high} must contain open "
            f"{open_} and close {close}"
        )
    return LiveBar(
        symbol=symbol,
        exchange=exchange,
        ts_open_ns=ts_open,
        ts_close_ns=ts_close,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=_candle_volume(row, what),
        ticks=0,
        source=CANDLE_SOURCE,
    )


@dataclass(frozen=True)
class BarDrift:
    """How a broker candle differs from the tick-built bar of its symbol.

    `open`, `high`, `low` and `close` are the candle's price minus the tick bar's
    (rupees), `volume` the candle's volume minus the tick bar's (units); they are kept
    even when zero. `ts_open_ns` and `ts_close_ns` are the tick bar's slice (the
    expected one), `candle_ts_open_ns` and `candle_ts_close_ns` the candle's (UTC
    nanoseconds); `ticks` counts the tick bar's trades.
    """

    symbol: str
    exchange: str
    ts_open_ns: int
    ts_close_ns: int
    candle_ts_open_ns: int
    candle_ts_close_ns: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    ticks: int

    @property
    def same_slice(self) -> bool:
        """True when the candle opens and closes with the tick bar (same interval)."""
        candle = (self.candle_ts_open_ns, self.candle_ts_close_ns)
        return candle == (self.ts_open_ns, self.ts_close_ns)


class CandleCheck:
    """Checks tick-built bars against the broker's candles (08-live-trading section 5,
    rule 2) and keeps the drift of every candle it accepts, for the drift log."""

    def __init__(self) -> None:
        self._drifts: list[BarDrift] = []

    @property
    def drifts(self) -> list[BarDrift]:
        """The drift of every accepted candle, in the order they were accepted."""
        return list(self._drifts)

    def compare(self, tick_bar: LiveBar, candle: LiveBar) -> BarDrift:
        """The differences between a candle and the tick bar of its symbol; records
        nothing. Raises ConfigError unless `tick_bar` is built from ticks and `candle`
        is a candle of the same symbol and exchange."""
        if (tick_bar.source, candle.source) != (TICKS_SOURCE, CANDLE_SOURCE):
            raise ConfigError(
                f"compare takes a bar built from ticks and then a broker candle, got "
                f"sources {tick_bar.source!r} and {candle.source!r}"
            )
        if (tick_bar.symbol, tick_bar.exchange) != (candle.symbol, candle.exchange):
            raise ConfigError(
                f"the candle of {candle.exchange}:{candle.symbol} cannot check the bar of "
                f"{tick_bar.exchange}:{tick_bar.symbol}"
            )
        return BarDrift(
            symbol=tick_bar.symbol,
            exchange=tick_bar.exchange,
            ts_open_ns=tick_bar.ts_open_ns,
            ts_close_ns=tick_bar.ts_close_ns,
            candle_ts_open_ns=candle.ts_open_ns,
            candle_ts_close_ns=candle.ts_close_ns,
            open=candle.open - tick_bar.open,
            high=candle.high - tick_bar.high,
            low=candle.low - tick_bar.low,
            close=candle.close - tick_bar.close,
            volume=candle.volume - tick_bar.volume,
            ticks=tick_bar.ticks,
        )

    def accepted(self, tick_bar: LiveBar, candle: LiveBar) -> LiveBar:
        """The candle when it covers the tick bar's slice (same open and close, so a
        candle of another interval never checks the bar), its drift then kept in
        `drifts`; else the tick bar, with nothing kept (the session asks the broker
        again). Raises ConfigError as `compare` does."""
        drift = self.compare(tick_bar, candle)
        if not drift.same_slice:
            return tick_bar
        self._drifts.append(drift)
        return candle


__all__ = [
    "CANDLE_SOURCE",
    "TICKS_SOURCE",
    "BarDrift",
    "CandleCheck",
    "candle_bar",
    "check_interval",
]
