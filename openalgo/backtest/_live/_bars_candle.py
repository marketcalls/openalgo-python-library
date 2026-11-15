"""Broker candles as live bars, and the check of tick-built bars against them
(08-live-trading section 5, rule 2; D27). Part of `bars`, which re-exports it.

The broker's 1-minute candle of a slice verifies the bar built from that slice's ticks:
every difference goes to the session's drift log, and indicators continue from the
candle when it starts on time, so small differences do not accumulate. Times are UTC
nanoseconds, prices rupees, volume units.
"""

from __future__ import annotations

import datetime as dt
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .._layout import NANOS_PER_SECOND
from ..errors import ConfigError, DataError, TimestampError
from ._types import LiveBar

#: `LiveBar.source` of a bar built from WebSocket ticks.
TICKS_SOURCE = "ticks"
#: `LiveBar.source` of a broker candle.
CANDLE_SOURCE = "candle"

_PRICE_FIELDS = ("open", "high", "low", "close")


def check_interval(interval_ns: int) -> int:
    """A bar interval in nanoseconds as an int. Raises ConfigError unless it is a
    positive whole number of seconds (the calendar builds its grid from whole seconds)."""
    whole = isinstance(interval_ns, numbers.Integral) and not isinstance(interval_ns, bool)
    if not whole or interval_ns <= 0 or interval_ns % NANOS_PER_SECOND:
        raise ConfigError(
            "interval_ns is the bar interval in nanoseconds and must be a positive whole "
            f"number of seconds, for example 60_000_000_000 for one minute; got {interval_ns!r}"
        )
    return int(interval_ns)


def _candle_open_ns(row: Mapping[str, Any] | pd.Series, what: str) -> int:
    """The candle's open as UTC nanoseconds: its "timestamp" field, or the row's name
    when it is a row of the frame `client.history` returns (indexed by timestamp)."""
    stamp = row.get("timestamp") if "timestamp" in row else getattr(row, "name", None)
    if isinstance(stamp, dt.datetime):
        if stamp.tzinfo is None:
            raise TimestampError(
                f"{what}: the timestamp {stamp} has no time zone, and a zone is never "
                "guessed: pass the row as client.history returns it (IST-aware)"
            )
        return int(pd.Timestamp(stamp).value)
    epoch = isinstance(stamp, numbers.Real) and not isinstance(stamp, bool)
    if epoch and math.isfinite(stamp) and stamp > 0 and stamp == int(stamp):
        return int(stamp) * NANOS_PER_SECOND
    raise TimestampError(
        f"{what}: expected a time-zone-aware timestamp or whole epoch seconds (the bar's "
        f"open), got {stamp!r}"
    )


def _price(row: Mapping[str, Any] | pd.Series, name: str, what: str) -> float:
    if name not in row:
        raise DataError(f"{what} has no {name!r} field")
    value = row[name]
    usable = isinstance(value, numbers.Real) and not isinstance(value, bool)
    if not (usable and math.isfinite(value) and value > 0):
        raise DataError(f"{what}: {name} must be a positive price in rupees, got {value!r}")
    return float(value)


def _candle_volume(row: Mapping[str, Any] | pd.Series, what: str) -> float:
    """The candle's volume; 0.0 when the row carries none (no field, None or NaN)."""
    value = row.get("volume")
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    usable = isinstance(value, numbers.Real) and not isinstance(value, bool)
    if not (usable and math.isfinite(value) and value >= 0):
        raise DataError(f"{what}: volume must be a number of units, got {value!r}")
    return float(value)


def candle_bar(
    symbol: str, exchange: str, row: Mapping[str, Any] | pd.Series, interval_ns: int
) -> LiveBar:
    """A broker candle as a live bar with `source` "candle" and `ticks` 0.

    `row` is one row of the frame `client.history` returns (a pandas Series named by its
    IST-aware timestamp, the instant the candle opens) or a mapping with a "timestamp"
    field: an aware datetime, or the whole epoch seconds of the server's payload (UTC,
    the candle's open). Prices are rupees; the volume is 0.0 when the row has none. The
    bar closes `interval_ns` (nanoseconds) after its open.

    Raises ConfigError for a bad interval, TimestampError for a timestamp without a time
    zone or not in whole seconds, and DataError for a missing or unusable price or
    volume, or a high and low that do not contain the open and close.
    """
    interval_ns = check_interval(interval_ns)
    what = f"the {exchange}:{symbol} candle"
    ts_open = _candle_open_ns(row, what)
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
        ts_close_ns=ts_open + interval_ns,
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
    even when zero. `ts_open_ns` is the tick bar's open (the expected time) and
    `candle_ts_open_ns` the candle's (UTC nanoseconds); `ticks` counts the tick bar's
    ticks.
    """

    symbol: str
    exchange: str
    ts_open_ns: int
    candle_ts_open_ns: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    ticks: int

    @property
    def on_time(self) -> bool:
        """True when the candle starts at the tick bar's open."""
        return self.candle_ts_open_ns == self.ts_open_ns


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
            candle_ts_open_ns=candle.ts_open_ns,
            open=candle.open - tick_bar.open,
            high=candle.high - tick_bar.high,
            low=candle.low - tick_bar.low,
            close=candle.close - tick_bar.close,
            volume=candle.volume - tick_bar.volume,
            ticks=tick_bar.ticks,
        )

    def accepted(self, tick_bar: LiveBar, candle: LiveBar) -> LiveBar:
        """The bar indicators continue from: the candle when it starts at the tick bar's
        open, its drift then kept in `drifts`; else the tick bar, with nothing kept (the
        session asks the broker again). Raises ConfigError as `compare` does."""
        drift = self.compare(tick_bar, candle)
        if not drift.on_time:
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
