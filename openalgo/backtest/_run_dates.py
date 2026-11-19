"""The dates of a run: `start` and `end` as the run reads them (a date, or a datetime with a
time of day, naive in IST), the IST dates and UTC nanosecond instants they name, and the
dates of own data (`data=`) when they are not given.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

from ._series import IST
from .data.loader import own_reader
from .errors import ConfigError


def bound(value: Any) -> Any:
    """A `start` or `end` as the run reads it: a string with a time of day other than
    midnight ("2026-03-10 11:00", or with an offset) is the datetime it names, naive in
    IST, as a datetime object is; any other string is the date it names, and anything
    else is as given."""
    if not isinstance(value, str):
        return value
    moment = pd.Timestamp(value)
    if moment.tzinfo is None and moment == moment.normalize():
        return moment.date()
    return moment.to_pydatetime()


def ist_day(value: Any) -> dt.date:
    """The IST date of a `start` or `end` (`bound`)."""
    value = bound(value)
    if isinstance(value, dt.datetime):
        return (value.astimezone(IST) if value.tzinfo is not None else value).date()
    return value if isinstance(value, dt.date) else pd.Timestamp(value).date()


def instant(value: Any) -> int | None:
    """UTC nanoseconds of a datetime bound (naive is IST, a string with a time of day
    too, `bound`), or None for a plain date."""
    value = bound(value)
    if isinstance(value, dt.datetime):
        moment = value if value.tzinfo is not None else value.replace(tzinfo=IST)
        return int(pd.Timestamp(moment).value)
    return None


def _frame_range(strategy: type, args: dict[str, Any]) -> tuple[dt.date, dt.date]:
    """The IST dates the own data (`data=`) of the strategy's symbols covers, its
    timestamps read as the data layer reads them (naive ones in `tz=`). Raises the data
    layer's errors for naive timestamps without `tz=` or a file it cannot read, and
    ConfigError for data given by a provider that is not own data."""
    data = args["data"]
    symbols = strategy.traded_symbols()
    frames = data if isinstance(data, dict) else dict.fromkeys(symbols, data)
    stamps: list[int] = []
    for symbol in symbols:
        if symbol not in frames:
            raise ConfigError(f"data= has no bars for {symbol}")
        reader = own_reader(frames[symbol], args["tz"])
        if reader is None:
            raise ConfigError(
                "pass start= and end= (IST dates): the run reads its dates only from own "
                "data (a DataFrame, a file path or bt.data.Frame or Files), or a dict of "
                "them by symbol"
            )
        stamps += reader.stamp_range(symbol, strategy.exchange, strategy.interval)
    days = [pd.Timestamp(stamp, tz="UTC").tz_convert(IST).date() for stamp in stamps]
    return min(days), max(days)


def run_dates(strategy: type, args: dict[str, Any]) -> tuple[dt.date, dt.date]:
    """The first and last IST date of a run of `strategy` with the run arguments `args`:
    `start` and `end`, or the dates of own data for one not given. Raises ConfigError
    when neither gives them, or for a start after the end."""
    start, end = args["start"], args["end"]
    if start is None or end is None:
        if args["data"] is None:
            raise ConfigError("pass start= and end= (IST dates)")
        first, last = _frame_range(strategy, args)
        start = first if start is None else start
        end = last if end is None else end
    first_day, last_day = ist_day(start), ist_day(end)
    if first_day > last_day:
        raise ConfigError(f"start {first_day} is after end {last_day}")
    return first_day, last_day


__all__ = ["bound", "instant", "ist_day", "run_dates"]
