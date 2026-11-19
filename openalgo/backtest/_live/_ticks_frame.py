"""Recorded ticks read back as one table (the reader behind `bt.read_ticks`): a day's
recordings in the cache, merged in the exact order they were received (`recorder.
read_merged_ticks`: by recording number, then sequence number), one row per tick.

The columns, in order:

- `exchange`, `symbol`: the instrument, as OpenAlgo writes it.
- `exchange_time`: the last trade time the feed gave, IST; the receive time where it gave
  none (or one the tick adapter found implausible, `ticks`).
- `receive_time`: when the recording took the tick, IST: the order a session receives
  ticks in, which a replay follows.
- `exchange_time_given`: True where `exchange_time` is the feed's, False where it is the
  receive time.
- `ltp`: rupees; `volume` (the day's cumulative units) when the day's recordings kept
  one, as the quote and depth modes do; `bid` and `ask` (rupees) when they kept them, as
  the depth mode does. A row without a value of a kept field holds NaN.

A recording that is not finalized yet for one of the instruments read (it runs, or it
stopped before finalizing: the next recording of the day finalizes it) is left out for
all of them, so the table never holds half a recording; its numbers are in
`frame.attrs["left_out"]`.

Times are tz-aware IST timestamps, prices rupees, volume units. Never writes.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import pandas as pd

from .._calendar_convert import IST, DateLike, from_day, to_day
from ..errors import DataError
from ._tick_files import finalized_numbers, instrument_folders, instrument_of, ticks_root
from ._types import Tick
from .recorder import read_merged_ticks

#: The fields a recording keeps only in some feed modes: each a column once any tick of
#: the day has a value of it.
OPTIONAL_FIELDS = ("volume", "bid", "ask")


def ticks_frame(
    day: DateLike,
    instruments: Sequence[Mapping[str, str]] | None = None,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
) -> pd.DataFrame:
    """The recorded ticks of the IST `day` (module docstring), of `instruments`
    (`[{"exchange": "NSE", "symbol": "SBIN"}, ...]`; every instrument recorded that day
    when None), from `cache_dir` (the package's cache folder when None).

    Raises ConfigError for a day that is not a date, unusable instruments or a relative
    cache folder; DataError when the day has no recording (naming the folder), when a
    named instrument has none that day, and when a recorded file is damaged (naming it);
    OSError naming the path when a file cannot be read."""
    root = ticks_root(cache_dir)
    date = from_day(to_day(day))
    wanted = list(instruments) if instruments is not None else _recorded(root, date)
    if not wanted:
        raise DataError(
            f"no ticks were recorded on {date} in {root}: record them with bt.record, or "
            "pass cache_dir= the folder they were recorded in"
        )
    merged = read_merged_ticks(wanted, date, cache_dir=cache_dir)
    frame = _table(merged.ticks)
    frame.attrs["left_out"] = merged.left_out
    return frame


def _recorded(root: os.PathLike[str], date: object) -> list[dict[str, str]]:
    """Every instrument with a finalized file of the day, as `read_merged_ticks` takes
    them."""
    found = []
    for folder in instrument_folders(root):
        if finalized_numbers(folder, date):
            exchange, symbol = instrument_of(folder)
            found.append({"exchange": exchange, "symbol": symbol})
    return found


def _table(ticks: list[Tick]) -> pd.DataFrame:
    """The ticks as the module docstring's columns."""
    received = _ist([tick.ts_recv_ns for tick in ticks])
    given = [tick.ts_event_ns is not None for tick in ticks]
    event = [tick.ts_recv_ns if tick.ts_event_ns is None else tick.ts_event_ns for tick in ticks]
    columns: dict[str, object] = {
        "exchange": pd.Series([tick.exchange for tick in ticks], dtype=object),
        "symbol": pd.Series([tick.symbol for tick in ticks], dtype=object),
        "exchange_time": _ist(event),
        "receive_time": received,
        "exchange_time_given": pd.Series(given, dtype=bool),
        "ltp": pd.Series([tick.ltp for tick in ticks], dtype="float64"),
    }
    for field in OPTIONAL_FIELDS:
        values = [getattr(tick, field) for tick in ticks]
        if any(value is not None for value in values):
            columns[field] = pd.Series(values, dtype="float64")
    return pd.DataFrame(columns)


def _ist(nanos: list[int]) -> pd.Series:
    """UTC nanoseconds as tz-aware IST timestamps."""
    return pd.to_datetime(pd.Series(nanos, dtype="int64"), utc=True).dt.tz_convert(IST)


__all__ = ["OPTIONAL_FIELDS", "ticks_frame"]
