"""The column layouts and conversions `Result` builds its tables with (`result`).

The kernel's result columns cross the binding as lists and NumPy arrays: times as int64
UTC nanoseconds, a missing text as None. A `Result` table shows every time as a tz-aware
IST datetime at nanosecond resolution, a missing text as NaN, and has the same columns
whether or not it has rows.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from ._series import IST

# The keys an event can carry, in order: those of every kind of event the binding writes
# (`rust/oa_py/src/backtest/engine/tables.rs`, `event`), so the events table has the
# same columns whatever events a run had.
EVENT_COLUMNS = (
    "slice",
    "slice_ts",
    "step",
    "type",
    "name",
    "order_id",
    "quantity_units",
    "price_rupees",
    "reject_code",
    "instrument",
    "product",
    "position_quantity_units",
    "scope",
    "level_rupees",
    "target_index",
    "observed_rupees",
    "previous_state",
    "trading_state",
    "cause",
    "gtt_id",
    "stop_trigger_rupees",
    "target_trigger_rupees",
    "order_trigger_rupees",
    "fired_leg",
    "child_limit_rupees",
)
# The keys of a final position (`rust/oa_py/src/backtest/engine/results.rs`, `finals`).
FINAL_POSITION_COLUMNS = (
    "instrument",
    "product",
    "quantity_units",
    "average_price_rupees",
    "mark_rupees",
    "unrealized_rupees",
    "stale",
)
# The `Instrument.kind` of a futures contract, whose final position has no market value.
FUTURE_KIND = "future"
# The columns of the orders table that hold a string or nothing. A missing one is NaN in
# an object column, as the events table marks a missing key, whatever the other rows
# hold: left to pandas, a column of None alone stays None while one with a string turns
# None into NaN, so the marker would depend on the run.
ORDER_TEXT_COLUMNS = ("reject_code", "tag", "sizer")
# The `good_until` of a GTT order in the kernel (`exec::GTT_GOOD_UNTIL`, the largest
# int64): no session close ends it.
GTT_GOOD_UNTIL = 2**63 - 1


def text_column(values: list[str | None]) -> pd.Series:
    """An object column of strings with NaN, pandas' missing value, for each None."""
    return pd.Series([math.nan if value is None else value for value in values], dtype=object)


def ist_times(values: Any, open_ended: int | None = None) -> pd.DatetimeIndex:
    """IST times of int64 UTC nanoseconds, at nanosecond resolution even when there are
    none (pandas reads an empty column at second resolution); `open_ended`, the
    kernel's stand-in for "no end", becomes NaT."""
    nanos = pd.Series(values, dtype="int64")
    if open_ended is not None:
        nanos = nanos.astype("Int64").mask(nanos == open_ended)
    utc = pd.to_datetime(nanos, unit="ns", utc=True)
    return pd.DatetimeIndex(utc).as_unit("ns").tz_convert(IST)


def fixed_columns(rows: Any, columns: tuple[str, ...]) -> pd.DataFrame:
    """A table with `columns` first, in order, then any key the rows have beyond them."""
    frame = pd.DataFrame(rows)
    extra = [name for name in frame.columns if name not in columns]
    return frame.reindex(columns=[*columns, *extra])


__all__ = [
    "EVENT_COLUMNS",
    "FINAL_POSITION_COLUMNS",
    "FUTURE_KIND",
    "GTT_GOOD_UNTIL",
    "ORDER_TEXT_COLUMNS",
    "fixed_columns",
    "ist_times",
    "text_column",
]
