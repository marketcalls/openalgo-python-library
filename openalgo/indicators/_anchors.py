"""
Anchor periods for indicators that restart on a calendar boundary.

A session VWAP restarts every trading day; floor pivots are drawn from the
previous day, week or month. Both need the same thing: which anchor period each
bar belongs to. This module turns bar times and an anchor name into period
boundaries, and nothing else.

Where bar times come from (`bar_times`): an explicit `timestamps` argument, or
the index of the first pandas Series among the price inputs when that index is a
DatetimeIndex, a PeriodIndex, or holds only date or datetime objects. NumPy
arrays, lists and Series with any other index (a RangeIndex, strings) carry no
bar times.

Time is read as local wall-clock time: the date and time of day that a
timezone-aware index shows in its own zone, or the values themselves for a
naive index. A trading day is therefore the exchange's calendar day, not a UTC
day. Plain numbers are Unix epoch seconds, which carry no zone, so they split
on UTC days.

Everything here is causal: the period of bar i depends only on the times of
bars up to i.
"""

import numpy as np
import pandas as pd

__all__ = [
    "ANCHORS",
    "anchor_starts",
    "bar_times",
    "check_anchor",
    "period_ids",
    "previous_period_hlc",
    "wall_seconds",
]

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400
DAYS_PER_WEEK = 7
# 1970-01-01, day 0 of the epoch, was a Thursday. Adding this many days before
# dividing by seven makes week numbers change on Monday.
EPOCH_DAY_TO_MONDAY_WEEK = 3

# The anchors `ta.vwap` documents, spelled as documented: name -> (kind, size).
# "day" and "week" follow the calendar, "months" counts calendar months from
# January, "seconds" is an intraday span measured from the day's first bar.
ANCHORS = {
    "Session": ("day", 1),
    "D": ("day", 1),
    "Week": ("week", 1),
    "Month": ("months", 1),
    "3M": ("months", 3),
    "Quarter": ("months", 3),
    "6M": ("months", 6),
    "12M": ("months", 12),
    "Year": ("months", 12),
    "4H": ("seconds", 4 * SECONDS_PER_HOUR),
    "1H": ("seconds", SECONDS_PER_HOUR),
    "30m": ("seconds", 30 * SECONDS_PER_MINUTE),
    "15m": ("seconds", 15 * SECONDS_PER_MINUTE),
    "5m": ("seconds", 5 * SECONDS_PER_MINUTE),
    "1m": ("seconds", SECONDS_PER_MINUTE),
}

# pandas names for an object index that holds dates or datetimes and nothing else.
_DATE_LIKE = ("date", "datetime", "datetime64")


def check_anchor(anchor):
    """Return (kind, size) for an anchor name; raise ValueError for any other value.

    anchor: one of the names in `ANCHORS`, spelled exactly.
    """
    if isinstance(anchor, str) and anchor in ANCHORS:
        return ANCHORS[anchor]
    raise ValueError(f"Unknown anchor {anchor!r}. Use None or one of: {', '.join(ANCHORS)}.")


def bar_times(*inputs):
    """Bar times carried by the price inputs themselves, or None.

    inputs: the price and volume arguments, in order.
    Returns a DatetimeIndex taken from the index of the first pandas Series
    among them, when that index is a DatetimeIndex, a PeriodIndex (the start of
    each period) or holds only date or datetime objects. Returns None for any
    other index and when no input is a Series.
    """
    index = next((item.index for item in inputs if isinstance(item, pd.Series)), None)
    if isinstance(index, pd.DatetimeIndex):
        return index
    if isinstance(index, pd.PeriodIndex):
        return index.to_timestamp()
    if index is not None and pd.api.types.infer_dtype(index, skipna=False) in _DATE_LIKE:
        try:
            return pd.DatetimeIndex(index)
        except (TypeError, ValueError):
            return None  # for instance, datetimes of several time zones
    return None


def wall_seconds(timestamps):
    """Local wall-clock time of every bar, in whole seconds since 1970-01-01.

    timestamps: one per bar. A DatetimeIndex or datetime Series (naive or
                timezone-aware), a PeriodIndex, a datetime64 array, a list of
                datetimes or date strings, or plain numbers (Unix epoch seconds).
    Returns an int64 array. A timezone-aware input keeps the time its own zone
    shows; numbers are returned as they are, so they are UTC.
    Raises ValueError when a timestamp is missing (NaT or NaN).
    """
    if isinstance(timestamps, pd.PeriodIndex):
        index = timestamps.to_timestamp()
    elif isinstance(timestamps, pd.DatetimeIndex):
        index = timestamps
    else:
        is_pandas = isinstance(timestamps, (pd.Series, pd.Index))
        if is_pandas and pd.api.types.is_datetime64_any_dtype(timestamps):
            index = pd.DatetimeIndex(timestamps)
        else:
            values = timestamps.to_numpy() if is_pandas else np.asarray(timestamps)
            if values.dtype.kind in "iuf":
                if values.dtype.kind == "f" and np.isnan(values).any():
                    raise ValueError("timestamps contain NaN")
                return np.floor(values).astype(np.int64)
            index = pd.DatetimeIndex(values)
    if index.hasnans:
        raise ValueError("timestamps contain NaT")
    if index.tz is not None:
        # Drop the zone but keep what a clock on the wall shows there.
        index = index.tz_localize(None)
    return np.asarray(index.values).astype("datetime64[s]").astype(np.int64)


def period_ids(seconds, anchor):
    """Number the anchor period each bar falls in.

    seconds: wall-clock seconds since 1970-01-01, one per bar (`wall_seconds`).
    anchor:  an anchor name (`check_anchor`).
    Returns an int64 array: two bars are in the same period exactly when their
    ids are equal. Days, weeks (Monday to Sunday) and months follow the
    calendar; quarters, half years and years are 3, 6 and 12 calendar months
    counted from January. Intraday periods ("30m", "4H") restart every day and
    are measured from that day's first bar, so a 4H anchor on a 09:15 open runs
    09:15 to 13:15, the way charts draw it.
    """
    kind, size = check_anchor(anchor)
    seconds = np.asarray(seconds, dtype=np.int64)
    day = seconds // SECONDS_PER_DAY
    if kind == "day":
        return day
    if kind == "week":
        return (day + EPOCH_DAY_TO_MONDAY_WEEK) // DAYS_PER_WEEK
    if kind == "months":
        month = seconds.astype("datetime64[s]").astype("datetime64[M]").astype(np.int64)
        return month // size
    bars = seconds.size
    if bars == 0:
        return day
    opens_day = np.ones(bars, dtype=bool)
    opens_day[1:] = day[1:] != day[:-1]
    first_bar_of_day = np.maximum.accumulate(np.where(opens_day, np.arange(bars), 0))
    span = (seconds - seconds[first_bar_of_day]) // size
    # A day holds fewer than SECONDS_PER_DAY + 1 spans, so (day, span) pairs
    # map to distinct ids.
    return day * (SECONDS_PER_DAY + 1) + span


def anchor_starts(timestamps, anchor):
    """Mark the first bar of every anchor period.

    timestamps: one per bar (see `wall_seconds`).
    anchor:     an anchor name (see `check_anchor`).
    Returns a boolean array, True on bar 0 and on every bar whose period
    differs from the bar before it.
    """
    ids = period_ids(wall_seconds(timestamps), anchor)
    starts = np.ones(ids.size, dtype=bool)
    starts[1:] = ids[1:] != ids[:-1]
    return starts


def previous_period_hlc(high, low, close, starts):
    """High, low and close of the previous anchor period, repeated on every bar.

    high, low, close: prices, one per bar.
    starts:           boolean array, True on the first bar of each period
                      (`anchor_starts`). Bar 0 always opens a period.
    Returns three float64 arrays. On a bar of period k they hold the highest
    high, the lowest low and the last close of period k - 1, so they use no
    bar of the current period. Bars of the first period are NaN. A NaN high
    (or low) anywhere in a period makes that period's high (or low) NaN.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    bars = high.size
    if bars == 0:
        return high.copy(), low.copy(), close.copy()
    starts = np.asarray(starts, dtype=bool).copy()
    starts[0] = True
    first_bar = np.flatnonzero(starts)
    last_bar = np.append(first_bar[1:], bars) - 1
    period_high = np.maximum.reduceat(high, first_bar)
    period_low = np.minimum.reduceat(low, first_bar)
    period_close = close[last_bar]
    previous = np.cumsum(starts) - 2  # period of each bar, minus one
    known = previous >= 0
    previous = np.maximum(previous, 0)
    return (
        np.where(known, period_high[previous], np.nan),
        np.where(known, period_low[previous], np.nan),
        np.where(known, period_close[previous], np.nan),
    )
