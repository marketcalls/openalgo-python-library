"""Session-aligned resampling of bars, equal to OpenAlgo Historify aggregation.

    from openalgo.backtest.resample import resample_bars

    five = resample_bars(one_minute, "5m", exchange="NSE")
    daily = resample_bars(one_minute, "D", exchange="NSE")
    weekly = resample_bars(daily, "W", exchange="NSE")

Bars go in and come out in the canonical layout of `_layout` (`ts_open`, `ts_close` as
int64 UTC nanoseconds; open, high, low, close, volume, oi as float64). A bucket of
interval N starts at IST midnight plus the exchange's regular open plus whole
multiples of N on every date, special sessions included, as on the OpenAlgo server:
buckets are decided by the clock, so missing 1-minute bars and a short last bucket are
handled, and the first bucket of a special session that opens off the lattice starts
before its open. Open is the first, high the maximum, low the minimum, close the last,
volume the sum and open interest the last known value; NaN means unknown and is
skipped.

`ts_close` of a resampled bar is the bucket end, cut at the session close; for daily
and longer bars the bounds are the open of the first session and the close of the last
session of the period, from the calendar.

Where the result differs from the server (the full list is in the kernel's
`oa_bt::data::resample` documentation): bars before the session open or at and after
its close are never merged into a session bucket but dropped; a bucket without a known
price emits no bar; W, M, Q and Y bars carry session bounds rather than 00:00 UTC of
the period's first date. What was dropped is counted in `frame.attrs["dropped"]`:

    pre_open_bars    base bars that end at or before the session open
    post_close_bars  base bars that open at or after the session close
    closed_day_bars  base bars on a date with no session (with on_closed_day="drop")
    empty_buckets    buckets whose bars had no known price
"""

import datetime as dt

import numpy as np
import pandas as pd

from ._calendar_convert import SECONDS_PER_DAY, call_native, check_exchange, from_day, native_attr
from ._layout import BAR_COLUMNS, IST_OFFSET_NANOS, NANOS_PER_SECOND, parse_interval
from ._protocols import SessionCalendar
from .calendar import Calendar, default_calendar
from .errors import ConfigError, DataError, TimestampError

_PERIOD_UNITS = ("W", "M", "Q", "Y")
_CLOSED_DAY_POLICIES = ("error", "drop")
_VALUE_COLUMNS = tuple(column for column in BAR_COLUMNS if not column.startswith("ts_"))
_NANOS_PER_DAY = SECONDS_PER_DAY * NANOS_PER_SECOND
_ONE_WEEK = dt.timedelta(days=7)
_DROP_KEYS = ("pre_open_bars", "post_close_bars", "closed_day_bars", "empty_buckets")


def resample_bars(
    bars: pd.DataFrame,
    interval: str,
    *,
    exchange: str,
    calendar: SessionCalendar | None = None,
    source_interval: str | None = None,
    on_closed_day: str = "error",
) -> pd.DataFrame:
    """Resample canonical bars to a coarser `interval`.

    Supported: an intraday interval to a coarser intraday interval that is a whole
    multiple of it (or the same interval, which only drops bars outside the session
    and recomputes `ts_close`); intraday to "D"; "D" to "W", "M", "Q" or "Y"; and
    intraday to those through daily bars. "D" to "D" returns the bars unchanged after
    checking their order.

    - `exchange`: the OpenAlgo exchange whose sessions bucket the bars. It must match
      `bars.attrs["exchange"]` when that is set.
    - `calendar`: the session calendar, default `default_calendar()`. Any
      `SessionCalendar` works: one that is not a `Calendar` is copied date by date,
      sessions and coverage, over the dates the bars span.
    - `source_interval`: the interval of `bars`, default `bars.attrs["interval"]`.
    - `on_closed_day`: "error" raises for bars on a date with no session; "drop"
      leaves them out and counts them.

    Returns a new frame in the canonical layout with a RangeIndex and the input's
    `attrs`, with "interval" set to `interval`, "exchange" to `exchange` and "dropped"
    to the counts described in the module documentation.

    Raises ConfigError for unknown or incompatible intervals and options, DataError
    for bars that are not in the canonical layout, not sorted or duplicated,
    TimestampError for a missing timestamp (NaT), and CalendarError for an unknown
    exchange or (by default) bars on a date the calendar has no session for; that
    message says how to add the session.
    """
    exchange = _checked_exchange(bars, exchange)
    if on_closed_day not in _CLOSED_DAY_POLICIES:
        raise ConfigError(
            f"on_closed_day must be one of {_CLOSED_DAY_POLICIES}, got {on_closed_day!r}"
        )
    steps = _plan(_source_interval(bars, source_interval), interval)
    if steps:
        columns = _input_columns(bars)
        kernel_calendar = _kernel_calendar(calendar, exchange, columns[0], steps)
        frame, dropped = _run_steps(kernel_calendar, exchange, columns, steps, on_closed_day)
    else:  # D to D: nothing to aggregate
        _check_sorted(_input_columns(bars)[0])
        frame, dropped = bars.loc[:, list(BAR_COLUMNS)].reset_index(drop=True), {}
    frame.attrs.update(bars.attrs)
    frame.attrs.update(exchange=exchange, interval=interval)
    frame.attrs["dropped"] = {key: int(dropped.get(key, 0)) for key in _DROP_KEYS}
    return frame


def _source_interval(bars: pd.DataFrame, source_interval: str | None) -> str:
    source = bars.attrs.get("interval") if source_interval is None else source_interval
    if source is None:
        raise ConfigError(
            "the interval of the input bars is unknown: set bars.attrs['interval'] or pass "
            "source_interval, for example source_interval='1m'"
        )
    return source


def _ist_date(ts_nanos: int) -> dt.date:
    """The IST calendar date of an instant in UTC nanoseconds."""
    return from_day((int(ts_nanos) + IST_OFFSET_NANOS) // _NANOS_PER_DAY)


def _kernel_calendar(
    calendar: SessionCalendar | None, exchange: str, ts_open: np.ndarray, steps: list[tuple]
) -> Calendar:
    """The calendar the kernel resamples with: `calendar` itself, or a copy of its
    sessions and coverage."""
    if calendar is None:
        return default_calendar()
    if isinstance(calendar, Calendar):
        return calendar
    if not callable(getattr(calendar, "session", None)):
        raise ConfigError(
            "calendar must provide session(exchange, day) returning (open_ns, close_ns) or "
            f"None (see SessionCalendar), got {type(calendar).__name__}"
        )
    valid = ts_open[ts_open != np.iinfo(np.int64).min]
    if len(valid) == 0:
        return Calendar(builtin=False)
    first, last = _ist_date(valid.min()), _ist_date(valid.max())
    if any(kind == "period" for kind, *_ in steps):
        # Weekly to yearly bars take their bounds from every session of their period,
        # and a week can reach one week past either end of a year.
        first = dt.date(first.year, 1, 1) - _ONE_WEEK
        last = dt.date(last.year, 12, 31) + _ONE_WEEK
    return Calendar.from_sessions(calendar, exchange, first, last)


def _run_steps(
    calendar: Calendar, exchange: str, columns: tuple, steps: list[tuple], on_closed_day: str
) -> tuple[pd.DataFrame, dict]:
    """Run the native resampling steps in order. Returns canonical bars and the counts
    of what was dropped, summed over the steps.

    `columns` is (ts_open, open, high, low, close, volume, oi) as contiguous arrays.
    """
    dropped = dict.fromkeys(_DROP_KEYS, 0)
    for kind, *arguments in steps:
        function = native_attr(f"resample_{kind}")
        native = calendar._cal  # the package's own handle on the kernel calendar
        output, counts = call_native(function, native, exchange, columns, *arguments, on_closed_day)
        for key in _DROP_KEYS:
            dropped[key] += int(counts[key])
        # A step returns ts_close as its second column; the next step takes none.
        columns = (output[0], *output[2:])
    return pd.DataFrame(dict(zip(BAR_COLUMNS, output, strict=True))), dropped


def _checked_exchange(bars: pd.DataFrame, exchange: str) -> str:
    if not isinstance(bars, pd.DataFrame):
        raise DataError(
            f"bars must be a pandas DataFrame in the canonical layout, got {type(bars).__name__}"
        )
    exchange = check_exchange(exchange)
    declared = bars.attrs.get("exchange")
    if declared is not None and declared != exchange:
        raise ConfigError(
            f"exchange={exchange!r} does not match bars.attrs['exchange']={declared!r}: "
            "resample bars with the calendar of the exchange they trade on"
        )
    return exchange


def _check_sorted(ts_open: np.ndarray) -> None:
    """Raise TimestampError for a missing value (NaT) and DataError unless `ts_open`
    is strictly increasing."""
    missing = ts_open == np.iinfo(np.int64).min
    if np.any(missing):
        row = int(np.argmax(missing))
        raise TimestampError(
            f"row {row} has no usable ts_open (NaT): drop or repair the row before resampling"
        )
    steps = np.diff(ts_open)
    if np.any(steps <= 0):
        row = int(np.argmax(steps <= 0)) + 1
        problem = "repeats" if steps[row - 1] == 0 else "is earlier than"
        raise DataError(
            f"bars are not sorted by ts_open: row {row} {problem} the row before it; sort "
            "the bars and drop duplicates before resampling"
        )


def _input_columns(bars: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """(ts_open, open, high, low, close, volume, oi) as contiguous arrays."""
    missing = [column for column in BAR_COLUMNS if column not in bars.columns]
    if missing:
        raise DataError(
            f"bars are missing the columns {missing}: the canonical layout is {BAR_COLUMNS}"
        )
    ts_open = bars["ts_open"].to_numpy()
    if ts_open.dtype.kind not in "iu":
        raise DataError(
            f"ts_open must be int64 UTC nanoseconds, got dtype {ts_open.dtype}: normalise "
            "timestamps before resampling"
        )
    values = (
        np.ascontiguousarray(bars[column].to_numpy(), dtype=np.float64) for column in _VALUE_COLUMNS
    )
    return (np.ascontiguousarray(ts_open, dtype=np.int64), *values)


def _plan(source: str, target: str) -> list[tuple]:
    """The native resampling steps from `source` to `target`, each (kind, *arguments)."""
    source_kind, source_seconds = parse_interval(source)
    target_kind, target_seconds = parse_interval(target)
    if source_kind == "intraday":
        base_nanos = source_seconds * NANOS_PER_SECOND
        if target_kind == "intraday":
            if target_seconds < source_seconds or target_seconds % source_seconds:
                raise ConfigError(
                    f"cannot resample {source} bars to {target}: the target must be a whole "
                    "multiple of the source interval; fetch finer bars for a finer target"
                )
            return [("intraday", base_nanos, target_seconds * NANOS_PER_SECOND)]
        daily = [("daily", base_nanos)]
        return daily if target_kind == "D" else [*daily, ("period", target_kind)]
    if source_kind == "D" and target_kind in _PERIOD_UNITS:
        return [("period", target_kind)]
    if source_kind == "D" and target_kind == "D":
        return []
    raise ConfigError(
        f"cannot resample {source} bars to {target}: resample from intraday bars to a "
        "coarser intraday interval or to D, W, M, Q, Y, or from D bars to W, M, Q, Y"
    )


__all__ = ["resample_bars"]
