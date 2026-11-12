"""Conversions between Python values and what the native calendar takes.

The native calendar (`openalgo._oaindicators.backtest.Calendar`) speaks in exchange
codes, IST dates as days since 1970-01-01, session times as seconds since IST
midnight and instants as int64 UTC nanoseconds. Everything that turns dates, clock
times, intervals and bar columns into those forms lives here, together with the
translation of native errors into the package's public error types.

The native side checks every day number against its supported range (1677 to 2262,
where every instant fits in int64 nanoseconds) and every timestamp array for missing
values (NaT is the int64 minimum); both raise the public error types through
`call_native`, never a crash.
"""

import datetime as dt
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ._layout import NANOS_PER_SECOND, parse_interval
from ._native import native
from ._protocols import COVERAGE_LEVELS
from .errors import CalendarError, ConfigError, DataError, TimestampError

DateLike = dt.date | str | np.datetime64

IST = dt.timezone(dt.timedelta(hours=5, minutes=30), "IST")
SECONDS_PER_DAY = 86_400
_EPOCH_ORDINAL = dt.date(1970, 1, 1).toordinal()


def native_attr(name: str) -> Any:
    """An attribute of the native backtest module, with a clear error when it is absent."""
    try:
        return getattr(native, name)
    except AttributeError:
        raise ImportError(
            f"the installed openalgo native module has no backtest.{name}: it is older "
            "than this Python package. Reinstall openalgo, or rebuild it from source with "
            "`maturin develop`."
        ) from None


# The exchange codes come from the kernel's own list, so they cannot drift from it.
EXCHANGES: tuple[str, ...] = tuple(native_attr("EXCHANGE_CODES"))
# Exchanges with their own holiday list (the index exchanges follow NSE and BSE).
HOLIDAY_EXCHANGES: tuple[str, ...] = tuple(native_attr("CALENDAR_EXCHANGE_CODES"))
# The exchange whose calendar each exchange follows.
_OWNERS: dict[str, str] = dict(native_attr("CALENDAR_OWNERS"))


def call_native(func: Callable[..., Any], *args: Any) -> Any:
    """Call into the native module, converting its errors to the public types.

    Raises CalendarError for a calendar argument the kernel rejects, TimestampError for
    a series that fails validation, CalendarError for bars on a date with no session
    and DataError for bars that cannot be resampled.
    """
    try:
        return func(*args)
    except native_attr("NativeCalendarError") as exc:
        raise CalendarError(str(exc)) from None
    except native_attr("NativeSeriesError") as exc:
        raise TimestampError(str(exc)) from None
    except native_attr("NativeClosedDayError") as exc:
        raise CalendarError(str(exc)) from None
    except native_attr("NativeResampleError") as exc:
        raise DataError(str(exc)) from None


def check_exchange(value: str) -> str:
    """Return `value` when it is an OpenAlgo exchange code, else raise CalendarError."""
    if isinstance(value, str) and value in EXCHANGES:
        return value
    raise CalendarError(
        f"unknown exchange {value!r}: use one of {', '.join(EXCHANGES)} (upper case)"
    )


def calendar_owner(exchange: str) -> str:
    """The exchange whose calendar `exchange` follows: NSE for NSE_INDEX, BSE for
    BSE_INDEX, else the exchange itself. Raises CalendarError for an unknown code."""
    return _OWNERS[check_exchange(exchange)]


def exchange_list(value: str | list[str] | tuple[str, ...]) -> list[str]:
    """One exchange or a sequence of exchanges as a checked, non-empty list."""
    if isinstance(value, str):
        return [check_exchange(value)]
    exchanges = [check_exchange(item) for item in value]
    if not exchanges:
        raise CalendarError("no exchanges given: pass at least one exchange code")
    return exchanges


def to_day(value: DateLike) -> int:
    """Days since 1970-01-01 for an IST date.

    Accepts a `datetime.date`, an ISO string ("2026-04-01"), a `numpy.datetime64`, or
    a `datetime.datetime` or `pandas.Timestamp` (a time zone aware value is converted
    to IST first; a naive one is taken as IST). Raises ConfigError for anything else.
    """
    if isinstance(value, dt.datetime):  # includes pandas.Timestamp
        if value.tzinfo is not None:
            value = value.astimezone(IST)
        return value.date().toordinal() - _EPOCH_ORDINAL
    if isinstance(value, dt.date):
        return value.toordinal() - _EPOCH_ORDINAL
    if isinstance(value, np.datetime64):
        return int(value.astype("datetime64[D]").astype(np.int64))
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value).toordinal() - _EPOCH_ORDINAL
        except ValueError:
            pass
    raise ConfigError(
        f"cannot read {value!r} as a date: pass a datetime.date or an ISO string such "
        "as '2026-04-01'"
    )


def from_day(day: int) -> dt.date:
    """The `datetime.date` for a day number (days since 1970-01-01)."""
    return dt.date.fromordinal(int(day) + _EPOCH_ORDINAL)


def to_seconds(value: str | dt.time | int, *, what: str) -> int:
    """Seconds since IST midnight from "HH:MM", "HH:MM:SS", a `datetime.time` or an int.

    `what` names the value in the error message ("session open"). 24:00 is accepted so
    a session may close at midnight. Raises ConfigError for anything else.
    """
    seconds = _parse_seconds(value)
    if seconds is None or not 0 <= seconds <= SECONDS_PER_DAY:
        raise ConfigError(
            f"cannot read {value!r} as the {what} time: use 'HH:MM' in IST, for example "
            "'09:15' (24:00 is allowed as a close)"
        )
    return seconds


def _parse_seconds(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, dt.time):
        return value.hour * 3600 + value.minute * 60 + value.second
    if not isinstance(value, str):
        return None
    parts = value.split(":")
    # isdecimal, not isdigit: a superscript digit is a digit int() cannot read.
    if len(parts) not in (2, 3) or not all(part.isdecimal() for part in parts):
        return None
    hours, minutes, seconds = (*map(int, parts), 0)[:3]
    if minutes > 59 or seconds > 59:
        return None
    return hours * 3600 + minutes * 60 + seconds


def clock(seconds: int) -> str:
    """ "HH:MM" for seconds since IST midnight."""
    return f"{seconds // 3600:02d}:{(seconds // 60) % 60:02d}"


def format_ist(ts_nanos: int) -> str:
    """ "YYYY-MM-DD HH:MM IST" for an instant in UTC nanoseconds."""
    epoch = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)
    stamp = epoch + dt.timedelta(microseconds=ts_nanos // 1000)
    return stamp.astimezone(IST).strftime("%Y-%m-%d %H:%M IST")


def coverage_code(level: str) -> int:
    """The native code of a coverage level name (its position in COVERAGE_LEVELS).

    Raises ConfigError for an unknown name."""
    if level in COVERAGE_LEVELS:
        return COVERAGE_LEVELS.index(level)
    raise ConfigError(f"unknown coverage level {level!r}: use one of {COVERAGE_LEVELS}")


def interval_nanos(interval: str) -> int | None:
    """Nanoseconds of an intraday interval string, or None for "D".

    Raises ConfigError for an interval that cannot be a base interval ("W", "M", ...).
    """
    kind, count = parse_interval(interval)
    if kind == "intraday":
        return count * NANOS_PER_SECOND
    if kind == "D":
        return None
    raise ConfigError(
        f"interval {interval!r} cannot be a base interval: use an intraday interval "
        "such as '1m' or '15m', or 'D'"
    )


def ts_open_column(bars: pd.DataFrame | np.ndarray | pd.Series | list) -> np.ndarray:
    """Bar opens as a contiguous int64 array of UTC nanoseconds.

    `bars` is a bars frame (its `ts_open` column is used) or an array of instants as
    integers or datetime64. Raises ConfigError for a missing column or another dtype.
    """
    values = bars
    if isinstance(bars, pd.DataFrame):
        if "ts_open" not in bars.columns:
            raise ConfigError("bars must have a ts_open column (int64 UTC nanoseconds)")
        values = bars["ts_open"]
    array = np.asarray(values)
    if array.dtype.kind == "M":
        array = array.astype("datetime64[ns]").astype(np.int64)
    elif array.dtype.kind not in "iu":
        raise ConfigError(
            f"ts_open must be int64 UTC nanoseconds, got dtype {array.dtype}: convert "
            "timestamps before using them here"
        )
    return np.ascontiguousarray(array, dtype=np.int64)


__all__ = [
    "COVERAGE_LEVELS",
    "EXCHANGES",
    "HOLIDAY_EXCHANGES",
    "IST",
    "SECONDS_PER_DAY",
    "DateLike",
    "calendar_owner",
    "call_native",
    "check_exchange",
    "clock",
    "coverage_code",
    "exchange_list",
    "format_ist",
    "from_day",
    "interval_nanos",
    "native_attr",
    "to_day",
    "to_seconds",
    "ts_open_column",
]
