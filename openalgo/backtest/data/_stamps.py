"""Convert the timestamps of own data to UTC nanoseconds, never guessing a convention.

The three things a user must say when the data does not say them itself:

- the time zone of naive timestamps (`tz`), with no default;
- the unit of numeric timestamps (`unit`);
- the pattern of timestamp strings that are not ISO 8601 (`fmt`), so that a
  day-first date is never mistaken for a month-first one.

Numeric timestamps are epoch values: they count from 1970-01-01 UTC, so they need no
`tz`, and a `tz` other than UTC contradicts them.
"""

import datetime as _dt
import warnings

import numpy as np
import pandas as pd

from ..errors import ConfigError, TimestampError

EPOCH_UNITS = ("s", "ms", "us", "ns")

# What pandas, zoneinfo and the conversion to nanoseconds raise for bad input.
_CONVERSION_ERRORS = (ValueError, TypeError, KeyError, OverflowError)


def to_utc_ns(
    values: object,
    *,
    tz: str | None = None,
    unit: str | None = None,
    fmt: str | None = None,
    what: str = "timestamps",
) -> tuple[np.ndarray, str]:
    """Convert user-supplied timestamps to int64 UTC nanoseconds.

    values: a DatetimeIndex, or a Series, Index, list or array of datetimes, epoch
        numbers or strings.
    tz: time zone of naive values, for example "Asia/Kolkata". Ignored for tz-aware
        values that agree with it, and for epoch numbers when it is UTC.
    unit: "s", "ms", "us" or "ns" for numeric values (epoch values, UTC).
    fmt: strptime pattern for strings that are not ISO 8601.
    what: how to name the values in error messages.

    Returns (nanoseconds, a description of the convention that was applied).

    Raises ConfigError when a needed declaration (`tz`, `unit`, `fmt`) is missing or
    contradicts the data, and TimestampError when values are missing, cannot be
    parsed, do not exist in the time zone or do not fit in nanosecond precision.
    """
    index = _to_datetime_index(values, unit=unit, fmt=fmt, tz=tz, what=what)
    if index.hasnans:
        missing = int(index.isna().sum())
        raise TimestampError(f"{what}: {missing} values are missing or unparseable")
    if index.tz is not None:
        _check_declared_zone(index, tz, what)
        described = f"tz-aware ({index.tz})"
    else:
        index = _localise(index, tz, what)
        described = f"naive, declared tz={tz}"
    try:
        nanos = index.tz_convert("UTC").as_unit("ns").asi8
    except _CONVERSION_ERRORS as exc:
        raise TimestampError(f"{what} do not fit in nanosecond precision: {exc}") from exc
    return np.asarray(nanos, dtype="int64"), described


def _check_epoch_zone(index: pd.DatetimeIndex, tz: str | None, what: str) -> None:
    """Epoch values are UTC instants: `tz` may be omitted or name a zone at UTC for
    every one of them, anything else contradicts them."""
    if tz is None or len(index) == 0:
        return
    try:
        local = index.tz_convert(tz)
    except _CONVERSION_ERRORS as exc:
        raise ConfigError(f"tz={tz!r} is not a known time zone ({exc})") from exc
    if (local.tz_localize(None) != index.tz_localize(None)).any():
        raise ConfigError(
            f"{what} are epoch numbers, which count from 1970-01-01 UTC, and tz={tz!r} says "
            "something else. Drop tz= (epoch numbers need none), or convert the data to "
            "true epoch values first."
        )


def _check_declared_zone(index: pd.DatetimeIndex, tz: str | None, what: str) -> None:
    if tz is None or len(index) == 0:
        return
    try:
        declared_offset = index[:1].tz_convert(tz)[0].utcoffset()
    except _CONVERSION_ERRORS as exc:
        raise ConfigError(f"tz={tz!r} is not a known time zone ({exc})") from exc
    if declared_offset != index[0].utcoffset():
        raise ConfigError(
            f"{what} are already time-zone aware ({index.tz}) and tz={tz!r} says something "
            "else. Drop tz=, or convert the data first."
        )


def _localise(index: pd.DatetimeIndex, tz: str | None, what: str) -> pd.DatetimeIndex:
    if tz is None:
        raise ConfigError(
            f"{what} carry no time zone. Pass tz= (for example tz='Asia/Kolkata' for IST "
            "clock times, tz='UTC' for UTC): there is no default, because a wrong guess "
            "shifts every bar by 5 hours 30 minutes."
        )
    try:
        return index.tz_localize(tz, ambiguous="raise", nonexistent="raise")
    except _CONVERSION_ERRORS as exc:
        raise TimestampError(f"{what} cannot be placed in time zone {tz!r}: {exc}") from exc


def _to_datetime_index(
    values: object, *, unit: str | None, fmt: str | None, tz: str | None, what: str
) -> pd.DatetimeIndex:
    if isinstance(values, pd.DatetimeIndex):
        return values
    series = values if isinstance(values, (pd.Series, pd.Index)) else pd.Series(values)
    dtype = series.dtype
    if isinstance(dtype, pd.DatetimeTZDtype) or pd.api.types.is_datetime64_any_dtype(dtype):
        return pd.DatetimeIndex(series)
    if pd.api.types.is_bool_dtype(dtype):
        raise TimestampError(f"{what} are booleans, not timestamps")
    if pd.api.types.is_numeric_dtype(dtype):
        index = _from_numbers(series, unit, what)
        _check_epoch_zone(index, tz, what)
        return index
    if len(series) == 0:
        return pd.DatetimeIndex([])
    if isinstance(series.iloc[0], (_dt.datetime, _dt.date, np.datetime64)):
        return _from_datetime_objects(series, what)
    return _from_text(series, fmt, what)


def _from_numbers(series: pd.Series, unit: str | None, what: str) -> pd.DatetimeIndex:
    if unit not in EPOCH_UNITS:
        raise ConfigError(
            f"{what} are numbers. Pass unit= ('s', 'ms', 'us' or 'ns') to say what they "
            "count; they are read as epoch values, from 1970-01-01 UTC."
        )
    try:
        return pd.DatetimeIndex(pd.to_datetime(series.to_numpy(), unit=unit, utc=True))
    except _CONVERSION_ERRORS as exc:
        raise TimestampError(f"{what} could not be read as epoch {unit}: {exc}") from exc


def _from_datetime_objects(series: pd.Series, what: str) -> pd.DatetimeIndex:
    try:
        return pd.DatetimeIndex(pd.to_datetime(list(series)))
    except _CONVERSION_ERRORS as exc:
        raise TimestampError(f"{what} mix time zones or types: {exc}") from exc


def _from_text(series: pd.Series, fmt: str | None, what: str) -> pd.DatetimeIndex:
    if series.isna().any():
        raise TimestampError(f"{what}: {int(series.isna().sum())} values are missing")
    texts = series.astype(str).to_numpy(dtype=object)
    if fmt is not None:
        try:
            return pd.DatetimeIndex(pd.to_datetime(texts, format=fmt))
        except _CONVERSION_ERRORS as exc:
            raise TimestampError(f"{what} do not match timestamp_format={fmt!r}: {exc}") from exc
    try:
        return _parse_iso(texts)
    except _CONVERSION_ERRORS as exc:
        raise ConfigError(
            f"{what} are not ISO 8601 (first value {texts[0]!r}). Pass timestamp_format= with "
            "a strptime pattern such as '%d-%m-%Y %H:%M:%S'; the day and month order is "
            "never guessed."
        ) from exc


def _parse_iso(texts: np.ndarray) -> pd.DatetimeIndex:
    """Parse ISO 8601 strings; strings with more than one UTC offset become UTC."""
    with warnings.catch_warnings():
        # pandas 2.2 warns, and pandas 3 raises, when the strings mix UTC offsets;
        # both cases are settled by the utc=True parse below.
        warnings.simplefilter("ignore", FutureWarning)
        try:
            parsed = pd.to_datetime(texts, format="ISO8601")
            if isinstance(parsed, pd.DatetimeIndex):
                return parsed
        except ValueError:
            pass
        return pd.DatetimeIndex(pd.to_datetime(texts, format="ISO8601", utc=True))
