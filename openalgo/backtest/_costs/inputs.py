"""Turning user input into what the kernel takes.

Scalars: a trade date becomes an IST day number (days since 1970-01-01), a side its
kernel id, a segment its upper-case code. Arrays: day numbers as int32, sides and
segments as uint8 ids, numbers as float64, all contiguous and of one common length.
Anything that cannot be read raises CostError; nothing is guessed.

Two rules have exactly one implementation here, used by the scalar and the array
paths alike: `to_day` reads a date (a string must be an ISO date, YYYY-MM-DD), and
`side_ids` turns a side into the kernel's 0 (buy) and 1 (sell).
"""

import datetime as dt
import re
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from .._layout import IST_OFFSET_NANOS
from .._messages import counted
from ..errors import CostError
from .kernel import MAX_DAY, MIN_DAY, SEGMENTS, call, native

__all__ = [
    "DateLike",
    "common_length",
    "day_to_date",
    "floats",
    "number",
    "segment_code",
    "segment_ids",
    "side_ids",
    "to_day",
    "to_days",
]

#: What a trade date may be given as. A string must be an ISO date ("2026-04-01").
DateLike = dt.date | np.datetime64 | str

_IST = dt.timezone(dt.timedelta(microseconds=IST_OFFSET_NANOS // 1000))
_EPOCH_ORDINAL = dt.date(1970, 1, 1).toordinal()
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

# Ids of the kernel's Side enum. `side_ids` is the only place that produces them.
_SIDE_IDS = MappingProxyType({"BUY": 0, "SELL": 1})
_SIDE_HELP = "use 'BUY' or 'SELL', or the signed numbers +1 (buy) and -1 (sell)"


def number(name: str, value: Any) -> float:
    """A real number as a float.

    Raises CostError for anything that is not a number (booleans included) or is too
    large for a float.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise CostError(f"{name} must be a number, got {value!r}")
    try:
        return float(value)
    except OverflowError:
        raise CostError(f"{name} is too large to be a number of rupees or units") from None


def day_to_date(day: int | None) -> dt.date | None:
    """The date of an IST day number, or None for None (an open-ended range)."""
    return None if day is None else dt.date.fromordinal(int(day) + _EPOCH_ORDINAL)


def _supported(day: int, name: str) -> int:
    if not MIN_DAY <= day <= MAX_DAY:
        first, last = day_to_date(MIN_DAY), day_to_date(MAX_DAY)
        raise CostError(f"{name} is outside the supported range {first} to {last}")
    return day


def _unchecked_day(value: Any, name: str) -> int:
    if value is None or value is pd.NaT or (isinstance(value, np.datetime64) and np.isnat(value)):
        raise CostError(f"{name} is missing")
    if isinstance(value, dt.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(_IST)
        return value.date().toordinal() - _EPOCH_ORDINAL
    if isinstance(value, dt.date):
        return value.toordinal() - _EPOCH_ORDINAL
    if isinstance(value, np.datetime64):
        return int(value.astype("datetime64[D]").astype(np.int64))
    if isinstance(value, str) and _ISO_DATE.fullmatch(value.strip()):
        try:
            return dt.date.fromisoformat(value.strip()).toordinal() - _EPOCH_ORDINAL
        except ValueError:
            pass  # A well-shaped string that is not a real date, such as 2026-02-30.
    raise CostError(
        f"{name} {value!r} is not a date: use a date, a datetime, a pandas Timestamp, a numpy "
        "datetime64, or an ISO date string such as '2026-04-01' (year first)"
    )


def to_day(value: DateLike, name: str = "date") -> int:
    """An IST calendar date as days since 1970-01-01.

    Accepts a date, a datetime or pandas Timestamp (a timezone-aware one is converted
    to IST first; a naive one is taken as IST), a numpy datetime64, or a string in
    ISO form, YYYY-MM-DD. Day-first and month-first strings are rejected, not
    guessed. Raises CostError for a missing value, any other type, or a date outside
    the supported range.
    """
    return _supported(_unchecked_day(value, name), name)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, bytes, dt.date, np.datetime64)) or np.ndim(value) == 0


def common_length(*values: Any) -> int:
    """The shared length of the array-like arguments; scalars and length 1 broadcast.

    Raises CostError when two arrays have different lengths.
    """
    length = 1
    for value in values:
        if _is_scalar(value):
            continue
        size = len(value)
        if length == 1:
            length = size
        elif size not in (1, length):
            raise CostError(f"inputs have different lengths ({length} and {size})")
    return length


def _fit(array: np.ndarray, length: int, name: str) -> np.ndarray:
    """A contiguous one-dimensional array of `length`, repeating a single element."""
    if array.ndim != 1:
        raise CostError(f"{name} must be one-dimensional")
    if array.shape[0] == 1 and length != 1:
        array = np.repeat(array, length)
    if array.shape[0] != length:
        raise CostError(f"{name} has {counted(array.shape[0], 'element')}, expected {length}")
    return np.ascontiguousarray(array)


def floats(values: Any, length: int, name: str) -> np.ndarray:
    """A float64 array of `length` from a number or an array-like of numbers.

    Raises CostError when the values are not numbers or do not fit a float.
    """
    try:
        array = np.atleast_1d(np.asarray(values, dtype=np.float64))
    except (ValueError, TypeError, OverflowError):
        raise CostError(f"{name} must be numeric") from None
    return _fit(array, length, name)


def _datetime_days(values: Any, name: str) -> np.ndarray | None:
    """Day numbers (int64) of an array that already holds datetimes, else None."""
    if isinstance(values, (pd.Series, pd.Index)) and pd.api.types.is_datetime64_any_dtype(values):
        index = pd.DatetimeIndex(values)
        if index.tz is not None:
            index = index.tz_convert(_IST).tz_localize(None)
        stamps = index.values
    else:
        stamps = np.asarray(values)
        if stamps.dtype.kind != "M":
            return None
    if np.isnat(stamps).any():
        raise CostError(f"{name} contains missing dates (NaT)")
    return stamps.astype("datetime64[D]").astype(np.int64)


def to_days(values: Any, length: int, name: str = "date") -> np.ndarray:
    """An int32 array of IST day numbers of `length` from a date or an array-like of dates.

    A datetime64 array, DatetimeIndex or datetime Series is converted in bulk
    (timezone-aware values to IST first). Anything else is read element by element
    with `to_day`, so the same rules apply as for a single date: a string must be an
    ISO date. Raises CostError for a missing or unreadable date, a date outside the
    supported range, or a length mismatch.
    """
    if _is_scalar(values):
        return np.full(length, to_day(values, name), dtype=np.int32)
    days = _datetime_days(values, name)
    if days is None:
        elements = np.asarray(values, dtype=object)
        if elements.ndim != 1:
            raise CostError(f"{name} must be one-dimensional")
        days = np.array([_unchecked_day(element, name) for element in elements], dtype=np.int64)
    if days.size:
        _supported(int(days.min()), name)
        _supported(int(days.max()), name)
    return _fit(days.astype(np.int32), length, name)


def _side_id(side: Any) -> int:
    """The kernel id of one side given as text or as exactly +1 or -1."""
    if isinstance(side, str):
        text = side.strip().upper()
        if text in _SIDE_IDS:
            return _SIDE_IDS[text]
    signed = isinstance(side, (int, float, np.signedinteger, np.floating)) and not isinstance(
        side, bool
    )
    if signed and side == 1:
        return _SIDE_IDS["BUY"]
    if signed and side == -1:
        return _SIDE_IDS["SELL"]
    raise CostError(f"unknown side {side!r}: {_SIDE_HELP}")


def side_ids(side: Any, length: int) -> np.ndarray:
    """A uint8 array of the kernel's side ids (0 buy, 1 sell) of `length`.

    This is the one place a side is converted. A side is "BUY" or "SELL" (any case),
    or the signed number +1 (buy) or -1 (sell). Anything else raises CostError,
    including 0, other magnitudes, booleans and unsigned integers: an array of
    unsigned 0s and 1s looks like the kernel's own ids and is refused rather than
    read as two buys.
    """
    if _is_scalar(side):
        return np.full(length, _side_id(side), dtype=np.uint8)
    if isinstance(side, (pd.Series, pd.Index)):
        side = side.to_numpy()
    # A list keeps its elements as they are, so ["BUY", -1] is not turned into text.
    array = side if isinstance(side, np.ndarray) else np.asarray(side, dtype=object)
    if array.dtype.kind in "ub":
        raise CostError(f"side: unsigned and boolean arrays are not accepted; {_SIDE_HELP}")
    if array.dtype.kind in "if":
        buys, sells = array == 1, array == -1
        if not np.all(buys | sells):
            raise CostError(f"side: every numeric side must be +1 or -1; {_SIDE_HELP}")
        ids = np.where(buys, _SIDE_IDS["BUY"], _SIDE_IDS["SELL"]).astype(np.uint8)
    else:
        ids = np.array([_side_id(element) for element in array.ravel()], dtype=np.uint8)
        ids = ids.reshape(array.shape)
    return _fit(ids, length, "side")


def segment_code(segment: Any) -> str:
    """A cost segment code, upper-cased and checked against the kernel's table.

    Raises CostError for a code that is unknown or names a segment that is not costed
    yet (currency derivatives, MCX options).
    """
    if not isinstance(segment, str):
        raise CostError(f"segment must be a code such as 'NSE_EQ_DELIVERY', got {segment!r}")
    return SEGMENTS[call(native.cost_segment_id, segment.strip().upper())]


def segment_ids(segment: Any, length: int) -> np.ndarray:
    """A uint8 array of segment ids of `length` from one code or an array-like.

    An array may hold codes or integer ids. Raises CostError for an unknown code or an
    id outside the segment table.
    """
    if isinstance(segment, str):
        return np.full(length, SEGMENTS.index(segment_code(segment)), dtype=np.uint8)
    array = np.atleast_1d(np.asarray(segment))
    if array.dtype.kind in "iu":
        if array.size and (array.min() < 0 or array.max() >= len(SEGMENTS)):
            raise CostError(f"segment ids must be between 0 and {len(SEGMENTS) - 1}")
        ids = array.astype(np.uint8)
    else:
        codes, inverse = np.unique(array.astype(str), return_inverse=True)
        known = [SEGMENTS.index(segment_code(code)) for code in codes]
        ids = np.array(known, dtype=np.uint8)[inverse]
    return _fit(ids, length, "segment")
