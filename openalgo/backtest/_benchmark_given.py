"""Closes given as `benchmark=`: a pandas Series, or a DataFrame with a close column,
read into increasing UTC nanosecond stamps and positive levels (the rules are in
`_benchmark`).

An index whose times are all midnight (naive, or in its own time zone) holds dates: each
date's close is stamped at the close of that date's session of the strategy's exchange,
the instant the run's last equity mark of the session is taken, and a date with no
session there is left out. Any other index must be time-zone-aware (or int64 UTC
nanoseconds) and is used as given: a naive time could be read 5 hours 30 minutes off.
A DataFrame with a `ts_close` column (the data layer's bars layout) is stamped by it. A
missing (NaN) close is left out; the coverage warning of `_benchmark` names any session
of the run that is then without one.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .errors import ConfigError
from .report._inputs import benchmark_columns, utc_nanos

# The bars layout's close stamp (`_layout`), used when a DataFrame of bars is given.
TS_CLOSE = "ts_close"
CLOSE = "close"


@dataclass(frozen=True)
class GivenCloses:
    """Given closes, read: `stamps_ns` (int64 UTC nanoseconds, increasing), `levels`
    (positive index or price levels) and how many dates had no session of the
    strategy's exchange and were left out."""

    stamps_ns: np.ndarray
    levels: np.ndarray
    off_session_dates: int

    def content_hash(self) -> str:
        """SHA-256 of the stamps and levels as read, for the run manifest."""
        digest = hashlib.sha256(self.stamps_ns.astype(np.int64).tobytes())
        digest.update(self.levels.astype(np.float64).tobytes())
        return f"sha256:{digest.hexdigest()}"


def read_given(value: pd.Series | pd.DataFrame, exchange: str, calendar: Any) -> GivenCloses:
    """Read closes given as `benchmark=` (see the module docstring); dates are stamped
    with the sessions of `exchange` in `calendar`. Raises ConfigError for a DataFrame
    without one close column, closes that are not numbers or not positive levels, a
    missing or naive (not midnight) timestamp, and stamps that do not increase."""
    stamps, closes = _columns(value)
    levels = _numbers(closes)
    stamps_ns, on_session = _stamps(stamps, exchange, calendar)
    keep = on_session & ~np.isnan(levels)
    read = GivenCloses(stamps_ns[keep], levels[keep], int(np.sum(~on_session)))
    if len(read.levels):
        benchmark_columns(pd.Series(read.levels, index=pd.Index(read.stamps_ns)))
    return read


def _columns(value: pd.Series | pd.DataFrame) -> tuple[Any, pd.Series]:
    """(the stamps, the closes) of a given Series or DataFrame."""
    if isinstance(value, pd.Series):
        return value.index, value
    names = [column for column in value.columns if str(column).lower() == CLOSE]
    if len(names) != 1:
        raise ConfigError(
            "benchmark= as a DataFrame needs one close column (positive index or price "
            f"levels), got the columns {list(value.columns)}"
        )
    stamps = value[TS_CLOSE] if TS_CLOSE in value.columns else value.index
    return stamps, value[names[0]]


def _numbers(closes: pd.Series) -> np.ndarray:
    try:
        return np.asarray(pd.to_numeric(closes, errors="raise"), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"benchmark closes must be numbers: {exc}") from None


def _stamps(stamps: Any, exchange: str, calendar: Any) -> tuple[np.ndarray, np.ndarray]:
    """(UTC nanoseconds, True where usable): times as given, or dates stamped at their
    session close (False for a date with no session)."""
    if pd.api.types.is_integer_dtype(getattr(stamps, "dtype", None)):
        nanos, missing = utc_nanos(stamps, what="benchmark timestamps")
        _refuse_missing(bool(missing.any()))
        return nanos, np.ones(len(nanos), dtype=bool)
    try:
        times = pd.DatetimeIndex(stamps)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"benchmark timestamps must be times, dates or int64 UTC nanoseconds: {exc}"
        ) from None
    _refuse_missing(bool(times.hasnans))
    if len(times) and bool((times == times.normalize()).all()):
        return _session_closes(times, exchange, calendar)
    nanos, _ = utc_nanos(times, what="benchmark timestamps")
    return nanos, np.ones(len(nanos), dtype=bool)


def _refuse_missing(missing: bool) -> None:
    if missing:
        raise ConfigError("benchmark has a missing timestamp: every close needs one")


def _session_closes(
    times: pd.DatetimeIndex, exchange: str, calendar: Any
) -> tuple[np.ndarray, np.ndarray]:
    """The session close of each date in UTC nanoseconds (0 where the date has no
    session), and True where it has one."""
    sessions = [calendar.session(exchange, day) for day in times.date]
    on_session = np.array([session is not None for session in sessions], dtype=bool)
    closes = [0 if session is None else int(session[1]) for session in sessions]
    return np.array(closes, dtype=np.int64), on_session


__all__ = ["GivenCloses", "read_given"]
