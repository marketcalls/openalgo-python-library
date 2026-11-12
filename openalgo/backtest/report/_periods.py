"""Session dates, the annualisation factor, period tables and benchmark alignment.

Session dates are IST calendar dates, held as int64 days since 1970-01-01 and shown
as a naive `DatetimeIndex` of dates named "date" (a naive date index cannot be moved
to the previous day by a time-zone conversion).
"""

import datetime as dt
import math
import warnings

import numpy as np
import pandas as pd

from .._calendar_types import CalendarWarning, SessionsPerYear
from .._layout import IST_OFFSET_NANOS, NANOS_PER_SECOND
from ..calendar import Calendar

IST = "Asia/Kolkata"
NANOS_PER_DAY = 86_400 * NANOS_PER_SECOND


def ist_days(nanos: np.ndarray) -> np.ndarray:
    """IST dates (days since 1970-01-01) of UTC nanosecond instants."""
    return (np.asarray(nanos, dtype=np.int64) + IST_OFFSET_NANOS) // NANOS_PER_DAY


def date_index(days: np.ndarray) -> pd.DatetimeIndex:
    """A naive index of IST dates named "date"."""
    return pd.DatetimeIndex(np.asarray(days, dtype=np.int64).astype("datetime64[D]"), name="date")


def ist_times(nanos: np.ndarray) -> pd.DatetimeIndex:
    """UTC nanosecond instants as an IST DatetimeIndex. The unit is named: without it
    pandas reads the integers one by one, which costs a 1M-point equity series 0.15 s."""
    utc = pd.to_datetime(np.asarray(nanos, dtype=np.int64), unit="ns", utc=True)
    return utc.tz_convert(IST)


def ist_instant(nanos: int | None) -> pd.Timestamp | None:
    """One UTC nanosecond instant as an IST Timestamp, None for None."""
    return None if nanos is None else pd.Timestamp(int(nanos), tz="UTC").tz_convert(IST)


def ist_date(nanos: int) -> dt.date:
    """The IST calendar date of one UTC nanosecond instant."""
    return date_index(ist_days(np.array([nanos]))).date[0]


def annualisation(
    calendar: Calendar, exchanges: list[str], first_ns: int, last_ns: int
) -> tuple[SessionsPerYear, str]:
    """Sessions per year over the whole calendar years from the IST date of `first_ns`
    to that of `last_ns`, on the union calendar of `exchanges`, and those years as text.

    Whole years make the figure independent of where the run starts and ends inside a
    year, and never shorter than the calendar's minimum window. When a year has no
    holiday list the count is overstated: a CalendarWarning says so, naming the years.
    """
    first, last = ist_date(first_ns), ist_date(last_ns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CalendarWarning)
        result = calendar.sessions_per_year(
            exchanges, dt.date(first.year, 1, 1), dt.date(last.year, 12, 31)
        )
    years = str(first.year) if first.year == last.year else f"{first.year} to {last.year}"
    if result.template_only:
        named = ", ".join(f"{code} {year}" for code, year in result.template_only)
        warnings.warn(
            f"sessions per year counts every weekday of {named} as a session (no holiday "
            "list is loaded), so volatility, Sharpe, Sortino and alpha are annualised with "
            "an overstated count",
            CalendarWarning,
            stacklevel=3,
        )
    return result, years


def missing_sessions(calendar: Calendar, exchanges: list[str], days: np.ndarray) -> int:
    """Calendar sessions of the union of `exchanges` between the first and last of the
    session dates `days` on which the series has no point."""
    if len(days) == 0:
        return 0
    first, last = date_index(days[[0, -1]]).date
    scheduled: set[dt.date] = set()
    for code in exchanges:
        scheduled.update(calendar.sessions(code, first, last)["date"])
    return len(scheduled - set(date_index(days).date))


def period_returns(session_returns: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Compounded returns per IST calendar month and per year, as fractions.

    The monthly table has one row per year and columns 1 to 12, NaN for a month with
    no session; a month's return is the product of (1 + session return), minus 1.
    """
    growth = 1.0 + session_returns
    index = session_returns.index
    monthly = growth.groupby([index.year, index.month]).prod() - 1.0
    table = monthly.unstack().reindex(columns=range(1, 13))
    table.index.name, table.columns.name = "year", "month"
    yearly = growth.groupby(index.year).prod() - 1.0
    yearly.index.name = "year"
    return table, yearly


def session_end_times(marks: np.ndarray, days: np.ndarray) -> np.ndarray:
    """UTC nanoseconds of the last of `marks` on each session date in `days`.

    `marks` are the equity points that belong to a session, increasing: every point
    when an opening wealth is given, every point but the first (the start) otherwise.
    """
    point_days = ist_days(marks)
    return marks[np.searchsorted(point_days, days, side="right") - 1]


def benchmark_path(
    bench_ts: np.ndarray, bench_values: np.ndarray, start_ns: int, end_ns: np.ndarray
) -> np.ndarray:
    """Benchmark levels as of `start_ns` and each session end: the last benchmark value
    stamped at or before each instant, NaN where it has none yet (never back-filled,
    which would book a later move)."""
    instants = np.concatenate(([start_ns], end_ns))
    position = np.searchsorted(bench_ts, instants, side="right") - 1
    return np.where(position >= 0, bench_values[np.maximum(position, 0)], np.nan)


def benchmark_growth(returns: np.ndarray) -> float:
    """The benchmark's return over a span of sessions, by the one rule of the run's
    `benchmark_total_return`, the horizon summary and `benchmark_comparison`: the
    product of (1 + r) over the sessions whose benchmark return is known, minus 1, so
    from the benchmark's first value in the span to its end; NaN when no session of
    the span has one. Only sessions before the benchmark's first value lack a return
    (`benchmark_path` never back-fills, and carries the last value forward after it)."""
    known = np.asarray(returns, dtype=float)
    known = known[np.isfinite(known)]
    return float(np.prod(1.0 + known)) - 1.0 if len(known) else math.nan


__all__ = [
    "IST",
    "NANOS_PER_DAY",
    "annualisation",
    "benchmark_growth",
    "benchmark_path",
    "date_index",
    "ist_date",
    "ist_days",
    "ist_instant",
    "ist_times",
    "missing_sessions",
    "period_returns",
    "session_end_times",
]
