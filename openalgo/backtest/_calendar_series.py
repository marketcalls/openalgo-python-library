"""The parts of `Calendar` that read bars: argument checks, the check report and the grid.

The report of `Calendar.check_intraday` is a dict with these keys. Counts are ints;
"session day" means an IST date with bars on which the exchange has a session.

    bars                      bars in the series
    days                      IST dates with at least one bar
    session_days              of those, dates with a session
    closed_days               dates with bars but no session
    first_closed_day          the first such date (datetime.date) or None
    closed_day_bars           bars on those dates
    days_starting_at_open     session days whose first session bar is the session's first bar
    days_starting_late        session days whose first session bar is a later one
    first_late_day            the first such date (datetime.date) or None
    days_without_session_bars session days with bars, none of them in the session
    days_with_bar_at_close    session days with a bar that opens exactly at the session close
    first_bar_at_close        the first such bar (int UTC nanoseconds) or None
    pre_open_bars             bars that end at or before the session open
    post_close_bars           bars that open at or after the session close
    off_lattice_bars          bars that overlap the session but do not open at the
                              exchange's regular open plus whole intervals
    first_off_lattice         the first such bar (int UTC nanoseconds) or None
    bars_in_session           session bars on the lattice
    missing_bars              lattice bars absent on the session days with data
    missing_days              sessions between the first and last date with no bar
    duplicate_bars            bars whose timestamp repeats the previous one
    template_only_days        session days with data that rest on the weekday rule alone
    unverified_hours_days     session days with data whose session times are unconfirmed
    close_labelled            True when the bars look stamped at their close
    offset                    "late_5h30" or "early_5h30" when the series is shifted, else None
    offset_check              "aligned", "shifted" or "inconclusive"
    offset_direction          the shift that fits best when not aligned, else None
    offset_reason             the offset outcome in words, "" when aligned
    finding                   "late_5h30", "early_5h30", "close_labelled" or None: what is
                              wrong with the timestamps, to choose advice by
    placement                 uint8 array, one code per bar, names in PLACEMENTS

The frame of `Calendar.union_grid` has one row per base-bar open of any exchange:

    ts             int64 UTC nanoseconds, strictly increasing
    date           IST session date (datetime64)
    active         uint16 mask of exchanges with a bar opening at ts
    session_open   uint16 mask of exchanges whose session opens with this bar
    session_close  uint16 mask of exchanges whose session ends with this bar
    close_<EXCH>   int64 UTC nanoseconds at which that exchange's bar opening at ts
                   closes, or NO_BAR (the int64 minimum) when it has none there

Bit `i` stands for `frame.attrs["exchanges"][i]` (`grid_mask` reads one out). Rows are
ordered by bar open only. Bars of different exchanges overlap whenever their lattices
differ: with the daily base the MCX bar opens at 09:00 and closes at 23:30, while the
NSE bar of the same date opens at 09:15 and closes at 15:30, so the NSE row comes after
the MCX row although NSE closes first. Order decisions and fills by the close columns,
never by row position.
"""

import datetime as dt

import numpy as np
import pandas as pd

from ._calendar_convert import (
    COVERAGE_LEVELS,
    check_exchange,
    clock,
    from_day,
    interval_nanos,
    ts_open_column,
)
from ._calendar_types import SESSION_KINDS, SessionsPerYear
from .errors import ConfigError

_DAY_KEYS = ("first_closed_day", "first_late_day")


def intraday_nanos(interval: str) -> int:
    """Nanoseconds of an intraday interval. Raises ConfigError for "D" and longer."""
    nanos = interval_nanos(interval)
    if nanos is None:
        raise ConfigError(
            f"interval {interval!r} is not intraday: session times only apply to intraday "
            "bars such as '1m' or '15m'"
        )
    return nanos


def series_arguments(
    bars, exchange: str | None, interval: str | None
) -> tuple[str, np.ndarray, int]:
    """(exchange, ts_open, interval nanoseconds) of a bars frame or a ts_open array.

    `exchange` and `interval` default to `bars.attrs`. Raises ConfigError when either
    is missing or the interval is not intraday, CalendarError for an unknown exchange.
    """
    attrs = bars.attrs if isinstance(bars, pd.DataFrame) else {}
    exchange = attrs.get("exchange") if exchange is None else exchange
    interval = attrs.get("interval") if interval is None else interval
    if exchange is None or interval is None:
        raise ConfigError(
            "exchange and interval are required: pass them, or set bars.attrs['exchange'] "
            "and bars.attrs['interval']"
        )
    return check_exchange(exchange), ts_open_column(bars), intraday_nanos(interval)


def check_report(counts: dict) -> dict:
    """The native report with day numbers turned into dates (keys in the module doc)."""
    report = dict(counts)
    for key in _DAY_KEYS:
        if report[key] is not None:
            report[key] = from_day(report[key])
    return report


def sessions_frame(columns: tuple, exchange: str) -> pd.DataFrame:
    """The native session columns as the frame `Calendar.sessions` returns."""
    days, open_, close, kind, coverage, unverified = columns
    frame = pd.DataFrame(
        {
            "date": pd.Series([from_day(day) for day in days], dtype=object),
            "open": open_,
            "close": close,
            "kind": pd.Series([SESSION_KINDS[code] for code in kind], dtype=object),
            "coverage": pd.Series([COVERAGE_LEVELS[code] for code in coverage], dtype=object),
            "hours_unverified": unverified,
        }
    )
    frame.attrs["exchange"] = exchange
    return frame


def holidays_frame(rules: list, names: dict[dt.date, str]) -> pd.DataFrame:
    """The native dated rules as the frame `Calendar.holidays` returns."""
    rows = []
    for day, window, unverified, level in rules:
        open_sec, close_sec = (None, None) if window is None else window
        date = from_day(day)
        open_ = None if open_sec is None else clock(open_sec)
        close = None if close_sec is None else clock(close_sec)
        level_name = COVERAGE_LEVELS[level]
        rows.append((date, names.get(date, ""), open_, close, level_name, bool(unverified)))
    columns = ["date", "name", "open", "close", "coverage", "hours_unverified"]
    # Object columns keep None as None on every pandas version.
    frame = pd.DataFrame(rows, columns=columns, dtype=object)
    return frame.astype({"hours_unverified": bool})


def annualisation(row: tuple) -> SessionsPerYear:
    """The native sessions-per-year tuple as a `SessionsPerYear`."""
    value, sessions, calendar_days, coverage, template_only = row
    level = COVERAGE_LEVELS[coverage]
    return SessionsPerYear(value, sessions, calendar_days, level, list(template_only))


def grid_frame(columns: tuple, codes: list[str], base: str) -> pd.DataFrame:
    """The union grid as a DataFrame (see `Calendar.union_grid`)."""
    ts, days, active, opens, closes, bar_closes = columns
    frame = pd.DataFrame(
        {
            "ts": ts,
            "date": days.astype("datetime64[D]"),
            "active": active,
            "session_open": opens,
            "session_close": closes,
        }
    )
    for code, close in zip(codes, bar_closes, strict=True):
        frame[f"close_{code}"] = close
    frame.attrs.update(exchanges=codes, base=base)
    return frame


__all__ = [
    "annualisation",
    "check_report",
    "grid_frame",
    "holidays_frame",
    "intraday_nanos",
    "series_arguments",
    "sessions_frame",
]
