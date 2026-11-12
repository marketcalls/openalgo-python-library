"""Value types the session calendar returns, and its warning category."""

import datetime as dt
from typing import NamedTuple

import numpy as np
import pandas as pd

from .errors import ConfigError

#: Session kinds, by native code.
SESSION_KINDS = ("regular", "special")
#: Where a bar sits against the session of its IST date, by the native code that
#: `Calendar.check_intraday` returns in `placement`.
PLACEMENTS = ("in_session", "pre_open", "post_close", "closed_day", "off_lattice")


class CalendarWarning(UserWarning):
    """A calendar answer rests on missing or unconfirmed data, or a check could not decide.

    Raised as a warning, never as an error: the result is still returned.
    """


class SessionInfo(NamedTuple):
    """One session. `open` and `close` are UTC nanoseconds, `date` is the IST date.

    `kind` is "regular" or "special"; `coverage` is "template", "unverified" or
    "sourced" (how well the date is backed by holiday data); `hours_unverified` is True
    when the session times are not confirmed by an exchange or regulator document.
    """

    exchange: str
    date: dt.date
    open: int
    close: int
    kind: str
    coverage: str
    hours_unverified: bool


class IntradayTimes(NamedTuple):
    """Default intraday times of a session, as UTC nanoseconds.

    An intraday entry is rejected when its fill bar opens at or after `entry_until`;
    positions are squared off at `square_off`. `relative_to_close` says the times were
    taken before that day's close rather than from the segment's clock times;
    `unverified` that they rest on an unconfirmed row or close.
    """

    entry_until: int
    square_off: int
    session_open: int
    session_close: int
    relative_to_close: bool
    unverified: bool


class SessionsPerYear(NamedTuple):
    """The annualisation factor of a date range and what it rests on.

    value: sessions per year (the session count scaled to 365.25 calendar days).
    sessions: dates in the range on which at least one of the exchanges trades.
    calendar_days: calendar days in the range, both ends included.
    coverage: the weakest year coverage among the exchanges and years of the range.
    template_only: (exchange, year) pairs without a holiday list; for each, the count
        includes that year's holidays, so `value` is overstated.
    """

    value: float
    sessions: int
    calendar_days: int
    coverage: str
    template_only: list[tuple[str, int]]


def grid_mask(grid: pd.DataFrame, exchange: str, column: str = "active") -> np.ndarray:
    """Boolean array: which rows of a `Calendar.union_grid` frame involve `exchange`.

    `column` is "active", "session_open" or "session_close". Raises ConfigError when
    the exchange is not part of the grid.
    """
    exchanges = grid.attrs.get("exchanges", [])
    if exchange not in exchanges:
        raise ConfigError(f"exchange {exchange!r} is not part of this grid (it holds {exchanges})")
    return (grid[column].to_numpy() & (1 << exchanges.index(exchange))) != 0


__all__ = [
    "PLACEMENTS",
    "SESSION_KINDS",
    "CalendarWarning",
    "IntradayTimes",
    "SessionInfo",
    "SessionsPerYear",
    "grid_mask",
]
