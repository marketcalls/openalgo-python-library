"""The off-lattice error for candles of a special session that opens off the bar lattice.

Bars of interval N open at each date's regular open plus whole intervals, special
sessions included (`rust/oa_bt/src/market/calendar/lattice.rs`), and the engine builds
its grid on that lattice. A special session can open between two lattice points (NSE
Muhurat trading at 14:20 IST, with 15-minute bars from 09:15), and a source that starts
that session's candles at its own open puts every one of them off the lattice. The
calendar's general message (bars anchored to the regular open; check the interval and
the resampling) does not tell a beginner why or what to do, so when every off-lattice
bar of a series lies in such a session the error names the session from the calendar
data, gives its open and the lattice's, and the two ways out: 1-minute data, from which
the engine builds the session's bars, or leaving the session's bars out.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from .._calendar_convert import IST, call_native, from_day, to_day
from .._calendar_types import PLACEMENTS
from .._layout import NANOS_PER_SECOND
from .._messages import counted
from ._sessions import Sessions, format_span, ist_days

OFF_LATTICE = PLACEMENTS.index("off_lattice")


def _clock(instant_ns: int) -> str:
    """The IST clock time "HH:MM" of UTC nanoseconds."""
    seconds, _ = divmod(instant_ns, NANOS_PER_SECOND)
    return dt.datetime.fromtimestamp(seconds, tz=IST).strftime("%H:%M")


def _bar_length(step_ns: int) -> str:
    """The interval as an adjective, "15-minute" or "1-hour"."""
    count, unit = format_span(step_ns).split(" ", 1)
    return f"{count}-{unit.removesuffix('s')}"


def _name(sessions: Sessions, day: dt.date) -> str | None:
    """The calendar's name for the dated rule of `day`, or None when it has none."""
    rules = sessions.native.holidays(sessions.exchange, day.year)
    named = rules.loc[(rules["date"] == day) & rules["name"].notna(), "name"]
    return str(named.iloc[0]) if len(named) else None


def _session_clause(sessions: Sessions, day: dt.date, step_ns: int) -> str | None:
    """The clause "the session of 2026-11-08 (its name) opens at 14:20 IST, off the
    09:15 bar lattice", or None when the session of `day` opens on the lattice (its
    off-lattice bars are then wrong for another reason) or there is none."""
    session = sessions.native.session_info(sessions.exchange, day)
    anchor = call_native(sessions.native._cal.lattice_anchor, sessions.exchange, to_day(day))
    if session is None or anchor is None or (session.open - anchor) % step_ns == 0:
        return None
    name = _name(sessions, day)
    named = f"the session of {day}" + ("" if name is None else f" ({name})")
    return f"{named} opens at {_clock(session.open)} IST, off the {_clock(anchor)} bar lattice"


def special_session_error(
    report: dict, ts_open: np.ndarray, step_ns: int, sessions: Sessions, tag: str
) -> str | None:
    """The error text when every off-lattice bar of a series lies in a session that
    opens off the bar lattice, else None (the calendar's own message stands). `report`
    is the calendar's check report of the sorted bar opens `ts_open` (UTC nanoseconds,
    `Calendar.check_intraday`); `step_ns` the interval; `tag` names the series."""
    off_lattice = report["placement"] == OFF_LATTICE
    if report["finding"] is not None or not off_lattice.any():
        return None
    days = [from_day(int(day)) for day in np.unique(ist_days(ts_open[off_lattice]))]
    clauses = [_session_clause(sessions, day, step_ns) for day in days]
    if any(clause is None for clause in clauses):
        return None
    one, length = len(days) == 1, _bar_length(step_ns)
    these = "that session" if one else "those sessions"
    built, left = ("the session's", "that session's") if one else ("their", "those sessions'")
    return (
        f"{tag}: {counted(int(off_lattice.sum()), 'bar is', 'bars are')} not on the "
        f"{sessions.exchange} bar lattice: {'; '.join(clauses)}, so {length} candles of "
        f"{these} cannot be placed. Pass 1-minute data (the engine builds {built} {length} "
        f"bars from it), or leave {left} bars out."
    )


__all__ = ["special_session_error"]
