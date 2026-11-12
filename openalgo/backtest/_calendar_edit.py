"""Copying the sessions of any `SessionCalendar` into a native calendar.

The kernel runs on its own calendar object. When the caller supplies another
implementation of `openalgo.backtest._protocols.SessionCalendar` (a user's own
calendar, a test fake), its sessions are copied date by date into a native calendar
that starts from the built-in regular hours and no holiday data.

What a copied date carries:

- the session the source returns, or a closure when it returns None;
- the source's coverage of that date, from `coverage(exchange, day)`. A source without
  that method has every date marked "unverified": a copy is never "sourced" unless its
  source says so. A "template" date whose session equals the built-in weekday rule is
  not copied at all, so it reads back as "template"; a "template" date that differs
  from it is stored as "unverified", since a dated rule is never a template;
- session times flagged as unverified unless the source marks the date "sourced".
"""

from ._calendar_convert import (
    COVERAGE_LEVELS,
    SECONDS_PER_DAY,
    call_native,
    coverage_code,
    format_ist,
    from_day,
)
from ._layout import IST_OFFSET_NANOS, NANOS_PER_SECOND
from ._protocols import SessionCalendar
from .errors import CalendarError

_UNVERIFIED = coverage_code("unverified")


def session_seconds(exchange: str, day: int, session: tuple[int, int]) -> tuple[int, int]:
    """(open, close) in seconds since IST midnight of `day` for a session in UTC nanoseconds.

    Raises CalendarError unless the session lies inside that IST date on whole seconds.
    """
    open_ns, close_ns = (int(value) for value in session)
    midnight_ns = day * SECONDS_PER_DAY * NANOS_PER_SECOND - IST_OFFSET_NANOS
    open_sec, open_rest = divmod(open_ns - midnight_ns, NANOS_PER_SECOND)
    close_sec, close_rest = divmod(close_ns - midnight_ns, NANOS_PER_SECOND)
    if open_rest or close_rest or not 0 <= open_sec < close_sec <= SECONDS_PER_DAY:
        raise CalendarError(
            f"the calendar returned a {exchange} session for {from_day(day)} that runs from "
            f"{format_ist(open_ns)} to {format_ist(close_ns)}: a session must lie inside "
            "its IST date, on whole seconds, with the open before the close"
        )
    return open_sec, close_sec


def _source_coverage(source: SessionCalendar, exchange: str, day: int) -> str:
    """The source's coverage of a date: its `coverage` answer, else "unverified"."""
    coverage = getattr(source, "coverage", None)
    if not callable(coverage):
        return "unverified"
    level = coverage(exchange, from_day(day))
    if level not in COVERAGE_LEVELS:
        raise CalendarError(
            f"the calendar returned coverage {level!r} for {exchange} on {from_day(day)}: "
            f"coverage must be one of {COVERAGE_LEVELS}"
        )
    return level


def copy_sessions(native, source: SessionCalendar, exchange: str, first: int, last: int) -> None:
    """Copy the sessions of `exchange` from `source` into `native`, day numbers
    `first` to `last` inclusive (see the module documentation for what is copied).

    Raises CalendarError for a session outside its IST date or an unknown coverage.
    """
    for day in range(first, last + 1):
        session = source.session(exchange, from_day(day))
        level = _source_coverage(source, exchange, day)
        if level == "template" and _same_as_template(native, exchange, day, session):
            continue
        code = max(coverage_code(level), _UNVERIFIED)
        if session is None:
            call_native(native.set_closed, exchange, day, code)
            continue
        open_sec, close_sec = session_seconds(exchange, day, session)
        hours_unverified = level != "sourced"
        call_native(native.set_session, exchange, day, open_sec, close_sec, hours_unverified, code)


def _same_as_template(native, exchange: str, day: int, session) -> bool:
    own = call_native(native.session, exchange, day)
    if own is None or session is None:
        return own is None and session is None
    return (own[1], own[2]) == tuple(int(value) for value in session)


__all__ = ["copy_sessions", "session_seconds"]
