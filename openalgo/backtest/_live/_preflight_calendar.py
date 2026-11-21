"""The preflight's check of the package calendar against OpenAlgo's market holidays
(`preflight`; 18-multi-strategy 12.2; unit M11).

A session trades by the package calendar: its slices, its entry cut-off and square-off,
its expiry days. OpenAlgo keeps its own holiday table (`holidays(year)`, kept by hand on
the server). For each exchange the session trades, every day from today to the year's end
is set side by side: the calendar's session (or none) against a calendar built from
OpenAlgo's rows alone on the same regular hours (`Calendar.add_holidays`, which reads the
server's rows as the server applies them: closed days, special sessions, MCX's evening on
a holiday at its regular close). A day on which the two differ is a disagreement:

- about today, it refuses the start: the session would trade a closed market or skip an
  open one;
- about a later day, it is a warning (the banner prints it, the result keeps it) and the
  event `calendar_disagrees` names the days (`calendar_disagrees` of the report); the
  session trades by the package calendar.

An answer that cannot be read (an error, a server without the endpoint, rows the calendar
cannot load) is a warning: the calendar was not checked. One `holidays` call is made per
preflight, for the current IST year. Times are UTC nanoseconds, dates IST.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .._calendar_convert import EXCHANGES, format_ist
from .._series import ist_time
from ..calendar import Calendar
from ..errors import CalendarError
from ..reference.symbols import INDEX_EXCHANGES
from ._guard_answers import SUCCESS
from ._preflight_plan import WARNING, Check

#: The name of the check.
HOLIDAYS = "holidays"
#: Where the OpenAlgo calendar's rows come from, as the calendar records them.
SOURCE = "OpenAlgo holidays (preflight)"
#: What to do about a disagreement.
_ADVICE = (
    "check the exchange's circular; correct OpenAlgo's Holidays page, or upgrade openalgo "
    "(pip install -U openalgo) if the package calendar is the one that is wrong"
)
_ONE_DAY = dt.timedelta(days=1)


@dataclass(frozen=True)
class Disagreement:
    """One day on which the package calendar and OpenAlgo's holidays differ for one
    exchange: each one's session as (open, close) UTC nanoseconds, None when closed."""

    exchange: str
    day: dt.date
    ours: tuple[int, int] | None
    theirs: tuple[int, int] | None

    def words(self) -> str:
        """The disagreement in words."""
        return (
            f"{self.exchange} on {self.day}: the package calendar {_session(self.ours)}, "
            f"OpenAlgo's holidays {_session(self.theirs)}"
        )


def holiday_check(
    client: Any, exchanges: Iterable[str], now_ns: int, calendar: Calendar
) -> tuple[Check, tuple[str, ...]]:
    """The check of `calendar` against OpenAlgo's holidays (module docstring) for the
    session's `exchanges` from the IST day of `now_ns` to the year's end, and the later
    days that disagree, in words (none when the check failed or could not read them)."""
    today = ist_time(now_ns).date()
    answer = client.holidays(year=today.year)
    theirs = _openalgo_calendar(answer, today.year)
    if isinstance(theirs, str):
        return Check(HOLIDAYS, True, f"{WARNING}{theirs}"), ()
    found = _disagreements(calendar, theirs, exchanges, today)
    on_today = [item.words() for item in found if item.day == today]
    later = tuple(item.words() for item in found if item.day != today)
    if on_today:
        return Check(
            HOLIDAYS,
            False,
            f"OpenAlgo's holidays and the package calendar disagree about today "
            f"({'; '.join(on_today)}): the session would trade a closed market or skip an "
            f"open one. {_ADVICE}",
        ), ()
    if later:
        return Check(
            HOLIDAYS,
            True,
            f"{WARNING}OpenAlgo's holidays and the package calendar disagree on later days "
            f"({'; '.join(later)}): the session trades by the package calendar; {_ADVICE}",
        ), later
    to = dt.date(today.year, 12, 31)
    return Check(HOLIDAYS, True, f"OpenAlgo's holidays agree with the package calendar to {to}"), ()


def _openalgo_calendar(answer: Any, year: int) -> Calendar | str:
    """A calendar of OpenAlgo's `answer` alone, on the regular hours, or why it cannot be
    read."""
    if not isinstance(answer, Mapping) or answer.get("status") != SUCCESS:
        message = answer.get("message") if isinstance(answer, Mapping) else repr(answer)
        return (
            f"OpenAlgo's holidays could not be read ({message}): the package calendar was "
            "not checked against them"
        )
    theirs = Calendar(builtin=False)
    try:
        theirs.add_holidays(dict(answer), year=year, source=SOURCE, verified=True, strict=True)
    except CalendarError as error:
        return (
            f"OpenAlgo's holidays could not be loaded ({error}): the package calendar was "
            "not checked against them"
        )
    return theirs


def _disagreements(
    ours: Calendar, theirs: Calendar, exchanges: Iterable[str], today: dt.date
) -> list[Disagreement]:
    """Each day from `today` to the year's end on which the two calendars give one of
    `exchanges` (each once, in order) different sessions. An index or an exchange the
    calendar does not know is left out: its own preflight checks refuse it."""
    found = []
    for exchange in dict.fromkeys(exchanges):
        if exchange not in EXCHANGES or exchange in INDEX_EXCHANGES:
            continue
        for day in _days(today):
            mine, other = ours.session(exchange, day), theirs.session(exchange, day)
            if mine != other:
                found.append(Disagreement(exchange, day, mine, other))
    return found


def _days(today: dt.date) -> Iterable[dt.date]:
    """Every day from `today` to the end of its year."""
    day = today
    while day.year == today.year:
        yield day
        day += _ONE_DAY


def _session(session: tuple[int, int] | None) -> str:
    """A day's session in words."""
    if session is None:
        return "is closed"
    return f"opens {format_ist(session[0])[-9:-4]} to {format_ist(session[1])[-9:-4]} IST"


__all__ = ["HOLIDAYS", "SOURCE", "Disagreement", "holiday_check"]
