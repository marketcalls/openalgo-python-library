"""Small interfaces that let the areas of the package depend on each other loosely."""

import datetime as _dt
from typing import Protocol, runtime_checkable

#: Values `SessionCalendar.coverage` returns, weakest first.
#: "template": no holiday list is loaded for that exchange and year, so open days are
#: only the weekday rule and holidays are unknown. "unverified": a holiday list is
#: loaded but not confirmed. "sourced": the list comes from a cited source or the user.
COVERAGE_LEVELS = ("template", "unverified", "sourced")


@runtime_checkable
class SessionCalendar(Protocol):
    """What the data layer and the reference tables need from a session calendar.

    `openalgo.backtest.calendar.Calendar` implements this; tests may pass a fake.
    """

    def session(self, exchange: str, day: _dt.date) -> tuple[int, int] | None:
        """The regular session of `exchange` on the IST date `day`.

        Returns (open, close) as UTC nanoseconds, or None when the exchange is closed
        on that date.
        """
        ...

    def coverage(self, exchange: str, day: _dt.date) -> str:
        """How well the answer of `session` for that date is backed by data.

        One of COVERAGE_LEVELS. A caller whose result would be wrong on an unknown
        holiday (an expiry date moved to the previous session, for example) must not
        rely on a "template" date.
        """
        ...


__all__ = ["COVERAGE_LEVELS", "SessionCalendar"]
