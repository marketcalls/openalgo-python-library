"""Small interfaces that let the areas of the package depend on each other loosely."""

import datetime as _dt
from typing import Protocol, runtime_checkable


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


__all__ = ["SessionCalendar"]
