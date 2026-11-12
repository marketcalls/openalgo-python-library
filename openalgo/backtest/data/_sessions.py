"""Instants, IST dates and session lookups for the data layer.

Units: an instant is int64 UTC nanoseconds; a day is an IST calendar date, as a
`datetime.date` or as the day number of `_calendar_convert` (days since 1970-01-01).

Date conversions come from `openalgo.backtest._calendar_convert`. This module adds
what the data layer needs on top: the injectable clock, the IST day of an instant
and `Sessions`, which answers session questions for whole arrays at once and hands
the kernel a `Calendar` whatever `SessionCalendar` the caller supplied.
"""

import datetime as dt
import time
from collections.abc import Callable

import numpy as np

from .._calendar_convert import IST, SECONDS_PER_DAY, DateLike, from_day, to_day
from .._layout import IST_OFFSET_NANOS, NANOS_PER_SECOND
from .._protocols import SessionCalendar
from ..calendar import Calendar, default_calendar
from ..errors import ConfigError

NANOS_PER_DAY = SECONDS_PER_DAY * NANOS_PER_SECOND
IST_OFFSET_SECONDS = IST_OFFSET_NANOS // NANOS_PER_SECOND
ONE_DAY = dt.timedelta(days=1)
# A copied calendar reaches this far past the years it is asked for, so that a week
# that crosses a year end and a 5 hour 30 minute shift across midnight stay inside.
_PAD = dt.timedelta(days=7)
_SPAN_UNITS = ((SECONDS_PER_DAY, "day"), (3_600, "hour"), (60, "minute"), (1, "second"))

NowLike = int | dt.datetime | str | Callable[[], object] | None


def resolve_now(now: NowLike = None) -> int:
    """The current time as UTC nanoseconds.

    `now` is None (the system clock), an int of UTC nanoseconds, a datetime or pandas
    Timestamp (a naive one is IST), an ISO date-time string (naive is IST) or a
    callable returning one of these. Tests pass a fixed value.

    Raises ConfigError for a float, a bool or any other type.
    """
    if now is None:
        return time.time_ns()
    if callable(now):
        return resolve_now(now())
    if isinstance(now, int | np.integer) and not isinstance(now, bool):
        return int(now)
    if isinstance(now, str):
        try:
            now = dt.datetime.fromisoformat(now.strip())
        except ValueError as exc:
            raise ConfigError(f"now={now!r} is not an ISO date-time") from exc
    if not isinstance(now, dt.datetime):
        raise ConfigError("now must be UTC nanoseconds (int), a datetime or an ISO string")
    if now.tzinfo is None:
        now = now.replace(tzinfo=IST)
    elapsed = now - dt.datetime(1970, 1, 1, tzinfo=dt.UTC)
    whole_seconds = elapsed.days * SECONDS_PER_DAY + elapsed.seconds
    return whole_seconds * NANOS_PER_SECOND + elapsed.microseconds * 1_000


def to_date(value: DateLike, name: str) -> dt.date:
    """Read `value` as an IST date (see `_calendar_convert.to_day`).

    Raises ConfigError that names the argument `name`.
    """
    try:
        return from_day(to_day(value))
    except ConfigError as exc:
        raise ConfigError(f"{name}: {exc}") from exc


def ist_days(instants_ns: np.ndarray | int) -> np.ndarray:
    """Day number of the IST date each UTC nanosecond instant falls on."""
    shifted = np.asarray(instants_ns, dtype="int64") + IST_OFFSET_NANOS
    return np.floor_divide(shifted, NANOS_PER_DAY)


def day_start_ns(day: dt.date | int | np.ndarray) -> int | np.ndarray:
    """UTC nanoseconds of 00:00 IST on a date, a day number or an array of day numbers."""
    if isinstance(day, np.ndarray):
        return day.astype("int64") * NANOS_PER_DAY - IST_OFFSET_NANOS
    number = to_day(day) if isinstance(day, dt.date) else int(day)
    return number * NANOS_PER_DAY - IST_OFFSET_NANOS


def format_span(nanos: int) -> str:
    """A duration in nanoseconds as words, for example '5 minutes'."""
    seconds = abs(int(nanos)) // NANOS_PER_SECOND
    parts = []
    for size, word in _SPAN_UNITS:
        count, seconds = divmod(seconds, size)
        if count:
            parts.append(f"{count} {word}" + ("" if count == 1 else "s"))
    return " ".join(parts) if parts else "0 seconds"


class Sessions:
    """The sessions of one exchange over the years `first` to `last`.

    calendar: any SessionCalendar; None is the package default calendar.
    native: a `Calendar` with the same sessions, for the kernel (validation, bar
        closes, resampling). It is the calendar itself when that is a `Calendar`,
        else a date by date copy that reaches one week beyond those years.

    Raises ConfigError when `calendar` has no `session` method, and CalendarError
    for an unknown exchange or a session the calendar reports outside its date.
    """

    def __init__(
        self, calendar: SessionCalendar | None, exchange: str, first: dt.date, last: dt.date
    ) -> None:
        if calendar is None:
            calendar = default_calendar()
        elif not callable(getattr(calendar, "session", None)):
            raise ConfigError(
                "calendar must provide session(exchange, day) returning (open_ns, close_ns) "
                "or None; see openalgo.backtest._protocols.SessionCalendar"
            )
        self.calendar = calendar
        self.exchange = exchange
        self.first = dt.date(first.year, 1, 1) - _PAD
        self.last = dt.date(last.year, 12, 31) + _PAD
        if isinstance(calendar, Calendar):
            self.native = calendar
        else:
            self.native = Calendar.from_sessions(calendar, exchange, self.first, self.last)
        table = self.native.sessions(exchange, self.first, self.last)
        self._days = np.array([to_day(day) for day in table["date"]], dtype="int64")
        self._opens = table["open"].to_numpy(dtype="int64")
        self._closes = table["close"].to_numpy(dtype="int64")

    def covering(self, first: dt.date, last: dt.date) -> "Sessions":
        """These sessions, or a wider copy when the dates reach beyond them."""
        if first >= self.first and last <= self.last:
            return self
        return Sessions(self.calendar, self.exchange, min(first, self.first), max(last, self.last))

    def bounds(self, instants_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(open_ns, close_ns, has_session) of the IST date of each instant.

        Rows on a date without a session get zeros and has_session False.
        """
        return self.bounds_of_days(ist_days(instants_ns))

    def bounds_of_days(self, days: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(open_ns, close_ns, has_session) for an array of day numbers."""
        return self.period_bounds(days, days)

    def period_bounds(
        self, first_days: np.ndarray, last_days: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(open of the first session, close of the last session, has_session) of the
        inclusive day ranges `first_days` to `last_days`. A range without a session
        gets zeros and has_session False."""
        low = np.searchsorted(self._days, first_days, side="left")
        high = np.searchsorted(self._days, last_days, side="right") - 1
        has = low <= high
        if not has.any():  # also the case of a calendar with no session at all
            nothing = np.zeros(len(low), dtype="int64")
            return nothing, nothing.copy(), has
        opens = np.where(has, self._opens[np.minimum(low, len(self._days) - 1)], 0)
        closes = np.where(has, self._closes[np.maximum(high, 0)], 0)
        return opens, closes, has

    def session(self, day: dt.date) -> tuple[int, int] | None:
        """(open_ns, close_ns) of the session on `day`, or None when closed. Dates
        beyond the copied years are asked from the calendar itself."""
        if not self.first <= day <= self.last:
            return self.calendar.session(self.exchange, day)
        opens, closes, has = self.bounds_of_days(np.array([to_day(day)], dtype="int64"))
        return (int(opens[0]), int(closes[0])) if has[0] else None

    def days(self, first: dt.date, last: dt.date) -> list[dt.date]:
        """The session dates from `first` to `last`, both inclusive."""
        inside = (self._days >= to_day(first)) & (self._days <= to_day(last))
        return [from_day(day) for day in self._days[inside]]

    def last_complete_day(self, now_ns: int) -> dt.date:
        """The latest IST date whose session, if it has one, has closed at `now_ns`."""
        today = from_day(int(ist_days(now_ns)))
        session = self.session(today)
        if session is None or now_ns >= session[1]:
            return today
        return today - ONE_DAY
