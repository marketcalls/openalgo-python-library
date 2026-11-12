"""Clock conditions of the condition grammar (14-strategy-api 4.2): `bt.time_between`
and `bt.weekday`.

    self.entry_when(fast.crossed_above(slow) & bt.time_between("09:30", "14:30"))

A clock condition is decided by a bar's close, its decision instant, in IST: the time
of day for `time_between` (at or after the start and before the end, the half-open rule
of the entry window) and the date for `weekday`. It belongs to no series until it meets
one: combined with a view by `&` or `|` it becomes a view of that view's bars, and
declared alone it is evaluated on the strategy bars. Clock conditions combine with each
other by `&`, `|` and `~`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ._calendar_convert import to_seconds
from ._layout import IST_OFFSET_NANOS, NANOS_PER_SECOND
from ._views import TimelineView
from .errors import ConfigError

SECONDS_PER_DAY = 86_400
NANOS_PER_DAY = SECONDS_PER_DAY * NANOS_PER_SECOND
# 1970-01-01, day 0 of the epoch, was a Thursday; Monday is 0 as in date.weekday().
EPOCH_WEEKDAY = 3
DAY_NAMES = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
DAY_NUMBERS = {
    **{name: number for number, name in enumerate(DAY_NAMES)},
    **{name[:3]: number for number, name in enumerate(DAY_NAMES)},
}

# The test of a clock condition: bar closes (int64 UTC nanoseconds) to booleans.
ClockTest = Callable[[np.ndarray], np.ndarray]


class ClockCondition:
    """A condition on bar close times; see the module docstring."""

    def __init__(self, test: ClockTest, label: str) -> None:
        self._test = test
        self.label = label

    def values(self, closes: np.ndarray) -> np.ndarray:
        """The condition at bars closing at `closes` (int64 UTC nanoseconds)."""
        return np.asarray(self._test(np.asarray(closes, dtype=np.int64)), dtype=bool)

    def on(self, store: Any) -> TimelineView:
        """The condition as a view of a store's bars."""
        return TimelineView(store, self.values(store.column("ts_close")), self.label)

    def _combine(self, other: Any, function: Callable, symbol: str) -> Any:
        if isinstance(other, ClockCondition):

            def test(closes: np.ndarray) -> np.ndarray:
                return function(self.values(closes), other.values(closes))

            return ClockCondition(test, f"({self.label} {symbol} {other.label})")
        if isinstance(other, TimelineView):
            return function(other, self.on(other._store))
        return NotImplemented

    def __and__(self, other: Any) -> Any:
        return self._combine(other, lambda a, b: a & b, "&")

    __rand__ = __and__

    def __or__(self, other: Any) -> Any:
        return self._combine(other, lambda a, b: a | b, "|")

    __ror__ = __or__

    def __invert__(self) -> ClockCondition:
        return ClockCondition(lambda closes: ~self.values(closes), f"~{self.label}")

    def __bool__(self) -> bool:
        raise TypeError(
            f"{self.label} is a condition over the run: declare it with entry_when or "
            "exit_when, and combine it with &, | and ~ (not and, or, not)"
        )

    def __repr__(self) -> str:
        return f"ClockCondition({self.label})"


def _seconds_of_day(closes: np.ndarray) -> np.ndarray:
    return ((closes + IST_OFFSET_NANOS) % NANOS_PER_DAY) // NANOS_PER_SECOND


def _weekdays(closes: np.ndarray) -> np.ndarray:
    return ((closes + IST_OFFSET_NANOS) // NANOS_PER_DAY + EPOCH_WEEKDAY) % 7


def time_between(start: str, end: str) -> ClockCondition:
    """True at bars whose close (the decision instant) is at or after `start` and before
    `end`, IST "HH:MM". Raises ConfigError for an unreadable time or unless start is
    before end."""
    first = to_seconds(start, what="time_between start")
    last = to_seconds(end, what="time_between end")
    if first >= last:
        raise ConfigError(f"time_between needs start before end, got {start!r} and {end!r}")

    def test(closes: np.ndarray) -> np.ndarray:
        seconds = _seconds_of_day(closes)
        return (seconds >= first) & (seconds < last)

    return ClockCondition(test, f"time_between({start!r}, {end!r})")


def weekday(*days: str | int) -> ClockCondition:
    """True at bars that close on one of `days` (IST): names ("Mon", "monday", any case)
    or numbers, Monday 0 to Sunday 6. Raises ConfigError otherwise."""
    if not days:
        raise ConfigError("weekday needs at least one day, for example bt.weekday('Mon')")
    numbers = sorted({_day_number(day) for day in days})

    def test(closes: np.ndarray) -> np.ndarray:
        return np.isin(_weekdays(closes), numbers)

    return ClockCondition(test, f"weekday{tuple(days)!r}")


def _day_number(day: str | int) -> int:
    if isinstance(day, int) and not isinstance(day, bool) and 0 <= day <= 6:
        return day
    if isinstance(day, str) and day.strip().lower() in DAY_NUMBERS:
        return DAY_NUMBERS[day.strip().lower()]
    raise ConfigError(f"weekday takes day names (Mon to Sun) or 0 (Monday) to 6, got {day!r}")


__all__ = ["ClockCondition", "time_between", "weekday"]
