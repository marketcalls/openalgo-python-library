"""`self.at("HH:MM", fn)` and `self.every(period, fn)`: the Python half of the kernel's
timers (`rust/oa_bt/src/engine/timers.rs`).

The kernel decides when a timer is due and pauses in step 2 of the slice
(`Pause::Timers`); the driver then calls the timers' functions in registration order,
and every order they place fills at that same open (03-architecture 4.2 step 2).

- `self.at("09:20", fn)` runs `fn` at the first base-bar open at or after 09:20 IST in
  every session. A session with no base bar opening at or after that time skips it, and
  the result's warnings say in how many sessions that happened.
- `self.every("month", fn)` (daily or longer strategies) runs `fn` for the close of the
  first session of each week, month, quarter or year: at the next session's first
  open, where its orders fill, as an order decided at that close would
  (14-strategy-api 2.3).

Timers are registered in `init`; registering later raises ConfigError.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from ._engine import call
from ._messages import counted
from .errors import BacktestError, ConfigError

# `self.every` periods and the kernel's calendar unit codes.
PERIODS = {"week": "W", "month": "M", "quarter": "Q", "year": "Y"}
_TIME = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
MINUTES_PER_HOUR = 60


def minute_of_day(time_ist: Any) -> int:
    """Minutes since IST midnight of an "HH:MM" string. Raises ConfigError otherwise."""
    match = _TIME.match(time_ist) if isinstance(time_ist, str) else None
    if match is None:
        raise ConfigError(
            f"self.at takes an IST time 'HH:MM' from 00:00 to 23:59, got {time_ist!r}"
        )
    return int(match.group(1)) * MINUTES_PER_HOUR + int(match.group(2))


def _callable(fn: Any, what: str) -> Callable[[], Any]:
    if not callable(fn):
        raise ConfigError(f"{what} takes a function to call, got {fn!r}")
    return fn


class Timers:
    """The timers of one run: registration with the engine, and their functions."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._timers: list[tuple[str, Callable[[], Any]]] = []

    def _add(self, label: str, register: Callable[..., int], value: Any, fn: Any) -> None:
        function = _callable(fn, label)
        timer_id = call(register, value)
        if timer_id != len(self._timers):
            raise BacktestError(
                f"{label}: the engine numbered this timer {timer_id}, not {len(self._timers)}; "
                "please report this"
            )
        self._timers.append((label, function))

    def at(self, time_ist: str, fn: Any) -> None:
        """Register `fn` at an IST time each session. Raises ConfigError for a bad time,
        a function that is not callable, or a call after the run started."""
        minute = minute_of_day(time_ist)
        self._add(f"self.at({time_ist!r})", self._engine.add_timer_at, minute, fn)

    def every(self, period: str, fn: Any) -> None:
        """Register `fn` for the first session of each period. Raises ConfigError for an
        unknown period, an intraday strategy or a call after the run started."""
        if period not in PERIODS:
            raise ConfigError(f"self.every takes one of {list(PERIODS)}, got {period!r}")
        label = f"self.every({period!r})"
        self._add(label, self._engine.add_timer_every, PERIODS[period], fn)

    def fire(self, timer_ids: list[int]) -> None:
        """Call the functions of the timers due, in the kernel's order."""
        for timer_id in timer_ids:
            self._timers[timer_id][1]()

    def warnings(self) -> list[str]:
        """A sentence per `self.at` timer that skipped sessions (call before `finish`)."""
        notes = []
        for timer_id, fired, missed in call(self._engine.timer_counts):
            if missed:
                label = self._timers[timer_id][0]
                notes.append(
                    f"{label} did not run in {missed} of {counted(fired + missed, 'session')}: "
                    f"no base bar opens at or after that time in {'it' if missed == 1 else 'them'}"
                )
        return notes


__all__ = ["PERIODS", "Timers", "minute_of_day"]
