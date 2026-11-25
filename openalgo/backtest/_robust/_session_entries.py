"""Random entries of an all-intraday run, kept within one session (design 17, section
2.4; lead, 2026-11-18).

When every closed trade of the run opens and closes in one session, its random trades do
too: each is placed within one session's tradable strategy bars, from the entry window to
the square-off, so none falls after the entry window or is cut short by the square-off,
and the random runs hold the run's trade count.

A session's tradable stretch (`Window`), positions in a symbol's strategy-bar closes of
the run: an entry decided at the close of bar i fills at the next open, the same
instant, so it may be decided at a close from `entry_from` up to before `entry_until`
(`first` to `last_entry`); an exit decided at the close of bar j fills at that open
before the square-off acts on it, so it may be decided at a close up to `square_off`
and before the session's close (`last_exit`). Products other than MIS have no entry
window or square-off: the whole session, its last close aside (a decision there fills
in the next session).

Groups: the run's trades of one session (by their places in the run's sequence, after
the draw of which trade of a direction takes each place, `_random_entries`) stay
together, with their sequence of directions and the waits between them. Each group is
moved to a session of its own, drawn among the unused sessions it fits in, the groups
with the most bars to place first (a group fits a session when its trades, held one bar
each, fit between its first entry and its last exit). Taking the largest first, each
group always finds a session when the run's needs, largest first, fit the sessions'
stretches, largest first (`fits`), which `EntryPlan.unplaced` asks. Within the session
the group keeps its holding times where they fit; while they do not, the longest is cut
by a bar (the earliest of equal ones). The slack left is cut at the group's trades'
points as across the whole run (`_random_entries`).

Draws: run r reads 3 draws a trade (draws r x 3 x trades onwards), symbols in
registration order, each symbol's in three parts of its trade count: the order of its
trades within their direction, the sessions of its groups (one draw a group, the rest
unread), the cut points.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .._calendar_convert import to_seconds
from .._layout import NANOS_PER_SECOND
from .._products import MIS
from ..data._sessions import day_start_ns
from . import _random
from ._ohlc import session_days

#: The least strategy bars a random trade is held (`_random_entries.LEAST_HOLD`).
LEAST_HOLD = 1
#: The draws a trade reads in a run of an all-intraday plan (module docstring).
DRAWS_PER_TRADE = 3


@dataclass(frozen=True)
class SessionTimes:
    """One session's instants, UTC nanoseconds: its open and close, and the entry window
    and square-off its run's rules give it (module docstring)."""

    day: int
    open: int
    close: int
    entry_from: int
    entry_until: int
    square_off: int


def _clock(day: dt.date, given: str) -> int:
    """The instant of the IST clock time `given` ("HH:MM") on `day`, UTC nanoseconds."""
    return int(day_start_ns(day)) + to_seconds(given, what="intraday") * NANOS_PER_SECOND


def session_times(plan: Any, calendar: Any, days: Sequence[dt.date]) -> tuple[SessionTimes, ...]:
    """The `SessionTimes` of each of `days` the calendar has a session for, from the run's
    `plan` (`_settings.RunPlan`): the times it was given, else the session's open and the
    calendar's dated defaults, as the kernel resolves them for MIS; the whole session for
    any other product."""
    found = []
    for day in days:
        hours = calendar.session(plan.exchange, day)
        if hours is None:
            continue
        opens, closes = hours
        start, until, square = opens, closes, closes
        if plan.product == MIS:
            defaults = calendar.intraday_times(plan.exchange, day)
            start = _clock(day, plan.entry_from) if plan.entry_from else opens
            until = _clock(day, plan.entry_until) if plan.entry_until else defaults.entry_until
            square = _clock(day, plan.square_off) if plan.square_off else defaults.square_off
        number = int(session_days(np.array([opens]))[0])
        found.append(
            SessionTimes(number, int(opens), int(closes), int(start), int(until), int(square))
        )
    return tuple(found)


@dataclass(frozen=True)
class Window:
    """A session's tradable stretch of a symbol, positions in its closes (module
    docstring): entries decided from `first` to `last_entry`, exits up to `last_exit`."""

    first: int
    last_entry: int
    last_exit: int

    @property
    def room(self) -> int:
        """The bars a group may spend before its last entry: from `first` to the later of
        its last entry and the last exit less a one-bar hold."""
        return min(self.last_entry, self.last_exit - LEAST_HOLD) - self.first


def windows(closes: np.ndarray, times: Sequence[SessionTimes]) -> tuple[Window, ...]:
    """The `Window` of each session of `times` in which `closes` (a symbol's strategy-bar
    closes of the run, UTC nanoseconds) leave room for one trade, in session order."""
    closes = np.asarray(closes, dtype=np.int64)
    days = session_days(closes - 1)
    found = []
    for session in times:
        inside = np.flatnonzero(days == session.day)
        if not len(inside):
            continue
        at = closes[inside]
        entries = inside[(at >= session.entry_from) & (at < session.entry_until)]
        exits = inside[(at < session.close) & (at <= session.square_off)]
        if not len(entries) or not len(exits):
            continue
        window = Window(int(entries[0]), int(entries[-1]), int(exits[-1]))
        if window.room >= 0:
            found.append(window)
    return tuple(found)


def all_intraday(trade_list: pd.DataFrame) -> bool:
    """Whether every closed trade of `trade_list` (`Result.trade_list()`) opens and
    closes on one IST date, so in one session; False without a closed trade."""
    closed = trade_list[trade_list["exit_time"].notna()]
    if closed.empty:
        return False
    entered = pd.DatetimeIndex(closed["entry_time"]).date
    left = pd.DatetimeIndex(closed["exit_time"]).date
    return bool((entered == left).all())


def groups(days: Sequence[int]) -> list[list[int]]:
    """The places of each session's trades: runs of equal `days`, in order."""
    found: list[list[int]] = []
    for place, day in enumerate(days):
        if found and days[found[-1][0]] == day:
            found[-1].append(place)
        else:
            found.append([place])
    return found


def need(holds: Sequence[int], waits: Sequence[int]) -> int:
    """The bars a group spends before its last entry when every trade is held one bar:
    `waits` holds the wait after each trade but the last."""
    return sum(LEAST_HOLD + wait for wait in waits[: len(holds) - 1])


def fits(needs: Sequence[int], rooms: Sequence[int]) -> bool:
    """Whether groups with `needs` can each take a session of its own among stretches
    with `rooms` (`Window.room`): the k-th largest need within the k-th largest room."""
    needs, rooms = sorted(needs, reverse=True), sorted(rooms, reverse=True)
    return len(needs) <= len(rooms) and all(n <= r for n, r in zip(needs, rooms, strict=False))


def _cut(holds: list[int], waits: Sequence[int], window: Window) -> int:
    """The slack of a group in `window`, cutting the longest of `holds` (in place) by a
    bar while it is below zero (module docstring)."""
    while True:
        before = sum(h + w for h, w in zip(holds[:-1], waits, strict=False))
        slack = min(window.last_entry, window.last_exit - holds[-1]) - window.first - before
        if slack >= 0 or max(holds) <= LEAST_HOLD:
            return slack
        holds[holds.index(max(holds))] -= 1


@dataclass(frozen=True)
class Placed:
    """A trade placed: its position in the trades placed, and the positions in the
    symbol's closes of its entry and exit decisions."""

    place: int
    entry: int
    exit: int


def place(
    holds: Sequence[int],
    waits: Sequence[int],
    days: Sequence[int],
    stretches: Sequence[Window],
    draws: np.ndarray,
) -> list[Placed]:
    """Where a symbol's trades go (module docstring): `holds` and `days` of each place,
    `waits` after each but the last, its session `stretches`, and `draws` its 2 x trades
    draws (the sessions of its groups, then its cut points)."""
    count = len(holds)
    chosen, cuts = draws[:count], draws[count:]
    sets = groups(days)
    needs = [need([holds[p] for p in group], [waits[p] for p in group[:-1]]) for group in sets]
    rooms = [stretch.room for stretch in stretches]
    used: set[int] = set()
    placed: list[Placed] = []
    for rank, g in enumerate(sorted(range(len(sets)), key=lambda g: -needs[g])):
        free = [s for s in range(len(stretches)) if s not in used and rooms[s] >= needs[g]]
        session = free[int(_random.below(chosen[rank : rank + 1], len(free))[0])]
        used.add(session)
        placed += _within(sets[g], holds, waits, stretches[session], cuts)
    return sorted(placed, key=lambda trade: trade.place)


def _within(
    group: list[int],
    holds: Sequence[int],
    waits: Sequence[int],
    window: Window,
    cuts: np.ndarray,
) -> list[Placed]:
    """The group's trades placed in `window` with the cut points of its places."""
    kept = [holds[p] for p in group]
    between = [waits[p] for p in group[:-1]]
    slack = _cut(kept, between, window)
    points = np.sort(_random.below(cuts[group], slack + 1))
    gaps = np.diff(np.concatenate(([0], points)))
    bar, placed = window.first, []
    for k, (gap, place_) in enumerate(zip(gaps, group, strict=True)):
        bar += int(gap)
        placed.append(Placed(place_, bar, bar + kept[k]))
        bar += kept[k] + (between[k] if k < len(between) else 0)
    return placed


def stretches_by_symbol(
    closes: Mapping[str, np.ndarray], times: Sequence[SessionTimes]
) -> dict[str, tuple[Window, ...]]:
    """The session stretches of each symbol's closes."""
    return {symbol: windows(values, times) for symbol, values in closes.items()}


__all__ = [
    "DRAWS_PER_TRADE",
    "Placed",
    "SessionTimes",
    "Window",
    "all_intraday",
    "fits",
    "groups",
    "need",
    "place",
    "session_times",
    "stretches_by_symbol",
    "windows",
]
