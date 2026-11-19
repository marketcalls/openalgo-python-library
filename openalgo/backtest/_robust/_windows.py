"""Windows of a run's sessions in time: the in-sample and out-of-sample parts of the out-of-
sample check (design 17, section 2.1) and the folds of a walk-forward (section 2.3).

A window is a run of whole sessions, positions in the run's session dates, both ends
included. A rerun of a window starts at its first session and ends at its last (the
run's own `start` or `end` when the window reaches it, so a run bounded by a time of day
keeps it), and loads the history its indicators need from the bars before it, as any
backtest does.

Out of sample (`split`): the first round(`in_sample` x sessions) sessions are in sample,
the rest out of sample (halves to even, Python's `round`).

Walk-forward folds (`folds`): with `in_sample` the share of a fold in sample and r =
in_sample / (1 - in_sample) out-of-sample windows to an in-sample one, the boundaries
b_j = round(sessions x (r + j) / (r + count)), j = 0 to `count`, cut the sessions after
the first b_0 into `count` out-of-sample windows of about equal length, [b_(j-1), b_j)
for fold j, which together cover the rest of the run once. Rolling folds put the b_0
sessions before each out-of-sample window in sample, so every fold is about `in_sample`
in sample (70/30 by default); anchored folds put every session before it in sample, so
the in-sample window grows from the first session. Both share the out-of-sample windows,
so their stitched out-of-sample runs cover the same sessions.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..errors import ConfigError

#: The fewest sessions in each part of a window a run is measured on: a ratio of session
#: returns needs two (`_part_objective.RATIO_SESSIONS`).
FEWEST_SESSIONS = 2


@dataclass(frozen=True)
class Window:
    """Sessions `first` to `last` of a run, positions, both included."""

    first: int
    last: int

    @property
    def sessions(self) -> int:
        """How many sessions it holds."""
        return self.last - self.first + 1


@dataclass(frozen=True)
class Fold:
    """Fold `number` (from 1) of a walk-forward: its in-sample and out-of-sample
    windows."""

    number: int
    in_sample: Window
    out_of_sample: Window


def split(sessions: int, in_sample: float) -> tuple[Window, Window]:
    """(in sample, out of sample) of a run of `sessions` (module docstring). Raises
    ConfigError when either part would hold fewer than `FEWEST_SESSIONS`."""
    cut = round(in_sample * sessions)
    if cut < FEWEST_SESSIONS or sessions - cut < FEWEST_SESSIONS:
        raise ConfigError(
            f"{sessions} sessions at {in_sample:g} in sample leave {cut} in sample and "
            f"{sessions - cut} out of sample; each part needs {FEWEST_SESSIONS}"
        )
    return Window(0, cut - 1), Window(cut, sessions - 1)


def folds(sessions: int, count: int, in_sample: float, anchored: bool) -> tuple[Fold, ...]:
    """The `count` folds of a run of `sessions` (module docstring). Raises ConfigError
    when a window would hold fewer than `FEWEST_SESSIONS`."""
    ratio = in_sample / (1.0 - in_sample)
    bounds = [round(sessions * (ratio + j) / (ratio + count)) for j in range(count + 1)]
    made = []
    for number in range(1, count + 1):
        first, end = bounds[number - 1], bounds[number]
        start = 0 if anchored else first - bounds[0]
        made.append(Fold(number, Window(start, first - 1), Window(first, end - 1)))
    shortest = min(min(f.in_sample.sessions, f.out_of_sample.sessions) for f in made)
    if shortest < FEWEST_SESSIONS:
        raise ConfigError(
            f"{sessions} sessions cut into {count} folds at {in_sample:g} in sample leave a "
            f"window of {shortest} sessions; each needs {FEWEST_SESSIONS}: give fewer folds or "
            "a longer run"
        )
    return tuple(made)


def window_arguments(
    window: Window, dates: Sequence[dt.date], start: Any, end: Any
) -> tuple[tuple[str, Any], ...]:
    """The `start` and `end` run arguments of a rerun of `window` of a run whose session
    dates are `dates` and whose own bounds are `start` and `end` (module docstring)."""
    first = start if window.first == 0 else dates[window.first]
    last = end if window.last == len(dates) - 1 else dates[window.last]
    return (("start", first), ("end", last))


__all__ = ["FEWEST_SESSIONS", "Fold", "Window", "folds", "split", "window_arguments"]
