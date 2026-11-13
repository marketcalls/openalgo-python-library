"""How far before `start` the history of a symbol is asked for (decision D25; the rules
of the history are in `_warm_up_history`, the loads in `_warm_up_load`).

The needs of a symbol's series (`_warm_up.SeriesNeed`) become requests (`requests`),
each the first IST date asked for, tried in order until one meets them:

- **Fixed days**: an anchored need asks from the first day of the first anchor period
  it reads (`_warm_up_anchor.first_day`), and a datetime `start` from its own day, so the
  morning before it is history.
- **Steps back**, for a need of n bars: first the calendar days that hold them at full
  session density (`_calendar_days`): for an intraday interval, the sessions that hold
  them at the regular session length near `start`; of daily bars, n sessions; either at
  five sessions a week, with a holiday allowance and a week more; of a period bar (W, M,
  Q, Y), the longest such period n + 1 times, as the oldest period is usually cut by the
  first day. That span is rounded up to a step of a fixed ladder (`SPANS`: from
  FIRST_SPAN_DAYS, each step a quarter longer than the one before), and while the bars
  returned fall short of a need and the source has bars from that step's first day
  (thinly traded shares whose candles are missing for minutes without a trade), the next
  step is asked for, up to `HISTORY_YEARS_CAP` years. A need longer than the longest
  window measured asks from the cap at once.

The ladder is the same for every set of needs, so a smaller need is met on a step no
later than a larger one: a sweep that loads for its largest need holds every step its
combinations ask for, each as its single run asks for it (`_warm_up_load`).
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._layout import NANOS_PER_SECOND, parse_interval
from ._warm_up import HISTORY_YEARS_CAP, SeriesNeed
from ._warm_up_anchor import first_day
from .data._sessions import day_start_ns

DAYS_PER_YEAR = 366
# Calendar days per session: five sessions a week.
CALENDAR_DAYS_PER_SESSION = 7 / 5
# Extra sessions asked for, as a fraction: NSE closes on about 15 weekdays of some 250
# a year (6 percent), and a year without a holiday list counts them as sessions.
HOLIDAY_ALLOWANCE = 0.10
# A week more than the sessions need, so that a short need is not cut by a weekend or
# by the partial session of a datetime `start`.
EXTRA_DAYS = 7
# The longest period of each period bar in calendar days, so that n bars fit; one more
# bar is asked for, as the oldest period is usually cut by the first day.
PERIOD_DAYS = {"W": 7, "M": 31, "Q": 92, "Y": 366}
# Days around `start` whose sessions give the regular session length of an intraday run.
SESSION_SAMPLE_DAYS = 30
# The ladder of spans asked for, in calendar days before `start` (module docstring): the
# first step, and each next one longer by its own length divided by this, rounded up.
FIRST_SPAN_DAYS = 16
SPAN_GROWTH_DIVISOR = 4
CAP_DAYS = HISTORY_YEARS_CAP * DAYS_PER_YEAR
# A first bar within a week of the day asked for is the first session on or after it
# (weekends, holidays); later, the source has no bars between the two.
FIRST_BAR_SLACK_DAYS = 7


def _ladder() -> tuple[int, ...]:
    """The spans of the ladder in calendar days, the last one the cap."""
    spans, span = [], FIRST_SPAN_DAYS
    while span < CAP_DAYS:
        spans.append(span)
        span += -(-span // SPAN_GROWTH_DIVISOR)
    return (*spans, CAP_DAYS)


SPANS = _ladder()


@dataclass(frozen=True)
class Request:
    """One request of a symbol's history: the first IST date asked for; `step` the first
    day of its step of the ladder, whose bars tell whether the source has more (None for
    a request of fixed days only); True when `HISTORY_YEARS_CAP` set it."""

    day: dt.date
    step: dt.date | None = None
    capped: bool = False


def run_from(plan: Any) -> int:
    """The first instant of the run (UTC nanoseconds): the datetime `start`, or IST
    midnight of the `start` date. History bars open before it."""
    return plan.first_slice if plan.first_slice is not None else int(day_start_ns(plan.start))


def _bars_per_session(interval: str, plan: Any, calendar: Any) -> int:
    """Bars of an intraday `interval` in a regular session: the median session length
    near `start` over the interval, rounded up (1 when no session is known)."""
    _, seconds = parse_interval(interval)
    span = dt.timedelta(days=SESSION_SAMPLE_DAYS)
    sessions = calendar.sessions(plan.exchange, plan.start - span, plan.start + span)
    if sessions.empty:
        return 1
    lengths = (sessions["close"] - sessions["open"]).to_numpy() // NANOS_PER_SECOND
    return max(1, math.ceil(float(np.median(lengths)) / seconds))


def _calendar_days(interval: str, bars: int, plan: Any, calendar: Any) -> int:
    """Calendar days before `start` that hold `bars` bars of `interval` at full session
    density."""
    kind, _ = parse_interval(interval)
    if kind in PERIOD_DAYS:
        return PERIOD_DAYS[kind] * (bars + 1)
    sessions = bars
    if kind == "intraday":
        sessions = math.ceil(bars / _bars_per_session(interval, plan, calendar))
    return math.ceil(sessions * CALENDAR_DAYS_PER_SESSION * (1 + HOLIDAY_ALLOWANCE)) + EXTRA_DAYS


def _fixed_day(needs: Sequence[SeriesNeed], plan: Any, calendar: Any) -> dt.date:
    """The first day the anchored needs and a datetime `start` ask for: `plan.start`, or
    the first day of the first anchor period an anchored need reads."""
    days = [plan.start]
    days += [
        first_day(plan.start, need.anchor, calendar, plan.exchange)
        for need in needs
        if need.anchor is not None
    ]
    return min(days)


def requests(needs: Sequence[SeriesNeed], plan: Any, calendar: Any) -> list[Request]:
    """The requests a symbol's `needs` make, in the order they are tried (module
    docstring). A request of fixed days only when no need counts bars; `plan.start`
    when nothing is needed."""
    fixed = _fixed_day(needs, plan, calendar)
    counted = [need for need in needs if need.anchor is None]
    if not counted:
        return [Request(fixed)]
    if any(need.more for need in counted):
        wanted = CAP_DAYS
    else:
        wanted = max(_calendar_days(n.interval, n.bars, plan, calendar) for n in counted)
    steps = [span for span in SPANS if span >= min(wanted, CAP_DAYS)]
    found = []
    for span in steps:
        step = plan.start - dt.timedelta(days=span)
        found.append(Request(min(fixed, step), step, span >= CAP_DAYS))
    return found


def by_symbol(needs: Sequence[SeriesNeed]) -> dict[str, list[SeriesNeed]]:
    """The needs of each symbol, in order."""
    found: dict[str, list[SeriesNeed]] = {}
    for need in needs:
        found.setdefault(need.symbol, []).append(need)
    return found


__all__ = [
    "CALENDAR_DAYS_PER_SESSION",
    "FIRST_BAR_SLACK_DAYS",
    "HOLIDAY_ALLOWANCE",
    "SPANS",
    "Request",
    "by_symbol",
    "requests",
    "run_from",
]
