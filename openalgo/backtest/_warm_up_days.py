"""How far before `start` the history of a symbol is asked for (decision D25; the rules
of the history are in `_warm_up_history`).

The needs of a symbol's series (`_warm_up.SeriesNeed`) become the first IST date asked
for (`first_requested_day`): for a need of n bars of an intraday interval, the sessions
that hold them at the regular session length near `start`; of daily bars, n sessions;
either at five sessions a week, with a holiday allowance and a week more; of a period
bar (W, M, Q, Y), the longest such period n + 1 times, as the oldest period is usually
cut by the first day. An anchored need asks from the start of its anchor period
(`_warm_up_anchor.period_start`), a need longer than the longest window measured from
`HISTORY_YEARS_CAP` years back, and every need is held to that cap. A datetime `start`
always asks from its own day, so the morning before it is history.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ._layout import NANOS_PER_SECOND, parse_interval
from ._warm_up import HISTORY_YEARS_CAP, SeriesNeed
from ._warm_up_anchor import period_start
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
    """Calendar days before `start` that hold `bars` bars of `interval`."""
    kind, _ = parse_interval(interval)
    if kind in PERIOD_DAYS:
        return PERIOD_DAYS[kind] * (bars + 1)
    sessions = bars
    if kind == "intraday":
        sessions = math.ceil(bars / _bars_per_session(interval, plan, calendar))
    return math.ceil(sessions * CALENDAR_DAYS_PER_SESSION * (1 + HOLIDAY_ALLOWANCE)) + EXTRA_DAYS


def first_requested_day(
    needs: Sequence[SeriesNeed], plan: Any, calendar: Any
) -> tuple[dt.date, bool]:
    """(the first IST date asked for so that every series of a symbol holds its need,
    see the module docstring; True when `HISTORY_YEARS_CAP` set it). `plan.start` when
    nothing is needed."""
    cap_days = HISTORY_YEARS_CAP * DAYS_PER_YEAR
    days = [plan.start]
    capped = False
    for need in needs:
        if need.anchor is not None:
            days.append(period_start(plan.start, need.anchor))
            continue
        span = cap_days if need.more else _calendar_days(need.interval, need.bars, plan, calendar)
        capped |= span >= cap_days
        days.append(plan.start - dt.timedelta(days=min(span, cap_days)))
    return min(days), capped


def by_symbol(needs: Sequence[SeriesNeed]) -> dict[str, list[SeriesNeed]]:
    """The needs of each symbol, in order."""
    found: dict[str, list[SeriesNeed]] = {}
    for need in needs:
        found.setdefault(need.symbol, []).append(need)
    return found


__all__ = [
    "CALENDAR_DAYS_PER_SESSION",
    "HOLIDAY_ALLOWANCE",
    "by_symbol",
    "first_requested_day",
    "run_from",
]
