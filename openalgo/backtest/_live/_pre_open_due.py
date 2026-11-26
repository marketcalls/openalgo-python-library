"""The bar a daily or longer strategy may decide before today's open in a live session
(design 20-positional-live section 8, decision 11.1; review finding 1 of unit N5b). Part of
`_session_pre_open`.

A backtest decides each bar of the strategy's interval at its close and fills its orders at
the next session's open. So the one bar a start can decide before today's open as the
backtest decides it is the bar due today: the bar that closed at the close of the
exchange's last session before today. A daily strategy has one every day, that session's
daily candle; a weekly, monthly, quarterly or yearly strategy has one only when that session
was its period's last (today begins a new period), and it is whole once that session's daily
candle is in. The period is the kernel's (`CalendarUnit::range`): a week from Monday to
Sunday, a calendar month, quarter or year.

- **Due and whole**: the kernel decides it (`Engine.pre_open_decision`). When the journal
  says an earlier start decided it (only a start of today can: it closed at the last
  session's close), it is not decided again, and a warning says so (`DECIDED_TODAY`): a
  stop may have cancelled that decision's orders before they reached the broker, and said
  so then.
- **Due, its candle missing** (the broker has not served the last session's daily candle
  yet, or its read came back empty): nothing is decided, not an older bar in its place, and
  a warning names the candle (`MISSING_CANDLE`). A start once the broker serves it decides
  that bar, after the open with its orders flagged `late_decision`.
- **None due** (a weekly or longer strategy after the first session of a period) while a
  whole bar that closed before the last session was never decided (a first start in the
  middle of a period, or sessions no session ran): it is not decided at an open later than
  its own; a warning names it and the open it was due at (`DUE_EARLIER`).

When nothing is to be decided the kernel is still given the bars, with the latest whole one
as the last decided, so they feed its last closes and none is decided; the journal's last
decided bar is not moved, so a later start decides the next bar due.

Times are UTC nanoseconds; days are IST dates.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from dataclasses import dataclass

from .._calendar_convert import format_ist
from .._layout import parse_interval
from ..calendar import Calendar
from ._plan_daily import DAILY, RECENT_DAYS

#: The warning of a due bar whose daily candle the broker has not served (module docstring).
MISSING_CANDLE = (
    "{name}: the broker has not served the daily candle of {day} (the last session) yet, so "
    "its {interval} bar closing on {day} is not decided now, nor an older bar in its place; "
    "start again once the broker serves it to decide that bar (after the open its orders are "
    "flagged late_decision)"
)
#: The warning of a due bar an earlier start of today decided (module docstring).
DECIDED_TODAY = (
    "{name}: its {interval} bar closing on {day} was decided by an earlier start today and is "
    "not decided again today"
)
#: The warning of a whole bar that was due at an earlier open (module docstring).
DUE_EARLIER = (
    "{name}: its {interval} bar closing on {closed} was due at the open of {due}, not this "
    "one: it feeds its indicators only and is not decided; the next is decided at the first "
    "start after its period's last session"
)
#: How far back the exchange's last session before today is looked for: the calendar days of
#: the daily candles read before today (`_plan_daily.RECENT_DAYS`).
_LOOKBACK = dt.timedelta(days=RECENT_DAYS[DAILY])
_ONE_DAY = dt.timedelta(days=1)


@dataclass(frozen=True)
class DueBar:
    """What a start decides of a daily or longer strategy (module docstring):
    `last_decided_ns`, the last decided bar the kernel is told (UTC nanoseconds, None for
    none), and `note`, the start's warning naming the bar due it does not decide, or None."""

    last_decided_ns: int | None
    note: str | None = None


@dataclass(frozen=True)
class StrategyBars:
    """A daily or longer strategy's bars before today's open: its `name` and `interval`, its
    `exchange`, the closes of its bars of its interval (`closes`) and of its daily candles
    (`daily_closes`), UTC nanoseconds, and today's open (`open_ns`)."""

    name: str
    interval: str
    exchange: str
    closes: Iterable[int]
    daily_closes: Iterable[int]
    open_ns: int


def due_bar(
    bars: StrategyBars, day: dt.date, last_decided_ns: int | None, calendar: Calendar
) -> DueBar:
    """What a start on `day` decides of the strategy of `bars`, whose journal's last decided
    bar closed at `last_decided_ns` (None for none), on `calendar` (module docstring)."""
    whole = [int(close) for close in bars.closes if close <= bars.open_ns]
    latest = max(whole, default=None)
    told = max((ns for ns in (latest, last_decided_ns) if ns is not None), default=None)
    decided = last_decided_ns is not None and latest is not None and latest <= last_decided_ns
    previous = _last_session(calendar, bars.exchange, day)
    if previous is not None and _ends_period(previous[0], day, bars.interval):
        last_day, last_close = previous
        words = {"name": bars.name, "interval": bars.interval, "day": last_day.isoformat()}
        if last_decided_ns is not None and last_close <= last_decided_ns:
            return DueBar(last_decided_ns, DECIDED_TODAY.format(**words))
        if latest == last_close and last_close in {int(c) for c in bars.daily_closes}:
            return DueBar(last_decided_ns)
        return DueBar(told, MISSING_CANDLE.format(**words))
    if latest is None or decided:
        return DueBar(last_decided_ns)
    note = DUE_EARLIER.format(
        name=bars.name,
        interval=bars.interval,
        closed=format_ist(latest)[:10],
        due=_next_session(calendar, bars.exchange, latest, day).isoformat(),
    )
    return DueBar(told, note)


def _last_session(calendar: Calendar, exchange: str, day: dt.date) -> tuple[dt.date, int] | None:
    """(the IST day, the close in UTC nanoseconds) of `exchange`'s last session before `day`,
    or None when the calendar has none in `_LOOKBACK`."""
    sessions = calendar.sessions(exchange, day - _LOOKBACK, day - _ONE_DAY)
    if sessions.empty:
        return None
    return sessions["date"].iloc[-1], int(sessions["close"].iloc[-1])


def _next_session(calendar: Calendar, exchange: str, close_ns: int, day: dt.date) -> dt.date:
    """The IST day of `exchange`'s first session after the bar closing at `close_ns`, the
    open it was due at; `day` when the calendar has none before it."""
    closed = dt.date.fromisoformat(format_ist(close_ns)[:10])
    sessions = calendar.sessions(exchange, closed + _ONE_DAY, day)
    return sessions["date"].iloc[0] if not sessions.empty else day


def _ends_period(previous: dt.date, day: dt.date, interval: str) -> bool:
    """Whether the session of `previous` was the last of its period of `interval` before the
    session of `day`: always for a daily strategy; for a weekly or longer one when `day`
    begins another period (module docstring)."""
    unit = parse_interval(interval)[0]
    return unit == DAILY or _period(previous, unit) != _period(day, unit)


def _period(day: dt.date, unit: str) -> tuple[int, int]:
    """The kernel's period of `unit` ("W", "M", "Q" or "Y") holding `day`, as a key."""
    if unit == "W":
        year, week, _ = day.isocalendar()
        return year, week
    if unit == "M":
        return day.year, day.month
    if unit == "Q":
        return day.year, (day.month - 1) // 3
    return day.year, 0


__all__ = ["DECIDED_TODAY", "DUE_EARLIER", "MISSING_CANDLE", "DueBar", "StrategyBars", "due_bar"]
