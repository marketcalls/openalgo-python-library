"""The warm-up of an anchored indicator (decision D25; the rule is in `_warm_up`).

An anchored indicator restarts at every anchor period and reads all the bars of a
period, so no rolling window measures it. The causality registry says which `ta`
functions are anchored and which periods a value reads (`causality.Anchor`): a session
or monthly VWAP reads its own period since it began, floor pivots the whole period
before their own. Its need (`PeriodAnchor`) is the bars of its series since the start
of the first period it reads:

- **The history** reaches back to the first day of the anchor period that holds `start`
  (`period_start`), and for each whole period before it that a value reads, to the
  first day of the period that holds the session before (`first_day`): the first of the
  month for "Month", the Monday for "Week", the start's own day for "Session" (so a
  datetime `start` in the middle of a session loads that session's morning, and a date
  `start` loads nothing), the session before for daily pivots.
- **Ready**: when the series begins with a whole anchor period (its first bar opens at or
  before the open of the first session of that period, by the run's calendar), the
  indicator is ready from the first bar of the period that follows the whole periods it
  reads (from the series' first bar for a VWAP). Otherwise its first period is partial
  (own data that begins mid-month, a source without the bars) and it is ready one period
  later (`anchored_ready_at`).

Anchor periods are those of `openalgo.indicators._anchors` (`ANCHORS`, `period_ids`), the
definitions the indicators use: days and weeks (Monday to Sunday) follow the calendar,
months, quarters, half years and years count calendar months from January, and an
intraday span ("1H", "15m") restarts every day, so its period always starts in its own
day. A period with no session is skipped: the period before Monday's session is Friday's.
The calendar is the run's, in IST.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from openalgo.indicators._anchors import check_anchor, period_ids, wall_seconds

from ._series import ist_time

# The longest anchor period in days: a year, the longest of `ANCHORS`.
LONGEST_PERIOD_DAYS = 366


@dataclass(frozen=True)
class PeriodAnchor:
    """The anchor of an anchored indicator call: its period ("Month", an anchor name of
    `_anchors.ANCHORS`) and the whole periods before the current one that a value reads
    (`causality.Anchor.periods_before`)."""

    period: str
    periods_before: int = 0

    def since(self) -> str:
        """Where its need begins, for messages: "the start of its Month", or "the start
        of the Month before its own"."""
        if not self.periods_before:
            return f"the start of its {self.period}"
        if self.periods_before == 1:
            return f"the start of the {self.period} before its own"
        return f"the start of the {self.periods_before} {self.period} periods before its own"


def _midnights(days: list[dt.date]) -> np.ndarray:
    """Wall-clock seconds of 00:00 on each day (`_anchors.wall_seconds`)."""
    return wall_seconds(pd.DatetimeIndex([pd.Timestamp(day) for day in days]))


def period_start(day: dt.date, anchor: str) -> dt.date:
    """The first calendar day of the `anchor` period that holds `day`. Raises ValueError
    for an anchor name `ta.vwap` does not take."""
    check_anchor(anchor)
    days = [day - dt.timedelta(days=back) for back in range(LONGEST_PERIOD_DAYS, -1, -1)]
    ids = period_ids(_midnights(days), anchor)
    return days[int(np.argmax(ids == ids[-1]))]


def _session_before(day: dt.date, calendar: Any, exchange: str) -> dt.date:
    """The last session day before `day` by the calendar; the day before when the
    calendar knows none in the year before it."""
    earlier = calendar.sessions(
        exchange, day - dt.timedelta(days=LONGEST_PERIOD_DAYS), day - dt.timedelta(days=1)
    )
    return earlier["date"].iloc[-1] if not earlier.empty else day - dt.timedelta(days=1)


def first_day(start: dt.date, anchor: PeriodAnchor, calendar: Any, exchange: str) -> dt.date:
    """The first day of the first anchor period an indicator anchored on `anchor` reads
    at `start` (module docstring)."""
    day = period_start(start, anchor.period)
    for _ in range(anchor.periods_before):
        day = period_start(_session_before(day, calendar, exchange), anchor.period)
    return day


def _covers_first_period(opens: np.ndarray, anchor: str, calendar: Any, exchange: str) -> bool:
    """True when the series that opens at `opens` (UTC nanoseconds) begins with a whole
    anchor period (module docstring)."""
    first = int(opens[0])
    day = ist_time(first).date()
    sessions = calendar.sessions(exchange, period_start(day, anchor), day)
    if sessions.empty:
        return True
    return first <= int(sessions["open"].iloc[0])


def anchored_ready_at(store: Any, anchor: PeriodAnchor, calendar: Any, exchange: str) -> int | None:
    """The close (UTC nanoseconds) of the bar from which an indicator anchored on
    `anchor` on `store`'s bars is ready (module docstring); None when it is never ready
    (no bar, or too few periods)."""
    opens = store.column("ts_open")
    closes = store.column("ts_close")
    if not len(opens):
        return None
    ids = period_ids(wall_seconds(store.times()), anchor.period)
    firsts = np.flatnonzero(np.concatenate(([True], ids[1:] != ids[:-1])))
    whole = _covers_first_period(opens, anchor.period, calendar, exchange)
    ready = anchor.periods_before + (0 if whole else 1)
    return int(closes[firsts[ready]]) if ready < len(firsts) else None


__all__ = ["PeriodAnchor", "anchored_ready_at", "first_day", "period_start"]
