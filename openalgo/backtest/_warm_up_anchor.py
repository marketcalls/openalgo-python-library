"""The warm-up of an anchored indicator (decision D25; the rule is in `_warm_up`).

An anchored indicator (`ta.vwap`, whose registry notes say "Anchored") restarts at every
anchor period and depends on all the bars since the period began, so no rolling window
measures it. Its need is the bars of its series since the start of the anchor period:

- **The history** reaches back to the first day of the anchor period that holds `start`
  (`period_start`): the first of the month for "Month", the Monday for "Week", the
  start's own day for "Session" (so a datetime `start` in the middle of a session loads
  that session's morning, and a date `start` loads nothing).
- **Ready**: when the series begins with a whole anchor period (its first bar opens at or
  before the open of the first session of that period, by the run's calendar), the
  indicator is ready from the series' first bar. Otherwise its first period is partial
  (own data that begins mid-month, a source without the bars) and it is ready at the
  close of the first bar of the next period, where it restarts (`anchored_ready_at`).

Anchor periods are those of `openalgo.indicators._anchors` (`ANCHORS`, `period_ids`), the
definitions the indicators use: days and weeks (Monday to Sunday) follow the calendar,
months, quarters, half years and years count calendar months from January, and an
intraday span ("1H", "15m") restarts every day, so its period always starts in its own
day. The calendar is the run's, in IST.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd

from openalgo.indicators._anchors import check_anchor, period_ids, wall_seconds

from ._series import ist_time

# The longest anchor period in days: a year, the longest of `ANCHORS`.
LONGEST_PERIOD_DAYS = 366


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


def _covers_first_period(opens: np.ndarray, anchor: str, calendar: Any, exchange: str) -> bool:
    """True when the series that opens at `opens` (UTC nanoseconds) begins with a whole
    anchor period (module docstring)."""
    first = int(opens[0])
    day = ist_time(first).date()
    sessions = calendar.sessions(exchange, period_start(day, anchor), day)
    if sessions.empty:
        return True
    return first <= int(sessions["open"].iloc[0])


def anchored_ready_at(store: Any, anchor: str, calendar: Any, exchange: str) -> int | None:
    """The close (UTC nanoseconds) of the bar from which an anchored indicator on
    `store`'s bars is ready (module docstring); None when it is never ready (no bar, or
    a partial first period and no later one)."""
    opens = store.column("ts_open")
    closes = store.column("ts_close")
    if not len(opens):
        return None
    if _covers_first_period(opens, anchor, calendar, exchange):
        return int(closes[0])
    ids = period_ids(wall_seconds(store.times()), anchor)
    later = np.flatnonzero(ids != ids[0])
    return int(closes[later[0]]) if len(later) else None


__all__ = ["anchored_ready_at", "period_start"]
