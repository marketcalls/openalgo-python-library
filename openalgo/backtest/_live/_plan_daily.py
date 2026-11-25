"""A daily or longer strategy in a live session (design 20-positional-live section 8,
decision 11.1, review M8; the lead's ruling N5b-1). Part of `plan`.

Such a strategy decides once a day, before the open, on the broker's candle of the session
that closed, as a backtest of it on daily bars decides at each close and fills at the next
open: its bars are the broker's daily candles (external bars, 08-live-trading 5), never
bars built from the day's ticks, so the decision is a backtest's on the same candles.

- **Its bars** (`DailySeries`): the warm-up's history read with a daily base, as a backtest
  of the strategy reads it (`history_plan`), and the broker's daily candles of the days
  before today (`recent_bars`, `RECENT_DAYS` back: enough for the latest whole bar of the
  strategy's interval); a weekly or longer bar is built from them as a backtest builds it
  (`resample.resample_bars`). Today's ticks and 1-minute candles still reach the kernel
  session, on its 1-minute grid, for its stops, targets and exits, through the plan's own
  1-minute series; none of them reaches the strategy's bars, since it decides nothing in
  the session.
- **The bars it is asked to decide** (`decision_bars`): its bars of the strategy's interval
  closing after the journal's last decided bar, for `Engine.pre_open_decision`, which
  decides the latest whole one and feeds the rest to the indicators only.

Times are UTC nanoseconds, prices rupees, volume units.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any

import numpy as np
import pandas as pd

from .._layout import BAR_COLUMNS, parse_interval
from .._settings import RunPlan
from .._timeframes import _check_interval
from .._warm_up_history import empty_base
from ..calendar import Calendar
from ..data import load_bars
from ..errors import NoDataError
from ..resample import resample_bars
from ._series_store import COLUMNS, Clock, LiveStore
from .data_worker import HISTORY_SOURCE

#: The base of a daily or longer strategy's own bars: the broker's daily candles.
DAILY = "D"
#: Calendar days of daily candles read before today for the pre-open decision, by the
#: strategy's interval: the latest whole bar's sessions and the holidays around them.
RECENT_DAYS = {"D": 10, "W": 21, "M": 70, "Q": 200, "Y": 750}
#: A bar's prices and volume, in the order `Engine.pre_open_decision` takes them.
PRICES = ("open", "high", "low", "close", "volume")
_ONE_DAY = dt.timedelta(days=1)


def decides_before_open(interval: str) -> bool:
    """Whether a strategy of `interval` decides before the open (module docstring): a
    daily or longer one ("D", "W", "M", "Q", "Y")."""
    return parse_interval(interval)[0] != "intraday"


def history_plan(plan: RunPlan) -> RunPlan:
    """`plan` with the base its warm-up history is read at: daily for a daily or longer
    strategy, as a backtest of it reads it, else the plan's own (the live 1-minute base)."""
    if decides_before_open(plan.interval):
        return dataclasses.replace(plan, base=DAILY)
    return plan


def recent_bars(
    symbol: str, exchange: str, plan: RunPlan, client: Any, calendar: Calendar, read_ns: int
) -> pd.DataFrame:
    """The broker's daily candles of `symbol` on `exchange` from `RECENT_DAYS` before the
    plan's day to the day before it (the bars layout, empty when the broker has none),
    read through `client` at `read_ns` (UTC nanoseconds), as a backtest of the strategy
    loads its own bars. Raises the data layer's errors but NoDataError."""
    first = plan.start - dt.timedelta(days=RECENT_DAYS[parse_interval(plan.interval)[0]])
    try:
        result = load_bars(
            symbol,
            exchange,
            DAILY,
            first,
            plan.start - _ONE_DAY,
            source=HISTORY_SOURCE,
            client=client,
            calendar=calendar,
            now=read_ns,
        )
    except NoDataError:
        return empty_base()
    return result.bars


def joined(history: pd.DataFrame | None, recent: pd.DataFrame) -> pd.DataFrame:
    """The daily bars of `history` (the warm-up's, None for none) and `recent` together,
    one per open, sorted; a bar both hold is the recent read's."""
    frames = [_layout(frame) for frame in (history, recent) if frame is not None and len(frame)]
    if not frames:
        return _layout(recent)
    both = pd.concat(frames, ignore_index=True)
    both = both.drop_duplicates("ts_open", keep="last").sort_values("ts_open")
    return both.reset_index(drop=True)


def _layout(frame: pd.DataFrame) -> pd.DataFrame:
    """`frame`'s columns of the bars layout (`BAR_COLUMNS`), its open interest unknown
    (NaN) where it gives none, as resampling takes them."""
    if "oi" not in frame:
        frame = frame.assign(oi=np.nan)
    return frame.loc[:, list(BAR_COLUMNS)]


class DailySeries:
    """A daily or longer strategy's own bars in a live session (module docstring): the
    broker's daily candles `bars` (the bars layout) before the session's day, of `symbol`
    on `exchange`, read as stores of the strategy's interval or any coarser calendar
    interval (`store`), the way `series.LiveSeries` is read: a store holds a bar once the
    decision instant is at or after its close. The session's day adds none: the strategy
    decides before the open only."""

    base = DAILY

    def __init__(self, symbol: str, exchange: str, bars: pd.DataFrame, calendar: Calendar) -> None:
        self.symbol, self.exchange = symbol, exchange
        self._bars = bars.reset_index(drop=True)
        self._calendar = calendar
        self._columns: dict[str, dict[str, np.ndarray]] = {}

    @property
    def version(self) -> int:
        """Always 0: the bars never change once the plan is built."""
        return 0

    def store(self, interval: str, clock: Clock, *, name: str | None = None) -> LiveStore:
        """The store of these bars at `interval`, visible up to the decision instant `clock`
        gives. Raises ConfigError for an interval daily bars cannot give."""
        _check_interval(interval, DAILY)
        return LiveStore(self, interval, clock, name)  # type: ignore[arg-type]

    def history_columns(self, interval: str) -> dict[str, np.ndarray]:
        """Every bar of `interval` the daily bars make, as store columns."""
        columns = self._columns.get(interval)
        if columns is None:
            frame = self._bars
            if interval != DAILY and len(frame):
                frame = resample_bars(
                    frame,
                    interval,
                    exchange=self.exchange,
                    calendar=self._calendar,
                    source_interval=DAILY,
                    on_closed_day="drop",
                )
            columns = {name: frame[name].to_numpy() for name in COLUMNS}
            self._columns[interval] = columns
        return columns

    def today_columns(self, interval: str) -> dict[str, np.ndarray]:
        """None of the session's day (module docstring), as empty store columns."""
        del interval  # every interval holds none of today
        daily = self.history_columns(DAILY)
        return {name: np.array([], dtype=daily[name].dtype) for name in COLUMNS}


def decision_bars(
    store: Any, last_decided_ns: int | None, instrument: int
) -> list[tuple[int, int, int, float, float, float, float, float]]:
    """The strategy's bars in `store` (its interval's) closing after `last_decided_ns` (UTC
    nanoseconds; every bar when None), as `Engine.pre_open_decision` takes them for the
    instrument registered at `instrument`: (instrument, ts_open, ts_close, open, high, low,
    close, volume)."""
    opens, closes = store.column("ts_open"), store.column("ts_close")
    fields = [store.column(name) for name in PRICES]
    bars = []
    for index, closed in enumerate(closes):
        if last_decided_ns is not None and int(closed) <= last_decided_ns:
            continue
        values = tuple(float(field[index]) for field in fields)
        bars.append((instrument, int(opens[index]), int(closed), *values))
    return bars


__all__ = [
    "DAILY",
    "RECENT_DAYS",
    "DailySeries",
    "decides_before_open",
    "decision_bars",
    "history_plan",
    "joined",
    "recent_bars",
]
