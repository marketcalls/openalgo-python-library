"""The history a backtest loads before `start`, so that its indicators are ready when
trading begins (decision D25; the rule is in `_warm_up`).

Two steps, so that every run of the same strategy and parameters keeps the same bars:

1. **Load** (`load_history`): for each traded symbol the need of each of its series, in
   that series' own interval (`_warm_up_probe.history_needs`), is turned into calendar
   days before `start` (`first_requested_day`: sessions of intraday bars, five sessions
   a week with a holiday allowance and a week more, the longest week, month, quarter or
   year of a period bar, and at most `HISTORY_YEARS_CAP` years). The run's own source
   (`HistorySource`) is asked for the base bars from that day to the last bar before
   `start` (with a datetime `start`, the bars of its day before it): the source the
   run's bars came from (Historify or the broker, never both), the user's `data=`, or
   the run's data provider. A single run loads after its `init`, for its own needs; a
   sweep loads once in `_pipeline.prepare`, for the largest need of its combinations
   (probed on the reference bars), as its worker processes load no data.
2. **Keep**, per run (`LoadedHistory.for_run`, from `_pipeline.execute`). A run keeps,
   of the bars loaded, those from the first day its own needs ask for, cut to the whole
   sessions that hold the last `need` bars of each of its series. A combination of a
   sweep therefore keeps exactly the bars its single run loads, and a rerun keeps them
   again.

Not loaded, and said so: a futures series rolled from contract to contract (its
contracts and their roll adjustments begin at `start`; building its history from the
contracts before `start` would need their switches and spreads, which the kernel plans
from `start` only, so the strategy waits inside the run). A run on one explicit
contract (`symbol = "NIFTY23NOV26FUT"`) is an ordinary symbol and loads its own bars
before `start` when the source has them. A source with fewer bars gives what it has,
own data that begins at `start` none. A load that fails (a quality error, a suspected
unadjusted split in the history) gives no history and says why; the run's own bars are
unaffected. The load's own warnings join the run's. The strategy then waits inside the
run (`_warm_up`).
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ._layout import IST_OFFSET_NANOS, NANOS_PER_SECOND, parse_interval
from ._messages import counted
from ._series import FIELDS, TIMES
from ._warm_up import SeriesNeed
from .data._finish import _hash_by_month
from .resample import resample_bars

# How far back history reaches at most: thirty years before `start`. Longer than the
# indicators in common use need on daily bars (a 200-day EMA needs 2,048 sessions, about
# eight years) and a bound on the load of an intraday base.
HISTORY_YEARS_CAP = 30
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
# A first bar within a week of the day asked for is the first session on or after it
# (weekends, holidays); later, the source has no bars between the two, which is said.
FIRST_BAR_SLACK_DAYS = 7
NANOS_PER_DAY = 86_400 * NANOS_PER_SECOND
FUTURES_NOTE = "none (a futures series is built from start: its contracts and their rolls)"
FUTURES_REASON = "a futures series is built from start, from its contracts and their rolls"
NO_NEED = "none (no indicator of it needs bars before start)"
NOT_PROBED = (
    "the sweep loaded no history for it: init did not register it when it ran on the "
    "engine's reference bars"
)


def _ist_date(ts_ns: int) -> dt.date:
    return dt.date(1970, 1, 1) + dt.timedelta(days=(ts_ns + IST_OFFSET_NANOS) // NANOS_PER_DAY)


def _ist_midnight(day: dt.date) -> int:
    """UTC nanoseconds of IST midnight at the start of `day`."""
    return int(pd.Timestamp(day).value) - IST_OFFSET_NANOS


def run_from(plan: Any) -> int:
    """The first instant of the run (UTC nanoseconds): the datetime `start`, or IST
    midnight of the `start` date. History bars open before it."""
    return plan.first_slice if plan.first_slice is not None else _ist_midnight(plan.start)


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


def first_requested_day(needs: Mapping[str, int], plan: Any, calendar: Any) -> tuple[dt.date, bool]:
    """(the first IST date asked for so that every series of a symbol holds its need,
    `needs` being bars by interval; True when `HISTORY_YEARS_CAP` cut it)."""
    days = max(_calendar_days(interval, bars, plan, calendar) for interval, bars in needs.items())
    cap = HISTORY_YEARS_CAP * DAYS_PER_YEAR
    return plan.start - dt.timedelta(days=min(days, cap)), days > cap


def by_symbol(needs: tuple[SeriesNeed, ...]) -> dict[str, dict[str, int]]:
    """The bars each series needs, by symbol and interval."""
    found: dict[str, dict[str, int]] = {}
    for need in needs:
        series = found.setdefault(need.symbol, {})
        series[need.interval] = max(series.get(need.interval, 0), need.bars)
    return found


def _frame(base: pd.DataFrame, interval: str, plan: Any, calendar: Any) -> pd.DataFrame:
    """The bars of `interval` the base bars make, built as the run builds them."""
    if interval == plan.base or base.empty:
        return base
    return resample_bars(
        base,
        interval,
        exchange=plan.exchange,
        calendar=calendar,
        source_interval=plan.base,
        on_closed_day="drop",
    )


def _closed(frame: pd.DataFrame, plan: Any) -> pd.DataFrame:
    """The bars closed by the run's first instant (a bar it cuts belongs to the run)."""
    return frame[frame["ts_close"].to_numpy() <= run_from(plan)].reset_index(drop=True)


def _count_back(
    base: pd.DataFrame, interval: str, bars: int, plan: Any, calendar: Any
) -> tuple[int, bool]:
    """(the open of the `bars`-th last bar of `interval` closed before the run, UTC
    nanoseconds, or of the first such bar when there are fewer; True when there are
    `bars` of them)."""
    closed = _closed(_frame(base, interval, plan, calendar), plan)
    if closed.empty:
        return int(base["ts_open"].iloc[0]), False
    return int(closed["ts_open"].iloc[max(len(closed) - bars, 0)]), len(closed) >= bars


@dataclass(frozen=True)
class SymbolHistory:
    """One symbol's history kept by a run: its base bars (canonical layout, possibly
    empty), the strategy bars they make (the columns of `_series`, for `BarStore`), what
    the Assumptions block says of it, the manifest hashes of the bars kept, what to
    say when they fall short of a need (`shortfall`): `kind` is "own" (the user's
    data), "source" (a client source), "failed" or "unprobed", `detail` the failure;
    and the load's warnings for the run."""

    symbol: str
    base: pd.DataFrame
    strategy: dict[str, np.ndarray]
    note: str
    kind: str
    detail: str = ""
    capped: bool = False
    partitions: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def shortfall(self, first_day: dt.date | None, start: Any) -> str:
        """Why the history of this symbol did not meet a need; `first_day` is the IST
        date of its first bar, history or run."""
        if self.kind == "failed":
            return f"its bars before {start} could not be loaded ({self.detail})"
        if self.kind == "unprobed":
            return NOT_PROBED
        if self.capped:
            return f"the history reaches back at most {HISTORY_YEARS_CAP} years before start"
        if self.kind == "own":
            return (
                f"the bars given for {self.symbol} begin on {first_day}; give earlier ones to "
                f"decide from {start}"
            )
        return f"the source has no bars of {self.symbol} before {first_day}"


def empty_base() -> pd.DataFrame:
    """No base bars, in the canonical columns."""
    base = pd.DataFrame({name: np.array([], dtype=np.int64) for name in TIMES})
    for name in FIELDS:
        base[name] = np.array([], dtype=np.float64)
    return base


def _empty(symbol: str, note: str, kind: str, detail: str = "") -> SymbolHistory:
    base = empty_base()
    strategy = {name: base[name].to_numpy() for name in (*TIMES, *FIELDS)}
    return SymbolHistory(symbol, base, strategy, note, kind, detail)


@dataclass(frozen=True)
class RunHistory:
    """The history one run keeps, by traded symbol in registration order; `futures`
    is True for a futures series, which has none."""

    symbols: dict[str, SymbolHistory]
    start: Any
    futures: bool = False

    @property
    def warnings(self) -> list[str]:
        """The loads' warnings, symbol by symbol."""
        return [note for history in self.symbols.values() for note in history.warnings]

    def strategy_bars(self) -> dict[str, dict[str, np.ndarray]]:
        """The strategy bars before `start` by symbol, for the run context."""
        return {symbol: history.strategy for symbol, history in self.symbols.items()}

    def with_base(self, symbol: str, run_bars: pd.DataFrame) -> pd.DataFrame:
        """The run's base bars of `symbol` after its history (canonical layout)."""
        history = self.symbols.get(symbol)
        if history is None or history.base.empty:
            return run_bars
        return pd.concat([history.base, run_bars], ignore_index=True)

    def partitions(self) -> dict[str, dict[str, str]]:
        """The partition hashes of the bars kept, as the manifest's data records them:
        under "<symbol> history", for each symbol that has some."""
        return {
            f"{symbol} history": history.partitions
            for symbol, history in self.symbols.items()
            if history.partitions
        }

    def describe(self) -> str:
        """What was kept, symbol by symbol, for the Assumptions block."""
        if self.futures:
            return FUTURES_NOTE
        return ", ".join(f"{symbol} {history.note}" for symbol, history in self.symbols.items())

    def shortfall(self, symbol: str, first_day: dt.date | None) -> str:
        """Why the history of `symbol` (or a futures underlying) did not meet a need."""
        if self.futures or symbol not in self.symbols:
            return FUTURES_REASON
        return self.symbols[symbol].shortfall(first_day, self.start)


@dataclass(frozen=True)
class SymbolLoad:
    """One symbol's bars before `start` as loaded (`_warm_up_load`): base bars that
    open before the run's first instant (canonical layout), the source's kind ("own" or
    "source"), and, when there are none, the failure that stopped the load ("" when the
    source simply had none). `key` prefixes its manifest partition keys. `notes` are the
    load's warnings, and `coverage` the one that says the source returned less than was
    asked for, which a run gives only when its history falls short of a need: the
    Assumptions block says what was asked for and what came back."""

    base: pd.DataFrame
    kind: str
    failure: str = ""
    key: str = ""
    notes: tuple[str, ...] = ()
    coverage: str = ""


def _asked(day: dt.date, base: pd.DataFrame, capped: bool) -> str:
    """How the Assumptions block says what was asked for and what came back."""
    told = f"asked from {day}"
    if capped:
        told += f", {HISTORY_YEARS_CAP} years before start at most"
    first = _ist_date(int(base["ts_open"].iloc[0])) if not base.empty else None
    if first is not None and first > day + dt.timedelta(days=FIRST_BAR_SLACK_DAYS):
        told += f"; the source returned bars from {first}"
    return told


def keep(
    symbol: str, load: SymbolLoad, needs: Mapping[str, int], plan: Any, calendar: Any
) -> SymbolHistory:
    """The history one run keeps of a symbol's load for its own `needs` (bars by
    interval); see the module docstring."""
    if not needs:
        return _empty(symbol, NO_NEED, load.kind)
    if load.failure:
        note = f"none (the bars before {plan.start} could not be loaded: {load.failure})"
        return _empty(symbol, note, "failed", load.failure)
    day, capped = first_requested_day(needs, plan, calendar)
    base = load.base[load.base["ts_open"].to_numpy() >= _ist_midnight(day)]
    if base.empty:
        note = f"none ({_asked(day, base, capped)}; the source has no bars before {plan.start})"
        return _empty(symbol, note, load.kind)
    asked = _asked(day, base, capped)
    counted_back = [
        _count_back(base, interval, bars, plan, calendar) for interval, bars in needs.items()
    ]
    cut = min(instant for instant, _ in counted_back)
    base = base[base["ts_open"].to_numpy() >= _ist_midnight(_ist_date(cut))].reset_index(drop=True)
    frame = _closed(_frame(base, plan.interval, plan, calendar), plan)
    strategy = {name: frame[name].to_numpy() for name in (*TIMES, *FIELDS)}
    shown = counted(len(frame), "strategy bar")
    if len(frame):
        opens = frame["ts_open"]
        shown += f" from {_ist_date(int(opens.iloc[0]))} to {_ist_date(int(opens.iloc[-1]))}"
    hashes = {entry["key"]: entry["sha256"] for entry in _hash_by_month(base, load.key)}
    short = not all(enough for _, enough in counted_back)
    warnings = (*load.notes, *((load.coverage,) if short and load.coverage else ()))
    return SymbolHistory(
        symbol,
        base,
        strategy,
        f"{shown} ({asked})",
        load.kind,
        capped=capped,
        partitions=hashes,
        warnings=warnings,
    )


@dataclass(frozen=True)
class LoadedHistory:
    """The bars before `start` loaded for a run or a sweep (`_warm_up_load`): each
    traded symbol's load (none for a symbol nothing needed, nor for a futures series,
    `futures` True)."""

    loads: dict[str, SymbolLoad]
    futures: bool = False

    def for_run(self, needs: tuple[SeriesNeed, ...], plan: Any, calendar: Any) -> RunHistory:
        """The history a run with `needs` keeps (module docstring)."""
        if self.futures:
            return RunHistory({}, plan.start, futures=True)
        wanted = by_symbol(needs)
        kept = {}
        for symbol in plan.symbols:
            load = self.loads.get(symbol)
            if load is None:
                kept[symbol] = _empty(symbol, NO_NEED, "unprobed")
                continue
            kept[symbol] = keep(symbol, load, wanted.get(symbol, {}), plan, calendar)
        return RunHistory(kept, plan.start)


__all__ = [
    "CALENDAR_DAYS_PER_SESSION",
    "HISTORY_YEARS_CAP",
    "HOLIDAY_ALLOWANCE",
    "LoadedHistory",
    "RunHistory",
    "SymbolHistory",
    "SymbolLoad",
    "by_symbol",
    "empty_base",
    "first_requested_day",
    "keep",
    "run_from",
]
