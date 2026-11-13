"""The history a backtest loads before `start`, so that its indicators are ready when
trading begins (decision D25; the rule is in `_warm_up`).

Two steps, so that every run of the same strategy and parameters keeps the same bars:

1. **Load** (`_warm_up_load`): for each traded symbol the needs of its series
   (`_warm_up_probe`) are turned into the first calendar day asked for
   (`_warm_up_days.first_requested_day`), and the run's own source is asked for the base
   bars from that day to the last bar before `start`, once: for the run's needs, or for
   the largest needs of a sweep's combinations.
2. **Keep**, per run (`LoadedHistory.for_run`, from `_pipeline.execute`). A run keeps,
   of the bars loaded, those from the first day its own needs ask for, cut to the whole
   sessions that hold the last `need` bars of each of its series (from the start of an
   anchor period, the morning of a datetime start). The kept bars and the run's first
   session are then checked as one series (`data.quality.check_bars`): a quality error
   there, such as a suspected unadjusted split between the history and the run, leaves
   the run without history and says why. A combination of a sweep therefore keeps and
   checks exactly the bars its single run does, whatever else the sweep loaded.

The kept bars are joined before `init` runs (`RunHistory.joined`, `with_base`): the base
bars in front of the run's (every other interval is built from both), and the strategy
bars they make in front of the kernel's. A strategy bar that opens before a datetime
`start` and closes after it is the aggregate of all its base bars, the morning's
included (`SymbolHistory.opening`).

Not loaded, and said so: a futures series rolled from contract to contract (its
contracts and their roll adjustments begin at `start`; building its history from the
contracts before `start` would need their switches and spreads, which the kernel plans
from `start` only, so the strategy waits inside the run). A run on one explicit
contract (`symbol = "NIFTY23NOV26FUT"`) is an ordinary symbol and loads its own bars
before `start` when the source has them. A source with fewer bars gives what it has,
own data that begins at `start` none. The load's own warnings join the run's. The
strategy then waits inside the run (`_warm_up`).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ._messages import counted
from ._series import FIELDS, TIMES, ist_time
from ._warm_up import HISTORY_YEARS_CAP, SeriesNeed
from ._warm_up_anchor import period_start
from ._warm_up_days import by_symbol, first_requested_day, run_from
from ._warm_up_join import Opening, join_failure, opening_part
from .data._finish import _hash_by_month
from .data._sessions import day_start_ns
from .resample import resample_bars

# A first bar within a week of the day asked for is the first session on or after it
# (weekends, holidays); later, the source has no bars between the two, which is said.
FIRST_BAR_SLACK_DAYS = 7
FUTURES_NOTE = "none (a futures series is built from start: its contracts and their rolls)"
FUTURES_REASON = "a futures series is built from start, from its contracts and their rolls"
NO_NEED = "none (no indicator of it needs bars before start)"
NOT_PROBED = (
    "no history was loaded for it: init did not register it when it ran on the engine's "
    "reference bars to learn what history to load"
)


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


def _first_kept(
    base: pd.DataFrame, needs: Sequence[SeriesNeed], plan: Any, calendar: Any
) -> tuple[int, bool]:
    """(the first instant a run keeps, UTC nanoseconds, before its day is taken whole;
    True when the bar needs are met), see the module docstring."""
    instants, enough = [], True
    if plan.first_slice is not None:
        instants.append(int(day_start_ns(plan.start)))
    for need in needs:
        if need.anchor is not None:
            instants.append(int(day_start_ns(period_start(plan.start, need.anchor))))
        elif need.more:
            instants.append(int(base["ts_open"].iloc[0]))
            enough = False
        else:
            instant, met = _count_back(base, need.interval, need.bars, plan, calendar)
            instants.append(instant)
            enough &= met
    return min(instants), enough


@dataclass(frozen=True)
class SymbolHistory:
    """One symbol's history kept by a run: its base bars (canonical layout, possibly
    empty), the strategy bars they make (the columns of `_series`, for `BarStore`), what
    the Assumptions block says of it, the manifest hashes of the bars kept, what to
    say when they fall short of a need (`shortfall`): `kind` is "own" (the user's
    data), "source" (a client source), "failed", or "unprobed" (the needs learnt on the
    reference bars asked for none), `detail` the failure;
    the load's warnings for the run; and the part before `start` of the strategy bar
    that straddles a datetime `start` (`opening`, None without one)."""

    symbol: str
    base: pd.DataFrame
    strategy: dict[str, np.ndarray]
    note: str
    kind: str
    detail: str = ""
    capped: bool = False
    partitions: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    opening: Opening | None = None

    def shortfall(self, first_day: dt.date | None, start: Any) -> str:
        """Why the history of this symbol did not meet a need; `first_day` is the IST
        date of its first bar, history or run."""
        if self.kind == "failed":
            return f"its bars before {start} could not be used ({self.detail})"
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

    def joined(self, symbol: str, run_bars: dict[str, np.ndarray]) -> tuple[dict, int, bool]:
        """(the strategy bars of `symbol` with its history in front, the number of
        history bars, True when the first run bar was rebuilt with the base bars before
        a datetime `start`) from the kernel's strategy bars `run_bars`."""
        history = self.symbols.get(symbol)
        if history is None:
            return run_bars, 0, False
        columns = {name: np.asarray(run_bars[name]) for name in (*TIMES, *FIELDS)}
        rebuilt = history.opening is not None and history.opening.rebuild(columns)
        before = history.strategy
        joined = {name: np.concatenate((before[name], columns[name])) for name in columns}
        return joined, len(before["ts_open"]), rebuilt

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
    Assumptions block says what was asked for and what came back. `declared` are the
    corporate actions the run declares for the symbol, which the join is checked with."""

    base: pd.DataFrame
    kind: str
    failure: str = ""
    key: str = ""
    notes: tuple[str, ...] = ()
    coverage: str = ""
    declared: Any = None


def _asked(day: dt.date, base: pd.DataFrame, capped: bool) -> str:
    """How the Assumptions block says what was asked for and what came back."""
    told = f"asked from {day}"
    if capped:
        told += f", {HISTORY_YEARS_CAP} years before start at most"
    first = ist_time(int(base["ts_open"].iloc[0])).date() if not base.empty else None
    if first is not None and first > day + dt.timedelta(days=FIRST_BAR_SLACK_DAYS):
        told += f"; the source returned bars from {first}"
    return told


def _shown(frame: pd.DataFrame) -> str:
    """The strategy bars kept, as the Assumptions block counts them."""
    shown = counted(len(frame), "strategy bar")
    if len(frame):
        first, last = (ist_time(int(frame["ts_open"].iloc[k])).date() for k in (0, -1))
        shown += f" from {first} to {last}"
    return shown


def keep(
    symbol: str,
    load: SymbolLoad,
    needs: Sequence[SeriesNeed],
    plan: Any,
    calendar: Any,
    run_base: pd.DataFrame,
) -> SymbolHistory:
    """The history one run keeps of a symbol's load for its own `needs`, checked joined
    to the run's base bars `run_base`; see the module docstring."""
    kind = load.kind if needs else "unprobed"
    if not needs and plan.first_slice is None:
        return _empty(symbol, NO_NEED, kind)
    if load.failure:
        note = f"none (the bars before {plan.start} could not be loaded: {load.failure})"
        return _empty(symbol, note, "failed", load.failure)
    day, capped = first_requested_day(needs, plan, calendar)
    base = load.base[load.base["ts_open"].to_numpy() >= day_start_ns(day)]
    if base.empty:
        note = f"none ({_asked(day, base, capped)}; the source has no bars before {plan.start})"
        return _empty(symbol, note, kind)
    asked = _asked(day, base, capped)
    cut, enough = _first_kept(base, needs, plan, calendar)
    kept_from = day_start_ns(ist_time(cut).date())
    base = base[base["ts_open"].to_numpy() >= kept_from].reset_index(drop=True)
    failure = join_failure(symbol, base, run_base, load.declared, plan, calendar)
    if failure:
        note = f"none (the bars before {plan.start} could not be used: {failure})"
        return _empty(symbol, note, "failed", failure)
    frame = _frame(base, plan.interval, plan, calendar)
    strategy = _closed(frame, plan)
    hashes = {entry["key"]: entry["sha256"] for entry in _hash_by_month(base, load.key)}
    warnings = (*load.notes, *((load.coverage,) if not enough and load.coverage else ()))
    return SymbolHistory(
        symbol,
        base,
        {name: strategy[name].to_numpy() for name in (*TIMES, *FIELDS)},
        f"{_shown(strategy)} ({asked})",
        kind,
        capped=capped,
        partitions=hashes,
        warnings=warnings,
        opening=opening_part(frame, run_from(plan)),
    )


@dataclass(frozen=True)
class LoadedHistory:
    """The bars before `start` loaded for a run or a sweep (`_warm_up_load`): each
    traded symbol's load (none for a symbol nothing needed, nor for a futures series,
    `futures` True)."""

    loads: dict[str, SymbolLoad]
    futures: bool = False

    def for_run(
        self,
        needs: Sequence[SeriesNeed],
        plan: Any,
        calendar: Any,
        run_bars: Mapping[str, pd.DataFrame],
    ) -> RunHistory:
        """The history a run with `needs` keeps (module docstring); `run_bars` are the
        run's base bars by symbol."""
        if self.futures:
            return RunHistory({}, plan.start, futures=True)
        wanted = by_symbol(needs)
        kept = {}
        for symbol in plan.symbols:
            load = self.loads.get(symbol)
            if load is None:
                kept[symbol] = _empty(symbol, NO_NEED, "unprobed")
                continue
            mine = wanted.get(symbol, [])
            kept[symbol] = keep(symbol, load, mine, plan, calendar, run_bars[symbol])
        return RunHistory(kept, plan.start)


__all__ = [
    "LoadedHistory",
    "RunHistory",
    "SymbolHistory",
    "SymbolLoad",
    "empty_base",
    "keep",
]
