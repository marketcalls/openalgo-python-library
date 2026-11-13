"""The history a backtest keeps before `start`, so that its indicators are ready when
trading begins (decision D25; the rule is in `_warm_up`).

Two steps, so that every run of the same strategy and parameters keeps the same bars:

1. **Load** (`_warm_up_load`): for each traded symbol the needs of its series
   (`_warm_up_probe`) are turned into requests (`_warm_up_days.requests`), and the run's
   own source is asked for the base bars from the first day of a request to the last bar
   before `start`, further back while they fall short of a need: for the run's needs,
   or once for the largest needs of a sweep's combinations, in which each combination
   finds the request its single run makes.
2. **Keep**, per run (`keep`, from `_warm_up_load.LoadedHistory.for_run`). A run keeps,
   of the bars of its request, the whole sessions that hold the last `need` bars of each
   of its series (from the start of the first anchor period an anchored need reads, the
   morning of a datetime start). The kept bars and the run's first base bar are then
   checked as one series (`_warm_up_join`): a quality error there, such as a suspected
   unadjusted split between the history and the run, leaves the run without history and
   says why, and a declared split or bonus the history is not adjusted for is adjusted
   for in it (prices before its ex-date divided by its factor, volume multiplied), which
   the Assumptions line says. A combination of a sweep therefore keeps and checks exactly
   the bars its single run does, whatever else the sweep loaded.

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
strategy then waits inside the run (`_warm_up`). A need the run's `init` registers but
did not register on the reference bars was loaded for by nothing, and its wait says so.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from ._messages import counted
from ._series import FIELDS, TIMES, ist_time
from ._warm_up import HISTORY_YEARS_CAP, SeriesNeed
from ._warm_up_anchor import first_day
from ._warm_up_days import FIRST_BAR_SLACK_DAYS, Request, run_from
from ._warm_up_join import Opening, join, opening_part
from .data._finish import _hash_by_month
from .data._sessions import day_start_ns
from .resample import resample_bars

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


def first_kept(
    base: pd.DataFrame, needs: Sequence[SeriesNeed], plan: Any, calendar: Any
) -> tuple[int, bool]:
    """(the first instant a run keeps of the non-empty base bars `base`, UTC nanoseconds,
    before its day is taken whole; True when every need that counts bars has them), see
    the module docstring."""
    instants, enough = [], True
    if plan.first_slice is not None:
        instants.append(int(day_start_ns(plan.start)))
    for need in needs:
        if need.anchor is not None:
            day = first_day(plan.start, need.anchor, calendar, plan.exchange)
            instants.append(int(day_start_ns(day)))
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
    empty, adjusted for the declared splits and bonuses inside it), the strategy bars
    they make (the columns of `_series`, for `BarStore`), what the Assumptions block says
    of it, the manifest hashes of the bars kept as the source gave them, what to say when
    they fall short of a need (`shortfall`): `kind` is "own" (the user's data), "source"
    (a client source), "failed" or "unprobed" (no need of it was learnt on the reference
    bars), `detail` the failure; the load's warnings for the run; and the part before
    `start` of the strategy bar that straddles a datetime `start` (`opening`, None
    without one)."""

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
        """Why the history of this symbol did not meet a need its run learnt on the
        reference bars; `first_day` is the IST date of the first bar of the need's
        series, history or run."""
        if self.kind == "failed":
            return f"its bars before {start} could not be used ({self.detail})"
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


def empty_history(symbol: str, note: str, kind: str, detail: str = "") -> SymbolHistory:
    """A symbol's history of no bars: `note` for the Assumptions block, its kind and
    failure (`SymbolHistory`)."""
    base = empty_base()
    strategy = {name: base[name].to_numpy() for name in (*TIMES, *FIELDS)}
    return SymbolHistory(symbol, base, strategy, note, kind, detail)


def _covered(need: SeriesNeed, probed: Sequence[SeriesNeed]) -> bool:
    """True when a need learnt on the reference bars (`probed`) asked for at least
    `need`: so its history was loaded for it."""
    return any(
        held.key() == need.key() and (held.bars, held.more) >= (need.bars, need.more)
        for held in probed
    )


@dataclass(frozen=True)
class RunHistory:
    """The history one run keeps, by traded symbol in registration order; the needs it
    was kept for (`probed`, learnt on the reference bars); `futures` is True for a
    futures series, which has none."""

    symbols: dict[str, SymbolHistory]
    start: Any
    futures: bool = False
    probed: tuple[SeriesNeed, ...] = ()

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

    def shortfall(self, need: SeriesNeed, first_day: dt.date | None) -> str:
        """Why the history did not meet `need` of a symbol (or a futures underlying);
        `first_day` is the IST date of the first bar of its series."""
        if self.futures or need.symbol not in self.symbols:
            return FUTURES_REASON
        if need.computed and not _covered(need, self.probed):
            return NOT_PROBED
        return self.symbols[need.symbol].shortfall(first_day, self.start)


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

    def since(self, day: dt.date) -> SymbolLoad:
        """This load's bars from `day` on (IST), as a load from `day` gives them."""
        opens = self.base["ts_open"].to_numpy()
        first = int(np.searchsorted(opens, day_start_ns(day), side="left"))
        if not first:
            return self
        return replace(self, base=self.base.iloc[first:].reset_index(drop=True))


def _asked(request: Request, base: pd.DataFrame) -> str:
    """How the Assumptions block says what was asked for and what came back."""
    told = f"asked from {request.day}"
    if request.capped:
        told += f", {HISTORY_YEARS_CAP} years before start at most"
    first = ist_time(int(base["ts_open"].iloc[0])).date() if not base.empty else None
    if first is not None and first > request.day + dt.timedelta(days=FIRST_BAR_SLACK_DAYS):
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
    request: Request,
) -> SymbolHistory:
    """The history one run keeps of a symbol's load for its own `needs` (the bars of
    its `request`), checked and adjusted where it meets the run's base bars `run_base`;
    see the module docstring."""
    kind = load.kind if needs else "unprobed"
    if not needs and plan.first_slice is None:
        return empty_history(symbol, NO_NEED, kind)
    if load.failure:
        note = f"none (the bars before {plan.start} could not be loaded: {load.failure})"
        return empty_history(symbol, note, "failed", load.failure)
    base = load.base
    if base.empty:
        note = f"none ({_asked(request, base)}; the source has no bars before {plan.start})"
        return empty_history(symbol, note, kind)
    cut, enough = first_kept(base, needs, plan, calendar)
    kept_from = day_start_ns(ist_time(cut).date())
    base = base[base["ts_open"].to_numpy() >= kept_from].reset_index(drop=True)
    joined = join(symbol, base, run_base, load.declared, plan, calendar)
    if joined.failure:
        note = f"none (the bars before {plan.start} could not be used: {joined.failure})"
        return empty_history(symbol, note, "failed", joined.failure)
    frame = _frame(joined.base, plan.interval, plan, calendar)
    strategy = _closed(frame, plan)
    hashes = {entry["key"]: entry["sha256"] for entry in _hash_by_month(base, load.key)}
    warnings = (*load.notes, *((load.coverage,) if not enough and load.coverage else ()))
    told = "; ".join((_asked(request, load.base), *joined.adjusted))
    return SymbolHistory(
        symbol,
        joined.base,
        {name: strategy[name].to_numpy() for name in (*TIMES, *FIELDS)},
        f"{_shown(strategy)} ({told})",
        kind,
        capped=request.capped,
        partitions=hashes,
        warnings=warnings,
        opening=opening_part(frame, run_from(plan)),
    )


__all__ = [
    "NO_NEED",
    "NOT_PROBED",
    "RunHistory",
    "SymbolHistory",
    "SymbolLoad",
    "empty_base",
    "empty_history",
    "first_kept",
    "keep",
]
