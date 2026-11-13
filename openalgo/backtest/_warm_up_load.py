"""Loading the bars before `start` that a run's indicators need (decision D25; the
history's rules are in `_warm_up_history`).

For each traded symbol with a need, and for every symbol after a datetime `start` (the
morning before it), the run's own source is asked for the base bars from the first day
of a request (`_warm_up_days.requests`) to the last bar before `start`: the source the
run's bars came from (Historify or the broker, never both), the user's `data=`, or the
run's data provider, as `HistorySource` records it. While the bars returned fall short
of a need and the source has bars from the request's step, the next request asks
further back (`reach`). A source with fewer bars gives what it has, none when it has
none (a new listing, own data that begins at `start`). The load is not stopped by a
quality error: each run checks the bars it keeps, joined to its own (`_warm_up_join`),
so a problem outside them does not cost it its history. A load that fails otherwise (a
timestamp the source gets wrong) gives no history and says why; the run's own bars are
unaffected. The load's own warnings join the run's, each named "<symbol> history before
start". A futures series loads none (`_warm_up_history`).

**A sweep** loads once per symbol, for the largest need of each series over its
combinations (the covering load), and each combination finds its own request by
stepping through that load as its single run steps through its source: the ladder of
requests is the same for every set of needs, so every step a combination takes is
inside the covering load. When the covering load fails, as one bad bar far before
`start` makes it, each set of needs is loaded on its own, request by request, so a
combination keeps exactly what its single run keeps whatever fails outside its range.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ._base_interval import _data_for
from ._run_corporate_actions import declared_for
from ._warm_up import SeriesNeed
from ._warm_up_days import FIRST_BAR_SLACK_DAYS, Request, by_symbol, requests, run_from
from ._warm_up_history import (
    NO_NEED,
    RunHistory,
    SymbolLoad,
    empty_base,
    empty_history,
    first_kept,
    keep,
)
from ._warm_up_probe import largest
from .data import BarsResult, load_bars
from .data._sessions import day_start_ns
from .errors import DataError, NoDataError

SERVER_SOURCES = ("db", "api")
# A load from a first IST day.
Fetch = Callable[[dt.date], SymbolLoad]


@dataclass(frozen=True)
class HistorySource:
    """Where a run's history is read from: the run's `data=`, `source=`, `tz=` and
    `corporate_actions=` arguments, its client and calendar, the (source, base
    interval) each symbol's run bars came from, and True for a futures series (none
    is read). It holds the client, so it stays in the process that prepared the run."""

    args: Mapping[str, Any]
    client: Any
    calendar: Any
    origins: Mapping[str, tuple[str | None, str | None]]
    futures: bool = False


def history_source(
    args: Mapping[str, Any],
    client: Any,
    calendar: Any,
    loaded: Mapping[str, BarsResult],
    futures: Any,
) -> HistorySource:
    """The source of a run's history from its arguments, client, calendar, its own
    loads by symbol and its futures plan (None without one)."""
    names = ("data", "source", "tz", "corporate_actions")
    origins = {
        symbol: (result.meta.source, result.meta.base_interval) for symbol, result in loaded.items()
    }
    given = {name: args[name] for name in names}
    return HistorySource(given, client, calendar, origins, futures is not None)


def _source(args: Mapping[str, Any], origin: str | None) -> Any:
    """The source the history is read from: the one the run's bars came from."""
    if args["data"] is None and args["source"] == "auto" and origin in SERVER_SOURCES:
        return origin
    return args["source"]


def _partition_key(result: BarsResult) -> str:
    """The partition key prefix of a load (its keys without the month)."""
    partitions = result.meta.partitions
    return partitions[0]["key"].rsplit("/", 1)[0] if partitions else ""


def _failure(error: DataError) -> str:
    """Why a load failed, for the Assumptions block: the error's kind and message."""
    return f"{type(error).__name__}: {error}"


def load_symbol(symbol: str, day: dt.date, plan: Any, source: HistorySource) -> SymbolLoad:
    """One symbol's base bars from `day` to the last bar before `start`. Raises
    ConfigError for arguments the data layer refuses; a failed load is a note, never an
    error."""
    args = source.args
    kind = "source" if args["data"] is None else "own"
    declared = declared_for(args["corporate_actions"], symbol, plan.symbols)
    last_day = plan.start if plan.first_slice is not None else plan.start - dt.timedelta(days=1)
    if day > last_day:
        return SymbolLoad(empty_base(), kind, declared=declared)
    origin, base_interval = source.origins[symbol]
    try:
        result = load_bars(
            symbol,
            plan.exchange,
            plan.base,
            day,
            last_day,
            source=_source(args, origin),
            client=source.client,
            data=_data_for(args["data"], symbol),
            calendar=source.calendar,
            tz=args["tz"],
            base=base_interval,
            now=plan.present,
            strict=False,
            corporate_actions=declared,
        )
    except NoDataError:
        return SymbolLoad(empty_base(), kind, declared=declared)
    except DataError as error:
        return SymbolLoad(empty_base(), kind, _failure(error), declared=declared)
    base = result.bars[result.bars["ts_open"].to_numpy() < run_from(plan)]
    notes = [f"{symbol} history before start: {note}" for note in result.meta.warnings]
    # A source that returned less than was asked for says so in the first warning
    # (`data._finish`, short_history); a run repeats it only when it is short of a need.
    coverage = notes.pop(0) if result.meta.short_history and notes else ""
    key = _partition_key(result)
    return SymbolLoad(base.reset_index(drop=True), kind, "", key, tuple(notes), coverage, declared)


def _no_more(base: pd.DataFrame, step: dt.date) -> bool:
    """True when the source has no bar from the first day of a request's step to a week
    after it: asking further back would find none."""
    edge = day_start_ns(step + dt.timedelta(days=FIRST_BAR_SLACK_DAYS))
    return base.empty or int(base["ts_open"].iloc[0]) >= edge


def reach(
    needs: Sequence[SeriesNeed], plan: Any, calendar: Any, fetch: Fetch
) -> tuple[Request, SymbolLoad]:
    """(the request that meets a symbol's `needs`, its load from `fetch`): the first
    request whose load fails, has every bar the needs count, or shows that the source has
    no more, else the last (module docstring, `_warm_up_days.requests`)."""
    found = requests(needs, plan, calendar)
    for request in found[:-1]:
        load = fetch(request.day)
        if load.failure or request.step is None or _no_more(load.base, request.step):
            return request, load
        if first_kept(load.base, needs, plan, calendar)[1]:
            return request, load
    return found[-1], fetch(found[-1].day)


@dataclass(frozen=True)
class SymbolLoads:
    """One symbol's history as loaded (module docstring): the loads kept, by the first
    day each asked for, and for each set of the symbol's needs the request that meets it
    and the first day of the load that holds its bars."""

    loads: dict[dt.date, SymbolLoad]
    chosen: dict[tuple[SeriesNeed, ...], tuple[Request, dt.date]]
    covering: dt.date

    def _held(self, day: dt.date) -> SymbolLoad:
        """The bars from `day` on, from the load from that day or the covering one."""
        if day in self.loads:
            return self.loads[day]
        return self.loads[self.covering].since(day)

    def for_needs(
        self, needs: tuple[SeriesNeed, ...], plan: Any, calendar: Any
    ) -> tuple[Request, SymbolLoad]:
        """(the request of `needs`, the bars it holds). A set of needs not loaded for
        (parameters the run did not probe) steps through the loads kept."""
        if needs not in self.chosen:
            return reach(needs, plan, calendar, self._held)
        request, day = self.chosen[needs]
        return request, self.loads[day].since(request.day)


def _symbol_loads(
    symbol: str, need_sets: list[tuple[SeriesNeed, ...]], plan: Any, source: HistorySource
) -> SymbolLoads:
    """The loads of one symbol for each of its sets of needs (module docstring)."""
    fetched: dict[dt.date, SymbolLoad] = {}

    def fetch(day: dt.date) -> SymbolLoad:
        if day not in fetched:
            fetched[day] = load_symbol(symbol, day, plan, source)
        return fetched[day]

    widest, covering = reach(largest(need_sets), plan, source.calendar, fetch)

    def within(day: dt.date) -> SymbolLoad:
        if covering.failure or day < widest.day:
            return fetch(day)
        return covering.since(day)

    chosen = {}
    for needs in need_sets:
        request, _ = reach(needs, plan, source.calendar, within)
        loaded_alone = bool(covering.failure) or request.day < widest.day
        chosen[needs] = (request, request.day if loaded_alone else widest.day)
    kept = {day for _, day in chosen.values()} | {widest.day}
    return SymbolLoads({day: fetched[day] for day in sorted(kept)}, chosen, widest.day)


@dataclass(frozen=True)
class LoadedHistory:
    """The bars before `start` loaded for a run or a sweep: each traded symbol's loads
    (none for a symbol nothing needed, nor for a futures series, `futures` True)."""

    loads: dict[str, SymbolLoads]
    futures: bool = False

    def for_run(
        self,
        needs: Sequence[SeriesNeed],
        plan: Any,
        calendar: Any,
        run_bars: Mapping[str, pd.DataFrame],
    ) -> RunHistory:
        """The history a run with `needs` keeps (`_warm_up_history`); `run_bars` are
        the run's base bars by symbol."""
        if self.futures:
            return RunHistory({}, plan.start, futures=True, probed=tuple(needs))
        wanted = by_symbol(needs)
        kept = {}
        for symbol in plan.symbols:
            loads = self.loads.get(symbol)
            if loads is None:
                kept[symbol] = empty_history(symbol, NO_NEED, "unprobed")
                continue
            mine = tuple(wanted.get(symbol, ()))
            request, load = loads.for_needs(mine, plan, calendar)
            kept[symbol] = keep(symbol, load, mine, plan, calendar, run_bars[symbol], request)
        return RunHistory(kept, plan.start, probed=tuple(needs))


def load_history(
    plan: Any, source: HistorySource, parameter_needs: Iterable[tuple[SeriesNeed, ...]]
) -> LoadedHistory:
    """The history of every symbol of `plan` that the needs of the parameter sets probed
    name (one set for a run, every combination's for a sweep), and of every symbol after
    a datetime `start`. Raises ConfigError for an interval the base bars cannot build
    or arguments the data layer refuses."""
    # Imported here, as the run context does: a run on its own interval needs none.
    from ._timeframes import _check_interval

    if source.futures:
        return LoadedHistory({}, futures=True)
    by_set = [by_symbol(needs) for needs in parameter_needs]
    loads = {}
    for symbol in plan.symbols:
        need_sets = list(dict.fromkeys(tuple(found.get(symbol, ())) for found in by_set))
        if not any(need_sets) and plan.first_slice is None:
            continue
        for need in largest(need_sets):
            _check_interval(need.interval, plan.base)
        loads[symbol] = _symbol_loads(symbol, need_sets, plan, source)
    return LoadedHistory(loads)


__all__ = [
    "HistorySource",
    "LoadedHistory",
    "SymbolLoads",
    "history_source",
    "load_history",
    "load_symbol",
    "reach",
]
