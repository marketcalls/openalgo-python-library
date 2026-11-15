"""Loading the bars before `start` that a run's indicators need (decision D25; the
history's rules are in `_warm_up_history`).

For each traded symbol with a need, and for every symbol after a datetime `start` (the
morning before it), the run's own source is asked for the base bars from the first day
of a request (`_warm_up_days.requests`) to the last bar before `start`: the source the
run's bars came from (Historify or the broker, never both), the user's `data=`, or the
run's data provider, as `HistorySource` records it.

**Steps back** (`reach`): while the bars returned fall short of a need, the next request
of the ladder asks further back. It asks only for the days before those already held
(`_StepLoads`), and its bars are joined in front of them, so each day is asked for once.
Stepping stops at the first request whose bars meet the needs, at the last request (the
thirty-year cap), and at a request that returned no bar older than the one before it
while the bars held begin more than a week after its first day: the source has none
there, and a gap a request does not cover entirely (a suspension across the first day of
a step, a thinly traded share's minutes without a trade) is stepped over. A source with
fewer bars gives what it has, none when it has none (a new listing, own data that
begins at `start`). The worst case, a symbol whose bars never meet a need and reach back
the whole cap, makes one request per step of the ladder (31 for a need of a few bars,
fewer for a larger one) and asks for the thirty years before `start` once in all.

**Failures**: the load is not stopped by a quality error: each run checks the bars it
keeps, joined to its own (`_warm_up_join`), so a problem outside them does not cost it
its history. A request that fails otherwise (a timestamp the source gets wrong) ends the
steps: the run keeps the bars of the requests before it and says why it stopped, or has
no history when the first request failed; the run's own bars are unaffected. The load's
own warnings join the run's, each named "<symbol> history before start". A futures
series loads none (`_warm_up_history`).

**A server that cannot answer** stops the run: when the OpenAlgo server fails a request
(`ServerError`: it cannot be reached, keeps rate limiting, or answers with an error such
as an expired broker login), the run asks it once more for the symbol's most recent
sessions, bypassing the local cache (`_warm_up_source.check_server`). If that fails too,
the server cannot serve history at all just now, and a run without its history would
trade from bars the cache happened to hold, waiting inside them for its indicators:
`ServerError` is raised. If the server answers it, the failure is about the older days (a
broker that refuses dates before a listing or beyond the recent history it keeps), and
the steps end as above. A rate limit stops the run at once, with its own advice (raise
`pacing=`): its history exists but was not sent.

**A sweep** loads once per symbol, stepping for the largest need of each series over
its combinations first, and each combination then steps through the same loads as its
single run steps through its source: every day a combination asks for is either held
already or asked for as its single run asks for it, so it keeps the bars its single run
keeps, and a request that fails fails for every combination that makes it.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from ._base_interval import _data_for
from ._run_corporate_actions import declared_for
from ._series import ist_time
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
from ._warm_up_source import HistorySource, check_server, load_range
from .data import BarsResult
from .data._sessions import day_start_ns
from .data.loader import own_reader
from .errors import DataError, NoDataError, ServerError

ONE_DAY = dt.timedelta(days=1)
# The bars from a first IST day to the last bar before `start`.
Fetch = Callable[[dt.date], SymbolLoad]


def _partition_key(result: BarsResult) -> str:
    """The partition key prefix of a load (its keys without the month)."""
    partitions = result.meta.partitions
    return partitions[0]["key"].rsplit("/", 1)[0] if partitions else ""


def _given_from(symbol: str, plan: Any, source: HistorySource) -> dt.date | None:
    """The IST date of the first bar the user's own data holds for `symbol`, None for a
    source or for own data with no row of it: what a shortfall of own data names."""
    if source.kind() != "own":
        return None
    reader = own_reader(_data_for(source.args["data"], symbol), source.args["tz"])
    if reader is None:
        return None
    interval = source.origins[symbol][1] or plan.base
    try:
        first, _ = reader.stamp_range(symbol, plan.exchange, interval)
    except NoDataError:
        return None
    return ist_time(first).date()


def _failure(error: DataError) -> str:
    """Why a load failed, for the Assumptions block: the error's kind and message."""
    return f"{type(error).__name__}: {error}"


def load_symbol(
    symbol: str, day: dt.date, plan: Any, source: HistorySource, until: dt.date | None = None
) -> SymbolLoad:
    """One symbol's base bars from `day` to `until` (IST dates, both included; by
    default the last day before `start`, or `start`'s own day before a datetime
    `start`). Raises ConfigError for arguments the data layer refuses, and ServerError
    when the OpenAlgo server cannot answer at all (`check_server`); any other failed
    load is a note."""
    args = source.args
    kind = source.kind()
    declared = declared_for(args["corporate_actions"], symbol, plan.symbols)
    last_day = plan.start if plan.first_slice is not None else plan.start - ONE_DAY
    last_day = last_day if until is None else min(last_day, until)
    if day > last_day:
        return SymbolLoad(empty_base(), kind, declared=declared)
    try:
        result = load_range(symbol, day, last_day, plan, source, corporate_actions=declared)
    except NoDataError:
        return SymbolLoad(empty_base(), kind, declared=declared)
    except ServerError as error:
        check_server(symbol, plan, source, error)
        return SymbolLoad(empty_base(), kind, _failure(error), declared=declared)
    except DataError as error:
        return SymbolLoad(empty_base(), kind, _failure(error), declared=declared)
    base = result.bars[result.bars["ts_open"].to_numpy() < run_from(plan)]
    notes = [f"{symbol} history before start: {note}" for note in result.meta.warnings]
    # A source that returned less than was asked for says so in the first warning
    # (`data._finish`, short_history); a run repeats it only when it is short of a need.
    coverage = notes.pop(0) if result.meta.short_history and notes else ""
    key = _partition_key(result)
    return SymbolLoad(base.reset_index(drop=True), kind, "", key, tuple(notes), coverage, declared)


class _StepLoads:
    """One symbol's bars before `start`, extended further back on demand (module
    docstring): asked for the bars from a day before those held, it loads only the days
    before them and joins them in front. `held` are the bars loaded, from `since`;
    `failed` the loads that failed, by the first day asked for."""

    def __init__(self, symbol: str, plan: Any, source: HistorySource) -> None:
        self._symbol, self._plan, self._source = symbol, plan, source
        self._given_from = _given_from(symbol, plan, source)
        self.held: SymbolLoad | None = None
        self.since: dt.date | None = None
        self.failed: dict[dt.date, SymbolLoad] = {}

    def __call__(self, day: dt.date) -> SymbolLoad:
        """The bars from `day` to the last bar before `start` (a `Fetch`)."""
        if self.held is not None and self.since is not None and day >= self.since:
            return self.held.since(day)
        if day in self.failed:
            return self.failed[day]
        until = None if self.since is None else self.since - ONE_DAY
        piece = load_symbol(self._symbol, day, self._plan, self._source, until)
        piece = replace(piece, given_from=self._given_from)
        if piece.failure:
            self.failed[day] = piece
            return piece
        self.held = piece if self.held is None else self.held.after(piece)
        self.since = day
        return self.held


def _no_more(base: pd.DataFrame, step: dt.date | None) -> bool:
    """True when the bars held begin more than a week after the first day of a
    request's step (or there are none): the source has no bar there."""
    if base.empty or step is None:
        return True
    edge = day_start_ns(step + dt.timedelta(days=FIRST_BAR_SLACK_DAYS))
    return int(base["ts_open"].iloc[0]) >= edge


def _no_older(load: SymbolLoad, kept: tuple[Request, SymbolLoad] | None) -> bool:
    """True when a request's bars hold none older than the request's before it."""
    if load.base.empty:
        return True
    if kept is None or kept[1].base.empty:
        return False
    return int(load.base["ts_open"].iloc[0]) >= int(kept[1].base["ts_open"].iloc[0])


def _stopped(
    kept: tuple[Request, SymbolLoad] | None, request: Request, failed: SymbolLoad
) -> tuple[Request, SymbolLoad]:
    """The request before a failed one and its bars, with why the steps stopped; the
    failed request itself when it was the first."""
    if kept is None:
        return request, failed
    good_request, good = kept
    why = f"asking for its bars from {request.day} failed ({failed.failure})"
    return good_request, replace(good, stopped=why)


def reach(
    needs: Sequence[SeriesNeed], plan: Any, calendar: Any, fetch: Fetch
) -> tuple[Request, SymbolLoad]:
    """(the request that meets a symbol's `needs`, its bars from `fetch`): the requests
    are tried in order until one meets the needs, shows the source has no more, or is
    the last; when one fails, the one before it is kept (module docstring,
    `_warm_up_days.requests`)."""
    found = requests(needs, plan, calendar)
    kept: tuple[Request, SymbolLoad] | None = None
    for request in found[:-1]:
        load = fetch(request.day)
        if load.failure:
            return _stopped(kept, request, load)
        if _no_older(load, kept) and _no_more(load.base, request.step):
            return request, load
        if first_kept(load.base, needs, plan, calendar)[1]:
            return request, load
        kept = (request, load)
    last = found[-1]
    load = fetch(last.day)
    return _stopped(kept, last, load) if load.failure else (last, load)


@dataclass(frozen=True)
class SymbolLoads:
    """One symbol's history as loaded (module docstring): the bars held, from `since`,
    the loads that failed by the first day each asked for, the kind of source, and for
    each set of the symbol's needs the request that meets it and why its steps stopped
    early ("" when they did not)."""

    held: SymbolLoad | None
    since: dt.date | None
    failed: dict[dt.date, SymbolLoad]
    kind: str
    chosen: dict[tuple[SeriesNeed, ...], tuple[Request, str]]

    def _held_from(self, day: dt.date) -> SymbolLoad:
        """The bars from `day` on as loaded: the load that failed from that day, else
        those held (from `since`, when `day` is earlier: nothing older was loaded)."""
        if day in self.failed:
            return self.failed[day]
        if self.held is None:
            return SymbolLoad(empty_base(), self.kind)
        return self.held.since(day)

    def for_needs(
        self, needs: tuple[SeriesNeed, ...], plan: Any, calendar: Any
    ) -> tuple[Request, SymbolLoad]:
        """(the request of `needs`, the bars it holds). A set of needs not loaded for
        (parameters the run did not probe) steps through the bars held."""
        if needs not in self.chosen:
            return reach(needs, plan, calendar, self._held_from)
        request, stopped = self.chosen[needs]
        load = self._held_from(request.day)
        return request, replace(load, stopped=stopped) if stopped else load


def _symbol_loads(
    symbol: str, need_sets: list[tuple[SeriesNeed, ...]], plan: Any, source: HistorySource
) -> SymbolLoads:
    """The loads of one symbol for each of its sets of needs (module docstring)."""
    fetch = _StepLoads(symbol, plan, source)
    reach(largest(need_sets), plan, source.calendar, fetch)
    chosen = {}
    for needs in need_sets:
        request, load = reach(needs, plan, source.calendar, fetch)
        chosen[needs] = (request, load.stopped)
    return SymbolLoads(fetch.held, fetch.since, dict(fetch.failed), source.kind(), chosen)


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
    or arguments the data layer refuses, and ServerError when the OpenAlgo server
    cannot answer (`load_symbol`)."""
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
    "LoadedHistory",
    "SymbolLoads",
    "load_history",
    "load_symbol",
    "reach",
]
