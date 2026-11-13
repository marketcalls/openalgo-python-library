"""Loading the bars before `start` that a run's indicators need (decision D25; the
history's rules are in `_warm_up_history`).

For each traded symbol with a need, and for every symbol after a datetime `start` (the
morning before it), the run's own source is asked for the base bars from the first day
the needs ask for (`_warm_up_days.first_requested_day`) to the last bar before
`start`: the source the run's bars came from (Historify or the broker, never both), the
user's `data=`, or the run's data provider, as `HistorySource` records it. A source with
fewer bars gives what it has, none when it has none (a new listing, own data that begins
at `start`). The load is not stopped by a quality error: each run checks the bars it
keeps, joined to its own (`_warm_up_join`), so a problem outside them does not cost it
its history. A load that fails otherwise (a timestamp the source gets wrong) gives no
history and says why; the run's own bars are unaffected. The load's own warnings join
the run's, each named "<symbol> history before start". A futures series loads none
(`_warm_up_history`).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ._base_interval import _data_for
from ._run_corporate_actions import declared_for
from ._warm_up import SeriesNeed
from ._warm_up_days import by_symbol, first_requested_day, run_from
from ._warm_up_history import LoadedHistory, SymbolLoad, empty_base
from .data import BarsResult, load_bars
from .errors import DataError, NoDataError

SERVER_SOURCES = ("db", "api")


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


def load_symbol(
    symbol: str, needs: Sequence[SeriesNeed], plan: Any, source: HistorySource
) -> SymbolLoad:
    """One symbol's load for `needs`. Raises ConfigError for an interval the base bars
    cannot build or arguments the data layer refuses; a failed load is a note, never an
    error."""
    # Imported here, as the run context does: a run on its own interval needs none.
    from ._timeframes import _check_interval

    args = source.args
    for need in needs:
        _check_interval(need.interval, plan.base)
    kind = "source" if args["data"] is None else "own"
    declared = declared_for(args["corporate_actions"], symbol, plan.symbols)
    first_day, _ = first_requested_day(needs, plan, source.calendar)
    last_day = plan.start if plan.first_slice is not None else plan.start - dt.timedelta(days=1)
    if first_day > last_day:
        return SymbolLoad(empty_base(), kind, declared=declared)
    origin, base_interval = source.origins[symbol]
    try:
        result = load_bars(
            symbol,
            plan.exchange,
            plan.base,
            first_day,
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


def load_history(plan: Any, source: HistorySource, needs: Sequence[SeriesNeed]) -> LoadedHistory:
    """The history of every symbol of `plan` that `needs` name (the needs of a run, or
    the largest of a sweep's), and of every symbol after a datetime `start`. Raises
    ConfigError as `load_symbol` does."""
    if source.futures:
        return LoadedHistory({}, futures=True)
    wanted = by_symbol(needs)
    loads = {
        symbol: load_symbol(symbol, wanted.get(symbol, []), plan, source)
        for symbol in plan.symbols
        if symbol in wanted or plan.first_slice is not None
    }
    return LoadedHistory(loads)


__all__ = ["HistorySource", "history_source", "load_history", "load_symbol"]
