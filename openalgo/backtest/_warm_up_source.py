"""Where a run's history before `start` is read from (decision D25; the load is in
`_warm_up_load`): the source of each symbol's run bars, a load of a range of days from
it, and the check that its OpenAlgo server can answer at all.

The history comes from the source the run's bars came from: Historify or the broker,
never both, the user's `data=`, or the run's data provider.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._base_interval import _data_for
from ._series import ist_time
from .data import BarsResult, load_bars
from .errors import DataError, ServerError

SERVER_SOURCES = ("db", "api")
# The days up to the present the server check asks for: a week holds a completed
# session but in the longest market closures, and a broker that keeps only recent
# history (minute bars of the last weeks) still holds it.
RECENT_DAYS = dt.timedelta(days=7)


@dataclass(frozen=True)
class HistorySource:
    """Where a run's history is read from: the run's `data=`, `source=`, `tz=` and
    `corporate_actions=` arguments, its client and calendar, the (source, base
    interval) each symbol's run bars came from, and True for a futures series (none is
    read). It holds the client, so it stays in the process that prepared the run."""

    args: Mapping[str, Any]
    client: Any
    calendar: Any
    origins: Mapping[str, tuple[str | None, str | None]]
    futures: bool = False

    def kind(self) -> str:
        """The kind of source: "own" for the user's `data=`, else "source"."""
        return "source" if self.args["data"] is None else "own"


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


def _history_from(args: Mapping[str, Any], origin: str | None) -> Any:
    """The source the history is read from: the one the run's bars came from
    (`origin`, "db" or "api") for `source="auto"`, else the run's `source=`."""
    if args["data"] is None and args["source"] == "auto" and origin in SERVER_SOURCES:
        return origin
    return args["source"]


def load_range(
    symbol: str, first: dt.date, last: dt.date, plan: Any, source: HistorySource, **options: Any
) -> BarsResult:
    """`symbol`'s base bars from the IST day `first` to `last` from the run's source,
    as `load_bars` returns them (and with the errors it raises); `options` add to its
    arguments."""
    args = source.args
    origin, base_interval = source.origins[symbol]
    return load_bars(
        symbol,
        plan.exchange,
        plan.base,
        first,
        last,
        source=_history_from(args, origin),
        client=source.client,
        data=_data_for(args["data"], symbol),
        calendar=source.calendar,
        tz=args["tz"],
        base=base_interval,
        now=plan.present,
        strict=False,
        **options,
    )


def check_server(symbol: str, plan: Any, source: HistorySource, failure: ServerError) -> None:
    """Raise ServerError when the server that failed a request of `symbol`'s history
    (`failure`) cannot answer at all (`_warm_up_load`): at once for a rate limit, whose
    advice is to raise `pacing=`, and otherwise when it cannot answer for the symbol's
    most recent sessions either (the `RECENT_DAYS` up to the run's present), asked
    without the local cache. Returns when it answers them, rows or none."""
    if failure.rate_limited:
        raise ServerError(
            f"{symbol}: the bars before start that the indicators need could not be "
            f"loaded: {failure}",
            rate_limited=True,
        ) from failure
    last = ist_time(plan.present).date()
    first = last - RECENT_DAYS
    try:
        load_range(symbol, first, last, plan, source, cache=False)
    except ServerError as again:
        raise ServerError(
            f"{symbol}: the OpenAlgo server cannot answer history requests. The bars before "
            f"start that the indicators need failed ({failure}), and so did a request for "
            f"its most recent sessions ({first} to {last}), asked without the local cache. "
            "Check that OpenAlgo is running and logged in to the broker, then run the "
            "backtest again."
        ) from again
    except DataError:
        # The server answered (no rows, an unknown symbol, or rows that fail a check):
        # it is up, and the failure is about the older days, which the caller records
        # as a note.
        return


__all__ = ["HistorySource", "check_server", "history_source", "load_range"]
