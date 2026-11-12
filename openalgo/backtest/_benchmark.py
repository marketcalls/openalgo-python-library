"""The benchmark of a run (07-research-and-reporting 2.4 and 3, FR-REP-05): reference
closes the reports compare the strategy with. It never enters the engine, never trades
and never changes a fill, and benchmark data that cannot be had is a warning, never a
stop. `bt.backtest(benchmark=...)` takes one of:

    "NIFTY"                  an index symbol on the index exchange of the strategy's
                             exchange: NSE_INDEX for NSE and NFO, BSE_INDEX for BSE and
                             BFO (the server pairs an index exchange with its
                             derivatives exchange the same way)
    ("SENSEX", "BSE_INDEX")  an index symbol and its exchange, in the SDK's (symbol,
                             exchange) order, on an index exchange the session calendar
                             knows (BENCHMARK_EXCHANGES)
    a pandas Series          closes (positive index or price levels, never returns)
                             indexed by time, or a DataFrame with a close column; read
                             by `_benchmark_given`
    None                     no benchmark, the default: no index is assumed, since one
                             would cost a server call nobody asked for and a run on
                             own data has no server

A symbol's daily closes come through the data layer exactly as the strategy's bars do:
from `data=` when it is a dict holding the symbol (own intraday bars are resampled to
daily), otherwise from OpenAlgo through `client` and `source=` (Historify or the
broker, by the same "auto" rule), on the run's calendar, keeping only bars closed by
`now=`. A run on own data never reads a server, so a symbol `data=` does not hold has
no closes. The load starts at the index's last session before the run's start: that
close is the level at the opening, which `metrics.analyse` reads as of the start, so
the first session has a benchmark return.

Coverage: the run's sessions (the IST dates of the strategy's bars) and the session
before the first must each have a close; a warning names those that do not. The
metrics read the benchmark as of each session end and never take a close from a later
day, so such a session keeps the last earlier close (a zero benchmark return).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._benchmark_given import read_given
from ._calendar_convert import EXCHANGES, from_day
from ._layout import parse_interval
from ._slab_ticks import last_session_before
from .data import BarsResult, LoadMetadata, load_bars
from .errors import CalendarError, ConfigError, DataError, TimestampError
from .reference.symbols import INDEX_EXCHANGES
from .report._periods import ist_days, ist_times

# The index exchange of a bare symbol, by the strategy's exchange.
INDEX_EXCHANGE_OF = {
    "NSE": "NSE_INDEX",
    "NFO": "NSE_INDEX",
    "BSE": "BSE_INDEX",
    "BFO": "BSE_INDEX",
}
# Index exchanges whose sessions the calendar knows: a benchmark is placed on them.
BENCHMARK_EXCHANGES = tuple(code for code in INDEX_EXCHANGES if code in EXCHANGES)
DAILY = "D"
ONE_MINUTE = "1m"
# A coverage warning lists this many dates, then counts the rest.
LISTED_DATES = 5

# What `benchmark=` asked for, checked: (symbol, index exchange) or given closes.
Request = tuple[str, str] | pd.Series | pd.DataFrame | None


@dataclass(frozen=True)
class Benchmark:
    """A run's benchmark: its `name` ("NIFTY on NSE_INDEX"), where its closes came from
    (`origin`), the closes (positive levels indexed by IST time, None when none could
    be had), the warnings about it and its manifest entry (`record`: symbol, exchange,
    source and the partition hashes of its data)."""

    name: str
    origin: str
    closes: pd.Series | None
    warnings: tuple[str, ...]
    record: dict[str, Any]

    def assumption(self) -> str:
        """The Assumptions line: what the benchmark is and where its closes came from."""
        if self.closes is None:
            return f"{self.name}: no closes, so no benchmark figures (see the warnings)"
        first, last = self.closes.index[0].date(), self.closes.index[-1].date()
        return f"{self.name}: {len(self.closes)} {self.origin}, {first} to {last}"


def benchmark_request(value: Any, exchange: str) -> Request:
    """`benchmark=` checked before any data is loaded, for a strategy on `exchange`.
    Raises ConfigError for anything but the forms of the module docstring, a bare
    symbol on an exchange with no index exchange of its own, and an exchange the
    calendar has no sessions for."""
    if value is None or isinstance(value, pd.Series | pd.DataFrame):
        return value
    if isinstance(value, str) and value:
        if exchange not in INDEX_EXCHANGE_OF:
            raise ConfigError(
                f"benchmark={value!r}: a strategy on {exchange} has no index exchange of its "
                f"own; name one, for example benchmark=({value!r}, 'NSE_INDEX'), or pass the "
                "closes as a pandas Series"
            )
        return value, INDEX_EXCHANGE_OF[exchange]
    if (
        isinstance(value, tuple | list)
        and len(value) == 2
        and all(isinstance(part, str) and part for part in value)
    ):
        if value[1] not in BENCHMARK_EXCHANGES:
            raise ConfigError(
                f"benchmark={value!r}: the exchange must be one of {BENCHMARK_EXCHANGES}, "
                "whose sessions the calendar knows; pass any other series as a pandas "
                "Series of closes"
            )
        return value[0], value[1]
    raise ConfigError(
        "benchmark takes an index symbol ('NIFTY'), a (symbol, exchange) pair such as "
        "('SENSEX', 'BSE_INDEX'), a pandas Series of closes, a DataFrame with a close "
        f"column, or None; got {value!r}"
    )


def load_benchmark(
    request: Request,
    plan: Any,
    args: dict[str, Any],
    client: Any,
    calendar: Any,
    frames: list[pd.DataFrame],
) -> Benchmark | None:
    """The benchmark a checked request names, for the run of `plan` with the arguments
    `args`, whose strategy bars are `frames`; None without one. Raises ConfigError for
    unusable given closes; data that cannot be had is a warning of the result."""
    if request is None:
        return None
    if isinstance(request, tuple):
        symbol, exchange = request
        found = _from_symbol(symbol, exchange, plan, args, client, calendar)
    else:
        exchange = plan.exchange
        found = _from_given(request, exchange, calendar)
    if found.closes is None:
        return found
    missing, needed = _uncovered(found.closes, _run_days(frames), exchange, calendar)
    if not missing:
        return found
    note = _coverage_note(found.name, missing, needed)
    return dataclasses.replace(found, warnings=(*found.warnings, note))


def benchmark_partitions(manifest: dict[str, Any]) -> Any:
    """The partition hashes of the benchmark data a manifest records, None without a
    benchmark."""
    record = manifest["arguments"].get("benchmark")
    return None if record is None else record["partitions"]


def _absent(name: str, record: dict[str, Any], why: str) -> Benchmark:
    return Benchmark(name, "", None, (f"benchmark {name}: {why}; the run has none",), record)


def _from_symbol(
    symbol: str, exchange: str, plan: Any, args: dict[str, Any], client: Any, calendar: Any
) -> Benchmark:
    """An index's daily closes through the data layer, by the rules of the module
    docstring."""
    name = f"{symbol} on {exchange}"
    record: dict[str, Any] = {
        "symbol": symbol,
        "exchange": exchange,
        "source": None,
        "partitions": {},
    }
    data = args["data"]
    own = data.get(symbol) if isinstance(data, dict) else None
    if data is not None and own is None:
        why = (
            f"data= holds no bars for {symbol}, and a run on own data reads no server: add "
            "them to data= (a dict by symbol) or pass the closes as benchmark="
        )
        return _absent(name, record, why)
    try:
        first = last_session_before(plan.start, exchange, calendar) or plan.start
        loaded = _load(symbol, exchange, first, plan, args, client, calendar, own)
    except (DataError, CalendarError) as exc:
        return _absent(name, record, f"no closes could be loaded ({exc})")
    bars = loaded.bars
    closes = pd.Series(
        bars["close"].to_numpy(dtype=np.float64),
        index=ist_times(bars["ts_close"].to_numpy()),
        name="benchmark",
    )
    record.update(source=loaded.meta.source, partitions=loaded.meta.partition_hashes())
    notes = tuple(f"benchmark {name}: {note}" for note in loaded.meta.warnings)
    origin = f"daily closes from {_origin(loaded.meta, own is not None)}"
    return Benchmark(name, origin, closes, notes, record)


def _load(
    symbol: str,
    exchange: str,
    first: dt.date,
    plan: Any,
    args: dict[str, Any],
    client: Any,
    calendar: Any,
    own: Any,
) -> BarsResult:
    """Daily bars of the index from `first` to the run's end. Own data is read as daily
    bars, else resampled from 1-minute bars, else from the run's base interval."""

    def at(base: str | None) -> BarsResult:
        return load_bars(
            symbol,
            exchange,
            DAILY,
            first,
            plan.end,
            source=args["source"],
            client=client,
            data=own,
            calendar=calendar,
            tz=args["tz"],
            base=base,
            now=plan.present,
        )

    bases = [None]
    if own is not None:
        intraday = [b for b in (ONE_MINUTE, plan.base) if parse_interval(b)[0] == "intraday"]
        bases += list(dict.fromkeys(intraday))
    for base in bases[:-1]:
        try:
            return at(base)
        except TimestampError:
            continue  # the bars are not daily: read them again from a finer base
    return at(bases[-1])


def _origin(meta: LoadMetadata, own: bool) -> str:
    """Where a load's bars came from, as the Assumptions block says it."""
    if own:
        return "own data (data=)"
    if meta.source == "db":
        return "Historify (source='db')"
    if meta.source == "api":
        return f"the broker ({meta.broker or 'unknown'}, source='api')"
    return f"the data provider {meta.provider!r}"


def _from_given(value: pd.Series | pd.DataFrame, exchange: str, calendar: Any) -> Benchmark:
    """Closes given as a Series or DataFrame (`_benchmark_given`)."""
    kind = "Series" if isinstance(value, pd.Series) else "DataFrame"
    name = f"the {kind} given as benchmark="
    read = read_given(value, exchange, calendar)
    record = {
        "symbol": None,
        "exchange": None,
        "source": "given",
        "partitions": {"given": read.content_hash()},
    }
    notes = []
    if read.off_session_dates:
        notes.append(
            f"benchmark {name}: {read.off_session_dates} of its dates are not {exchange} "
            "sessions and were left out"
        )
    if not len(read.levels):
        return _absent(name, record, "it holds no usable close")
    closes = pd.Series(read.levels, index=ist_times(read.stamps_ns), name="benchmark")
    return Benchmark(name, "closes as given", closes, tuple(notes), record)


def _run_days(frames: list[pd.DataFrame]) -> list[dt.date]:
    """The IST dates of the strategy's bars, in order: the run's sessions."""
    days = set()
    for bars in frames:
        days.update(ist_days(bars["ts_close"].to_numpy()).tolist())
    return [from_day(day) for day in sorted(days)]


def _uncovered(
    closes: pd.Series, run_days: list[dt.date], exchange: str, calendar: Any
) -> tuple[list[dt.date], int]:
    """(the sessions the benchmark is read on that have no close, how many it is read
    on): the run's sessions and, on the benchmark's exchange, the one before them."""
    if not run_days:
        return [], 0
    opening = last_session_before(run_days[0], exchange, calendar)
    needed = ([] if opening is None else [opening]) + run_days
    held = set(closes.index.date)
    return [day for day in needed if day not in held], len(needed)


def _coverage_note(name: str, missing: list[dt.date], needed: int) -> str:
    listed = ", ".join(day.isoformat() for day in missing[:LISTED_DATES])
    if len(missing) > LISTED_DATES:
        listed += f" and {len(missing) - LISTED_DATES} more"
    return (
        f"benchmark {name}: no close on {len(missing)} of the {needed} sessions it is read "
        f"on (the run's sessions and the one before them): {listed}. The metrics never "
        "take a close from a later day: such a session keeps the last earlier close (a "
        "zero benchmark return), and before the first close there is no benchmark return"
    )


__all__ = [
    "BENCHMARK_EXCHANGES",
    "INDEX_EXCHANGE_OF",
    "Benchmark",
    "benchmark_partitions",
    "benchmark_request",
    "load_benchmark",
]
