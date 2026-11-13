"""The history a backtest loads before `start`, so that its indicators are ready when
trading begins (decision D25; the rule is in `_warm_up`).

For every traded symbol the run's own source is asked for its bars before `start` at
the run's base interval: the source the run's bars came from (Historify or the broker,
never both), the user's `data=`, or the run's data provider. Enough calendar days are
asked for to hold `HISTORY_BARS` strategy bars, and the bars obtained are cut to the
whole sessions that hold the last `HISTORY_BARS` of them. That length does not depend
on the strategy's parameters, so every combination of a sweep and a single run of one
of them see the same history, and a run reruns to the same result.

Not loaded, and said so: a futures series (its contracts and their roll adjustments
begin at `start`). A source with fewer bars gives what it has, own data that begins at
`start` none. A load that fails (a quality error, a suspected unadjusted split in the
history) gives no history and says why in the Assumptions block; the run's own bars
are unaffected. The strategy then waits inside the run (`_warm_up`).
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ._base_interval import _data_for
from ._layout import IST_OFFSET_NANOS, NANOS_PER_SECOND, parse_interval
from ._messages import counted
from ._run_corporate_actions import declared_for
from ._series import FIELDS, TIMES
from .data import BarsResult, load_bars
from .errors import DataError, NoDataError
from .resample import resample_bars

# Strategy bars of history loaded before `start` (decision D25). The window-stability
# check measures the warm-up of a 200-bar EMA, the longest lookback in common use, at
# 2,048 bars at its default tolerance (the window doubles from 16 bars, 03-architecture
# section 5); this is that length, so such an indicator is ready at `start`.
HISTORY_BARS = 2_048
# Calendar days per session: five sessions a week.
CALENDAR_DAYS_PER_SESSION = 7 / 5
# Extra sessions asked for, as a fraction: NSE closes on about 15 weekdays of some 250
# a year (6 percent), and a year without a holiday list counts them as sessions.
HOLIDAY_ALLOWANCE = 0.10
# Days around `start` whose sessions give the regular session length of an intraday run.
SESSION_SAMPLE_DAYS = 30
NANOS_PER_DAY = 86_400 * NANOS_PER_SECOND
SERVER_SOURCES = ("db", "api")
FUTURES_NOTE = "none (a futures series is built from start: its contracts and their rolls)"


@dataclass(frozen=True)
class SymbolHistory:
    """One symbol's history before `start`: its base bars (canonical layout, possibly
    empty), the strategy bars they make (the columns of `_series`, for `BarStore`), what
    the Assumptions block says of it, and the manifest hashes of the bars loaded."""

    base: pd.DataFrame
    strategy: dict[str, np.ndarray]
    note: str
    partitions: dict[str, str] = field(default_factory=dict)

    @property
    def bars(self) -> int:
        """Strategy bars of history."""
        return len(self.strategy["ts_open"])


@dataclass(frozen=True)
class History:
    """The history of every traded symbol (`symbols`, in registration order)."""

    symbols: dict[str, SymbolHistory]

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
        """The partition hashes of the history loaded, as the manifest's data records
        them: under "<symbol> history", for each symbol that has some."""
        return {
            f"{symbol} history": history.partitions
            for symbol, history in self.symbols.items()
            if history.partitions
        }

    def describe(self) -> str:
        """What was loaded, symbol by symbol, for the Assumptions block."""
        return ", ".join(f"{symbol} {history.note}" for symbol, history in self.symbols.items())

    def manifest(self) -> dict[str, Any]:
        """The history as the run manifest records it: strategy bars, first and last
        bar, and the partition hashes of the bars loaded, by symbol."""
        return {
            symbol: {
                "strategy_bars": history.bars,
                "first": _first_day(history.strategy),
                "partitions": history.partitions,
            }
            for symbol, history in self.symbols.items()
        }


def no_history(symbols: tuple[str, ...], note: str) -> History:
    """A history with no bars for any symbol, and why."""
    return History({symbol: _empty(note) for symbol in symbols})


def _empty(note: str) -> SymbolHistory:
    base = pd.DataFrame({name: np.array([], dtype=np.int64) for name in TIMES})
    for name in FIELDS:
        base[name] = np.array([], dtype=np.float64)
    strategy = {name: base[name].to_numpy() for name in (*TIMES, *FIELDS)}
    return SymbolHistory(base, strategy, note)


def _first_day(strategy: dict[str, np.ndarray]) -> str | None:
    opens = strategy["ts_open"]
    return None if len(opens) == 0 else _ist_date(int(opens[0])).isoformat()


def _ist_date(ts_ns: int) -> dt.date:
    return dt.date(1970, 1, 1) + dt.timedelta(days=(ts_ns + IST_OFFSET_NANOS) // NANOS_PER_DAY)


def _bars_per_session(plan: Any, calendar: Any) -> int:
    """Strategy bars in a regular session: 1 for a daily strategy; for an intraday one,
    the median session length near `start` over the interval, rounded up."""
    kind, seconds = parse_interval(plan.interval)
    if kind != "intraday":
        return 1
    span = dt.timedelta(days=SESSION_SAMPLE_DAYS)
    sessions = calendar.sessions(plan.exchange, plan.start - span, plan.start + span)
    if sessions.empty:
        return 1
    lengths = (sessions["close"] - sessions["open"]).to_numpy() // NANOS_PER_SECOND
    return max(1, math.ceil(float(np.median(lengths)) / seconds))


def first_requested_day(plan: Any, calendar: Any) -> dt.date:
    """The first IST date asked for: enough sessions for `HISTORY_BARS` strategy bars,
    with the holiday allowance, in calendar days before `start`."""
    sessions = math.ceil(HISTORY_BARS / _bars_per_session(plan, calendar))
    days = math.ceil(sessions * CALENDAR_DAYS_PER_SESSION * (1 + HOLIDAY_ALLOWANCE))
    return plan.start - dt.timedelta(days=days)


def _source(args: dict[str, Any], loaded: BarsResult) -> Any:
    """The source the history is read from: the one the run's bars came from."""
    if args["data"] is None and args["source"] == "auto" and loaded.meta.source in SERVER_SOURCES:
        return loaded.meta.source
    return args["source"]


def _strategy_frame(base: pd.DataFrame, plan: Any, calendar: Any) -> pd.DataFrame:
    """The strategy bars the base bars make, built as the kernel builds them."""
    if plan.base == plan.interval or base.empty:
        return base
    return resample_bars(
        base,
        plan.interval,
        exchange=plan.exchange,
        calendar=calendar,
        source_interval=plan.base,
        on_closed_day="drop",
    )


def _last_bars(base: pd.DataFrame, strategy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both cut to the whole sessions that hold the last `HISTORY_BARS` strategy bars."""
    if len(strategy) <= HISTORY_BARS:
        return base, strategy
    first = _ist_date(int(strategy["ts_open"].iloc[len(strategy) - HISTORY_BARS]))
    since = int(pd.Timestamp(first).value) - IST_OFFSET_NANOS
    keep_base = base[base["ts_open"].to_numpy() >= since].reset_index(drop=True)
    keep_strategy = strategy[strategy["ts_open"].to_numpy() >= since].reset_index(drop=True)
    return keep_base, keep_strategy


def _symbol_history(
    symbol: str,
    plan: Any,
    args: dict[str, Any],
    client: Any,
    calendar: Any,
    loaded: BarsResult,
) -> SymbolHistory:
    """One symbol's history (see the module docstring)."""
    first_day = first_requested_day(plan, calendar)
    last_day = plan.start - dt.timedelta(days=1)
    try:
        result = load_bars(
            symbol,
            plan.exchange,
            plan.base,
            first_day,
            last_day,
            source=_source(args, loaded),
            client=client,
            data=_data_for(args["data"], symbol),
            calendar=calendar,
            tz=args["tz"],
            base=loaded.meta.base_interval,
            now=plan.present,
            corporate_actions=declared_for(args["corporate_actions"], symbol, plan.symbols),
        )
    except NoDataError:
        return _empty(f"none (the source has no bars before {plan.start})")
    except DataError as exc:
        reason = str(exc).split(". ")[0].rstrip(".")
        return _empty(f"none (the bars before {plan.start} could not be loaded: {reason})")
    frame = _strategy_frame(result.bars, plan, calendar)
    if frame.empty:
        return _empty(f"none (the source has no bars before {plan.start})")
    base, frame = _last_bars(result.bars, frame)
    strategy = {name: frame[name].to_numpy() for name in (*TIMES, *FIELDS)}
    first, last = _first_day(strategy), _ist_date(int(frame["ts_open"].iloc[-1])).isoformat()
    note = f"{counted(len(frame), 'strategy bar')} from {first} to {last}"
    if len(frame) < HISTORY_BARS:
        note += f" (all the source has since {first_day})"
    return SymbolHistory(base, strategy, note, result.meta.partition_hashes())


def load_history(
    plan: Any,
    args: dict[str, Any],
    client: Any,
    calendar: Any,
    loaded: dict[str, BarsResult],
    futures: Any = None,
) -> History:
    """The history of every symbol of `plan` (see the module docstring); `loaded` are
    the run's own loads by symbol. Raises ConfigError for arguments the data layer
    refuses; a failed load is a note, never an error."""
    if futures is not None:
        return no_history(plan.symbols, FUTURES_NOTE)
    return History(
        {
            symbol: _symbol_history(symbol, plan, args, client, calendar, loaded[symbol])
            for symbol in plan.symbols
        }
    )


__all__ = [
    "CALENDAR_DAYS_PER_SESSION",
    "HISTORY_BARS",
    "HOLIDAY_ALLOWANCE",
    "History",
    "SymbolHistory",
    "first_requested_day",
    "load_history",
    "no_history",
]
