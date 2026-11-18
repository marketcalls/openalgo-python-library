"""The bars a run read, for the checks that rerun it on generated bars (design 17,
sections 2.4 to 2.9), and the arguments of those reruns.

A `Result` keeps no bars, so they are read again from the run's own arguments, through
its rerun record: the run is prepared once more (`_pipeline.prepare`, which reads the
run's source again: the user's `data=`, or OpenAlgo through the run's client), and each
symbol's base bars are joined after the history before `start` that its indicators keep
(`_pipeline._kept_history`), exactly the bars the run read.

A rerun goes through `bt.backtest`'s `data=` path with the warm-up the run needs
(`rerun_arguments`): the run's own arguments, with `data=` the generated bars by symbol
(history included, so the warm-up is read from them as from the user's own data), no
client, `start` and `end` the run's dates, `base` its base interval, `source` "auto",
`tz` None (the bars carry UTC nanoseconds) and the benchmark as the closes the run had.

A futures series (a strategy on an underlying) changes contract at each roll, which a
rerun on the run's bars as one series cannot hold yet: its bar checks raise
NotImplementedError.
"""

from __future__ import annotations

import datetime as dt
import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .._causality_checks import IndicatorChecks
from .._manifest import strategy_source_hash
from .._messages import not_supported
from .._pipeline import _kept_history, prepare
from ..errors import BacktestError
from ._ohlc import session_days
from ._session_entries import SessionTimes, session_times

#: What the bar checks of a futures series are refused with, and what to do instead.
FUTURES = (
    "the checks that rerun the backtest (vs_random, permutation, noise, shifted, "
    "other_markets, stress, random_exits, delayed, liquidity) for a strategy on a futures "
    "series",
    "a futures series changes contract at each roll, and the reruns read the run's bars "
    "as one series of its own, which cannot hold the rolls yet; run checks='quick', or a "
    "list of the trade-list checks",
)


@dataclass(frozen=True, eq=False)
class RunBars:
    """The bars a run read (module docstring): `symbols` in registration order, the base
    bars of each in `frames` (canonical layout, the history before `start` first), the
    tick of each in rupees, `first_run_ns`, the UTC nanosecond open of the run's first
    bar (the bars before it are history), the `sessions` of the run with their entry
    window and square-off (`_session_entries.SessionTimes`), and the `base` interval of
    the bars ("1m", "D")."""

    symbols: tuple[str, ...]
    frames: tuple[pd.DataFrame, ...]
    ticks: tuple[float, ...]
    first_run_ns: int
    sessions: tuple[SessionTimes, ...] = ()
    base: str = ""

    def run_part(self, position: int) -> pd.DataFrame:
        """The run's bars of the symbol at `position`, without the history."""
        frame = self.frames[position]
        return frame[frame["ts_open"].to_numpy() >= self.first_run_ns].reset_index(drop=True)

    def session_closes(self, sessions: pd.DatetimeIndex) -> tuple[str, np.ndarray]:
        """(the first symbol, its close at the last base bar of each of `sessions`, the run's
        session dates), rupees; a session without a bar keeps the close before it."""
        bars = self.run_part(0)
        days = session_days(bars["ts_open"].to_numpy())
        last = np.append(days[1:] != days[:-1], True)
        closes = pd.Series(
            bars["close"].to_numpy(dtype=float)[last],
            index=pd.to_datetime(days[last], unit="D"),
        )
        wanted = pd.DatetimeIndex(sessions).tz_localize(None).normalize()
        return self.symbols[0], closes.reindex(wanted, method="ffill").to_numpy(dtype=float)


def run_arguments(record: Any) -> dict[str, Any]:
    """The run's own arguments of `bt.backtest`, client and strategy aside."""
    from ..run import backtest

    names = set(inspect.signature(backtest).parameters) - {"strategy", "client"}
    return {name: value for name, value in record.arguments.items() if name in names}


def strategy_name(strategy: type) -> str:
    """The name its indicator checks are kept under: its source hash, else its name."""
    return strategy_source_hash(strategy) or strategy.__qualname__


def run_bars(result: Any) -> tuple[RunBars, dict[str, Any]]:
    """(the `RunBars` of `result`, the arguments of its reruns without `data=`), from its
    rerun record (module docstring). Raises BacktestError for a result not made by
    `bt.backtest`, NotImplementedError for a futures series, and what preparing the run
    raises (the data layer's errors among them)."""
    record = result._record
    if record is None:
        raise BacktestError("this result was not made by bt.backtest, so it has no bars to rerun")
    arguments = run_arguments(record)
    prepared = prepare(record.strategy, record.client, {**arguments, "client": record.client})
    if prepared.futures is not None:
        raise not_supported(*FUTURES)
    plan = prepared.plan
    checks = IndicatorChecks(strategy_name(record.strategy), measured=prepared.measured)
    history = _kept_history(record.strategy, prepared, arguments["params"], checks)
    frames = tuple(
        history.with_base(symbol, frame)
        for symbol, frame in zip(plan.symbols, prepared.frames, strict=True)
    )
    first = min(int(frame["ts_open"].iloc[0]) for frame in prepared.frames if len(frame))
    days = np.unique(
        np.concatenate([session_days(frame["ts_open"].to_numpy()) for frame in prepared.frames])
    )
    dates = [dt.date(1970, 1, 1) + dt.timedelta(days=int(day)) for day in days]
    bars = RunBars(
        tuple(plan.symbols),
        frames,
        tuple(float(instrument.tick_size) for instrument in prepared.instruments),
        first,
        session_times(plan, prepared.session_calendar(), dates),
        plan.base,
    )
    reruns = {
        **arguments,
        "data": None,
        "start": arguments["start"] if arguments["start"] is not None else plan.start,
        "end": arguments["end"] if arguments["end"] is not None else plan.end,
        "base": plan.base,
        "source": "auto",
        "tz": None,
        "benchmark": result._benchmark,
    }
    return bars, reruns


__all__ = ["FUTURES", "RunBars", "run_arguments", "run_bars", "strategy_name"]
