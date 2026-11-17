"""`SessionResult` from what a live session leaves (16-session-design 1.4 and 1.5; unit 13):
the engines' tables, the journal's day as the session counted it, and the session's
records (`_session_close.SessionOutcome`).

- **Each strategy's `bt.Result`** is built by the backtest's own result builder
  (`_pipeline._result`, the one `bt.backtest` returns), from its kernel session's
  `finish()` tables and its part of the session plan: the settings, instruments, market
  and history a backtest of the day would have, the strategy instance and its run
  context, the warm-up measured on its live stores. It covers this start only: a
  restarted engine's run begins at the restart. It cannot rerun (`Result.rerun` raises):
  a live session is not a backtest to repeat.
- **The day's summary** is the session's (`_session_figures`): every start of the day,
  each counted once.
- **The tables** are `_result_frames`'.
- **The warnings**, the mode first: the mode the session ran in and what it means; then
  the session's own warnings in order (the preflight's, the plan's, the clock's against
  the exchange, each first position opened with no stop at the broker, and the rest);
  then the seconds positions had no confirmed stop, and the positions left open.

Times are UTC nanoseconds turned into IST datetimes, amounts rupees, quantities units.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import pandas as pd

from .._pipeline import PreparedRun, _Finished, _result
from .._result_tables import IST
from .._warm_up import readiness
from ..reference import Instrument
from ..result import Result
from ..session_result import SessionResult
from ._plan_catch_up import StrategyPlan
from ._result_frames import (
    drift_frame,
    fills_frame,
    locks_frame,
    open_positions_frame,
    orders_frame,
    reconciliations_frame,
    skipped_frame,
    unprotected_frame,
)
from ._session_close import SessionOutcome
from ._types import Mode

#: How a live session's base interval was chosen, as the Assumptions block says it.
BASE_RULE = "a live session's bars are built from the feed's ticks on 1-minute slices"
#: The first warning of a result: the mode, and what it means.
MODE_WARNINGS = MappingProxyType(
    {
        Mode.LIVE: "LIVE mode: OpenAlgo sent these orders to {broker}, the broker",
        Mode.SANDBOX: (
            "SANDBOX mode: OpenAlgo simulated these fills at its own prices; nothing reached "
            "{broker}, the broker"
        ),
    }
)


def session_result(outcome: SessionOutcome) -> SessionResult:
    """The `SessionResult` of a session that ended with `outcome` (module docstring).
    Raises what the backtest's result builder raises for tables it cannot read."""
    results = {
        name: strategy_result(
            outcome.plans[name], outcome.outputs[name], outcome.timer_warnings.get(name, ())
        )
        for name in outcome.strategies
    }
    return SessionResult(
        mode=outcome.mode,
        broker=outcome.broker,
        account=outcome.account,
        day=outcome.day,
        started=_ist(outcome.started_ns),
        ended=_ist(outcome.ended_ns),
        stopped_by=outcome.stopped_by,
        restarts=outcome.restarts,
        strategies=outcome.strategies,
        results=MappingProxyType(results),
        summary=outcome.summary,
        orders=orders_frame(outcome.orders),
        fills=fills_frame(outcome.fills),
        drift=drift_frame(outcome.drift),
        unprotected=unprotected_frame(outcome.unprotected, outcome.ended_ns),
        locks=locks_frame(outcome.locks),
        reconciliations=reconciliations_frame(outcome.reconciliations),
        skipped_decisions=skipped_frame(outcome.skipped_decisions),
        open_positions=open_positions_frame(outcome.open_positions),
        mode_breach=outcome.mode_breach,
        warnings=result_warnings(outcome),
        paths=MappingProxyType(dict(outcome.paths)),
    )


def strategy_result(
    part: StrategyPlan, output: Mapping[str, Any], timer_warnings: Sequence[str] = ()
) -> Result:
    """One strategy's `bt.Result` (module docstring): its kernel session's `finish()`
    tables `output`, its part of the session plan `part` and its timers' notes, read
    before the kernel session finished, through the backtest's own result builder."""
    spec, setup, context = part.spec, part.setup, part.context
    state = readiness(context.run.checks.outcomes, context, setup.history)
    context.close(output["orders"])
    arguments = dict(spec.arguments)
    prepared = PreparedRun(
        plan=setup.plan,
        calendar=None,
        instruments=list(setup.instruments),
        frames=[setup.today],
        partitions={},
        notes={"base_rule": BASE_RULE, "instruments": instrument_lines(setup.instruments)},
        warnings=list(setup.warnings),
        reference_tables=None,
        market_args={
            "costs": arguments["costs"],
            "slippage": arguments["slippage"],
            "margin": setup.plan.margin,
        },
        market=setup.market,
    )
    finished = _Finished(
        part.strategy, context, setup.market, setup.history, state, output, list(timer_warnings)
    )
    return _result(spec.strategy, prepared, arguments, finished, None)


def instrument_lines(instruments: Sequence[Instrument]) -> str:
    """The Assumptions line of a session's instruments: each one's lot, tick and freeze
    quantity, and where each came from (OpenAlgo's facts, read at the start, or the
    reference tables)."""
    lines = []
    for instrument in instruments:
        sources = "; ".join(f"{name} from {source}" for name, source, _ in instrument.provenance)
        lines.append(
            f"{instrument.exchange}:{instrument.symbol} lot {instrument.lot_size}, tick "
            f"{instrument.tick_size}, freeze quantity {instrument.freeze_qty or 'none'}"
            + (f" ({sources})" if sources else "")
        )
    return "; ".join(lines)


def result_warnings(outcome: SessionOutcome) -> tuple[str, ...]:
    """The result's warnings (module docstring), each once."""
    notes = [MODE_WARNINGS[Mode(outcome.mode)].format(broker=outcome.broker)]
    notes += outcome.warnings
    seconds = outcome.summary.unprotected_seconds
    if seconds:
        notes.append(
            f"positions had no confirmed stop at the broker for {math.ceil(seconds)} s today "
            "in all: result.unprotected lists each window and why"
        )
    if outcome.open_positions:
        legs = ", ".join(f"{p.strategy} {p.units:g} {p.symbol}" for p in outcome.open_positions)
        notes.append(
            f"positions were still open when the session ended ({legs}): "
            "result.open_positions says what to do with each"
        )
    return tuple(dict.fromkeys(notes))


def _ist(ts_ns: int) -> pd.Timestamp:
    return pd.Timestamp(ts_ns, unit="ns", tz="UTC").tz_convert(IST)


__all__ = [
    "BASE_RULE",
    "MODE_WARNINGS",
    "instrument_lines",
    "result_warnings",
    "session_result",
    "strategy_result",
]
