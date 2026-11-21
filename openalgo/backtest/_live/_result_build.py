"""`SessionResult` from what a live session leaves (16-session-design 1.4 and 1.5; unit 13):
the engines' tables, the journal's day as the session counted it, and the session's
records (`_session_close.SessionOutcome`).

- **Each strategy's `bt.Result`** is built by the backtest's own result builder
  (`_pipeline_result.finished_result`, the one `bt.backtest` returns), from its kernel session's
  `finish()` tables and its part of the session plan: the settings, instruments, market
  and history a backtest of the day would have, the strategy instance and its run
  context, the warm-up measured on its live stores. It covers the trades this start
  closes, whole: a restarted engine's run begins at the restart, and its account carried
  the realized P&L and charges of the round trips the day's earlier starts closed (the
  kernel restores the cash), which are taken out of its equity rows after the restore
  and of its gross P&L (`this_start`), so its metrics are this start's. A position
  carried open across the restart is the trade this start closes: its P&L and its
  charges, its entry's before the restart included, stay in the result (the unit 13
  follow-up item 2), so the result's net profit is what an unbroken day's result shows
  for the same trades. Its trades table is the kernel's, with the charges the position
  carried open paid before the restart added to its trade (`Restored.carried`): a
  restored engine's charge book holds the fills of its own start only. Its kernel
  session's stop settled the contract notes of its start's fills and recorded the slice
  in progress (the kernel follow-up item 1), so its equity holds every fill and the
  notes' charges: its net profit is its trades' net (plus an open position's value), and
  its charges differ from the day summary's, each fill's rounded to the paisa, by the
  notes' rounding (`Result.contract_notes`). Its Assumptions
  block says
  OpenAlgo priced the fills (`LIVE_ASSUMPTIONS`, `LIVE_BASE_MEANING`): the backtest's
  fill model and slippage played no part. It cannot rerun (`Result.rerun` raises):
  a live session is not a backtest to repeat. `bt.live` ends with no end-of-run notice of
  orders not placed, as `bt.backtest` does (`_rejection_notice`): the session alerted
  each refusal as it came, and each result's warnings name every reject code, the live
  session's entry locks (`live_lock`) and in-flight gate (`live_in_flight`) among them.
- **The portfolio** is the strategies' results as one account (`_result_portfolio`).
- **The day's summary** is the session's (`_session_figures`): every start of the day,
  each counted once.
- **The tables** are `_result_frames`'.
- **The warnings**, the mode first: the mode the session ran in and what it means; then
  the session's own warnings in order (the preflight's, the plan's, the clock's against
  the exchange, each first position opened with no stop at the broker, and the rest);
  then the seconds positions had no confirmed stop, the positions left open (each leg, then
  each shared instrument as OpenAlgo's net for its members together, never as a position
  of its own), and each leg the close named that it could not confirm
  (`SessionResult.unconfirmed_legs`).

Times are UTC nanoseconds turned into IST datetimes, amounts rupees, quantities units.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from .._pipeline import PreparedRun
from .._pipeline_result import FinishedRun, finished_result
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
from ._result_portfolio import session_portfolio
from ._session_close import SessionOutcome
from ._session_open import OpenPosition
from ._session_records import Restored
from ._types import Mode

#: How a live session's base interval was chosen, as the Assumptions block says it.
BASE_RULE = "a live session's bars are built from the feed's ticks on 1-minute slices"
#: What the base interval means live, in place of the backtest's fills at bar opens.
LIVE_BASE_MEANING = (
    "the strategy decides at each bar's close and its orders are sent at once; OpenAlgo "
    "reports the price of each fill"
)
#: The Assumptions lines of a live result that replace the backtest's fill model: the
#: broker (or OpenAlgo's sandbox) filled every order at its own prices.
LIVE_ASSUMPTIONS = MappingProxyType(
    {
        "fill": (
            "at the prices OpenAlgo reported for each order, sent when the strategy decided "
            "(the broker's fills in LIVE mode, OpenAlgo's sandbox fills in SANDBOX mode)"
        ),
        "slippage": "not modelled: the fills are OpenAlgo's",
        "risk_exit_fill": (
            "at the price OpenAlgo reported: a stop is held by the broker, and a target or "
            "the daily loss limit is sent as an order on the tick that reaches it"
        ),
    }
)
#: How the result names a leg the close named that it could not confirm.
NOT_CONFIRMED_AT_CLOSE = (
    "not confirmed at the close: {words}; result.open_positions holds no confirmed figure "
    "of it: check the position and the order book in OpenAlgo"
)
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
            outcome.plans[name],
            outcome.outputs[name],
            outcome.timer_warnings.get(name, ()),
            outcome.restored.get(name),
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
        portfolio=session_portfolio(
            results,
            outcome.arguments,
            outcome.funds_at_start,
            shared=bool(outcome.summary.shared),
        ),
        versions=MappingProxyType(dict(outcome.versions)),
        summary=outcome.summary,
        orders=orders_frame(outcome.orders),
        fills=fills_frame(outcome.fills),
        drift=drift_frame(outcome.drift),
        unprotected=unprotected_frame(outcome.unprotected, outcome.ended_ns),
        locks=locks_frame(outcome.locks),
        reconciliations=reconciliations_frame(
            outcome.reconciliations, shared=bool(outcome.summary.shared)
        ),
        skipped_decisions=skipped_frame(outcome.skipped_decisions),
        open_positions=open_positions_frame(outcome.open_positions),
        mode_breach=outcome.mode_breach,
        warnings=result_warnings(outcome),
        paths=MappingProxyType(dict(outcome.paths)),
    )


def strategy_result(
    part: StrategyPlan,
    output: Mapping[str, Any],
    timer_warnings: Sequence[str] = (),
    restore: Restored | None = None,
) -> Result:
    """One strategy's `bt.Result` (module docstring): its kernel session's `finish()`
    tables `output`, its part of the session plan `part` and its timers' notes, read
    before the kernel session finished, through the backtest's own result builder; for a
    kernel session a restart restored (`restore`), this start's money only."""
    spec, setup, context = part.spec, part.setup, part.context
    output = dict(output) if restore is None else this_start(output, restore)
    state = readiness(context.run.checks.outcomes, context, setup.history)
    context.close(output["orders"])
    arguments, prepared = dict(spec.arguments), prepared_run(part)
    finished = FinishedRun(
        part.strategy, context, setup.market, setup.history, state, output, list(timer_warnings)
    )
    base_rule = f"base {setup.plan.base} ({BASE_RULE}): {LIVE_BASE_MEANING}"
    lines = LIVE_ASSUMPTIONS | {"base_rule": base_rule}
    return finished_result(spec.strategy, prepared, arguments, finished, None, assumptions=lines)


def prepared_run(part: StrategyPlan) -> PreparedRun:
    """The prepared run of a backtest of the day that one strategy's part of the session
    plan stands for (its settings, instruments, today's bars and market settings), as its
    result and its start's manifest (`_session_versions`) read it."""
    setup, arguments = part.setup, part.spec.arguments
    return PreparedRun(
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


def this_start(output: Mapping[str, Any], restore: Restored) -> dict[str, Any]:
    """`output` with the realized gross P&L and charges of the round trips the day's
    earlier starts closed (`restore`) taken out of the equity rows that close after the
    restore and of the gross P&L (module docstring): a position carried open keeps what it
    realized and paid. The rows before the restore, the catch-up's, hold none."""
    equity = dict(output["equity"])
    after = np.asarray(equity["ts"], dtype=np.int64) > restore.after_ns
    carried = {
        "equity_rupees": restore.realized - restore.charges,
        "realized_rupees": restore.realized,
        "costs_rupees": restore.charges,
    }
    for column, amount in carried.items():
        values = np.array(equity[column], dtype=float)
        values[after] -= amount
        equity[column] = values
    realized, unrealized, total = output["gross_pnl"]
    gross_pnl = (realized - restore.realized, unrealized, total - restore.realized)
    trades = _carried_charges(output["trades"], restore)
    return {**output, "equity": equity, "gross_pnl": gross_pnl, "trades": trades}


def _carried_charges(trades: Mapping[str, Any], restore: Restored) -> dict[str, Any]:
    """The kernel's trades table with the charges the position carried open across the
    restart paid before it (`Restored.carried`) added to the trade that position is: the
    first this start closes (a strategy holds one position, so it closes before another
    opens), opened before the restore. The restored engine's charge book holds the fills
    of its own start only (module docstring)."""
    table = dict(trades)
    entries = np.asarray(table["entry_ts"], dtype=np.int64)
    if not restore.carried or not len(entries) or entries[0] > restore.after_ns:
        return table
    costs = np.array(table["costs_rupees"], dtype=float)
    net = np.array(table["net_pnl_rupees"], dtype=float)
    costs[0] = round(costs[0] + restore.carried, 2)
    net[0] = round(net[0] - restore.carried, 2)
    table["costs_rupees"], table["net_pnl_rupees"] = costs, net
    return table


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
    if any(position.units for position in outcome.open_positions):
        notes.append(
            "positions were still open when the session ended "
            f"({_open_words(outcome.open_positions)}): result.open_positions says what to do "
            "with each"
        )
    summary = outcome.summary
    for words in (*summary.differing, *summary.unconfirmed):
        notes.append(NOT_CONFIRMED_AT_CLOSE.format(words=words))
    return tuple(dict.fromkeys(notes))


def _open_words(positions: Sequence[OpenPosition]) -> str:
    """The positions left open in words: each leg, then each shared instrument's row as
    OpenAlgo's net for its members together."""
    legs = ", ".join(f"{p.strategy} {p.units:g} {p.symbol}" for p in positions if not p.members)
    nets = ", ".join(
        f"OpenAlgo's net {p.units:g} {p.symbol} for {p.strategy} together"
        for p in positions
        if p.members
    )
    return "; ".join(words for words in (legs, nets) if words)


def _ist(ts_ns: int) -> pd.Timestamp:
    return pd.Timestamp(ts_ns, unit="ns", tz="UTC").tz_convert(IST)


__all__ = [
    "BASE_RULE",
    "LIVE_ASSUMPTIONS",
    "LIVE_BASE_MEANING",
    "MODE_WARNINGS",
    "NOT_CONFIRMED_AT_CLOSE",
    "instrument_lines",
    "prepared_run",
    "result_warnings",
    "session_result",
    "strategy_result",
    "this_start",
]
