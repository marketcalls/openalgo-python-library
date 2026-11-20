"""The `Result` of a finished run: its Assumptions block, its warnings and its manifest
(the last step of `_pipeline.execute`, shared by `bt.backtest`, a portfolio run's
strategies and a live session's results, `_live._result_build`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from ._base_interval import meaning
from ._causality_checks import RunChecks
from ._context import RunContext
from ._futures_margin import margin_assumption
from ._intraday_assumptions import one_clock_time
from ._manifest import build_manifest
from ._market import MarketSettings
from ._perturbation import manifest_settings
from ._rejections import reject_warnings
from ._run_corporate_actions import manifest_form
from ._run_end import run_end_warnings
from ._settings import assumptions
from ._warm_up import Readiness
from ._warm_up_history import RunHistory
from .result import Result

if TYPE_CHECKING:
    from ._pipeline import PreparedRun


def _assumptions(
    prepared: PreparedRun, calendar: Any, checks: RunChecks, market: MarketSettings
) -> dict:
    plan = prepared.plan
    block = assumptions(plan, calendar) | market.assumptions
    session = calendar.session(plan.exchange, plan.start)
    square_off = one_clock_time(block["square_off"])
    session_open = None
    if session is not None and square_off is not None:
        session_open = pd.Timestamp(session[0], tz="UTC").tz_convert("Asia/Kolkata")
        session_open = session_open.strftime("%H:%M")
    rule = meaning(plan.base, square_off, session_open)
    block["base_rule"] = f"base {plan.base} ({prepared.notes['base_rule']}): {rule}"
    block["instruments"] = prepared.notes["instruments"]
    if "timestamps" in prepared.notes:
        block["timestamps"] = prepared.notes["timestamps"]
    if prepared.futures is not None:
        block["futures"] = prepared.futures.assumption()
    futures_margin = margin_assumption(plan.margin, prepared.instruments, plan.start)
    if futures_margin is not None:
        block["futures_margin"] = futures_margin
    block["benchmark"] = None if prepared.benchmark is None else prepared.benchmark.assumption()
    indicator_checks = checks.assumption()
    if indicator_checks is not None:
        block["indicator_checks"] = indicator_checks
    return block


def _arguments(
    prepared: PreparedRun, args: dict[str, Any], market: MarketSettings
) -> dict[str, Any]:
    """Every argument of the run after defaults were resolved, for the manifest (the
    cost model by its label: it is a native object)."""
    resolved = manifest_settings(prepared.plan)
    resolved.update(
        costs=market.costs_label,
        slippage_rule=market.slippage_label,
        source=args["source"],
        tz=args["tz"],
        data="given" if args["data"] is not None else None,
        client="given" if args.get("client") is not None else None,
        calendar="default" if prepared.calendar is None else "given",
        benchmark=None if prepared.benchmark is None else prepared.benchmark.record,
        corporate_actions=manifest_form(args.get("corporate_actions")),
    )
    # The present only when `now=` fixed it: a run on the clock reruns to the same manifest.
    present = resolved.pop("present")
    resolved["now"] = None if args["now"] is None else present
    return resolved


@dataclass(frozen=True)
class _Finished:
    """A run the driver has finished: the strategy instance and its context, the market
    settings, the history kept, the readiness, the kernel's output and the timers'
    warnings."""

    instance: Any
    context: RunContext
    market: MarketSettings
    history: RunHistory
    state: Readiness
    output: dict[str, Any]
    timer_warnings: list[str]


def _result(
    strategy: type,
    prepared: PreparedRun,
    args: dict[str, Any],
    run: _Finished,
    record: Any,
    *,
    assumptions: Mapping[str, str] | None = None,
) -> Result:
    """The `Result` of a finished run, with its Assumptions block, warnings and manifest;
    `assumptions` replace lines of the block (a live session's, whose fills OpenAlgo
    priced)."""
    plan, calendar = prepared.plan, prepared.session_calendar()
    market, output = run.market, run.output
    parameters = {name: getattr(run.instance, name) for name in strategy.parameters()}
    manifest = build_manifest(
        strategy,
        parameters,
        _arguments(prepared, args, market) | {"warm_up": run.state.manifest()},
        [instrument.to_dict() for instrument in prepared.instruments],
        prepared.partitions | run.history.partitions(),
        prepared.reference_tables,
        market.cost_schedules,
    )
    block = _assumptions(prepared, calendar, run.context.run.checks, market)
    return Result(
        output,
        list(plan.symbols),
        block | {"warm_up": run.state.assumption()} | dict(assumptions or {}),
        [
            *run_end_warnings(output["run_end"]),
            *reject_warnings(output["orders"], [i.kind for i in prepared.instruments]),
            *run.state.warnings(plan.start),
            *prepared.warnings,
            *run.history.warnings,
            *run.timer_warnings,
        ],
        run.context.skipped,
        manifest.to_dict(),
        record=record,
        calendar=calendar,
        benchmark=None if prepared.benchmark is None else prepared.benchmark.closes,
        strategy_closes=[store.column("ts_close") for store in run.context.stores.values()],
    )


__all__ = ["_Finished", "_result"]
