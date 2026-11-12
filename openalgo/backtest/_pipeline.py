"""How one backtest runs, in two halves so that the combinations of a sweep share the
first (the Python half of the design in `rust/oa_bt/src/engine/mod.rs`).

`prepare` resolves everything that does not depend on the strategy's parameters: the
settings (`_settings.resolve`), the calendar, the futures series (`_futures`), the
instruments (`_instruments.resolve_instruments`) and, before any data is loaded, that the
margin table prices them all (`_margin_check`); then the base interval and its bars
(`_base_interval.choose_base`, through the data layer), the ticks chosen by a price slab
(`_slab_ticks`, from the data) and the market settings (`_market`: the cost model, the
slippage and the per-instrument facts). A sweep sends its `PreparedRun` to worker
processes without the market settings, which hold native objects, and each worker
resolves them again from the `costs=` and `slippage=` arguments. That needs those
arguments to pickle and the run to use the shared default calendar (a calendar of its
own is a native object too); otherwise the sweep runs in this process.

`execute` builds the native engine, binds a strategy instance to a `RunContext`, runs
`init` and the driver to the end of the grid (the Rust signal driver for a condition
strategy, `_conditions`), and returns the `Result` with its
Assumptions block, warnings and manifest.
"""

from __future__ import annotations

import dataclasses
import pickle
from dataclasses import dataclass
from typing import Any

import pandas as pd

from . import calendar as calendar_module
from ._base_interval import choose_base, meaning
from ._causality_checks import IndicatorChecks
from ._conditions import drive_conditions, has_conditions
from ._context import RunContext, RunData
from ._driver import drive
from ._engine import build_engine, call
from ._futures import attach, resolve_futures
from ._instruments import resolve_instruments
from ._manifest import build_manifest, strategy_source_hash
from ._margin_check import check_margin
from ._market import MarketSettings, market_settings
from ._settings import RunPlan, assumptions, resolve
from ._slab_ticks import resolve_slab_ticks
from .reference import Instrument
from .result import Result


@dataclass(frozen=True)
class PreparedRun:
    """A run resolved up to the engine: the plan (base interval decided), the calendar
    (None for the shared default one), the instruments and their base bars in
    registration order, the data partition hashes by symbol, the Assumptions lines, the
    warnings found so far, the reference-table version used, and the market settings
    (None in a worker, rebuilt from `market_args`, the `costs=` and `slippage=`
    arguments)."""

    plan: RunPlan
    calendar: Any
    instruments: list[Instrument]
    frames: list[pd.DataFrame]
    partitions: dict[str, dict[str, str]]
    notes: dict[str, str]
    warnings: list[str]
    reference_tables: str | None
    market_args: dict[str, Any]
    market: MarketSettings | None
    # The futures series (`_futures.FuturesPlan`), None for a strategy on no underlying.
    futures: Any = None

    @property
    def portable(self) -> bool:
        """True when the run can be sent to another process: the shared default
        calendar, and `costs=` and `slippage=` arguments that pickle."""
        return self.calendar is None and _picklable(self.market_args)

    def for_workers(self) -> PreparedRun:
        """This run without its market settings, to send to worker processes."""
        return dataclasses.replace(self, market=None)

    def market_settings(self) -> MarketSettings:
        """The market settings, resolved again when this run came without them."""
        if self.market is not None:
            return self.market
        return market_settings(self.plan, self.instruments, self.market_args)

    def session_calendar(self) -> calendar_module.Calendar:
        """The run's calendar."""
        return self.calendar or calendar_module.default_calendar()


def _picklable(value: Any) -> bool:
    """True when `value` pickles (native objects do not)."""
    try:
        pickle.dumps(value)
    except (pickle.PicklingError, TypeError, AttributeError):
        return False
    return True


def _calendar(given: Any, plan: RunPlan) -> calendar_module.Calendar | None:
    if given is None:
        return None
    if isinstance(given, calendar_module.Calendar):
        return given
    return calendar_module.Calendar.from_sessions(given, plan.exchange, plan.start, plan.end)


def _cut(bars: pd.DataFrame, plan: RunPlan) -> pd.DataFrame:
    """The bars that open inside the run's optional instant bounds (the kernel ignores
    the others; views of other intervals are built from these)."""
    keep = pd.Series(True, index=bars.index)
    if plan.first_slice is not None:
        keep &= bars["ts_open"] >= plan.first_slice
    if plan.last_slice is not None:
        keep &= bars["ts_open"] <= plan.last_slice
    return bars.loc[keep].reset_index(drop=True)


def prepare(strategy: type, client: Any, args: dict[str, Any]) -> PreparedRun:
    """Resolve a run of `strategy` with the arguments of `bt.backtest` up to the engine.
    Raises ConfigError for unusable arguments, the data layer's and the reference
    tables' errors, and NotImplementedError for settings of later work."""
    plan = resolve(strategy, args)
    own_calendar = _calendar(args["calendar"], plan)
    calendar = own_calendar or calendar_module.default_calendar()
    futures = resolve_futures(strategy, plan, args, calendar)
    windows = None
    if futures is not None:
        plan = dataclasses.replace(plan, symbols=futures.symbols)
        windows = futures.windows
    if futures is None:
        instruments, lines, version = resolve_instruments(plan, args["instrument"], calendar)
    else:
        instruments, lines, version = futures.instruments, futures.lines, futures.version
    # Before any data is loaded: an instrument the margin table cannot price stops here.
    check_margin(instruments, plan.product, (plan.start, plan.end))
    choice = choose_base(plan, args, client, calendar, windows)
    plan = dataclasses.replace(plan, base=choice.base)
    instruments, lines = resolve_slab_ticks(
        instruments, lines, plan, args, client, calendar, windows
    )
    if futures is not None:
        futures = futures.with_instruments(instruments)
    market_args = {"costs": args["costs"], "slippage": args["slippage"]}
    market = market_settings(plan, instruments, market_args)
    loaded = [choice.bars[symbol] for symbol in plan.symbols]
    warnings = [*plan.warnings]
    for result in loaded:
        warnings += [note for note in result.meta.warnings if note not in warnings]
    for instrument in instruments:
        if instrument.generic:
            warnings.append(
                f"{instrument.symbol}: generic cash-equity instrument assumed (lot 1, tick 0.05)"
            )
    warnings += market.warnings
    warnings += [] if futures is None else list(futures.warnings)
    return PreparedRun(
        plan=plan,
        calendar=own_calendar,
        instruments=instruments,
        frames=[_cut(result.bars, plan) for result in loaded],
        partitions={
            symbol: result.meta.partition_hashes()
            for symbol, result in zip(plan.symbols, loaded, strict=True)
        },
        notes={"base_rule": choice.reason, "instruments": "; ".join(lines.values())},
        warnings=warnings,
        reference_tables=version,
        market_args=market_args,
        market=market,
        futures=futures,
    )


def _assumptions(
    prepared: PreparedRun, calendar: Any, checks: IndicatorChecks, market: MarketSettings
) -> dict:
    plan = prepared.plan
    block = assumptions(plan, calendar) | market.assumptions
    session = calendar.session(plan.exchange, plan.start)
    session_open = None
    if session is not None and block["square_off"] is not None:
        session_open = pd.Timestamp(session[0], tz="UTC").tz_convert("Asia/Kolkata")
        session_open = session_open.strftime("%H:%M")
    rule = meaning(plan.base, block["square_off"], session_open)
    block["base_rule"] = f"base {plan.base} ({prepared.notes['base_rule']}): {rule}"
    block["instruments"] = prepared.notes["instruments"]
    if prepared.futures is not None:
        block["futures"] = prepared.futures.assumption()
    indicator_checks = checks.assumption()
    if indicator_checks is not None:
        block["indicator_checks"] = indicator_checks
    return block


def _arguments(
    prepared: PreparedRun, args: dict[str, Any], market: MarketSettings
) -> dict[str, Any]:
    """Every argument of the run after defaults were resolved, for the manifest (the
    cost model by its label: it is a native object)."""
    plan = prepared.plan
    resolved = {field.name: getattr(plan, field.name) for field in dataclasses.fields(plan)}
    resolved.update(
        costs=market.costs_label,
        slippage_rule=market.slippage_label,
        source=args["source"],
        tz=args["tz"],
        data="given" if args["data"] is not None else None,
        client="given" if args.get("client") is not None else None,
        calendar="default" if prepared.calendar is None else "given",
    )
    return resolved


def execute(
    strategy: type,
    prepared: PreparedRun,
    args: dict[str, Any],
    checks: IndicatorChecks | None = None,
    record: Any = None,
) -> Result:
    """Run `strategy` on a prepared run with the parameter values of `args["params"]`.
    `checks` carries indicator checks already done (a sweep); `record` lets the result
    rerun itself. Raises the kernel's and the strategy's errors unchanged."""
    plan = prepared.plan
    calendar = prepared.session_calendar()
    checks = checks or IndicatorChecks(strategy_source_hash(strategy) or strategy.__qualname__)
    market = prepared.market_settings()
    config = plan.native_config(deliver_events=strategy.wants_events()) | market.config
    engine = build_engine(calendar._cal, config, prepared.instruments, prepared.frames)
    futures = None
    if prepared.futures is not None:
        futures = (prepared.futures, attach(engine, prepared.futures, list(plan.symbols)))
    base_bars = dict(zip(plan.symbols, prepared.frames, strict=True))
    run = RunData(plan.exchange, plan.interval, plan.base, base_bars, calendar, checks)
    context = RunContext(engine, prepared.instruments, plan.product, strategy.size, run, futures)
    instance = strategy(**(args["params"] or {}))
    instance._ctx = context
    instance.init()
    if has_conditions(instance) and futures is not None:
        raise NotImplementedError(
            "condition strategies on a futures underlying are not implemented yet (Phase 1 "
            "work in progress): write on_bar"
        )
    if has_conditions(instance):
        drive_conditions(instance, engine)
    else:
        drive(instance, engine, context)
    timer_warnings = context.timers.warnings()
    output = call(engine.finish)
    context.close(output["orders"])
    parameters = {name: getattr(instance, name) for name in strategy.parameters()}
    manifest = build_manifest(
        strategy,
        parameters,
        _arguments(prepared, args, market),
        [instrument.to_dict() for instrument in prepared.instruments],
        prepared.partitions,
        prepared.reference_tables,
        market.cost_schedules,
    )
    return Result(
        output,
        list(plan.symbols),
        _assumptions(prepared, calendar, checks, market),
        [*prepared.warnings, *timer_warnings],
        context.skipped,
        manifest.to_dict(),
        record=record,
    )


__all__ = ["PreparedRun", "execute", "prepare"]
