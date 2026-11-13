"""How one backtest runs, in two halves so that the combinations of a sweep share the
first (the Python half of the design in `rust/oa_bt/src/engine/mod.rs`).

`prepare` resolves everything that does not depend on the strategy's parameters: the
settings (`_settings.resolve`), the calendar, the futures series (`_futures`), the
instruments (`_instruments.resolve_instruments`) and, before any data is loaded, that the
margin table prices them all (`_margin_check`) and what the indicators of each parameter
set to be run need before `start` (`_warm_up_probe`); then the base interval and its bars
(`_base_interval.choose_base`, through the data layer), the ticks chosen by a price slab
(`_slab_ticks`, from the data) and the market settings (`_market`: the cost model, the
slippage and the per-instrument facts), and the history before `start` for the largest
of those needs (`_warm_up_history`). A sweep sends its `PreparedRun` to worker
processes without the market settings, which hold native objects, and each worker
resolves them again from the `costs=` and `slippage=` arguments and the parsed `margin=`
(`_futures_margin`). That needs those arguments to pickle and the run to use the shared
default calendar (a calendar of its own is a native object too); otherwise the sweep
runs in this process. The benchmark's closes (`_benchmark`) are loaded last, after the
strategy's bars, and sent along.

`execute` keeps the history its parameters need (`_warm_up_history`), builds the native
engine, binds a strategy instance to a `RunContext`, runs `init` and the driver to the
end of the grid (the Rust signal driver for a condition strategy, `_conditions`),
deciding only once the strategy's indicators are ready (`_warm_up`), and returns the
`Result` with its Assumptions block, warnings and manifest.
"""

from __future__ import annotations

import dataclasses
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from . import calendar as calendar_module
from ._base_interval import choose_base, meaning
from ._benchmark import benchmark_request, load_benchmark
from ._causality_checks import IndicatorChecks, RunChecks
from ._conditions import drive_conditions, evaluation_path, has_conditions
from ._context import RunContext, RunData
from ._driver import drive
from ._engine import build_engine, call
from ._expired_contracts import expired_contracts
from ._futures import attach, resolve_futures
from ._futures_margin import margin_assumption, margin_warnings
from ._instruments import resolve_instruments
from ._intraday_assumptions import one_clock_time
from ._manifest import build_manifest, strategy_source_hash
from ._margin_check import check_margin
from ._market import MarketSettings, market_settings
from ._messages import not_supported
from ._rejections import reject_warnings
from ._run_corporate_actions import (
    UnadjustedEvent,
    check_crossings,
    manifest_form,
    run_warnings,
    unadjusted_events,
)
from ._run_end import run_end_warnings
from ._session_dates import session_dates_note
from ._settings import RunPlan, assumptions, resolve
from ._slab_ticks import resolve_slab_ticks
from ._warm_up import Readiness, readiness
from ._warm_up_attach import attach as join_history
from ._warm_up_history import LoadedHistory, RunHistory
from ._warm_up_load import HistorySource, history_source, load_history
from ._warm_up_probe import history_needs, probe_setup, sweep_needs
from .errors import BacktestError
from .reference import Instrument
from .result import Result


@dataclass(frozen=True)
class PreparedRun:
    """A run resolved up to the engine: the plan (base interval decided), the calendar
    (None for the shared default one), the instruments and their base bars in
    registration order, the data partition hashes by symbol, the Assumptions lines, the
    warnings found so far, the reference-table version used, and the market settings
    (None in a worker, rebuilt from `market_args`, the `costs=` and `slippage=`
    arguments and the parsed `margin=`)."""

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
    # The benchmark (`_benchmark.Benchmark`), reference closes for the reports; None
    # without one.
    benchmark: Any = None
    # Declared splits and bonuses the bars are not adjusted for, which nothing may be
    # held across (`_run_corporate_actions`).
    unadjusted: tuple[UnadjustedEvent, ...] = ()
    # A sweep's bars before `start`, loaded for the largest need of its combinations
    # (`_warm_up_load`); None for a single run, which loads its own after `init` from
    # `history_source`. Each run keeps what its own needs ask for (`_warm_up_history`).
    history: LoadedHistory | None = None
    history_source: HistorySource | None = None

    @property
    def portable(self) -> bool:
        """True when the run can be sent to another process: the shared default
        calendar, and `costs=` and `slippage=` arguments that pickle."""
        return self.calendar is None and _picklable(self.market_args)

    def for_workers(self) -> PreparedRun:
        """This run without its market settings and its history source (it holds the
        client), to send to worker processes: they keep the history loaded."""
        return dataclasses.replace(self, market=None, history_source=None)

    def market_settings(self) -> MarketSettings:
        """The market settings, resolved again when this run came without them."""
        if self.market is not None:
            return self.market
        # Unverified rates were announced when the run was prepared (`_market`).
        return market_settings(self.plan, self.instruments, self.market_args, announce=False)

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


def _load_warnings(
    plan: RunPlan, instruments: list[Instrument], loaded: list[Any], extra: list[Any]
) -> list[str]:
    """The warnings found while preparing: the plan's and the margin's, each load's
    (once each), the corporate actions', the generic instruments', then each of `extra`
    (the market's, the futures plan's and the benchmark's), in that order."""
    warnings = [*plan.warnings, *margin_warnings(plan.margin, instruments)]
    for result in loaded:
        warnings += [note for note in result.meta.warnings if note not in warnings]
    warnings += run_warnings(plan.symbols, loaded)
    for instrument in instruments:
        if instrument.generic:
            warnings.append(
                f"{instrument.symbol}: generic cash-equity instrument assumed (lot 1, tick 0.05)"
            )
    for notes in extra:
        warnings += list(notes)
    return warnings


def _instruments(plan: RunPlan, args: dict[str, Any], calendar: Any, futures: Any) -> tuple:
    """(instruments, Assumptions lines by symbol, reference-table version) of a run."""
    if futures is None:
        return resolve_instruments(plan, args["instrument"], calendar)
    return futures.instruments, futures.lines, futures.version


def prepare(
    strategy: type,
    client: Any,
    args: dict[str, Any],
    parameter_sets: list[dict[str, Any] | None] | None = None,
) -> PreparedRun:
    """Resolve a run of `strategy` with the arguments of `bt.backtest` up to the engine.
    `parameter_sets` are the `params=` of every combination of a sweep, whose largest
    need of history before `start` is loaded here (`_warm_up_probe.sweep_needs`); a
    single run loads its history after `init`. Raises ConfigError for unusable
    arguments, the data layer's and the reference tables' errors, NotImplementedError
    for settings of later work, and, for a sweep, what the strategy's `init` raises."""
    plan = resolve(strategy, args)
    # Checked before any data is loaded; loaded after the strategy's bars.
    wanted_benchmark = benchmark_request(args["benchmark"], plan.exchange)
    own_calendar = _calendar(args["calendar"], plan)
    calendar = own_calendar or calendar_module.default_calendar()
    futures = resolve_futures(strategy, plan, args, calendar)
    windows = None if futures is None else futures.windows
    plan = plan if futures is None else dataclasses.replace(plan, symbols=futures.symbols)
    instruments, lines, version = _instruments(plan, args, calendar, futures)
    # Before any data is loaded: an instrument the margin table cannot price stops here,
    # and a sweep's largest need before start is learnt on the reference bars.
    check_margin(instruments, plan.product, (plan.start, plan.end))
    needs = sweep_needs(strategy, plan, instruments, futures, parameter_sets)
    with expired_contracts(instruments, plan.present):
        choice = choose_base(plan, args, client, calendar, windows)
    plan = dataclasses.replace(plan, base=choice.base)
    instruments, lines = resolve_slab_ticks(
        instruments, lines, plan, args, client, calendar, windows
    )
    futures = None if futures is None else futures.with_instruments(instruments)
    market_args = {"costs": args["costs"], "slippage": args["slippage"], "margin": plan.margin}
    market = market_settings(plan, instruments, market_args)
    loaded = [choice.bars[symbol] for symbol in plan.symbols]
    by_symbol = dict(zip(plan.symbols, loaded, strict=True))
    source = history_source(args, client, calendar, by_symbol, futures)
    frames = [_cut(result.bars, plan) for result in loaded]
    benchmark = load_benchmark(wanted_benchmark, plan, args, client, calendar, frames)
    extra = [market.warnings, () if futures is None else futures.warnings]
    extra.append(() if benchmark is None else benchmark.warnings)
    return PreparedRun(
        plan=plan,
        calendar=own_calendar,
        instruments=instruments,
        frames=frames,
        partitions={symbol: result.meta.partition_hashes() for symbol, result in by_symbol.items()},
        notes={
            "base_rule": choice.reason,
            "instruments": "; ".join(lines.values()),
            **session_dates_note(plan.symbols, loaded),
        },
        warnings=_load_warnings(plan, instruments, loaded, extra),
        reference_tables=version,
        market_args=market_args,
        market=market,
        futures=futures,
        benchmark=benchmark,
        unadjusted=unadjusted_events(plan.symbols, loaded),
        history=None if needs is None else load_history(plan, source, needs),
        history_source=source,
    )


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
        benchmark=None if prepared.benchmark is None else prepared.benchmark.record,
        corporate_actions=manifest_form(args.get("corporate_actions")),
    )
    # The present only when `now=` fixed it: a run on the clock reruns to the same manifest.
    present = resolved.pop("present")
    resolved["now"] = None if args["now"] is None else present
    return resolved


def _kept_history(
    strategy: type, prepared: PreparedRun, context: RunContext, checks: IndicatorChecks
) -> RunHistory:
    """The history this run keeps: what the indicators its `init` registered need,
    loaded now for a single run, cut from the sweep's load otherwise
    (`_warm_up_history`)."""
    plan = prepared.plan
    setup = probe_setup(strategy, plan, prepared.instruments, prepared.futures)
    needs = history_needs(strategy, context.run.checks.outcomes, setup, checks)
    loaded = prepared.history
    if loaded is None:
        if prepared.history_source is None:
            raise BacktestError(
                "this prepared run has no history and no source to load it from; please report this"
            )
        loaded = load_history(plan, prepared.history_source, needs)
    return loaded.for_run(needs, plan, prepared.session_calendar())


def _bound(
    strategy: type, prepared: PreparedRun, market: MarketSettings, checks: Any
) -> tuple[Any, RunContext]:
    """(the native engine, the run context over it; the history before `start` is
    joined after `init`, `_warm_up_attach`)."""
    plan, calendar = prepared.plan, prepared.session_calendar()
    config = plan.native_config(deliver_events=strategy.wants_events()) | market.config
    engine = build_engine(calendar._cal, config, prepared.instruments, prepared.frames)
    futures = None
    if prepared.futures is not None:
        futures = (prepared.futures, attach(engine, prepared.futures, list(plan.symbols)))
    base_bars = dict(zip(plan.symbols, prepared.frames, strict=True))
    run = RunData(plan.exchange, plan.interval, plan.base, base_bars, calendar, checks)
    context = RunContext(engine, prepared.instruments, plan.product, strategy.size, run, futures)
    return engine, context


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
    strategy: type, prepared: PreparedRun, args: dict[str, Any], run: _Finished, record: Any
) -> Result:
    """The `Result` of a finished run, with its Assumptions block, warnings and manifest."""
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
        block | {"warm_up": run.state.assumption()},
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
    )


def execute(
    strategy: type,
    prepared: PreparedRun,
    args: dict[str, Any],
    checks: IndicatorChecks | None = None,
    record: Any = None,
    on_path: Callable[[str], Any] | None = None,
) -> Result:
    """Run `strategy` on a prepared run with the parameter values of `args["params"]`.
    `checks` carries indicator checks already done (a sweep); `record` lets the result
    rerun itself; `on_path`, when given, is told the evaluation path the run took
    (`_conditions.evaluation_path`). Raises the kernel's and the strategy's errors
    unchanged."""
    checks = checks or IndicatorChecks(strategy_source_hash(strategy) or strategy.__qualname__)
    market = prepared.market_settings()
    engine, context = _bound(strategy, prepared, market, checks.for_run())
    instance = strategy(**(args["params"] or {}))
    instance._ctx = context
    instance.init()
    if has_conditions(instance) and prepared.futures is not None:
        raise not_supported(
            "a condition strategy on a futures underlying",
            "write the strategy with on_bar",
        )
    history = _kept_history(strategy, prepared, context, checks)
    join_history(context, history)
    state = readiness(context.run.checks.outcomes, context, history)
    if has_conditions(instance):
        drive_conditions(instance, engine, state.ready_at)
    else:
        drive(instance, engine, context, state.ready_at)
    if on_path is not None:
        on_path(evaluation_path(instance))
    timer_warnings = context.timers.warnings()
    output = call(engine.finish)
    check_crossings(output, prepared.unadjusted)
    context.close(output["orders"])
    finished = _Finished(instance, context, market, history, state, output, timer_warnings)
    return _result(strategy, prepared, args, finished, record)


__all__ = ["PreparedRun", "execute", "prepare"]
