"""How one backtest runs, in two halves so that the combinations of a sweep share the
first (the Python half of the design in `rust/oa_bt/src/engine/mod.rs`).

`prepare` resolves everything that does not depend on the strategy's parameters, in two
stages. `_run_check.check_run` checks, before any data is loaded, the settings
(`_settings.resolve`), the calendar, the futures series (`_futures`), the instruments
(`_instruments.resolve_instruments`), that the margin table prices them all
(`_margin_check`) and what the indicators of each parameter set to be run need before
`start` (`_warm_up_probe`, `init` on the engine's reference bars: the run's parameters, or
every combination of a sweep). `load` then chooses the base interval and loads its bars
(`_base_interval.choose_base`, through the data layer), the ticks chosen by a price slab
(`_slab_ticks`, from the data) and the market settings (`_market`: the cost model, the
slippage and the per-instrument facts), and the history before `start` for those needs
(`_warm_up_load`: once for the largest, in which each parameter set finds the request
its single run makes). A portfolio run checks every strategy before it loads any
(`_portfolio_run`). A sweep sends its `PreparedRun` to worker
processes without the market settings, which hold native objects, and each worker
resolves them again from the `costs=` and `slippage=` arguments and the parsed `margin=`
(`_futures_margin`). That needs those arguments to pickle and the run to use the shared
default calendar (a calendar of its own is a native object too); otherwise the sweep
runs in this process. The benchmark's closes (`_benchmark`) are loaded last, after the
strategy's bars, and sent along.

`execute` keeps the history its parameters need (the needs `prepare` learnt for them,
`_warm_up_history`; checked where it meets the run's bars), builds the native engine,
binds a strategy instance to a `RunContext` whose series hold that history before the
run's bars, runs `init` once on them (`begin`) and the driver to the end of the grid (the
Rust signal driver for a condition strategy, `_conditions`), deciding only once the
strategy's indicators are ready (`_warm_up`), and returns the `Result` with its
Assumptions block, warnings and manifest (`conclude`, `_pipeline_result`). A portfolio
run calls `begin` and `conclude` for each strategy, with the fund's driver between them
(`_portfolio_driver`).
"""

from __future__ import annotations

import dataclasses
import pickle
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from . import calendar as calendar_module
from ._base_interval import choose_base
from ._benchmark import load_benchmark
from ._causality_checks import IndicatorChecks
from ._conditions import drive_conditions, evaluation_path, has_conditions
from ._context import RunContext, RunData
from ._driver import drive
from ._engine import build_engine, call
from ._expired_contracts import expired_contracts
from ._futures import attach
from ._futures_margin import margin_warnings
from ._manifest import strategy_source_hash
from ._market import MarketSettings, market_settings
from ._messages import not_supported
from ._pipeline_result import FinishedRun, finished_result
from ._run_check import CheckedRun, check_run
from ._run_corporate_actions import (
    UnadjustedEvent,
    check_crossings,
    run_warnings,
    unadjusted_events,
)
from ._session_dates import session_dates_note
from ._settings import RunPlan
from ._slab_ticks import resolve_slab_ticks
from ._warm_up import Readiness, SeriesNeed, readiness
from ._warm_up_history import RunHistory
from ._warm_up_load import LoadedHistory, load_history
from ._warm_up_measure import WarmUp
from ._warm_up_probe import combination_needs, params_key, probe_setup
from ._warm_up_source import history_source
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
    # The bars before `start`, loaded for the needs of the parameter sets probed
    # (`_warm_up_load`); each run keeps what its own needs ask for
    # (`_warm_up_history`). The needs of each parameter set probed, by
    # `_warm_up_probe.params_key`, and the warm-ups measured on the reference bars, by
    # indicator label, so that no run probes or measures again.
    history: LoadedHistory | None = None
    needs: dict[tuple, tuple[SeriesNeed, ...]] = dataclasses.field(default_factory=dict)
    measured: dict[str, WarmUp] = dataclasses.field(default_factory=dict)

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


def prepare(
    strategy: type,
    client: Any,
    args: dict[str, Any],
    parameter_sets: list[dict[str, Any] | None] | None = None,
    checks: IndicatorChecks | None = None,
) -> PreparedRun:
    """Resolve a run of `strategy` with the arguments of `bt.backtest` up to the engine:
    `_run_check.check_run`, then `load`. `parameter_sets` are the `params=` of every
    combination of a sweep (None for a single run, whose own `params=` are probed); the
    largest need of history before `start` over them is loaded. `checks` holds the
    indicator checks and warm-ups already done for the strategy (a rerun of the same
    strategy on other bars, `_robust`), None to do them here. Raises ConfigError for
    unusable arguments, the data layer's and the reference tables' errors,
    NotImplementedError for settings of later work, and what the strategy's `init`
    raises on the reference bars."""
    return load(check_run(strategy, args, parameter_sets, checks), client, args)


def load(checked: CheckedRun, client: Any, args: dict[str, Any]) -> PreparedRun:
    """The second stage of `prepare`: the base interval and its bars for the run
    `checked` (`_run_check.check_run`), through `client` or the run's own data, then the
    ticks chosen by a price slab, the market settings, the benchmark's closes and the
    history before `start` that the needs probed ask for. Raises the data layer's errors
    and ConfigError for data the run cannot use."""
    plan, futures, needs = checked.plan, checked.futures, checked.needs
    calendar = checked.session_calendar()
    windows = None if futures is None else futures.windows
    with expired_contracts(checked.instruments, plan.present):
        choice = choose_base(plan, args, client, calendar, windows)
    plan = dataclasses.replace(plan, base=choice.base)
    instruments, lines = resolve_slab_ticks(
        checked.instruments, checked.lines, plan, args, client, calendar, windows
    )
    futures = None if futures is None else futures.with_instruments(instruments)
    market_args = {"costs": args["costs"], "slippage": args["slippage"], "margin": plan.margin}
    market = market_settings(plan, instruments, market_args)
    loaded = [choice.bars[symbol] for symbol in plan.symbols]
    by_symbol = dict(zip(plan.symbols, loaded, strict=True))
    source = history_source(args, client, calendar, by_symbol, futures)
    frames = [_cut(result.bars, plan) for result in loaded]
    benchmark = load_benchmark(checked.benchmark, plan, args, client, calendar, frames)
    extra = [market.warnings, () if futures is None else futures.warnings]
    extra.append(() if benchmark is None else benchmark.warnings)
    return PreparedRun(
        plan=plan,
        calendar=checked.calendar,
        instruments=instruments,
        frames=frames,
        partitions={symbol: result.meta.partition_hashes() for symbol, result in by_symbol.items()},
        notes={
            "base_rule": choice.reason,
            "instruments": "; ".join(lines.values()),
            **session_dates_note(plan.symbols, loaded),
        },
        warnings=_load_warnings(plan, instruments, loaded, extra),
        reference_tables=checked.version,
        market_args=market_args,
        market=market,
        futures=futures,
        benchmark=benchmark,
        unadjusted=unadjusted_events(plan.symbols, loaded),
        history=load_history(plan, source, needs.values()),
        needs=needs,
        measured=checked.measured,
    )


def _kept_history(
    strategy: type, prepared: PreparedRun, params: Any, checks: IndicatorChecks
) -> RunHistory:
    """The history this run keeps: what its indicators need, learnt by its `init` on
    the reference bars (in `prepare`, or now for parameters it did not probe), cut from
    the history `prepare` loaded (`_warm_up_history`). Raises what the strategy's `init`
    raises."""
    plan = prepared.plan
    needs = prepared.needs.get(params_key(params))
    if needs is None:
        setup = probe_setup(strategy, plan, prepared.instruments, prepared.futures)
        needs = combination_needs(strategy, setup, params, checks)
    loaded = prepared.history or LoadedHistory({})
    run_bars = dict(zip(plan.symbols, prepared.frames, strict=True))
    return loaded.for_run(needs, plan, prepared.session_calendar(), run_bars)


def _bound(
    strategy: type,
    prepared: PreparedRun,
    market: MarketSettings,
    checks: Any,
    history: RunHistory,
) -> tuple[Any, RunContext]:
    """(the native engine, the run context over it, whose series hold `history`
    before the run's bars)."""
    plan, calendar = prepared.plan, prepared.session_calendar()
    config = plan.native_config(deliver_events=strategy.wants_events()) | market.config
    engine = build_engine(calendar._cal, config, prepared.instruments, prepared.frames)
    futures = None
    if prepared.futures is not None:
        futures = (prepared.futures, attach(engine, prepared.futures, list(plan.symbols)))
    frames = zip(plan.symbols, prepared.frames, strict=True)
    base_bars = {symbol: history.with_base(symbol, bars) for symbol, bars in frames}
    run = RunData(plan.exchange, plan.interval, plan.base, base_bars, calendar, checks, history)
    context = RunContext(engine, prepared.instruments, plan.product, strategy.size, run, futures)
    return engine, context


@dataclass(frozen=True)
class Begun:
    """A run bound and initialised, before its driver runs: the strategy instance (its
    `init` run), the engine, the run context, the market settings, the history kept and
    the readiness (`state.first_decision`: UTC nanoseconds, None from the first pause)."""

    instance: Any
    engine: Any
    context: RunContext
    market: MarketSettings
    history: RunHistory
    state: Readiness


def begin(
    strategy: type,
    prepared: PreparedRun,
    args: dict[str, Any],
    checks: IndicatorChecks | None = None,
) -> Begun:
    """The first half of `execute`: keep the history the parameters of `args["params"]`
    need, build the engine and the run context, run the strategy's `init` and learn when
    it is ready. Raises what `init` raises, ConfigError, BacktestError and
    NotImplementedError (a condition strategy on a futures underlying)."""
    if checks is None:
        name = strategy_source_hash(strategy) or strategy.__qualname__
        checks = IndicatorChecks(name, measured=prepared.measured)
    market = prepared.market_settings()
    history = _kept_history(strategy, prepared, args["params"], checks)
    engine, context = _bound(strategy, prepared, market, checks.for_run(), history)
    instance = strategy(**(args["params"] or {}))
    instance._ctx = context
    instance.init()
    if has_conditions(instance) and prepared.futures is not None:
        raise not_supported(
            "a condition strategy on a futures underlying",
            "write the strategy with on_bar",
        )
    state = readiness(context.run.checks.outcomes, context, history)
    context.waiting = state.waiting()
    return Begun(instance, engine, context, market, history, state)


def conclude(
    strategy: type,
    prepared: PreparedRun,
    args: dict[str, Any],
    begun: Begun,
    record: Any = None,
    *,
    assumptions: Mapping[str, str] | None = None,
) -> Result:
    """The second half of `execute`, once the driver has run to the end of the grid:
    finish the engine and build the `Result`; `assumptions` add lines to (or replace lines
    of) its Assumptions block. Raises the kernel's errors and CorporateActionError for a
    position held across an unadjusted split or bonus (`_run_corporate_actions`)."""
    context = begun.context
    timer_warnings = context.timers.warnings()
    output = call(begun.engine.finish)
    check_crossings(output, prepared.unadjusted)
    context.close(output["orders"])
    kept = (begun.market, begun.history, begun.state)
    finished = FinishedRun(begun.instance, context, *kept, output, timer_warnings)
    return finished_result(strategy, prepared, args, finished, record, assumptions=assumptions)


def execute(
    strategy: type,
    prepared: PreparedRun,
    args: dict[str, Any],
    checks: IndicatorChecks | None = None,
    record: Any = None,
    on_path: Callable[[str], Any] | None = None,
) -> Result:
    """Run `strategy` on a prepared run with the parameter values of `args["params"]`
    (`begin`, the driver, `conclude`). `checks` carries indicator checks already done (a
    sweep); `record` lets the result rerun itself; `on_path`, when given, is told the
    evaluation path the run took (`_conditions.evaluation_path`). Raises the kernel's and
    the strategy's errors unchanged."""
    begun = begin(strategy, prepared, args, checks)
    instance, ready_at = begun.instance, begun.state.first_decision
    if has_conditions(instance):
        drive_conditions(instance, begun.engine, ready_at)
    else:
        drive(instance, begun.engine, begun.context, ready_at)
    if on_path is not None:
        on_path(evaluation_path(instance))
    return conclude(strategy, prepared, args, begun, record)


__all__ = ["Begun", "PreparedRun", "begin", "conclude", "execute", "load", "prepare"]
