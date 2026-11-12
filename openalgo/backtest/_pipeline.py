"""How one backtest runs, in two halves so that the combinations of a sweep share the
first (the Python half of the design in `rust/oa_bt/src/engine/mod.rs`).

`prepare` resolves everything that does not depend on the strategy's parameters: the
settings (`_settings.resolve`), the calendar, the base interval and its bars
(`_base_interval.choose_base`, through the data layer) and the instruments
(`_instruments.resolve_instruments`). Its `PreparedRun` holds plain data, so a sweep
can send it to worker processes, unless the run has a calendar of its own (a native
object that does not cross processes; the shared default calendar is rebuilt there).

`execute` builds the native engine, binds a strategy instance to a `RunContext`, runs
`init` and the driver to the end of the grid, and returns the `Result` with its
Assumptions block, warnings and manifest.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import pandas as pd

from . import calendar as calendar_module
from . import costs as costs_module
from ._base_interval import choose_base, meaning
from ._causality_checks import IndicatorChecks
from ._context import RunContext, RunData
from ._driver import drive
from ._engine import build_engine, call
from ._instruments import resolve_instruments
from ._manifest import build_manifest, strategy_source_hash
from ._settings import RunPlan, assumptions, resolve
from .reference import Instrument
from .result import Result


@dataclass(frozen=True)
class PreparedRun:
    """A run resolved up to the engine: the plan (base interval decided), the calendar
    (None for the shared default one), the instruments and their base bars in
    registration order, the data partition hashes by symbol, the Assumptions lines and
    the warnings found so far."""

    plan: RunPlan
    calendar: Any
    instruments: list[Instrument]
    frames: list[pd.DataFrame]
    partitions: dict[str, dict[str, str]]
    notes: dict[str, str]
    warnings: list[str]
    reference_tables: str | None

    @property
    def portable(self) -> bool:
        """True when the run can be sent to another process (no calendar of its own)."""
        return self.calendar is None

    def session_calendar(self) -> calendar_module.Calendar:
        """The run's calendar."""
        return self.calendar or calendar_module.default_calendar()


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
    choice = choose_base(plan, args, client, calendar)
    plan = dataclasses.replace(plan, base=choice.base)
    instruments, lines, version = resolve_instruments(plan, args["instrument"], calendar)
    loaded = [choice.bars[symbol] for symbol in plan.symbols]
    warnings = [*plan.warnings]
    for result in loaded:
        warnings += [note for note in result.meta.warnings if note not in warnings]
    for instrument in instruments:
        if instrument.generic:
            warnings.append(
                f"{instrument.symbol}: generic cash-equity instrument assumed (lot 1, tick 0.05)"
            )
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
    )


def _assumptions(prepared: PreparedRun, calendar: Any, checks: IndicatorChecks) -> dict:
    plan = prepared.plan
    block = assumptions(plan, calendar)
    session = calendar.session(plan.exchange, plan.start)
    session_open = None
    if session is not None and block["square_off"] is not None:
        session_open = pd.Timestamp(session[0], tz="UTC").tz_convert("Asia/Kolkata")
        session_open = session_open.strftime("%H:%M")
    rule = meaning(plan.base, block["square_off"], session_open)
    block["base_rule"] = f"base {plan.base} ({prepared.notes['base_rule']}): {rule}"
    block["instruments"] = prepared.notes["instruments"]
    indicator_checks = checks.assumption()
    if indicator_checks is not None:
        block["indicator_checks"] = indicator_checks
    return block


def _cost_schedules(plan: RunPlan) -> list[str] | None:
    """Ids of the built-in cost schedule versions in force over the run, or None
    without charges."""
    if plan.costs is None:
        return None
    ids = []
    for schedule in costs_module.schedules():
        valid_from, valid_to = schedule["valid_from"], schedule["valid_to"]
        if valid_from <= plan.end and (valid_to is None or valid_to >= plan.start):
            ids.append(f"{plan.costs}:{schedule['id']}")
    return ids


def _arguments(prepared: PreparedRun, args: dict[str, Any]) -> dict[str, Any]:
    """Every argument of the run after defaults were resolved, for the manifest."""
    plan = prepared.plan
    resolved = dataclasses.asdict(plan)
    resolved.update(
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
    config = plan.native_config(deliver_events=strategy.wants_events())
    engine = build_engine(calendar._cal, config, prepared.instruments, prepared.frames)
    base_bars = dict(zip(plan.symbols, prepared.frames, strict=True))
    run = RunData(plan.exchange, plan.interval, plan.base, base_bars, calendar, checks)
    context = RunContext(engine, prepared.instruments, plan.product, strategy.size, run)
    instance = strategy(**(args["params"] or {}))
    instance._ctx = context
    instance.init()
    drive(instance, engine, context)
    timer_warnings = context.timers.warnings()
    output = call(engine.finish)
    context.close(output["orders"])
    parameters = {name: getattr(instance, name) for name in strategy.parameters()}
    manifest = build_manifest(
        strategy,
        parameters,
        _arguments(prepared, args),
        [instrument.to_dict() for instrument in prepared.instruments],
        prepared.partitions,
        prepared.reference_tables,
        _cost_schedules(plan),
    )
    return Result(
        output,
        list(plan.symbols),
        _assumptions(prepared, calendar, checks),
        [*prepared.warnings, *timer_warnings],
        context.skipped,
        manifest.to_dict(),
        record=record,
    )


__all__ = ["PreparedRun", "execute", "prepare"]
