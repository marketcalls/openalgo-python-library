"""The first stage of a run's preparation (`_pipeline.prepare`): everything about it that is
checked before any data is loaded. No client is called and no bars are read here.

`check_run` resolves the settings (`_settings.resolve`), the benchmark asked for
(`_benchmark.benchmark_request`; its closes are loaded after the strategy's bars), the
run's own calendar, the futures series (`_futures`) and the instruments
(`_instruments.resolve_instruments`), checks that the margin table prices every
instrument (`_margin_check`), and learns what the indicators of each parameter set to be
run need before `start` (`_warm_up_probe`: `init` on the engine's reference bars). The
second stage, `_pipeline.load`, loads the data from what this one found. A portfolio run
checks every strategy this way before it loads the data of any (`_portfolio_run`).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from . import calendar as calendar_module
from ._benchmark import benchmark_request
from ._causality_checks import IndicatorChecks
from ._futures import resolve_futures
from ._instruments import resolve_instruments
from ._manifest import strategy_source_hash
from ._margin_check import check_margin
from ._settings import RunPlan, resolve
from ._warm_up import SeriesNeed
from ._warm_up_measure import WarmUp
from ._warm_up_probe import probe_needs, probe_setup
from .reference import Instrument


@dataclass(frozen=True)
class CheckedRun:
    """A run checked before any data is loaded (`check_run`): the plan (its base interval
    not chosen yet), the run's own calendar (None for the shared default one), the
    futures series (`_futures.FuturesPlan`, None for a strategy on no underlying), the
    instruments in registration order with an Assumptions line by symbol and the
    reference-table version used (None when the tables were not), the benchmark asked
    for (`_benchmark.benchmark_request`), and the needs before `start` of each parameter
    set probed (by `_warm_up_probe.params_key`) with the warm-ups measured to learn them
    (by indicator label)."""

    plan: RunPlan
    calendar: calendar_module.Calendar | None
    futures: Any
    instruments: list[Instrument]
    lines: dict[str, str]
    version: str | None
    benchmark: Any
    needs: dict[tuple, tuple[SeriesNeed, ...]]
    measured: dict[str, WarmUp]

    def session_calendar(self) -> calendar_module.Calendar:
        """The run's calendar."""
        return self.calendar or calendar_module.default_calendar()


def _calendar(given: Any, plan: RunPlan) -> calendar_module.Calendar | None:
    """The run's own calendar from `calendar=`: None for the shared default one."""
    if given is None:
        return None
    if isinstance(given, calendar_module.Calendar):
        return given
    return calendar_module.Calendar.from_sessions(given, plan.exchange, plan.start, plan.end)


def _instruments(plan: RunPlan, args: dict[str, Any], calendar: Any, futures: Any) -> tuple:
    """(instruments, Assumptions lines by symbol, reference-table version) of a run."""
    if futures is None:
        return resolve_instruments(plan, args["instrument"], calendar)
    return futures.instruments, futures.lines, futures.version


def check_run(
    strategy: type,
    args: dict[str, Any],
    parameter_sets: list[dict[str, Any] | None] | None = None,
    checks: IndicatorChecks | None = None,
) -> CheckedRun:
    """Check a run of `strategy` with the arguments of `bt.backtest` before any data is
    loaded (module docstring). `parameter_sets` are the `params=` of every combination of
    a sweep (None for a single run, whose own `params=` are probed); `checks` holds the
    indicator checks and warm-ups already done for the strategy (None to do them here).
    Raises ConfigError for unusable arguments, the reference tables' errors,
    NotImplementedError for settings of later work, and what the strategy's `init` raises
    on the reference bars."""
    plan = resolve(strategy, args)
    benchmark = benchmark_request(args["benchmark"], plan.exchange)
    own_calendar = _calendar(args["calendar"], plan)
    calendar = own_calendar or calendar_module.default_calendar()
    futures = resolve_futures(strategy, plan, args, calendar)
    plan = plan if futures is None else dataclasses.replace(plan, symbols=futures.symbols)
    instruments, lines, version = _instruments(plan, args, calendar, futures)
    check_margin(instruments, plan.product, (plan.start, plan.end))
    sets = [args["params"]] if parameter_sets is None else parameter_sets
    setup = probe_setup(strategy, plan, instruments, futures)
    checks = checks or IndicatorChecks(strategy_source_hash(strategy) or strategy.__qualname__)
    needs = probe_needs(strategy, setup, sets, checks)
    return CheckedRun(
        plan=plan,
        calendar=own_calendar,
        futures=futures,
        instruments=instruments,
        lines=lines,
        version=version,
        benchmark=benchmark,
        needs=needs,
        measured=checks.measured(),
    )


__all__ = ["CheckedRun", "check_run"]
