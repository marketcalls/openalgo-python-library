"""`bt.optimize`: a parameter sweep over a grid (14-strategy-api 5.1; 07-research-and-
reporting 1; 03-architecture 5).

    result = bt.optimize(EmaCross, data=frame, capital=100_000, costs=None, slippage=0,
                         fast=range(5, 30, 5), slow=range(20, 100, 10))
    result.table, result.best, result.select(fast=10, slow=40), result.heatmap("fast", "slow")

The grid: every parameter named with a range in the call, plus every parameter declared
with `bt.param(default, values)` and not fixed by `params=`, in the order the class
declares them; the others keep their defaults (or their `params=` values). Every
other keyword is a run argument of `bt.backtest`, the same for every combination.

How it runs: the inputs (settings, base interval, bars, instruments) are prepared once.
The first combination runs in this process; it runs the indicator causality checks for
the whole sweep (they are never repeated per combination) and its duration gives the
estimate reported before the rest starts, through `progress` (a function taking one
line of text) or, without one, the `openalgo.backtest.optimize` logger at INFO. The rest
run in `workers` spawned processes (the CPU count by default; see `_sweep_pool`), or in
this process with `workers=1` or when the run has a calendar of its own. Every
combination is an ordinary backtest of the same deterministic kernel, so the table is
the same as that of the single runs, whatever the number of workers.

Only the event path exists before Phase 4: each combination calls `on_bar` in Python for
every strategy bar. A grid of more than `LARGE_EVENT_GRID` combinations gets a
suggestion to write the strategy as conditions.
"""

from __future__ import annotations

import inspect
import itertools
import logging
import os
import time
from collections.abc import Callable
from typing import Any

import pandas as pd

from ._causality_checks import IndicatorChecks
from ._manifest import strategy_source_hash
from ._objective import DEFAULT_OBJECTIVE, check_objective
from ._optimize_result import OptimizeResult
from ._pipeline import PreparedRun, execute, prepare
from ._rerun import RunRecord
from ._sweep_pool import Chunk, portable, run_combinations, run_in_pool, split
from .errors import ConfigError

EVENT_PATH = "event"
# Event strategies suit hundreds to a few thousand combinations (07-research 1).
LARGE_EVENT_GRID = 1000
LOGGER = logging.getLogger("openalgo.backtest.optimize")


def _run_arguments(client: Any, settings: dict[str, Any]) -> dict[str, Any]:
    """The arguments of `bt.backtest` with the given ones applied, as `backtest` passes
    them on. Raises ConfigError for a missing required argument."""
    from .run import backtest

    arguments: dict[str, Any] = {"client": client}
    for name, parameter in inspect.signature(backtest).parameters.items():
        if name in ("strategy", "client"):
            continue
        if name in settings:
            arguments[name] = settings[name]
        elif parameter.default is inspect.Parameter.empty:
            raise ConfigError(f"bt.optimize needs {name}=, as bt.backtest does")
        else:
            arguments[name] = parameter.default
    return arguments


def _values(name: str, values: Any) -> tuple[Any, ...]:
    if isinstance(values, str) or not hasattr(values, "__iter__"):
        values = (values,)
    values = tuple(values)
    if not values:
        raise ConfigError(f"the range of {name} is empty")
    return values


def grid(
    strategy: type, ranges: dict[str, Any], fixed: dict[str, Any] | None
) -> tuple[list[str], list[dict[str, Any]]]:
    """(swept parameter names, every combination as the full `params=` of a run), in
    grid order. Raises ConfigError for unknown names, a name both fixed and swept, an
    empty range, or nothing to sweep."""
    defaults = strategy.parameters()
    fixed = dict(fixed or {})
    unknown = sorted(set(fixed) - set(defaults))
    if unknown:
        raise ConfigError(f"params= names {unknown} that {strategy.__name__} does not have")
    both = sorted(set(fixed) & set(ranges))
    if both:
        raise ConfigError(f"{both} are both fixed by params= and swept")
    declared = strategy.parameter_ranges()
    swept: dict[str, tuple[Any, ...]] = {}
    for name in defaults:
        if name in ranges:
            swept[name] = _values(name, ranges[name])
        elif name in declared and name not in fixed:
            swept[name] = declared[name]
    if not swept:
        raise ConfigError(
            f"nothing to sweep: pass ranges named after {sorted(defaults)} (fast=range(5, 30)) "
            "or declare them with bt.param(default, values)"
        )
    names = list(swept)
    combinations = [
        {**fixed, **dict(zip(names, values, strict=True))}
        for values in itertools.product(*swept.values())
    ]
    return names, combinations


def _workers(requested: Any, remaining: int, prepared: PreparedRun) -> tuple[int, str]:
    """(worker processes for the rest of the sweep, why it runs in this process)."""
    if requested is not None and (isinstance(requested, bool) or not isinstance(requested, int)):
        raise ConfigError(f"workers takes a whole number of processes, got {requested!r}")
    if requested is not None and requested < 1:
        raise ConfigError(f"workers takes at least 1, got {requested}")
    count = min(requested or os.cpu_count() or 1, max(remaining, 1))
    if count > 1 and not prepared.portable:
        return 1, "the run's own calendar cannot be sent to worker processes"
    return count, ""


def _plan(strategy: type, combinations: int, workers: int, first: float, why: str) -> str:
    rest = (combinations - 1) * first / workers
    where = f"{workers} worker processes" if workers > 1 else "this process"
    text = (
        f"bt.optimize: {combinations} combinations of {strategy.__name__} on the event path "
        f"(on_bar runs in Python for every strategy bar), in {where}"
        f"{f' ({why})' if why else ''}. The first took {first:.2f} s; the rest should take "
        f"about {rest:.1f} s."
    )
    if combinations > LARGE_EVENT_GRID:
        text += (
            " A grid this large suits the condition form (entry_when, exit_when), which "
            "runs without a Python call per bar."
        )
    return text


def _split_settings(strategy: type, settings: dict[str, Any]) -> tuple[dict, dict]:
    """(run arguments, parameter ranges) of the keywords. Raises ConfigError for a name
    that is neither."""
    from .run import backtest

    run_names = set(inspect.signature(backtest).parameters) - {"strategy", "client"}
    parameters = strategy.parameters()
    unknown = sorted(set(settings) - run_names - set(parameters))
    if unknown:
        raise ConfigError(
            f"bt.optimize: {unknown} are neither run arguments of bt.backtest nor parameters "
            f"of {strategy.__name__} ({sorted(parameters)})"
        )
    run = {name: value for name, value in settings.items() if name in run_names}
    ranges = {name: value for name, value in settings.items() if name in parameters}
    return run, ranges


def optimize(
    strategy: type,
    client: Any = None,
    *,
    objective: str | Callable[[Any], float] = DEFAULT_OBJECTIVE,
    maximize: bool = True,
    workers: int | None = None,
    progress: Callable[[str], Any] | None = None,
    **settings: Any,
) -> OptimizeResult:
    """Sweep `strategy` over a grid; see the module docstring. Raises ConfigError for
    unusable arguments, a strategy or objective that worker processes cannot import
    (without cloudpickle), and whatever a single backtest raises."""
    from .strategy import Strategy

    if not (isinstance(strategy, type) and issubclass(strategy, Strategy)):
        raise ConfigError("bt.optimize takes a bt.Strategy subclass")
    check_objective(objective)
    run_settings, ranges = _split_settings(strategy, settings)
    arguments = _run_arguments(client, run_settings)
    names, combinations = grid(strategy, ranges, arguments["params"])
    prepared = prepare(strategy, client, arguments)
    checks = IndicatorChecks(strategy_source_hash(strategy) or strategy.__qualname__)
    light = {key: arguments[key] for key in ("source", "tz")}
    light["data"] = None if arguments["data"] is None else "given"
    started = time.perf_counter()
    rows = run_combinations(strategy, objective, prepared, light, combinations[:1], checks)
    count, why = _workers(workers, len(combinations) - 1, prepared)
    plan = _plan(strategy, len(combinations), count, time.perf_counter() - started, why)
    (progress or LOGGER.info)(plan)
    rest = combinations[1:]
    if count > 1:
        shipped = (portable(strategy, "the strategy"), portable(objective, "the objective"))
        chunks = [
            Chunk(*shipped, prepared, light, part, checks.strategy_hash, checks.keys)
            for part in split(rest, count)
        ]
        rows += run_in_pool(chunks, count)
    else:
        rows += run_combinations(strategy, objective, prepared, light, rest, checks)
    table = pd.DataFrame(
        [{**{n: c[n] for n in names}, **row} for c, row in zip(combinations, rows, strict=True)]
    )
    label = objective if isinstance(objective, str) else getattr(objective, "__name__", "custom")

    def runner(params: dict[str, Any]) -> Any:
        selected = {**arguments, "params": params}
        record = RunRecord(strategy, client, {k: v for k, v in selected.items() if k != "client"})
        return execute(strategy, prepared, selected, record=record)

    return OptimizeResult(table, combinations, names, label, maximize, EVENT_PATH, plan, runner)


__all__ = ["EVENT_PATH", "LARGE_EVENT_GRID", "grid", "optimize"]
