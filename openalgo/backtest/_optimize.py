"""`bt.optimize`: a parameter sweep over a grid (14-strategy-api 5.1; 07-research-and-
reporting 1; 03-architecture 5).

    result = bt.optimize(EmaCross, data=frame, capital=100_000, costs=None, slippage=0,
                         fast=range(5, 30, 5), slow=range(20, 100, 10))
    result.table, result.best, result.select(fast=10, slow=40), result.heatmap("fast", "slow")

The grid: every parameter named with a range in the call, plus every parameter declared
with `bt.param(default, values)` and not fixed by `params=`, in the order the class
declares them; the others keep their defaults (or their `params=` values). Every
other keyword is a run argument of `bt.backtest`, the same for every combination.

How it runs: the inputs (settings, base interval, bars, instruments, the benchmark's
closes) are prepared once, so every combination is measured against the same benchmark.
The first combination runs in this process. It runs the indicator causality checks for
the whole sweep (they are never repeated per combination) and shows which evaluation
path of 03-architecture 5 the strategy takes (`OptimizeResult.path`):

- "event": `on_bar` (and timers and callbacks) runs in Python at every decision; the
  first combination's duration gives the estimate.
- "conditions": the strategy declared `entry_when` and the like, and each combination
  turns them into arrays whose bars run in the Rust signal driver. Such a run is short
  next to the indicator checks, so the second combination runs in this process as well
  and its duration gives the estimate.

That plan (combinations, path, where the rest runs, the estimate) is reported before the
rest starts, through `progress` (a function taking one line of text) or, without one,
the `openalgo.backtest.optimize` logger at INFO. The rest run in `workers` spawned
processes (the CPU count by default; see `_sweep_pool`), or in this process with
`workers=1`, when the run has a calendar or cost model of its own, or for a strategy
built at run time by `bt.from_arrays` or `bt.from_signals`, which no worker could
import. Every combination is an ordinary backtest of the same deterministic kernel, so
the table is the same as that of the single runs, whatever the number of workers.

An event grid of more than `LARGE_EVENT_GRID` combinations gets a suggestion to write
the strategy as conditions.
"""

from __future__ import annotations

import inspect
import itertools
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ._arrays import made_by
from ._causality_checks import IndicatorChecks
from ._conditions import CONDITIONS_PATH, EVENT_PATH
from ._manifest import strategy_source_hash
from ._objective import DEFAULT_OBJECTIVE, check_objective
from ._optimize_result import OptimizeResult
from ._pipeline import PreparedRun, execute, prepare
from ._rerun import RunRecord
from ._sweep_pool import Chunk, portable, run_combinations, run_in_pool, split
from .errors import ConfigError

# Event strategies suit hundreds to a few thousand combinations (07-research 1).
LARGE_EVENT_GRID = 1000
# What runs in Python on each path, as the plan states it.
PATH_MEANINGS = {
    EVENT_PATH: "on_bar runs in Python for every strategy bar",
    CONDITIONS_PATH: (
        "the conditions become arrays once per combination and the bars run in Rust, "
        "with no Python call per bar"
    ),
}
LOGGER = logging.getLogger("openalgo.backtest.optimize")

# Runs combinations in this process; told each run's evaluation path when given a
# function (`_sweep_pool.run_combinations` with the sweep's fixed inputs bound).
Runner = Callable[..., list[dict[str, float]]]


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


@dataclass(frozen=True)
class Opening:
    """The combinations run in this process before the plan is reported: their rows in
    grid order, the evaluation path of the first, and the seconds each took (the first
    also ran the sweep's indicator checks)."""

    rows: list[dict[str, float]]
    path: str
    seconds: list[float]


def _opening(run: Runner, combinations: list[dict[str, Any]]) -> Opening:
    """Run the first combination, and on the conditions path the second as well (see
    the module docstring)."""
    rows: list[dict[str, float]] = []
    paths: list[str] = []

    def timed(position: int) -> float:
        started = time.perf_counter()
        rows.extend(run(combinations[position : position + 1], paths.append))
        return time.perf_counter() - started

    seconds = [timed(0)]
    if paths[0] == CONDITIONS_PATH and len(combinations) > 1:
        seconds.append(timed(1))
    return Opening(rows, paths[0], seconds)


def _workers(
    requested: Any, remaining: int, prepared: PreparedRun, strategy: type
) -> tuple[int, str]:
    """(worker processes for the rest of the sweep, why it runs in this process)."""
    if requested is not None and (isinstance(requested, bool) or not isinstance(requested, int)):
        raise ConfigError(f"workers takes a whole number of processes, got {requested!r}")
    if requested is not None and requested < 1:
        raise ConfigError(f"workers takes at least 1, got {requested}")
    count = min(requested or os.cpu_count() or 1, max(remaining, 1))
    maker = made_by(strategy)
    if count > 1 and maker is not None:
        return 1, f"a strategy made by {maker} is built in this process and is not sent to workers"
    if count > 1 and not prepared.portable:
        return 1, "the run's own calendar or cost model cannot be sent to worker processes"
    return count, ""


def _plan(
    strategy: type, combinations: int, workers: int, seconds: list[float], why: str, path: str
) -> str:
    """The line reported before the rest of the sweep runs. `seconds` are the durations
    of the combinations already run, in order; the last stands for each of the rest."""
    rest = (combinations - len(seconds)) * seconds[-1] / workers
    where = f"{workers} worker processes" if workers > 1 else "this process"
    took = f"The first took {seconds[0]:.2f} s"
    if len(seconds) > 1:
        took += f" with the sweep's indicator checks, the second took {seconds[1]:.2f} s"
    text = (
        f"bt.optimize: {combinations} combinations of {strategy.__name__} on the {path} path "
        f"({PATH_MEANINGS[path]}), in {where}{f' ({why})' if why else ''}. {took}; the "
        f"rest should take about {rest:.1f} s."
    )
    if path == EVENT_PATH and combinations > LARGE_EVENT_GRID:
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
    light["now"] = None if arguments["now"] is None else "given"

    def run(
        part: list[dict[str, Any]], on_path: Callable[[str], Any] | None = None
    ) -> list[dict[str, float]]:
        return run_combinations(strategy, objective, prepared, light, part, checks, on_path)

    opening = _opening(run, combinations)
    rest = combinations[len(opening.rows) :]
    count, why = _workers(workers, len(rest), prepared, strategy)
    plan = _plan(strategy, len(combinations), count, opening.seconds, why, opening.path)
    (progress or LOGGER.info)(plan)
    if count > 1:
        shipped = (portable(strategy, "the strategy"), portable(objective, "the objective"))
        chunks = [
            Chunk(*shipped, prepared.for_workers(), light, part, checks.strategy_hash, checks.keys)
            for part in split(rest, count)
        ]
        later = run_in_pool(chunks, count)
    else:
        later = run(rest)
    rows = [*opening.rows, *later]
    table = pd.DataFrame(
        [{**{n: c[n] for n in names}, **row} for c, row in zip(combinations, rows, strict=True)]
    )
    label = objective if isinstance(objective, str) else getattr(objective, "__name__", "custom")

    def runner(params: dict[str, Any]) -> Any:
        selected = {**arguments, "params": params}
        record = RunRecord(strategy, client, {k: v for k, v in selected.items() if k != "client"})
        return execute(strategy, prepared, selected, record=record)

    return OptimizeResult(table, combinations, names, label, maximize, opening.path, plan, runner)


__all__ = ["CONDITIONS_PATH", "EVENT_PATH", "LARGE_EVENT_GRID", "grid", "optimize"]
