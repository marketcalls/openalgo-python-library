"""Setting up the bar checks of a report: the run's bars, a first rerun on them, the
reruns' runner, and what each bar check reads (`_bar_common.BarRun`).

The first rerun runs the strategy on the run's own bars through the reruns' path
(`_rerun_one`): it does the indicator checks once for every rerun after it, times a
rerun for the plan's estimate, tells the evaluation path, and should give the run's own
objective. If it does not (a source whose bars changed since the run), the plan says so,
since every generated run is then measured against a run that read other bars.

How many reruns a report plans (`planned`): each bar check's runs (versus random's for
each of its families; noise and permutation theirs; stress at most its candidates; other
markets one a symbol), less the permutation test's runs that versus random shares.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._conditions import EVENT_PATH
from .._objective import evaluate
from . import _words as words
from ._bar_common import BarRun
from ._check_vs_random import RANDOM
from ._paths import fan_steps
from ._random_entries import EntryPlan, entry_plan
from ._registry import BY_NAME
from ._request import Request
from ._rerun_one import Known, rerun
from ._reruns import Reruns, Setup, Workers, choose_workers, shipped
from ._run_bars import RunBars, run_bars
from ._trades import TradeRun

#: Reruns on the event path above which the plan suggests the condition form, as
#: `bt.optimize` does for a large grid (`_optimize.LARGE_EVENT_GRID`).
LARGE_EVENT_RERUNS = 1000
#: How close the first rerun's objective must be to the run's to count as the same, a
#: fraction of the run's: the kernel is deterministic, so only a change of the bars
#: read makes them differ.
SAME_OBJECTIVE = 1e-9


def planned(request: Request) -> tuple[dict[str, int], int]:
    """(the reruns each bar check plans, by name; those versus random shares with the
    permutation test) of `request` (module docstring)."""
    counts: dict[str, int] = {}
    for check in request.reruns:
        settings, runs = request.settings[check.name], request.runs_of(check)
        if check.name == "vs_random":
            counts[check.name] = runs * len(settings["families"])
        elif check.name == "other_markets":
            counts[check.name] = len(request.others)
        elif check.name == "stress":
            counts[check.name] = settings["noise_candidates"] + settings["permutation_candidates"]
        else:
            counts[check.name] = runs
    shared = 0
    both = "vs_random" in counts and "permutation" in counts
    if both and "permuted" in request.settings["vs_random"]["families"]:
        shared = min(request.runs_of(BY_NAME["vs_random"]), counts["permutation"])
    return counts, shared


@dataclass(frozen=True, eq=False)
class BarContext:
    """The bar checks' shared state: the run's bars, the reruns' runner and workers, the
    run's objective (`real`), the first rerun's objective (`reference`) and its seconds,
    and the evaluation path it took."""

    bars: RunBars
    reruns: Reruns
    workers: Workers
    real: float
    reference: float
    seconds: float
    path: str

    @property
    def same(self) -> bool:
        """Whether the first rerun gave the run's own objective."""
        if math.isnan(self.real) and math.isnan(self.reference):
            return True
        return abs(self.reference - self.real) <= SAME_OBJECTIVE * max(1.0, abs(self.real))

    def subject(self, trades: TradeRun, result: Any, request: Request) -> BarRun:
        """What a bar check reads."""
        return BarRun(
            trades,
            result,
            self.real,
            self.bars,
            self.reruns,
            request.seed,
            request.others,
            result._record.client,
        )


def _entries(result: Any, bars: RunBars, request: Request) -> EntryPlan | None:
    """What random entries are drawn from, when versus random draws them."""
    if "vs_random" not in request.settings:
        return None
    if not set(request.settings["vs_random"]["families"]) & set(RANDOM):
        return None
    closes = {
        symbol: np.asarray(closes)[np.asarray(closes) > bars.first_run_ns]
        for symbol, closes in zip(bars.symbols, result._strategy_closes, strict=True)
    }
    return entry_plan(result.trade_list(), closes)


def bar_context(result: Any, request: Request, trades: TradeRun, total: int) -> BarContext:
    """The `BarContext` of `result` for `request`, planning `total` reruns. Raises what
    loading the run's bars and the first rerun raise, and ConfigError for a strategy or
    objective worker processes cannot import (`_reruns.shipped`)."""
    strategy, objective = result._record.strategy, request.objective
    bars, arguments = run_bars(result)
    paths: list[str] = []
    started = time.perf_counter()
    first, known = rerun(
        strategy, arguments, bars.symbols, list(bars.frames), Known(), paths.append
    )
    seconds = time.perf_counter() - started
    workers = choose_workers(request.workers, total, strategy, arguments)
    steps = fan_steps(len(trades.sessions))
    setup = Setup(
        shipped(strategy, "the strategy", workers),
        shipped(objective, "the objective", workers),
        arguments,
        bars,
        known,
        trades.sessions,
        steps,
        _entries(result, bars, request),
    )
    real = evaluate(result, objective)["objective"]
    reference = evaluate(first, objective)["objective"]
    return BarContext(bars, Reruns(setup, workers), workers, real, reference, seconds, paths[0])


def plan_words(counts: Mapping[str, int], shared: int, context: BarContext | None) -> str:
    """The plan's sentence on the reruns (deterministic: no time, no worker count)."""
    if not counts:
        return (
            "They read the run's trade list, so the backtest is not rerun and no worker "
            "process starts."
        )
    parts = [
        f"{name} {'up to ' if name == 'stress' else ''}{words.number(count, 0)}"
        for name, count in counts.items()
        if count
    ]
    total = words.number(sum(counts.values()) - shared, 0)
    text = (
        "The trade-list checks read the run's trade list; the bar checks rerun the backtest "
        f"{total} times or fewer on generated bars or other symbols ({', '.join(parts)}), "
        "each an ordinary backtest of the same kernel."
    )
    if shared:
        text += f" Versus random shares {words.number(shared, 0)} runs with the permutation test."
    if context is not None and not context.same:
        text += (
            f" A rerun on the run's own bars gave {context.reference!r} where the run gave "
            f"{context.real!r}: the source's bars changed since the run, and every rerun is "
            "measured against the run."
        )
    return text


def timing_words(counts: Mapping[str, int], shared: int, context: BarContext) -> str:
    """The plan's estimate of the reruns, reported before them and not kept."""
    total = sum(counts.values()) - shared
    workers = context.workers
    where = f"{workers.count} worker processes" if workers.count > 1 else "this process"
    why = f" ({workers.why})" if workers.why else ""
    estimate = total * context.seconds / max(workers.count, 1)
    text = (
        f" A rerun on the run's own bars took {context.seconds:.2f} s; in {where}{why} the "
        f"reruns should take about {estimate:.1f} s."
    )
    if context.path == EVENT_PATH and total > LARGE_EVENT_RERUNS:
        text += (
            " That many reruns suit the condition form (entry_when, exit_when), which runs "
            "without a Python call per bar."
        )
    return text


__all__ = [
    "LARGE_EVENT_RERUNS",
    "BarContext",
    "bar_context",
    "plan_words",
    "planned",
    "timing_words",
]
