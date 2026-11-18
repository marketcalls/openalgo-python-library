"""Setting up the checks of a report that rerun the backtest: the run's bars, a first rerun
on them, the reruns' runner, and what each check reads (`_bar_common.BarRun` for a bar
check, `_search_setup.SearchRun` for a searching check).

The first rerun runs the strategy on the run's own bars through the reruns' path
(`_rerun_one`): it does the indicator checks once for every rerun after it, tells the
evaluation path, and should give the run's own objective. If it does not (a source whose
bars changed since the run), the plan says so, since every generated run is then
measured against a run that read other bars. The searching checks read the run's bars
with the history of every combination they run (`_search_setup`); when bar checks run
too, a first rerun on those bars does their indicator checks, and their runner is a
sibling of the bar checks' that shares its worker processes (`_reruns.Reruns.sibling`).

The plan's estimate is timed on run 0 of the first family that a bar check will rerun (a
noisy one first, kept for the check that reads it), so on generated bars it counts making
the bars and measuring the rerun as well as the rerun; with no such family it is timed on
the first family a searching check will rerun, and with none on the first rerun. A
family on the run's own bars with a setting changed (a delay, a fill model, shifted
strategy bars, varied exits, other parameters or dates) is a rerun on the run's own bars.
What the plan counts is `_rerun_plan`'s.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._objective import evaluate
from ._bar_common import BarRun
from ._check_vs_random import RANDOM
from ._paths import fan_steps
from ._random_entries import EntryPlan, entry_plan
from ._request import Request
from ._rerun_one import Known, rerun
from ._rerun_plan import Plan, changed_only, plan_words, planned, timing_words
from ._reruns import NOISY, Family, Reruns, Setup, Workers, choose_workers, shipped
from ._run_bars import RunBars, run_bars
from ._search_setup import SearchRun, first_known, search_preview
from ._trades import TradeRun

#: How close the first rerun's objective must be to the run's to count as the same, a
#: fraction of the run's: the kernel is deterministic, so only a change of the bars
#: read makes them differ.
SAME_OBJECTIVE = 1e-9


@dataclass(frozen=True, eq=False)
class BarContext:
    """The reruns' shared state: the run's bars (the searching checks' when no bar check
    runs), the runner that owns the worker processes and the workers, the run's objective
    (`real`), the first rerun's objective (`reference`), the seconds of the timed rerun
    and whether it was on generated bars (module docstring), the evaluation path the
    first rerun took, the `Plan`, and what the searching checks read (None when none
    runs)."""

    bars: RunBars
    reruns: Reruns
    workers: Workers
    real: float
    reference: float
    seconds: float
    generated: bool
    path: str
    plan: Plan
    search: SearchRun | None = None

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
            self.reruns.setup.entries,
        )

    def words(self) -> str:
        """The plan's sentence on the reruns (`_rerun_plan.plan_words`)."""
        return plan_words(self.plan, self.same, self.reference, self.real)

    def timing(self) -> str:
        """The plan's estimate of the reruns (`_rerun_plan.timing_words`)."""
        return timing_words(self.plan, self.seconds, self.generated, self.workers, self.path)


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
    return entry_plan(result.trade_list(), closes, bars.sessions)


def _search(
    result: Any, request: Request, trades: TradeRun, real: float
) -> tuple[SearchRun | None, dict[str, Any] | None]:
    """The searching checks' subject with no runner yet and its reruns' arguments, when
    the request runs one."""
    if not any(check.searches for check in request.reruns):
        return None, None
    step = request.settings.get("parameters", {}).get("step")
    return search_preview(result, request.ranges, request.seed, trades, real, step)


def bar_context(result: Any, request: Request, trades: TradeRun) -> BarContext:
    """The `BarContext` of `result` for `request`. Raises what loading the run's bars and
    the first rerun raise, and ConfigError for a strategy or objective worker processes
    cannot import (`_reruns.shipped`)."""
    strategy, objective = result._record.strategy, request.objective
    real = evaluate(result, objective)["objective"]
    plain = any(not check.searches for check in request.reruns)
    search, search_arguments = _search(result, request, trades, real)
    bars, arguments = run_bars(result) if plain else (search.bars, search_arguments)
    paths: list[str] = []
    started = time.perf_counter()
    first, known = rerun(
        strategy, arguments, bars.symbols, list(bars.frames), Known(), paths.append
    )
    seconds = time.perf_counter() - started
    entries = _entries(result, bars, request) if plain else None
    client = result._record.client
    preview = BarRun(
        trades, result, real, bars, None, request.seed, request.others, client, entries
    )
    plan = planned(request, preview if plain else None, search)
    workers = choose_workers(request.workers, plan.total, strategy, arguments)
    steps = fan_steps(len(trades.sessions))
    ship = (
        shipped(strategy, "the strategy", workers),
        shipped(objective, "the objective", workers),
    )
    setup = Setup(*ship, arguments, bars, known, trades.sessions, steps, entries)
    reruns = Reruns(setup, workers)
    if search is not None:
        search_known = first_known(strategy, search_arguments, search.bars) if plain else known
        own = Setup(
            *ship, search_arguments, search.bars, search_known, trades.sessions, steps, None
        )
        runner = reruns.sibling(own) if plain else reruns
        search = dataclasses.replace(search, reruns=runner)
    reference = evaluate(first, objective)["objective"]
    timed, runner = _timed_family(plan, reruns, search)
    generated = False
    if timed is not None:
        started = time.perf_counter()
        runner.rows(timed, [0])
        seconds = time.perf_counter() - started
        generated = not changed_only(timed)
    return BarContext(
        bars, reruns, workers, real, reference, seconds, generated, paths[0], plan, search
    )


def _timed_family(
    plan: Plan, reruns: Reruns, search: SearchRun | None
) -> tuple[Family | None, Reruns]:
    """The family whose run 0 times the plan's estimate (module docstring) and the
    runner it runs on; None when no check will rerun."""
    families = sorted(plan.families, key=lambda family: family.bars != NOISY)
    if families:
        return families[0], reruns
    if plan.searched and search is not None:
        return plan.searched[0], search.reruns
    return None, reruns


__all__ = ["SAME_OBJECTIVE", "BarContext", "bar_context"]
