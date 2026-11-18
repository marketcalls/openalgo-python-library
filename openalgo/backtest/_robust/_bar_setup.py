"""Setting up the bar checks of a report: the run's bars, a first rerun on them, the
reruns' runner, and what each bar check reads (`_bar_common.BarRun`).

The first rerun runs the strategy on the run's own bars through the reruns' path
(`_rerun_one`): it does the indicator checks once for every rerun after it, tells the
evaluation path, and should give the run's own objective. If it does not (a source whose
bars changed since the run), the plan says so, since every generated run is then
measured against a run that read other bars. The plan's estimate is timed on run 0 of the
first family of generated bars the checks read (a noisy one first, kept for the check
that reads it), so it counts making the bars and measuring the rerun as well as the
rerun; with no generated family (other markets alone) it is timed on the first rerun.

How many reruns a report plans (`planned`): each bar check's runs (versus random's for
each of its families; noise and permutation theirs; stress its candidates; other markets
one a symbol), less the runs two checks share: a family of generated bars is rerun once
up to the most runs any check asks of it (`_bar_common`: the permutation test's permuted
paths, which versus random and stress read, and the noise check's noisy copies, which
stress reads at the default noise).
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
from . import _random
from . import _words as words
from ._bar_common import BarRun, noisy_family, permuted_family
from ._check_noise import noise_of
from ._check_vs_random import RANDOM
from ._check_vs_random import family as random_family
from ._generate_noisy import Noise
from ._paths import fan_steps
from ._random_entries import EntryPlan, entry_plan
from ._request import Request
from ._rerun_one import Known, rerun
from ._reruns import NOISY, Family, Reruns, Setup, Workers, choose_workers, shipped
from ._run_bars import RunBars, run_bars
from ._trades import TradeRun

#: Reruns on the event path above which the plan suggests the condition form, as
#: `bt.optimize` does for a large grid (`_optimize.LARGE_EVENT_GRID`).
LARGE_EVENT_RERUNS = 1000
#: How close the first rerun's objective must be to the run's to count as the same, a
#: fraction of the run's: the kernel is deterministic, so only a change of the bars
#: read makes them differ.
SAME_OBJECTIVE = 1e-9


def _families(request: Request, check: Any) -> list[tuple[Family | str, int]]:
    """(family, runs) of each family of reruns bar check `check` asks for; other markets
    has one family of its own (its symbols)."""
    settings, runs = request.settings[check.name], request.runs_of(check)
    if check.name == "vs_random":
        seed = _random.check_seed(request.seed, check.number)
        return [(random_family(name, seed, request.seed), runs) for name in settings["families"]]
    if check.name == "permutation":
        return [(permuted_family(request.seed), runs)]
    if check.name == "noise":
        return [(noisy_family(request.seed, noise_of(settings)), runs)]
    if check.name == "stress":
        return [
            (noisy_family(request.seed, Noise()), settings["noise_candidates"]),
            (permuted_family(request.seed), settings["permutation_candidates"]),
        ]
    return [("other symbols", len(request.others))]


def planned(request: Request) -> tuple[dict[str, int], int]:
    """(the reruns each bar check plans, by name; how many of them another check shares)
    of `request` (module docstring)."""
    counts: dict[str, int] = {}
    most: dict[Family | str, int] = {}
    for check in request.reruns:
        families = _families(request, check)
        counts[check.name] = sum(runs for _, runs in families)
        for family, runs in families:
            most[family] = max(most.get(family, 0), runs)
    return counts, sum(counts.values()) - sum(most.values())


@dataclass(frozen=True, eq=False)
class BarContext:
    """The bar checks' shared state: the run's bars, the reruns' runner and workers, the
    run's objective (`real`), the first rerun's objective (`reference`), the seconds of
    the timed rerun and whether it was on generated bars (module docstring), and the
    evaluation path the first rerun took."""

    bars: RunBars
    reruns: Reruns
    workers: Workers
    real: float
    reference: float
    seconds: float
    generated: bool
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
    reruns = Reruns(setup, workers)
    timed = _timed_family(request)
    if timed is not None:
        started = time.perf_counter()
        reruns.rows(timed, [0])
        seconds = time.perf_counter() - started
    generated = timed is not None
    return BarContext(bars, reruns, workers, real, reference, seconds, generated, paths[0])


def _timed_family(request: Request) -> Family | None:
    """The family whose run 0 times the plan's estimate (module docstring), None when
    the checks read no generated bars."""
    families = [
        family
        for check in request.reruns
        for family, runs in _families(request, check)
        if isinstance(family, Family) and runs
    ]
    families.sort(key=lambda family: family.bars != NOISY)
    return families[0] if families else None


def plan_words(counts: Mapping[str, int], shared: int, context: BarContext | None) -> str:
    """The plan's sentence on the reruns (deterministic: no time, no worker count)."""
    if not counts:
        return (
            "They read the run's trade list, so the backtest is not rerun and no worker "
            "process starts."
        )
    parts = [f"{name} {words.number(count, 0)}" for name, count in counts.items() if count]
    total = words.number(sum(counts.values()) - shared, 0)
    text = (
        "The trade-list checks read the run's trade list; the bar checks rerun the backtest "
        f"{total} times or fewer on generated bars or other symbols ({', '.join(parts)}), "
        "each an ordinary backtest of the same kernel."
    )
    if shared:
        text += (
            f" The checks share {words.number(shared, 0)} of the runs listed, which read the "
            "same generated bars, so the total counts each once."
        )
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
    timed = (
        "A rerun on generated bars took {:.2f} s, making its bars and measuring it included"
        if context.generated
        else "A rerun on the run's own bars took {:.2f} s"
    )
    text = (
        f" {timed.format(context.seconds)}; in {where}{why} the reruns should take about "
        f"{estimate:.1f} s."
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
