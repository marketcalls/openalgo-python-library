"""`bt.walk_forward` and `WalkForwardResult` (design 17, section 2.3).

    result = bt.walk_forward(EmaCross, data=frame, capital=100_000, folds=5,
                             fast=range(5, 30, 5), slow=range(20, 100, 10))
    result.folds, result.equity, result.efficiency, result.profitable_share, result.drift

The run's sessions are cut into folds (`_windows.folds`: rolling by default, or
anchored; 70 percent of each fold in sample), each fold searches the parameter ranges in
sample and trades the best out of sample, and the out-of-sample runs are stitched into
one equity curve (`_walk`, `_stitch`), as the walk-forward check of `bt.robustness` runs
it: the same settings give the same figures. Every run is an ordinary backtest of the
deterministic kernel on the run's bars, read once with the history of every
combination's warm-up (`_run_bars.bars_for`), so the result is the same whatever the
number of worker processes.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from .._conditions import EVENT_PATH
from .._messages import counted
from .._objective import check_objective
from .._optimize import _run_arguments, _split_settings
from ..errors import ConfigError
from ..report._performance import Performance
from ._check_walk_forward import SETTINGS as CHECK_SETTINGS
from ._grid import Grid, search_grid
from ._part_objective import objective_name
from ._paths import fan_steps
from ._rerun_one import Known, rerun
from ._rerun_plan import CONDITION_FORM, LARGE_EVENT_RERUNS
from ._reruns import Reruns, Setup, choose_workers, shipped
from ._run_bars import bars_for
from ._search_setup import SearchRun, parameter_sets, session_dates
from ._setting import resolve, whole
from ._walk import WalkOutcome, walk
from ._walk_tables import fold_frame
from ._windows import folds as cut_folds

LOGGER = logging.getLogger("openalgo.backtest.walk_forward")
#: What a walk-forward of a futures series is refused with, and what to do instead.
FUTURES = (
    "bt.walk_forward of a strategy on a futures series",
    "a futures series changes contract at each roll, and the walk-forward reruns the "
    "run's bars as one series of its own, which cannot hold the rolls yet; run "
    "bt.optimize on each in-sample window and bt.backtest on the window after it",
)
#: The settings of the check that `bt.walk_forward` takes as its own arguments.
FOLD_SETTINGS = tuple(s for s in CHECK_SETTINGS if s.name in ("folds", "anchored", "in_sample"))


class WalkForwardResult:
    """What `bt.walk_forward` returns: its folds, the stitched out-of-sample equity and
    performance, and the walk-forward's figures (module docstring of `_walk`)."""

    def __init__(self, outcome: WalkOutcome, dates: tuple, plan: str) -> None:
        self._outcome = outcome
        self._dates = dates
        self._plan = plan

    @property
    def folds(self) -> pd.DataFrame:
        """A row per fold, indexed by fold number: the first and last session of its
        in-sample and out-of-sample windows (IST dates), the parameters chosen in sample,
        the in-sample and out-of-sample objective per year (a total such as the net
        profit divided by the years its window lasted), their efficiency (from the
        objective's break-even), and the out-of-sample net P&L (rupees), maximum
        drawdown (a fraction, zero or negative), P&L over drawdown, closed trades and win
        rate. A fold that chose nothing has NaN figures."""
        return fold_frame(self._outcome, self._dates)

    @property
    def performance(self) -> Performance | None:
        """The stitched out-of-sample run measured as any run is (`metrics.analyse`, at
        session closes), None when no fold traded."""
        return self._outcome.performance

    @property
    def equity(self) -> pd.Series:
        """The stitched out-of-sample equity at each session close, rupees, from the
        run's capital (empty when no fold traded)."""
        performance = self._outcome.performance
        return pd.Series(dtype=float) if performance is None else performance.equity.copy()

    @property
    def efficiency(self) -> float:
        """The walk-forward efficiency: the stitched out-of-sample objective per year
        over the mean of the folds' in-sample ones, both from the objective's break-even
        (0, or 1 for a profit factor); NaN for an objective with none, a function of a
        `Result` (the stitched run has no `Result`), or an in-sample mean not above it."""
        return self._outcome.efficiency

    @property
    def profitable_share(self) -> float:
        """The share of the folds whose out-of-sample run made money, every fold counted:
        one that chose nothing in sample made none."""
        return self._outcome.profitable_share

    @property
    def net_profit(self) -> float:
        """The stitched out-of-sample net P&L, rupees."""
        return self._outcome.net_profit

    @property
    def drift(self) -> pd.Series:
        """Each parameter's drift from fold to fold, a fraction: for a number, the mean
        move between consecutive folds' choices over the span of its range; otherwise,
        the share of consecutive folds whose choice differs."""
        return pd.Series(self._outcome.drift(), name="drift", dtype=float)

    @property
    def objective(self) -> str:
        """The objective's name."""
        return objective_name(self._outcome.objective)

    @property
    def plan(self) -> str:
        """What was reported before the folds ran, with its estimate."""
        return self._plan

    def __repr__(self) -> str:
        return (
            f"WalkForwardResult({counted(len(self._outcome.folds), 'fold')}, objective "
            f"{self.objective!r}, efficiency {self.efficiency:.2f}, "
            f"{self.profitable_share:.0%} of folds profitable)"
        )


def _plan(
    name: str,
    made: tuple,
    settings: Mapping[str, Any],
    combos: int,
    workers: Any,
    timed: tuple[float, str],
) -> str:
    """The line reported before the reruns: `timed` is (the seconds a rerun of the whole
    run took, its evaluation path); a large total on the event path gets the condition
    form's suggestion, as `bt.robustness` gives it."""
    seconds, path = timed
    total = len(made) * (combos + 1)
    where = f"{workers.count} worker processes" if workers.count > 1 else "this process"
    why = f" ({workers.why})" if workers.why else ""
    mode = "anchored" if settings["anchored"] else "rolling"
    text = (
        f"bt.walk_forward: {counted(len(made), 'fold')} of {name}, {mode}, each "
        f"{settings['in_sample']:.0%} in sample, searching {counted(combos, 'combination')}: "
        f"{total} reruns ({len(made) * combos} in sample, {len(made)} out of sample), each an "
        f"ordinary backtest of the same kernel, in {where}{why}. A rerun of the whole run took "
        f"{seconds:.2f} s; the reruns should take about "
        f"{total * seconds / max(workers.count, 1):.1f} s."
    )
    if path == EVENT_PATH and total > LARGE_EVENT_RERUNS:
        text += CONDITION_FORM
    return text


def _checked(
    strategy: type, objective: Any, shape: dict[str, Any], workers: Any, progress: Any
) -> dict[str, Any]:
    """The fold settings `shape` checked. Raises ConfigError for an unusable argument."""
    from ..strategy import Strategy

    if not (isinstance(strategy, type) and issubclass(strategy, Strategy)):
        raise ConfigError("bt.walk_forward takes a bt.Strategy subclass")
    check_objective(objective)
    if workers is not None and whole(1)(workers) is None:
        raise ConfigError(f"workers takes a whole number, 1 or more, got {workers!r}")
    if progress is not None and not callable(progress):
        raise ConfigError(f"progress takes a function of one line of text, got {progress!r}")
    return resolve("bt.walk_forward", FOLD_SETTINGS, shape)


def _grid(strategy: type, ranges: dict[str, Any], given: Any) -> Grid:
    """The grid searched, `params=` `given` fixing the parameters it names as in
    `bt.optimize`. Raises ConfigError when there is nothing to search."""
    grid = search_grid(strategy, ranges, given, params_fix=True)
    if grid is None:
        raise ConfigError(
            f"bt.walk_forward: nothing to search: pass ranges named after the parameters of "
            f"{strategy.__name__} ({sorted(strategy.parameters())}, fast=range(5, 30, 5)) or "
            "declare them with bt.param(default, values)"
        )
    return grid


def walk_forward(
    strategy: type,
    client: Any = None,
    *,
    folds: int = 5,
    anchored: bool = False,
    in_sample: float = 0.7,
    objective: str | Callable[[Any], float] = "net_profit",
    workers: int | None = None,
    progress: Callable[[str], Any] | None = None,
    **settings: Any,
) -> WalkForwardResult:
    """Walk `strategy` forward over its parameter ranges (module docstring); see
    `bt.walk_forward`. Raises ConfigError for unusable arguments, too few sessions for
    the folds, or nothing to search, and whatever a backtest raises."""
    shape = {"folds": folds, "anchored": anchored, "in_sample": in_sample}
    shape = _checked(strategy, objective, shape, workers, progress)
    run_settings, ranges = _split_settings(strategy, settings, "bt.walk_forward")
    arguments = _run_arguments(client, run_settings, "bt.walk_forward")
    grid = _grid(strategy, ranges, arguments["params"])
    sets = parameter_sets(strategy, arguments["params"], grid, None)
    bars, rerun_arguments = bars_for(strategy, client, arguments, sets, FUTURES)
    dates = session_dates(bars)
    made = cut_folds(len(dates), shape["folds"], shape["in_sample"], shape["anchored"])
    started, paths = time.perf_counter(), []
    _, known = rerun(
        strategy, rerun_arguments, bars.symbols, list(bars.frames), Known(), paths.append
    )
    timed = (time.perf_counter() - started, paths[0])
    chosen = choose_workers(workers, len(made) * (len(grid.combos) + 1), strategy, rerun_arguments)
    ship = (shipped(strategy, "the strategy", chosen), shipped(objective, "the objective", chosen))
    steps = fan_steps(len(dates))
    setup = Setup(*ship, rerun_arguments, bars, known, pd.DatetimeIndex(dates), steps, None)
    plan = _plan(strategy.__name__, made, shape, len(grid.combos), chosen, timed)
    (progress or LOGGER.info)(plan)
    with Reruns(setup, chosen) as runner:
        capital = float(arguments["capital"])
        given, bounds = arguments["params"], (rerun_arguments["start"], rerun_arguments["end"])
        subject = SearchRun(
            strategy, capital, None, None, math.nan, grid, given, bars, runner, 0, dates, *bounds
        )
        outcome = walk(subject, runner, made, objective)
    return WalkForwardResult(outcome, dates, plan)


__all__ = ["FOLD_SETTINGS", "LOGGER", "WalkForwardResult", "walk_forward"]
