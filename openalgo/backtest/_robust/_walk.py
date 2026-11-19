"""The walk-forward (design 17, section 2.3), shared by `bt.walk_forward` and the walk-
forward check: each fold searches the grid on its in-sample window, trades the next
out-of-sample window with the combination that searched best, and the out-of-sample runs
are stitched into one equity curve (`_stitch`).

Each run is an ordinary backtest of the deterministic kernel on the run's bars
(`_reruns`): a combination over a window (`_windows`), its warm-up read from the bars
before the window. A fold chooses the combination with the highest objective in sample;
a tie goes to the first in grid order and an undefined objective is never chosen, as
`OptimizeResult.best` chooses. A fold with no defined objective in sample chooses none
and trades nothing.

Figures, each annualised so windows of different lengths compare (`annual`): an
objective that totals over time (net profit, net P&L, total return) is divided by the
years the run lasted; any other (a ratio such as Sharpe, a rate such as CAGR, a figure
per trade) is taken as it is.

    efficiency         the stitched out-of-sample objective over the mean of the folds'
                       in-sample objectives (of the combinations chosen), both measured
                       from the objective's break-even; NaN when the in-sample mean is
                       not above it or the objective has no break-even
    profitable_share   the share of the folds whose out-of-sample run made at least half
                       a paisa, every fold counted: one that chose nothing in sample (or
                       whose run did not run) made no money
    drift              for each parameter, how far the choice moves from fold to fold:
                       for a number, the mean move between consecutive folds over the
                       span of its values in the grid (0, the same choice every fold, to
                       1, from one end to the other every time); for anything else, the
                       share of consecutive folds whose choice differs
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..report._performance import Performance
from ._bar_common import break_even
from ._grid import Grid
from ._rerun_one import DAYS_PER_YEAR, RerunRow
from ._reruns import Reruns
from ._search_setup import SearchRun
from ._stitch import stitched
from ._trade_common import BREAK_EVEN_RUPEES
from ._windows import Fold

#: The objectives that total over time, annualised by the years a run lasted.
ANNUAL_TOTALS = ("net_profit", "net_pnl", "total_return")


def annual(value: float, years: float, objective: Any) -> float:
    """`value` of `objective` per year (module docstring)."""
    if totals(objective):
        return value / years if years > 0 else math.nan
    return value


def totals(objective: Any) -> bool:
    """Whether `objective` totals over time, so its figures are put per year."""
    return isinstance(objective, str) and objective in ANNUAL_TOTALS


def annual_name(objective: Any) -> str:
    """The objective's name as its compared figures read: "net_profit per year" for one
    that totals over time, else its name ("sharpe")."""
    name = objective if isinstance(objective, str) else getattr(objective, "__name__", "custom")
    return f"{name} per year" if totals(objective) else name


@dataclass(frozen=True, eq=False)
class FoldOutcome:
    """One fold: its windows, the objective of every combination in sample, the
    combination chosen (None for none), its in-sample row and its out-of-sample run
    (with its detail; None when none was chosen)."""

    fold: Fold
    in_sample: np.ndarray
    chosen: int | None
    in_row: RerunRow | None
    out: RerunRow | None

    def in_annual(self, objective: Any) -> float:
        """The chosen combination's in-sample objective per year."""
        row = self.in_row
        return math.nan if row is None else annual(row.objective, row.years, objective)

    def out_annual(self, objective: Any) -> float:
        """Its out-of-sample objective per year."""
        row = self.out
        return math.nan if row is None else annual(row.objective, row.years, objective)

    @property
    def traded(self) -> bool:
        """Whether it chose a combination and its out-of-sample run ran (its bars were
        not refused)."""
        return self.out is not None and not self.out.failure

    @property
    def profitable(self) -> bool:
        """Whether its out-of-sample run made at least half a paisa."""
        return self.traded and bool(self.out.net_profit >= BREAK_EVEN_RUPEES)


@dataclass(frozen=True, eq=False)
class WalkOutcome:
    """A walk-forward (module docstring): its folds, the grid, the objective, the
    stitched out-of-sample performance (None when no fold traded) and its figures."""

    folds: tuple[FoldOutcome, ...]
    grid: Grid
    objective: Any
    performance: Performance | None

    @property
    def traded(self) -> list[FoldOutcome]:
        """The folds that chose a combination and traded it out of sample."""
        return [fold for fold in self.folds if fold.traded]

    @property
    def stitched_annual(self) -> float:
        """The stitched out-of-sample objective per year (NaN for a function, which
        reads a whole `Result`)."""
        if self.performance is None or not isinstance(self.objective, str):
            return math.nan
        metrics = self.performance.metrics
        years = float(metrics["elapsed_days"]) / DAYS_PER_YEAR
        return annual(float(metrics[self.objective]), years, self.objective)

    @property
    def in_sample_annual(self) -> float:
        """The mean of the folds' in-sample objectives per year."""
        values = [fold.in_annual(self.objective) for fold in self.traded]
        known = [value for value in values if math.isfinite(value)]
        return math.fsum(known) / len(known) if known else math.nan

    @property
    def efficiency(self) -> float:
        """The walk-forward efficiency (module docstring)."""
        level = break_even(self.objective)
        inside = self.in_sample_annual
        if level is None or not inside > level:
            return math.nan
        return (self.stitched_annual - level) / (inside - level)

    @property
    def profitable_share(self) -> float:
        """The share of the folds whose out-of-sample run made money, every fold counted
        (module docstring)."""
        folds = self.folds
        return sum(fold.profitable for fold in folds) / len(folds) if folds else math.nan

    @property
    def net_profit(self) -> float:
        """The stitched out-of-sample net P&L, rupees."""
        return math.fsum(fold.out.net_profit for fold in self.traded)

    def drift(self) -> dict[str, float]:
        """The drift of each parameter (module docstring)."""
        chosen = [fold.chosen for fold in self.folds if fold.chosen is not None]
        return {name: _drift(self.grid, name, chosen) for name in self.grid.names}


def _drift(grid: Grid, name: str, chosen: Sequence[int]) -> float:
    values = [grid.combos[position][name] for position in chosen]
    if len(values) < 2:
        return 0.0
    pairs = list(zip(values[:-1], values[1:], strict=True))
    axis = grid.axis(name)
    if all(_number(value) for value in axis):
        span = float(max(axis)) - float(min(axis))
        moves = [abs(float(b) - float(a)) for a, b in pairs]
        return math.fsum(moves) / len(moves) / span if span > 0 else 0.0
    return sum(a != b for a, b in pairs) / len(pairs)


def _number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def best(values: np.ndarray) -> int | None:
    """The position of the highest defined value, the first of equal ones; None when no
    value is defined."""
    known = np.isfinite(values)
    if not known.any():
        return None
    return int(np.argmax(np.where(known, values, -np.inf)))


def walk(subject: SearchRun, runner: Reruns, folds: Sequence[Fold], objective: Any) -> WalkOutcome:
    """The walk-forward of `subject`'s grid over `folds` (module docstring), every run on
    `runner`. Raises what a rerun raises."""
    grid = subject.grid
    count = len(grid.combos)
    searched = [
        subject.family(grid, position, fold.in_sample)
        for fold in folds
        for position in range(count)
    ]
    rows = runner.firsts(searched)
    chosen, in_rows, scores = [], [], []
    for k, _ in enumerate(folds):
        mine = rows[k * count : (k + 1) * count]
        values = np.array([row.objective for row in mine], dtype=float)
        position = best(values)
        chosen.append(position)
        in_rows.append(None if position is None else mine[position])
        scores.append(values)
    traded = [
        (k, subject.family(grid, position, fold.out_of_sample, detail=True))
        for k, (fold, position) in enumerate(zip(folds, chosen, strict=True))
        if position is not None
    ]
    outs = dict(zip([k for k, _ in traded], runner.firsts([f for _, f in traded]), strict=True))
    outcomes = tuple(
        FoldOutcome(fold, scores[k], chosen[k], in_rows[k], outs.get(k))
        for k, fold in enumerate(folds)
    )
    details = [fold.out.detail for fold in outcomes if fold.traded]
    exchange = subject.strategy.exchange
    performance = stitched(details, subject.capital, exchange) if details else None
    return WalkOutcome(outcomes, grid, objective, performance)


__all__ = [
    "ANNUAL_TOTALS",
    "FoldOutcome",
    "WalkOutcome",
    "annual",
    "annual_name",
    "best",
    "totals",
    "walk",
]
