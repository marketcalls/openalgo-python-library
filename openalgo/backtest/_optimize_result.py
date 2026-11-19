"""`OptimizeResult`: what `bt.optimize` returns (14-strategy-api 5.2).

`.table` has one row per combination, in grid order: the parameter values, the
objective and the key metrics (`_objective.KEY_METRICS`). `.best` is the combination
with the highest objective (the lowest with `maximize=False`); a tie goes to the first
in grid order and a NaN objective is never best. `.select(**params)` runs one
combination again in this process and returns its full `Result`, bit for bit what the
sweep computed. `.heatmap(x, y)` is the objective over two parameters with the others
held at their best values. `.path` names the evaluation path that ran (03-architecture
5): "event" for a strategy whose `on_bar`, timers or callbacks run in Python at every
decision, "conditions" for one that declared `entry_when` and the like (or was made by
`bt.from_arrays` or `bt.from_signals`), whose bars run in the Rust signal driver.
`.plan` is the line reported before the sweep started, with its duration estimate.

From the grid it already ran, with no rerun (design 17, section 5; `_robust._optimizer_
checks`): `.neighbourhood`, the parameter-neighbourhood rule on the best combination's
neighbours in the grid, a `CheckResult`; `.warnings`, a line when the best is a lone
spike rather than a plateau; `.deflated_sharpe`, the probability that the best's true
Sharpe ratio is above what the best of that many tries would show by luck;
`.overfitting_probability`, the probability of backtest overfitting by combinatorially
symmetric cross-validation over the combinations' session returns (a figure near 0 does
not show the settings hold); `.overfitting`, both with what they rest on; and
`.tearsheet(path)`, the heatmap page that shows them. They are computed on first use
and never change the table.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cached_property
from os import PathLike
from typing import Any

import numpy as np
import pandas as pd

from ._messages import counted
from .errors import BacktestError, ConfigError


class OptimizeResult:
    """The result of one sweep; see the module docstring."""

    def __init__(
        self,
        table: pd.DataFrame,
        combinations: list[dict[str, Any]],
        parameters: list[str],
        objective: str,
        maximize: bool,
        path: str,
        plan: str,
        runner: Callable[[dict[str, Any]], Any],
        returns: np.ndarray,
        named: bool,
        strategy: str,
    ) -> None:
        self._table = table
        self._combinations = combinations
        self._parameters = parameters
        self._objective = objective
        self._maximize = maximize
        self._path = path
        self._plan = plan
        self._runner = runner
        # The session returns, (sessions, combinations) in grid order
        # (`_sweep_pool.returns_matrix`); whether the objective is a metric's name (not a
        # function); the strategy's name, for the page's title.
        self._returns = returns
        self._named = named
        self._strategy = strategy

    @property
    def table(self) -> pd.DataFrame:
        """One row per combination: parameters, `objective`, key metrics."""
        return self._table.copy()

    @property
    def path(self) -> str:
        """The evaluation path that ran: "event" or "conditions"."""
        return self._path

    @property
    def plan(self) -> str:
        """What was reported before the sweep: combinations, path, workers, estimate."""
        return self._plan

    @property
    def objective(self) -> str:
        """The objective's name."""
        return self._objective

    def _best_position(self) -> int:
        """The position of the best combination. Raises BacktestError when every
        objective is NaN."""
        scores = self._table["objective"].reset_index(drop=True)
        if scores.notna().sum() == 0:
            raise BacktestError("no combination has a finite objective, so none is best")
        return int(scores.idxmax() if self._maximize else scores.idxmin())

    def _swept(self, position: int) -> dict[str, Any]:
        """The swept values of a combination, as they were given."""
        return {name: self._combinations[position][name] for name in self._parameters}

    @property
    def best(self) -> dict[str, Any]:
        """The parameters of the best combination. Raises BacktestError when every
        objective is NaN."""
        return self._swept(self._best_position())

    def select(self, **params: Any) -> Any:
        """The full `Result` of one combination of the grid. Raises ConfigError for
        values that are not a combination of the sweep."""
        if set(params) != set(self._parameters):
            raise ConfigError(f"select takes a value for each of {self._parameters}")
        for position in range(len(self._combinations)):
            if self._swept(position) == params:
                return self._runner(self._combinations[position])
        raise ConfigError(f"{params} is not a combination of this sweep")

    def heatmap(self, x: str, y: str) -> pd.DataFrame:
        """The objective with `x` values as columns and `y` values as rows; any other
        parameter is held at its value in `best`. Raises ConfigError for unknown or
        equal names."""
        for name in (x, y):
            if name not in self._parameters:
                raise ConfigError(f"{name!r} is not a swept parameter: {self._parameters}")
        if x == y:
            raise ConfigError("heatmap takes two different parameters")
        rows = self._table
        best = self.best
        for name in self._parameters:
            if name not in (x, y):
                rows = rows[rows[name] == best[name]]
        return rows.pivot(index=y, columns=x, values="objective")

    @cached_property
    def _checks(self) -> Any:
        """The overfitting figures (module docstring), computed on first use."""
        from ._robust._optimizer_checks import optimizer_checks

        return optimizer_checks(
            self._table,
            self._combinations,
            self._parameters,
            (self._objective, self._named),
            self._maximize,
            self._returns,
            self._best_position(),
        )

    @property
    def neighbourhood(self) -> Any:
        """The parameter-neighbourhood rule on the best combination (design 17, 2.16): a
        `CheckResult` whose table has a row per combination (`best`, `neighbour`), its
        summary the share of neighbours profitable and the median and 25th percentile
        neighbour over the best, from the objective's break-even. Not judged for
        `maximize=False`, an objective with no break-even, or a grid of one combination.
        Raises BacktestError when no combination has a finite objective."""
        return self._checks.neighbourhood

    @property
    def warnings(self) -> list[str]:
        """A line when the best combination is a lone spike (its neighbourhood failed
        the rule), else none."""
        return list(self._checks.warnings)

    @property
    def deflated_sharpe(self) -> float:
        """The deflated Sharpe ratio of the best combination for the number of
        combinations tried (Bailey and Lopez de Prado 2014): the probability, 0 to 1,
        that its true Sharpe ratio is above what the best of that many tries would show
        by luck alone; NaN with fewer than two combinations or undefined ratios."""
        return self._checks.deflated.deflated

    @property
    def overfitting_probability(self) -> float:
        """The probability of backtest overfitting by combinatorially symmetric
        cross-validation over the combinations' session returns (Bailey, Borwein, Lopez
        de Prado and Zhu 2017): the share of the splits of the sessions into two halves
        in which the combination best in sample ranks below the median out of sample (a
        split at the median counting half), 0 to 1; NaN with too few sessions or one
        combination. Every split cuts the same history, so a combination that held a
        lasting exposure to its trend ranks well in all of them: near 0 does not show
        the settings hold. Near 0.5 or above, the search did not pick settings that held
        up across the blocks."""
        return self._checks.overfitting.probability

    @property
    def overfitting(self) -> dict[str, float]:
        """The deflated Sharpe ratio and the probability of backtest overfitting with
        what they rest on: the best's per-session Sharpe ratio, the Sharpe ratio the best
        of the tries would show by luck, the variance of the combinations' Sharpe
        ratios, the combinations tried, the best's sessions, and the blocks, splits and
        splits counted of the cross-validation."""
        return self._checks.figures()

    def tearsheet(
        self, path: str | PathLike | None = None, *, x: str | None = None, y: str | None = None
    ) -> str:
        """The heatmap page: the objective over parameters `x` and `y` (the first two
        swept by default) as a heatmap and a surface with the best outlined, its
        neighbourhood, the deflated Sharpe ratio and the probability of backtest
        overfitting. Returned as a string, or written to `path` as UTF-8 and the path
        returned. Raises ConfigError for an unknown or repeated parameter."""
        from ._robust._optimizer_page import render, write

        return write(render(self, x, y), path)

    def __repr__(self) -> str:
        return (
            f"OptimizeResult({counted(len(self._table), 'combination')}, objective "
            f"{self._objective!r}, "
            f"path {self._path!r})"
        )


__all__ = ["OptimizeResult"]
