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
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
    ) -> None:
        self._table = table
        self._combinations = combinations
        self._parameters = parameters
        self._objective = objective
        self._maximize = maximize
        self._path = path
        self._plan = plan
        self._runner = runner

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

    def _swept(self, position: int) -> dict[str, Any]:
        """The swept values of a combination, as they were given."""
        return {name: self._combinations[position][name] for name in self._parameters}

    @property
    def best(self) -> dict[str, Any]:
        """The parameters of the best combination. Raises BacktestError when every
        objective is NaN."""
        scores = self._table["objective"].reset_index(drop=True)
        if scores.notna().sum() == 0:
            raise BacktestError("no combination has a finite objective, so none is best")
        return self._swept(int(scores.idxmax() if self._maximize else scores.idxmin()))

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

    def __repr__(self) -> str:
        return (
            f"OptimizeResult({counted(len(self._table), 'combination')}, objective "
            f"{self._objective!r}, "
            f"path {self._path!r})"
        )


__all__ = ["OptimizeResult"]
