"""The overfitting figures of a `bt.optimize` sweep (design 17, section 5), from the grid it
already ran, with no rerun:

    neighbourhood   the parameter-neighbourhood rule (`_neighbourhood`, as the check of
                    section 2.16 applies it) on the best combination's neighbours in the
                    grid (`_grid.Grid.neighbours`), as a `CheckResult`; failing it is the
                    spike warning (`warnings`)
    deflated        the deflated Sharpe ratio of the best for the number of combinations
                    tried (`_deflated_sharpe`)
    overfitting     the probability of backtest overfitting by combinatorially symmetric
                    cross-validation over the grid's session returns
                    (`_overfitting_probability`)

The rule reads an objective that is better higher with a break-even value (0, or 1 for
a profit factor); a sweep with `maximize=False`, an objective with no break-even or a
function of a `Result` leaves the neighbourhood not judged, saying why. The deflated
Sharpe ratio and the probability rest on the session returns alone, whatever the
objective.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import _words as words
from ._bar_common import BREAK_EVEN
from ._deflated_sharpe import DeflatedSharpe, deflated_sharpe
from ._grid import Grid
from ._neighbourhood import DEFAULT_KEEP, figure_words, plateau, rule_words
from ._overfitting_probability import Overfitting, overfitting
from ._search_common import params_text
from ._trade_common import BREAK_EVEN_RUPEES
from ._verdict import FAILED, NOT_JUDGED, PASSED, CheckResult

TITLE, QUESTION = "Parameter neighbourhood", "A plateau, or one lucky spike?"


@dataclass(frozen=True, eq=False)
class OptimizerChecks:
    """The figures of the module docstring for one sweep: its grid, the position of its
    best combination, and the spike warning when the neighbourhood rule fails."""

    grid: Grid
    best: int
    neighbourhood: CheckResult
    deflated: DeflatedSharpe
    overfitting: Overfitting
    warnings: tuple[str, ...]

    def figures(self) -> dict[str, float]:
        """The deflated Sharpe ratio, the probability of backtest overfitting and what
        they rest on, by name (per-session Sharpe ratios, fractions as fractions)."""
        found, cv = self.deflated, self.overfitting
        return {
            "deflated_sharpe": found.deflated,
            "best_sharpe": found.best_sharpe,
            "expected_best_sharpe": found.expected,
            "sharpe_variance": found.variance,
            "trials": float(found.trials),
            "sessions": float(found.sessions),
            "overfitting_probability": cv.probability,
            "blocks": float(cv.blocks),
            "splits": float(cv.splits),
            "splits_counted": float(len(cv.logits)),
        }


def _result(verdict: str, figure: str, rule: str, note: str, table, summary) -> CheckResult:
    settings = {"keep": DEFAULT_KEEP}
    return CheckResult(
        "parameters", TITLE, QUESTION, verdict, figure, rule, note, table, summary, settings
    )


def _neighbourhood(
    grid: Grid, table: pd.DataFrame, best: int, name: str, level: float | None, maximize: bool
) -> CheckResult:
    """The rule on the best combination's neighbours (module docstring), for the
    objective `name` with break-even `level` (None for none)."""
    rule = rule_words(name, 0.0 if level is None else level, DEFAULT_KEEP)
    near = grid.neighbours(best)
    frame = table.assign(
        best=[k == best for k in range(len(table))],
        neighbour=[k in near for k in range(len(table))],
    )
    values = table["objective"].to_numpy(dtype=float)
    why = None
    if not maximize:
        why = "The rule reads a figure that is better higher; this sweep minimised its objective."
    elif level is None:
        why = f"The objective {name} has no value that says a run made nothing."
    elif not values[best] > level:
        shown = words.number(values[best])
        why = f"No edge to judge: the best {name} is {shown}, not above {level:g}."
    elif not near:
        why = "The best combination has no neighbour: the grid holds one combination."
    found = None
    if why is None:
        profits = table["net_profit"].to_numpy(dtype=float)[near] >= BREAK_EVEN_RUPEES
        others = np.delete(values, best)
        found = plateau(values[best], values[near], profits, others, level, DEFAULT_KEEP)
        why = None if found is not None else "No neighbour's objective is defined."
    if found is None:
        return _result(NOT_JUDGED, "n/a", rule, why, frame, {})
    verdict = PASSED if found.passed else FAILED
    note = (
        f"The best combination ({params_text(grid.swept(best))}) has "
        f"{words.whole(len(near), 'neighbour')} in the grid, the combinations at most one "
        f"step from it in every parameter. Ratios are measured from {level:g}, a run that "
        "makes nothing."
    )
    return _result(verdict, figure_words(found, name), rule, note, frame, found.summary())


def optimizer_checks(
    table: pd.DataFrame,
    combinations: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    objective: tuple[str, bool],
    maximize: bool,
    returns: np.ndarray,
    best: int,
) -> OptimizerChecks:
    """The `OptimizerChecks` of a sweep: its `table` (an objective and a net_profit column
    a combination), its combinations and swept `names`, its objective as (its name,
    whether that is a metric's name rather than a function's), the direction it was
    ranked in, the session returns (sessions, combinations; `_sweep_returns.ReturnsMatrix`)
    and the position of its best."""
    grid = Grid(tuple(names), tuple(combinations), dict(combinations[best]), True)
    name, metric = objective
    level = BREAK_EVEN.get(name) if metric else None
    found = _neighbourhood(grid, table, best, name, level, maximize)
    matrix = np.asarray(returns, dtype=float)
    deflated = deflated_sharpe(matrix[:, best], [matrix[:, k] for k in range(matrix.shape[1])])
    warnings = ()
    if found.verdict == FAILED:
        warnings = (
            f"The best combination ({params_text(grid.swept(best))}) looks like a lone spike, "
            f"not a plateau: {found.figure}, where the rule is {found.rule}. The settings next "
            "to it did not hold up, so it may be fitted to the past or a stroke of luck; "
            "prefer a setting inside a plateau.",
        )
    return OptimizerChecks(grid, best, found, deflated, overfitting(matrix), warnings)


__all__ = ["OptimizerChecks", "optimizer_checks"]
