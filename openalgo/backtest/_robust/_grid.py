"""The parameter combinations the searching checks run (design 17, sections 2.1, 2.3, 2.16
and 2.17): a grid of the parameter ranges given, or the neighbourhood of the run's own
parameters when none are.

The ranges: the keywords named after strategy parameters (`fast=range(5, 30, 5)`), and
the ranges the strategy declares with `bt.param(default, values)` for the others. The
run's own `params=` values are its chosen parameters, the point the parameter
neighbourhood judges: a parameter with a range, given or declared, is chosen in
`params=` and searched over its range, so a trader who checks the best of a sweep by
passing it as `params=` keeps the declared ranges searched (design 17, section 1: the
`bt.param` ranges when no range is given); a parameter with no range keeps its `params=`
value in every combination. `bt.walk_forward`,
which has no chosen point, reads `params=` as `bt.optimize` does (`params_fix`): a
declared parameter it names is fixed, not searched. The grid's combinations are in grid
order (the first parameter slowest), and each parameter's values in the order its range
gives them, which is the order its steps are counted in.

The neighbourhood (`cube`), when no range is given: every combination of each number
parameter of the run at minus, zero and plus `step` of its value (a whole number moves
by at least one unit, rounding half up, so 14 at 10 percent gives 13, 14 and 15, and
never below 1 from 1 or more nor below 0 from 0, since such a parameter is most often a
count or a period that a strategy cannot take at 0 or below: 1 gives 1 and 2; a
fraction by `step` of its value, so it keeps its sign and a zero stays zero); a
parameter that is not a number (a flag, a name) stays as the run has it.

A neighbour of a combination (`Grid.neighbours`) is another combination of the grid at
most one step from it in every parameter: the 8 around it on a grid of two parameters,
the 26 of three. Each combination's place on every axis is found once per grid, so
finding neighbours takes time in proportion to the combinations.
"""

from __future__ import annotations

import itertools
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from .._optimize import _values
from .._optimize import grid as sweep_grid

#: The default step of the neighbourhood, a fraction of each value (design 17, 2.16).
DEFAULT_STEP = 0.10


@dataclass(frozen=True, eq=False)
class Grid:
    """The combinations of a search (module docstring): `names` the parameters that vary,
    in the strategy's order; `combos` every combination as the `params=` of a run, in
    grid order; `run_params` the run's own parameters, every one; `given` True for a grid
    of ranges, False for the run's neighbourhood."""

    names: tuple[str, ...]
    combos: tuple[Mapping[str, Any], ...]
    run_params: Mapping[str, Any]
    given: bool

    def axis(self, name: str) -> tuple[Any, ...]:
        """The values of parameter `name`, in the order of its range."""
        return tuple(dict.fromkeys(combo[name] for combo in self.combos))

    def changes(self, position: int) -> tuple[tuple[str, Any], ...]:
        """(name, value) of the parameters of combination `position` that differ from
        the run's: what a rerun of it changes (`_reruns.Family.params`)."""
        combo = self.combos[position]
        return tuple((n, combo[n]) for n in self.names if combo[n] != self.run_params.get(n))

    def swept(self, position: int) -> dict[str, Any]:
        """The values of the varying parameters of combination `position`."""
        return {name: self.combos[position][name] for name in self.names}

    def chosen(self) -> int | None:
        """The position of the run's own parameters in the grid, None when they are not a
        combination of it."""
        wanted = {name: self.run_params.get(name) for name in self.names}
        found = [k for k in range(len(self.combos)) if self.swept(k) == wanted]
        return found[0] if found else None

    @cached_property
    def places(self) -> np.ndarray:
        """(combinations, parameters): the place of each combination on each parameter's
        axis, counted from 0."""
        columns = []
        for name in self.names:
            where = {value: k for k, value in enumerate(self.axis(name))}
            columns.append([where[combo[name]] for combo in self.combos])
        return np.array(columns, dtype=np.int64).T.reshape(len(self.combos), len(self.names))

    def steps(self, position: int) -> tuple[int, ...]:
        """The place of combination `position` on each parameter's axis."""
        return tuple(int(place) for place in self.places[position])

    def neighbours(self, position: int) -> list[int]:
        """The positions of the neighbours of combination `position` (module
        docstring), in grid order."""
        places = self.places
        near = np.all(np.abs(places - places[position]) <= 1, axis=1)
        near[position] = False
        return [int(k) for k in np.flatnonzero(near)]


def run_parameters(strategy: type, given: Mapping[str, Any] | None) -> dict[str, Any]:
    """Every parameter of a run of `strategy` with `params=` `given`: its defaults with the
    given values over them."""
    return {**strategy.parameters(), **dict(given or {})}


def search_grid(
    strategy: type,
    ranges: Mapping[str, Any],
    given: Mapping[str, Any] | None,
    *,
    params_fix: bool = False,
) -> Grid | None:
    """The grid of the ranges (module docstring) around a run with `params=` `given`;
    None when no range is given or declared. With `params_fix` a declared parameter that
    `given` names is fixed, as `bt.optimize` fixes it, and is not a range. Raises
    ConfigError for an empty range or a name in `given` the strategy does not have."""
    given = dict(given or {})
    declared = [name for name in strategy.parameter_ranges() if not (params_fix and name in given)]
    searched = set(ranges) | set(declared)
    if not searched:
        return None
    fixed = {name: value for name, value in given.items() if name not in searched}
    names, combos = sweep_grid(strategy, dict(ranges), fixed)
    return Grid(tuple(names), tuple(combos), run_parameters(strategy, given), True)


def _number(value: Any) -> bool:
    """Whether a parameter value is a number the neighbourhood moves (not a flag)."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def around(value: Any, step: float) -> tuple[Any, ...]:
    """The values of a parameter at minus, zero and plus `step` (module docstring)."""
    if isinstance(value, numbers.Integral):
        whole = int(value)
        move = max(1, math.floor(step * abs(whole) + 0.5))
        floor = 1 if whole >= 1 else 0 if whole == 0 else None
        low = whole - move if floor is None else max(whole - move, floor)
        return tuple(dict.fromkeys((low, whole, whole + move)))
    move = step * abs(float(value))
    return (float(value) - move, float(value), float(value) + move) if move else (value,)


def cube(strategy: type, given: Mapping[str, Any] | None, step: float) -> Grid | None:
    """The neighbourhood of the run's own parameters (module docstring); None when the
    strategy has no number parameter to move."""
    run_params = run_parameters(strategy, given)
    names = [name for name, value in run_params.items() if _number(value)]
    if not names:
        return None
    axes = [around(run_params[name], step) for name in names]
    combos = tuple(
        {**run_params, **dict(zip(names, values, strict=True))}
        for values in itertools.product(*axes)
    )
    return Grid(tuple(names), combos, run_params, False)


def checked_ranges(ranges: Mapping[str, Any]) -> dict[str, tuple[Any, ...]]:
    """The ranges as tuples of values. Raises ConfigError for an empty one."""
    return {name: _values(name, values) for name, values in ranges.items()}


__all__ = [
    "DEFAULT_STEP",
    "Grid",
    "around",
    "checked_ranges",
    "cube",
    "run_parameters",
    "search_grid",
]
