"""The parameter combinations the searching checks run (design 17, sections 2.1, 2.3, 2.16
and 2.17): a grid of the parameter ranges given, or the neighbourhood of the run's own
parameters when none are.

The ranges: the keywords named after strategy parameters (`fast=range(5, 30, 5)`), else
the ranges the strategy declares with `bt.param(default, values)` for the parameters
`params=` does not fix, as `bt.optimize` reads them (`_optimize.grid`). The run's own
`params=` values are its chosen parameters; a parameter both given in `params=` and as a
range is chosen there and searched over the range. The grid's combinations are in grid
order (the first parameter slowest), and each parameter's values in the order its range
gives them, which is the order its steps are counted in.

The neighbourhood (`cube`), when no range is given: every combination of each number
parameter of the run at minus, zero and plus `step` of its value (a whole number moves
by at least one unit, rounding half up, so 14 at 10 percent gives 13, 14 and 15; a
fraction by `step` of its value, so a zero stays zero); a parameter that is not a number
(a flag, a name) stays as the run has it.

A neighbour of a combination (`Grid.neighbours`) is another combination of the grid at
most one step from it in every parameter: the 8 around it on a grid of two parameters,
the 26 of three.
"""

from __future__ import annotations

import itertools
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

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

    def steps(self, position: int) -> tuple[int, ...]:
        """The place of combination `position` on each parameter's axis."""
        combo = self.combos[position]
        return tuple(self.axis(name).index(combo[name]) for name in self.names)

    def neighbours(self, position: int) -> list[int]:
        """The positions of the neighbours of combination `position` (module
        docstring), in grid order."""
        places = [self.steps(k) for k in range(len(self.combos))]
        mine = places[position]
        return [
            k
            for k, place in enumerate(places)
            if k != position and all(abs(a - b) <= 1 for a, b in zip(place, mine, strict=True))
        ]


def run_parameters(strategy: type, given: Mapping[str, Any] | None) -> dict[str, Any]:
    """Every parameter of a run of `strategy` with `params=` `given`: its defaults with the
    given values over them."""
    return {**strategy.parameters(), **dict(given or {})}


def search_grid(
    strategy: type, ranges: Mapping[str, Any], given: Mapping[str, Any] | None
) -> Grid | None:
    """The grid of the ranges (module docstring) around a run with `params=` `given`;
    None when no range is given and the strategy declares none that `params=` leaves
    free. Raises ConfigError for an empty range."""
    fixed = {name: value for name, value in dict(given or {}).items() if name not in ranges}
    declared = [name for name in strategy.parameter_ranges() if name not in fixed]
    if not ranges and not declared:
        return None
    names, combos = sweep_grid(strategy, dict(ranges), fixed)
    return Grid(tuple(names), tuple(combos), run_parameters(strategy, given), True)


def _number(value: Any) -> bool:
    """Whether a parameter value is a number the neighbourhood moves (not a flag)."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def around(value: Any, step: float) -> tuple[Any, ...]:
    """The values of a parameter at minus, zero and plus `step` (module docstring)."""
    if isinstance(value, numbers.Integral):
        move = max(1, math.floor(step * abs(int(value)) + 0.5))
        return (int(value) - move, int(value), int(value) + move)
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
