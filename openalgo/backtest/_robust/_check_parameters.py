"""Parameter neighbourhood (design 17, section 2.16): a plateau, or one lucky spike?

The strategy reruns on the run's own bars at the combinations around its chosen
parameters (the run's): with parameter ranges given (or declared with `bt.param`), every
combination of their grid; without, every combination of each number parameter at
minus, zero and plus `step` (10 percent, at least a unit for a whole number, `_grid.cube`).
The chosen parameters' neighbours are the combinations at most one step from them in
every parameter. Figures and rule: `_neighbourhood` (the share of neighbours profitable,
the median and 25th percentile neighbour over the chosen objective, the chosen one's
percentile among every other combination run). Pass: the 25th percentile neighbour's
objective above break-even and the median keeping at least `keep` (half) of the chosen.

Not judged when the run has too few closed trades or did not make money, when its
objective is undefined, has no break-even or is not above it, when the strategy has no
number parameter to move and no range is given, when the run's parameters are not a
combination of the ranges searched (a `params=` value outside the range given or
declared: the note names that range and the keyword form of a range around the value, or,
for a value off a value of its range by a float residue only, that value to pass), or
when no neighbour's objective is defined.

Charts: the heatmap of the first two parameters that vary (the others held at the chosen
values) with the chosen cell outlined; the same grid as an isometric surface; the
histogram of the neighbours' objectives with the chosen one, the keep line and break-even
marked; the equity of an equal-weight blend of the chosen setting and its neighbours,
"several settings from one plateau run together", against the run's own.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from . import _words as words
from ._bar_common import RerunSet, break_even, joined, real_path, rerun_set
from ._blocks import LIMIT, REAL, REFERENCE, HistogramBlock, Mark, PathGroup, PathsBlock
from ._grid import DEFAULT_STEP, Grid, around, cube, is_number
from ._neighbourhood import DEFAULT_KEEP, figure_words, plateau, rule_words
from ._part_objective import objective_name
from ._reruns import Family
from ._search_common import (
    grid_block,
    objective_unjudged,
    search_unjudged,
    value_text,
)
from ._search_setup import SearchRun
from ._setting import Setting, between
from ._trade_common import ENOUGH_TRADES
from ._verdict import FAILED, PASSED, Outcome, not_judged

#: The most values of a range a note lists one by one; a longer range reads as its ends.
LISTED_VALUES = 6
#: Significant digits a neighbour in a note's keyword range is written to: enough to keep
#: it apart from the run's value, not so many that a float residue shows (0.27, not
#: 0.26999999999999996).
NEIGHBOUR_DIGITS = 12

SETTINGS = (
    ENOUGH_TRADES,
    Setting(
        "step",
        DEFAULT_STEP,
        between(0.0, 1.0, low_open=True, high_open=True),
        "a fraction above 0 and below 1",
        "how far each number parameter moves when no range is given, a fraction of its value",
    ),
    Setting(
        "keep",
        DEFAULT_KEEP,
        between(0.0, 1.0, low_open=True, high_open=False),
        "a fraction above 0 and at most 1",
        "the share of the chosen objective the median neighbour must keep",
    ),
)


def rule(settings: Mapping[str, Any], objective: Any) -> str:
    """The pass rule, in words."""
    level = break_even(objective)
    return rule_words(objective_name(objective), 0.0 if level is None else level, settings["keep"])


def searched(subject: SearchRun, settings: Mapping[str, Any]) -> Grid | None:
    """The grid this check runs: the ranges', else the run's neighbourhood."""
    if subject.grid is not None:
        return subject.grid
    return cube(subject.strategy, subject.given, settings["step"])


def unjudged(subject: SearchRun, settings: Mapping[str, Any], objective: Any) -> Outcome | None:
    """The "not judged" outcome of the module docstring that needs no rerun, else None:
    the one source of whether the check reruns (`reruns`)."""
    text = rule(settings, objective)
    short = search_unjudged(subject, settings, text) or objective_unjudged(
        subject, objective, text, "neighbouring"
    )
    if short is not None:
        return short
    grid = searched(subject, settings)
    if grid is None:
        return not_judged(
            text,
            "No parameter to vary: the strategy has no number parameter and no range was "
            "given; give ranges named after its parameters (fast=range(5, 30, 5)).",
        )
    chosen = grid.chosen()
    if chosen is None:
        return not_judged(text, _off_the_grid(grid, settings["step"]))
    if not grid.neighbours(chosen):
        return not_judged(text, "The chosen parameters have no neighbour in the grid.")
    return None


def _off_the_grid(grid: Grid, step: float) -> str:
    """Why the run's parameters are not judged when they are not a combination of the
    ranges searched: each value its range does not hold and that range, then the keyword
    form of a range around each such value (the run's value and `step` either side,
    `_grid.around`) that would let the check judge it, or, for a value off a value of its
    range by a float residue only, that value to pass in `params=` instead. The grid holds
    every combination of its ranges' values, so a combination it lacks has at least one
    value off its range. Values are written as Python writes them (`_exact`), so the
    advice holds the very value the run has."""
    shown = ", ".join(f"{name} {_exact(grid.run_params[name])}" for name in grid.names)
    lead = f"The run's parameters ({shown}) are not a combination of the ranges searched"
    off = [name for name in grid.names if grid.run_params[name] not in grid.axis(name)]
    near = {name: _residue_of(grid.run_params[name], grid.axis(name)) for name in off}
    where = "; ".join(
        _off_words(name, grid.run_params[name], grid.axis(name), near[name]) for name in off
    )
    apart = [name for name in off if near[name] is None]
    advice = []
    if apart:
        examples = ", ".join(
            f"{name}={_keyword_range(grid.run_params[name], step)}" for name in apart
        )
        whom = "it" if len(apart) == 1 else "each"
        advice.append(
            f"Give {whom} a range that holds it as a keyword named after the parameter, for "
            f"example {examples}."
        )
    close = ", ".join(f"{name!r}: {_exact(near[name])}" for name in off if near[name] is not None)
    if close:
        advice.append(f"Pass the range's value in params=, for example params={{{close}}}.")
    return f"{lead}: {where}. {' '.join(advice)}"


def _off_words(name: str, value: Any, values: tuple[Any, ...], near: Any) -> str:
    """How parameter `name`'s run value `value` stands off its range `values`; `near` is
    the value of the range it is off by a float residue only, else None."""
    if near is None:
        return (
            f"{name} {_exact(value)} is not among the values of its range ({_range_text(values)})"
        )
    return (
        f"{name} {_exact(value)} differs from {_exact(near)}, a value of its range "
        f"({_range_text(values)}), by a float residue only"
    )


def _residue_of(value: Any, values: tuple[Any, ...]) -> Any:
    """The value of a range `value` is off by a float residue only (`math.isclose` at its
    default tolerance, a part in a thousand million), else None. For example 0.3 in
    `params=` against a range made by np.arange(0.1, 0.7, 0.1), whose third value is
    0.30000000000000004."""
    if not is_number(value):
        return None
    return next(
        (each for each in values if is_number(each) and math.isclose(float(value), float(each))),
        None,
    )


def _exact(value: Any) -> str:
    """A parameter value as Python writes it: a fraction to every digit it needs to read
    back as itself (0.1 + 0.2 is 0.30000000000000004, 2.3456789 stays whole), anything
    else as `value_text` writes it."""
    return repr(float(value)) if isinstance(value, float) else value_text(value)


def _range_text(values: tuple[Any, ...]) -> str:
    """A range's values in a note: each of them up to `LISTED_VALUES`, else its first and
    last and how many."""
    if len(values) <= LISTED_VALUES:
        return ", ".join(_exact(value) for value in values)
    return f"{_exact(values[0])} to {_exact(values[-1])}, {len(values)} values"


def _keyword_range(value: Any, step: float) -> str:
    """A range around `value` as a keyword's list: the value (as `_exact` writes it) and
    `step` either side (to `NEIGHBOUR_DIGITS`) for a number, the value alone (as Python
    writes it) otherwise."""
    if not is_number(value):
        return f"[{value!r}]"
    texts = [_exact(each) if each == value else _neighbour(each) for each in around(value, step)]
    return "[" + ", ".join(texts) + "]"


def _neighbour(value: Any) -> str:
    """A neighbour in a keyword range (`NEIGHBOUR_DIGITS`)."""
    return f"{value:.{NEIGHBOUR_DIGITS}g}" if isinstance(value, float) else value_text(value)


def _families(subject: SearchRun, grid: Grid) -> list[Family]:
    return [subject.family(grid, position) for position in range(len(grid.combos))]


def reruns(
    subject: SearchRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> list[tuple[Family, int]]:
    """(family, runs) of what the check reruns: one run of each combination, none when
    it is not judged before any rerun (`unjudged`)."""
    if unjudged(subject, settings, objective) is not None:
        return []
    return [(family, 1) for family in _families(subject, searched(subject, settings))]


def run(
    subject: SearchRun, settings: Mapping[str, Any], *, runs: int, seed: int, objective: Any
) -> Outcome:
    """The check of the module docstring."""
    text = rule(settings, objective)
    short = unjudged(subject, settings, objective)
    if short is not None:
        return short
    grid = searched(subject, settings)
    set_ = joined([rerun_set([row]) for row in subject.reruns.firsts(_families(subject, grid))])
    chosen = grid.chosen()
    near = grid.neighbours(chosen)
    others = [k for k in range(len(grid.combos)) if k != chosen]
    level, name = break_even(objective), objective_name(objective)
    found = plateau(
        set_.objective[chosen],
        set_.objective[near],
        set_.profitable[near],
        set_.objective[others],
        level,
        settings["keep"],
    )
    if found is None:
        return not_judged(text, "No neighbour's objective is defined." + set_.failure_note())
    where = "the grid of the ranges" if grid.given else _cube_words(grid, settings["step"])
    note = (
        f"Each combination of {where} reran on the run's own bars; the chosen parameters "
        f"have {words.whole(len(near), 'neighbour')}, the combinations at most one step from "
        f"them in every parameter, of {words.whole(len(grid.combos), 'combination')} run. "
        f"Ratios are measured from {level:g}, a run that makes nothing." + set_.failure_note()
    )
    table = _table(grid, set_, chosen, near)
    blocks = _blocks(subject, grid, set_, chosen, near, found, name)
    verdict = PASSED if found.passed else FAILED
    return Outcome(verdict, figure_words(found, name), text, note, table, found.summary(), blocks)


def _cube_words(grid: Grid, step: float) -> str:
    moved = ", ".join(grid.names)
    return f"{moved} at minus, zero and plus {words.percent(step, 0)} of the run's values"


def _table(grid: Grid, set_: RerunSet, chosen: int, near: list[int]) -> pd.DataFrame:
    """A row per combination: its parameters, its figures, and whether it is the chosen
    one or a neighbour."""
    frame = set_.table()
    for position, name in enumerate(grid.names):
        frame.insert(position, name, [combo[name] for combo in grid.combos])
    frame["chosen"] = [k == chosen for k in range(len(grid.combos))]
    frame["neighbour"] = [k in near for k in range(len(grid.combos))]
    frame.index.name = "combination"
    return frame


def _blocks(
    subject: SearchRun,
    grid: Grid,
    set_: RerunSet,
    chosen: int,
    near: list[int],
    found: Any,
    name: str,
) -> tuple:
    """The heatmap beside the surface, the histogram beside the blend."""
    level = found.level
    heat = grid_block(
        "Objective by parameter", f"The {name} of each setting", grid, set_.objective, level=level
    )
    surface = grid_block(
        "The same grid as a surface",
        f"The {name} as a height",
        grid,
        set_.objective,
        True,
        level=level,
    )
    keep_line = found.level + found.keep * (found.chosen - found.level)
    marks = (
        Mark(found.chosen, "the chosen setting", REAL),
        Mark(keep_line, f"the median keeps {words.percent(found.keep, 0)}", LIMIT),
        Mark(found.level, "break-even", REFERENCE),
    )
    values = set_.objective[near]
    histogram = HistogramBlock(
        "Neighbour objectives",
        f"The {name} of each neighbour",
        values[np.isfinite(values)],
        False,
        marks,
    )
    return ((heat, surface), (histogram, _blend(subject, set_, [chosen, *near])))


def _blend(subject: SearchRun, set_: RerunSet, plateau_positions: list[int]) -> PathsBlock:
    """The equal-weight blend of the chosen setting and its neighbours that ran: the mean
    of their cumulative returns (each a share of the capital), against the run's own."""
    setup = subject.reruns.setup
    ran = [k for k in plateau_positions if set_.ran[k]]
    blend = set_.paths[ran].mean(axis=0)[None, :]
    label = f"an equal-weight blend of the chosen setting and its {len(ran) - 1} neighbours"
    return PathsBlock(
        "Several settings run together",
        "The plateau's settings, each on an equal share of the capital",
        setup.steps,
        pd.DatetimeIndex(setup.dates),
        (PathGroup(label, blend),),
        real_path(subject.trades, setup.steps),
    )


__all__ = ["SETTINGS", "reruns", "rule", "run", "searched", "unjudged"]
