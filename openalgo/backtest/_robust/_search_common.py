"""What the searching checks share (design 17, sections 2.1, 2.3, 2.16 and 2.17): the
reasons they leave a run not judged before any rerun, a parameter value as text, and the
grid of an objective over two parameters that their heatmaps and surfaces draw.

The grid drawn (`grid_block`): the first two parameters that vary, across and down, the
others held at the values of a reference combination (the chosen or the best setting),
so a grid of three parameters shows the slice through the reference; one parameter is
drawn as a single row.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ._bar_common import break_even
from ._blocks import GridBlock
from ._grid import Grid
from ._trade_common import no_edge, too_few
from ._variant_common import objective_unjudged
from ._verdict import Outcome


def search_unjudged(subject: Any, settings: Mapping[str, Any], rule: str) -> Outcome | None:
    """The "not judged" outcome of a run with too few closed trades or no edge (its
    `trades`, `_trade_common`), else None."""
    return too_few(subject.trades, settings["min_trades"], rule) or no_edge(subject.trades, rule)


def value_text(value: Any) -> str:
    """A parameter value as text: a whole number as it is, a fraction to its shortest
    exact form, anything else as its text."""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def params_text(values: Mapping[str, Any]) -> str:
    """Parameter values as "fast 10, slow 40"."""
    return ", ".join(f"{name} {value_text(value)}" for name, value in values.items())


def grid_block(
    title: str,
    subtitle: str,
    grid: Grid,
    values: Sequence[float],
    surface: bool = False,
    marked: Sequence[tuple[int, str]] | None = None,
    level: float | None = None,
) -> GridBlock:
    """The `GridBlock` of `values` (one a combination of `grid`) over its first two
    parameters (module docstring), the reference the first `marked` position (the
    chosen setting when None); `level` is the break-even of the colour scale (0 when
    None)."""
    if marked is None:
        chosen = grid.chosen()
        marked = [] if chosen is None else [(chosen, "chosen")]
    reference = grid.combos[marked[0][0]] if marked else grid.run_params
    x = grid.names[0]
    y = grid.names[1] if len(grid.names) > 1 else ""
    held = [name for name in grid.names[2:]]
    columns = grid.axis(x)
    rows = grid.axis(y) if y else ("",)
    matrix = np.full((len(rows), len(columns)), math.nan)
    places: dict[int, tuple[int, int]] = {}
    for position, combo in enumerate(grid.combos):
        if any(combo[name] != reference[name] for name in held):
            continue
        place = (columns.index(combo[x]), rows.index(combo[y]) if y else 0)
        matrix[place[1], place[0]] = values[position]
        places[position] = place
    marks = tuple((*places[p], label) for p, label in marked if p in places)
    return GridBlock(
        title,
        subtitle,
        x,
        y,
        tuple(value_text(v) for v in columns),
        tuple(value_text(v) for v in rows),
        matrix,
        0.0 if level is None else float(level),
        marks,
        surface,
    )


def level_of(objective: Any) -> float:
    """The objective's break-even value, 0 when it has none (for the colour scale)."""
    found = break_even(objective)
    return 0.0 if found is None else found


__all__ = [
    "grid_block",
    "level_of",
    "objective_unjudged",
    "params_text",
    "search_unjudged",
    "value_text",
]
