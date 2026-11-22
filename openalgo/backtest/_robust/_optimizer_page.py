"""The heatmap page of a `bt.optimize` sweep (design 17, section 5): one self-contained page
in the tearsheet's look (inline SVG, no fetch, light and dark, phone width, readable
without colour).

Sections: the Search (the objective over two parameters as a heatmap and as an isometric
surface, the best outlined, and the sweep in tiles); the Neighbourhood (the rule's
verdict on the best, the spike warning when it fails, and the histogram of the
neighbours' objectives); the Overfitting figures (the deflated Sharpe ratio and the
probability of backtest overfitting, with the histogram of the cross-validation's
logits); and the Notes (how to read them, and the plan without its timings, so two
sweeps of one grid make the same page).
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from ... import __version__
from ..errors import ConfigError
from ..report import _format as fmt
from ..report._dashboard.layout import SMALL, Section, card, stack
from ..report._dashboard.page import document
from ._blocks import LIMIT, REAL, REFERENCE, HistogramBlock, Mark, Tile, TilesBlock
from ._grid import Grid
from ._html import row_html
from ._scorecard import verdict_badge
from ._search_common import grid_block, level_of, params_text
from ._verdict import FAILING, PASSING

#: How to read the overfitting figures, for the Notes.
READING = (
    PASSING,
    FAILING,
    "The deflated Sharpe ratio is the probability that the best combination's true Sharpe "
    "ratio is above what the best of that many tries would show by luck alone, given the "
    "spread of the tries' Sharpe ratios, the best's sessions and the shape of its returns "
    "(Bailey and Lopez de Prado 2014). Near 1 the best stands out from luck; near 0.5 or "
    "below it does not.",
    "The probability of backtest overfitting cuts the sessions into blocks and tries every "
    "way of putting half of them in sample: it is the share of those splits in which the "
    "combination with the best Sharpe ratio in sample ranks below the median out of "
    "sample, a split at the median counting half (Bailey, Borwein, Lopez de Prado and Zhu "
    "2017). Near 0 the best in sample also ranked well in the other blocks of this one "
    "history, which a lasting exposure to its trend gives as well; it does not show the "
    "settings hold. Near 0.5 or above, the search did not pick settings that held up "
    "across the blocks.",
)


def _axes(result: Any, x: str | None, y: str | None) -> Grid:
    """The sweep's grid with `x` and `y` first. Raises ConfigError for an unknown or
    repeated parameter."""
    names = list(result._parameters)
    x = x or names[0]
    y = y or next((name for name in names if name != x), None)
    for name in (x, y):
        if name is not None and name not in names:
            raise ConfigError(f"{name!r} is not a swept parameter: {names}")
    if x == y:
        raise ConfigError("the heatmap takes two different parameters")
    order = [x, *([y] if y else []), *(n for n in names if n not in (x, y))]
    checks = result._checks
    return Grid(tuple(order), checks.grid.combos, checks.grid.run_params, True)


def _search(result: Any, grid: Grid) -> Section:
    checks, name = result._checks, result.objective
    values = result._table["objective"].to_numpy(dtype=float)
    level = level_of(name if result._named else None)
    marked = [(checks.best, "best")]
    heat = grid_block(
        "Objective", f"The {name} of each combination", grid, values, False, marked, level
    )
    surface = grid_block(
        "Objective as a surface", f"The {name} as a height", grid, values, True, marked, level
    )
    tiles = TilesBlock(
        "The sweep",
        "",
        (
            Tile("Combinations", fmt.indian_digits(len(values), 0), "tried"),
            Tile("Best", params_text(grid.swept(checks.best), "\n"), f"by {name}", wrap=True),
            Tile("Best objective", fmt.number(values[checks.best]), name),
            Tile("Path", result.path, "evaluation"),
        ),
    )
    body = stack([("", row_html((heat, surface))), (SMALL, row_html((tiles,)))])
    return Section("search", "Search", "The objective over two parameters", body)


def _neighbourhood(result: Any) -> Section:
    checks = result._checks
    found = checks.neighbourhood
    lines = [
        f"<p class='lead'>{verdict_badge(found.verdict)} &#183; {fmt.text(found.figure)}</p>",
        f"<p class='lead'>Rule: {fmt.text(found.rule)}</p>",
        f"<p class='key'>{fmt.text(found.note)}</p>",
        *(f"<p class='key'>Warning: {fmt.text(line)}</p>" for line in checks.warnings),
    ]
    parts = [("", card("Verdict", "", "".join(lines)))]
    near = found.table["neighbour"].to_numpy(dtype=bool) if "neighbour" in found.table else []
    values = found.table["objective"].to_numpy(dtype=float)[near] if len(near) else np.empty(0)
    if found.summary:
        chosen, keep = found.summary["chosen"], found.settings["keep"]
        level = level_of(result.objective if result._named else None)
        marks = (
            Mark(chosen, "the best", REAL),
            Mark(level + keep * (chosen - level), f"the median keeps {keep:.0%}", LIMIT),
        )
        histogram = HistogramBlock(
            "Neighbour objectives", "The objective of each neighbour", values, False, marks
        )
        parts.append((SMALL, row_html((histogram,))))
    return Section("neighbourhood", "Neighbourhood", "A plateau, or one lucky spike?", stack(parts))


def _overfitting(result: Any) -> Section:
    figures = result.overfitting
    tiles = TilesBlock(
        "Deflated Sharpe ratio and backtest overfitting",
        "Per-session Sharpe ratios",
        (
            Tile("Deflated Sharpe", fmt.number(figures["deflated_sharpe"], 3), "a probability"),
            Tile("Best Sharpe", fmt.number(figures["best_sharpe"], 4), "per session"),
            Tile("By luck", fmt.number(figures["expected_best_sharpe"], 4), "best of the tries"),
            Tile("Tries", fmt.indian_digits(figures["trials"], 0), "combinations"),
            Tile(
                "Overfitting",
                fmt.number(figures["overfitting_probability"], 3),
                "a probability",
            ),
            Tile(
                "Splits",
                fmt.indian_digits(figures["splits_counted"], 0),
                f"of {fmt.indian_digits(figures['blocks'], 0)} blocks",
            ),
        ),
    )
    logits = result._checks.overfitting.logits
    histogram = HistogramBlock(
        "Rank of the best out of sample",
        "The logit of each split: below zero, the best in sample ranked below the median out "
        "of sample; at zero, at the median (counted half)",
        logits,
        False,
        (Mark(0.0, "the median", REFERENCE),),
        "splits",
    )
    body = stack([("", row_html((tiles,))), (SMALL, row_html((histogram,)))])
    return Section("overfitting", "Overfitting", "Corrected for the combinations tried", body)


def _notes(result: Any) -> Section:
    reading = "".join(f"<p>{fmt.text(line)}</p>" for line in READING)
    plan = f"<p>{fmt.text(result._untimed_plan)}</p>"
    body = card("Reading the figures", "", f"<div class='prose'>{reading}{plan}</div>")
    return Section("notes", "Notes", "Reading the figures &amp; the plan", body)


def render(result: Any, x: str | None, y: str | None) -> str:
    """The whole page of `result` (an `OptimizeResult`) over parameters `x` and `y`."""
    grid = _axes(result, x, y)
    parts = [_search(result, grid), _neighbourhood(result), _overfitting(result), _notes(result)]
    title = f"{result._strategy}: optimization"
    heading = (
        f"<div class='title'><h1 class='serif'>{fmt.text(result._strategy)}</h1>"
        f"<span class='nums'>{fmt.indian_digits(len(result._table), 0)} combinations "
        f"&#183; objective {fmt.text(result.objective)}</span></div>"
    )
    return document(title, heading, parts, __version__, "")


def write(page: str, path: str | PathLike | None) -> str:
    """`page` returned, or written to `path` as UTF-8 and the path returned."""
    if path is None:
        return page
    target = Path(path)
    target.write_text(page, encoding="utf-8")
    return str(target)


__all__ = ["READING", "render", "write"]
