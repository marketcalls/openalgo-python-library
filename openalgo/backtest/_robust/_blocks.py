"""What a check's section draws, as data: fans, histograms, the sessions chart and
tiles. A check returns rows of blocks (`Outcome.blocks`), one or two blocks a row, and
the HTML report renders them (`_charts`, `_html`). Text fields are plain text; the
renderer escapes them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._paths import Fan

#: The kinds of a mark on a histogram: the run's own value (solid), a limit of the pass
#: rule (dashed) and a reference value such as zero (dotted).
REAL, LIMIT, REFERENCE = "real", "limit", "reference"


@dataclass(frozen=True)
class Mark:
    """A value marked on a histogram, with its label and kind (`REAL`, `LIMIT` or
    `REFERENCE`)."""

    value: float
    label: str
    kind: str


@dataclass(frozen=True, eq=False)
class FanBlock:
    """A fan of paths: percentile bands, sampled thin paths and the run's own path bold
    (`_paths.Fan`)."""

    title: str
    subtitle: str
    fan: Fan


@dataclass(frozen=True, eq=False)
class HistogramBlock:
    """A histogram of one figure over the runs, with `marks`; `percent` when the values
    are fractions shown as percentages."""

    title: str
    subtitle: str
    values: np.ndarray
    percent: bool
    marks: tuple[Mark, ...]


@dataclass(frozen=True, eq=False)
class SessionsBlock:
    """A value at each session close, `values` (fractions shown as percentages when
    `percent`, else rupees, such as a close), with the sessions of `marked` shaded;
    `line` says in words what the line is, for the chart's key."""

    title: str
    subtitle: str
    dates: pd.DatetimeIndex
    values: np.ndarray
    marked: np.ndarray
    percent: bool
    line: str


@dataclass(frozen=True)
class Tile:
    """One figure as a tile: label, value and a subline, as text; `tone` "pos", "neg"
    or "" colours the value."""

    label: str
    value: str
    sub: str = ""
    tone: str = ""


@dataclass(frozen=True)
class TilesBlock:
    """A titled grid of tiles."""

    title: str
    subtitle: str
    tiles: tuple[Tile, ...]


__all__ = [
    "LIMIT",
    "REAL",
    "REFERENCE",
    "FanBlock",
    "HistogramBlock",
    "Mark",
    "SessionsBlock",
    "Tile",
    "TilesBlock",
]
