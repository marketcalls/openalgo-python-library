"""What a check's section draws, as data: fans, histograms, the sessions chart, candles,
lines over dates, groups of paths, tables and tiles. A check returns rows of blocks
(`Outcome.blocks`), one or two blocks a row, and the HTML report renders them (`_charts`,
`_charts_bars`, `_html`). Text fields are plain text; the renderer escapes them.
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
    (`_paths.Fan`). Its steps count trades booked, or with `dates` the run's sessions
    (step k the close of session k, `dates[k - 1]`, step 0 the start); `real` says what
    the bold path is."""

    title: str
    subtitle: str
    fan: Fan
    dates: pd.DatetimeIndex | None = None
    real: str = "the run's own trades in the order they closed"


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


@dataclass(frozen=True, eq=False)
class CandlesBlock:
    """Bars as candles (canonical layout: `ts_open`, open, high, low, close)."""

    title: str
    subtitle: str
    bars: pd.DataFrame


#: The styles of a line over dates: the run's own (bold), another run (thin), and a
#: buy-and-hold reference (dotted).
BOLD, THIN, DOTTED = "bold", "thin", "dotted"


@dataclass(frozen=True, eq=False)
class Line:
    """One line of a `LinesBlock`: its label, its dates and values (fractions), and its
    style (`BOLD`, `THIN` or `DOTTED`)."""

    label: str
    dates: pd.DatetimeIndex
    values: np.ndarray
    style: str


@dataclass(frozen=True, eq=False)
class LinesBlock:
    """Cumulative returns over dates, one line each, on one time axis."""

    title: str
    subtitle: str
    lines: tuple[Line, ...]


@dataclass(frozen=True, eq=False)
class PathGroup:
    """Paths drawn alike: a label and (paths, points) cumulative returns."""

    label: str
    paths: np.ndarray


@dataclass(frozen=True, eq=False)
class PathsBlock:
    """Groups of paths at the fan's steps of the run's sessions (`FanBlock`), each group in
    its own colour and dash, and the run's own path bold."""

    title: str
    subtitle: str
    steps: np.ndarray
    dates: pd.DatetimeIndex
    groups: tuple[PathGroup, ...]
    real: np.ndarray


@dataclass(frozen=True)
class TableBlock:
    """A table of text: its column names and rows; the first column is a label, the
    others are figures."""

    title: str
    subtitle: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


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
    "BOLD",
    "DOTTED",
    "LIMIT",
    "REAL",
    "REFERENCE",
    "THIN",
    "CandlesBlock",
    "FanBlock",
    "HistogramBlock",
    "Line",
    "LinesBlock",
    "Mark",
    "PathGroup",
    "PathsBlock",
    "SessionsBlock",
    "TableBlock",
    "Tile",
    "TilesBlock",
]
