"""What a check's section draws, as data: fans, histograms, the sessions chart, candles,
lines over dates, groups of paths, a parameter grid (a heatmap or a surface), the fold
diagram, tables (with a small equity line a row) and tiles. A check returns rows of blocks
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
    are fractions shown as percentages; `counted` names what each value is, plural, for
    the key ("runs", or the "splits" of a cross-validation)."""

    title: str
    subtitle: str
    values: np.ndarray
    percent: bool
    marks: tuple[Mark, ...]
    counted: str = "runs"


@dataclass(frozen=True, eq=False)
class SessionsBlock:
    """A value at each session close, `values` (fractions shown as percentages when
    `percent`, else rupees, such as a close), with the sessions of `marked` shaded;
    `line` says in words what the line is, and `shaded` what the shaded sessions are,
    for the chart's key."""

    title: str
    subtitle: str
    dates: pd.DatetimeIndex
    values: np.ndarray
    marked: np.ndarray
    percent: bool
    line: str
    shaded: str = "this assignment holds out of sample"


@dataclass(frozen=True, eq=False)
class CandlesBlock:
    """Bars as candles (canonical layout: `ts_open`, open, high, low, close)."""

    title: str
    subtitle: str
    bars: pd.DataFrame


#: The styles of a line over dates: the run's own (bold), another run (thin), and a
#: buy-and-hold reference (dotted).
BOLD, THIN, DOTTED = "bold", "thin", "dotted"
#: The groups of lines a `LinesBlock` tells apart, each by its colour and the shape at
#: the end of its lines (`_charts_bars.SHAPES`); groups 0 to LINE_GROUPS - 1.
LINE_GROUPS = 4


@dataclass(frozen=True, eq=False)
class Line:
    """One line of a `LinesBlock`: its label, its dates and values (fractions), its
    style (`BOLD`, `THIN` or `DOTTED`) and its `group`, from 0 to `LINE_GROUPS` - 1: the
    lines of one group (one symbol's run and its buy and hold) share a colour and the
    shape at their end."""

    label: str
    dates: pd.DatetimeIndex
    values: np.ndarray
    style: str
    group: int = 0


@dataclass(frozen=True, eq=False)
class LinesBlock:
    """Returns over dates, one line each, on one time axis; `note` ends the key, saying
    what the lines measure."""

    title: str
    subtitle: str
    lines: tuple[Line, ...]
    note: str


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


@dataclass(frozen=True, eq=False)
class GridBlock:
    """An objective over two parameters: `x` and `y` name them, `columns` and `rows` are
    their values, `values` (rows, columns) the objective of each cell (NaN where none
    ran), measured from `level`, its break-even; `marked` the (column, row) of each cell
    marked, with its label ("chosen", "best"). `surface` draws it as an isometric
    surface of columns instead of a heatmap (`_charts_grid`)."""

    title: str
    subtitle: str
    x: str
    y: str
    columns: tuple[str, ...]
    rows: tuple[str, ...]
    values: np.ndarray
    level: float
    marked: tuple[tuple[int, int, str], ...]
    surface: bool = False


@dataclass(frozen=True)
class FoldSpan:
    """One fold of a walk-forward on the fold diagram: its label and the positions, in
    the run's sessions, of its in-sample and out-of-sample windows (both ends
    included)."""

    label: str
    in_first: int
    in_last: int
    out_first: int
    out_last: int


@dataclass(frozen=True, eq=False)
class FoldsBlock:
    """The fold diagram: a row per fold, its in-sample and out-of-sample windows on the
    time axis of the run's session `dates`."""

    title: str
    subtitle: str
    dates: pd.DatetimeIndex
    folds: tuple[FoldSpan, ...]


@dataclass(frozen=True, eq=False)
class SparkTableBlock:
    """A table of text with a small equity line in each row: its column names and rows
    (the first column a label, the others figures), the column the line goes in
    (`spark_at`, its name `spark_name`) and each row's line (`sparks`, rupees of P&L,
    an empty array for none)."""

    title: str
    subtitle: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    spark_at: int
    spark_name: str
    sparks: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class Tile:
    """One figure as a tile: label, value and a subline, as text; `tone` "pos", "neg"
    or "" colours the value; `wrap` lets a long value (parameters, one a line) take more
    lines instead of being cut short."""

    label: str
    value: str
    sub: str = ""
    tone: str = ""
    wrap: bool = False


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
    "LINE_GROUPS",
    "REAL",
    "REFERENCE",
    "THIN",
    "CandlesBlock",
    "FanBlock",
    "FoldSpan",
    "FoldsBlock",
    "GridBlock",
    "HistogramBlock",
    "Line",
    "LinesBlock",
    "Mark",
    "PathGroup",
    "PathsBlock",
    "SessionsBlock",
    "SparkTableBlock",
    "TableBlock",
    "Tile",
    "TilesBlock",
]
