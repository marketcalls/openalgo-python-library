"""The walk-forward's charts as inline SVG and HTML, in the style of `_charts`:

    folds     the fold diagram: a row per fold on the time axis of the run's sessions,
              its in-sample window hollow (outlined, lightly filled) and its
              out-of-sample window solid, so it reads without colour; each window
              carries a `<title>` with its dates
    sparks    a table of text with a small equity line in one column: a fold's
              out-of-sample P&L over its sessions, from zero, with the zero line dotted

Each mark carries a `<title>`, the browser's hover readout.
"""

from __future__ import annotations

import numpy as np

from ..report import _format as fmt
from ..report._dashboard.bars import AXIS_TEXT, MUTED_TEXT, Margin, frame
from ..report._dashboard.tables import head
from ._blocks import FoldsBlock, SparkTableBlock
from ._charts import WIDTH, points

#: The height of one fold's row and the bar inside it, pixels.
ROW_PX, BAR_PX = 26, 16
#: The diagram's margin: room for the fold labels on the left and the dates below.
FOLDS_MARGIN = Margin(10, 16, 34, 58)
#: A sparkline's size, pixels, and the padding inside it.
SPARK_W, SPARK_H, SPARK_PAD = 120, 28, 3


def folds_chart(block: FoldsBlock) -> str:
    """The fold diagram of `block`, with its key."""
    count = max(len(block.dates), 1)
    height = FOLDS_MARGIN.top + FOLDS_MARGIN.bottom + ROW_PX * len(block.folds)
    width = WIDTH - FOLDS_MARGIN.left - FOLDS_MARGIN.right
    step = width / count
    marks = []
    for k, fold in enumerate(block.folds):
        top = k * ROW_PX + (ROW_PX - BAR_PX) / 2
        marks.append(
            f"<text x='-8' y='{top + BAR_PX / 2 + 3:.1f}' text-anchor='end' "
            f"style='{MUTED_TEXT}'>{fmt.text(fold.label)}</text>"
        )
        windows = (
            ("fold-in", fold.in_first, fold.in_last, "in sample"),
            ("fold-out", fold.out_first, fold.out_last, "out of sample"),
        )
        for css, first, last, what in windows:
            tip = f"{fmt.text(fold.label)}, {what}: {_day(block, first)} to {_day(block, last)}"
            marks.append(
                f"<rect class='{css}' x='{first * step:.1f}' y='{top:.1f}' "
                f"width='{max(1.0, (last - first + 1) * step):.1f}' height='{BAR_PX}'>"
                f"<title>{tip}</title></rect>"
            )
    bottom = ROW_PX * len(block.folds)
    if len(block.dates):
        marks.append(
            f"<text x='0' y='{bottom + 16}' style='{AXIS_TEXT}'>{_day(block, 0)}</text>"
            f"<text x='{width:.1f}' y='{bottom + 16}' text-anchor='end' style='{AXIS_TEXT}'>"
            f"{_day(block, count - 1)}</text>"
        )
    svg = frame(WIDTH, height, FOLDS_MARGIN, fmt.text(block.title), "".join(marks))
    key = (
        "Hollow: the in-sample window where a fold chose its parameters; solid: the "
        "out-of-sample window that traded them"
    )
    return f"{svg}<p class='key'>{key}.</p>"


def _day(block: FoldsBlock, position: int) -> str:
    return f"{block.dates[position]:%Y-%m-%d}"


def sparkline(values: np.ndarray) -> str:
    """A small line of `values` (rupees of P&L, the first at the start) from zero, with
    the zero line dotted; empty for fewer than two points."""
    known = np.asarray(values, dtype=float)
    if len(known) < 2 or not np.isfinite(known).all():
        return ""
    low, high = min(0.0, float(known.min())), max(0.0, float(known.max()))
    span = high - low or 1.0
    xs = np.linspace(SPARK_PAD, SPARK_W - SPARK_PAD, len(known))

    def at(value: float) -> float:
        return SPARK_H - SPARK_PAD - (value - low) / span * (SPARK_H - 2 * SPARK_PAD)

    line = points(xs, [at(v) for v in known], 1)
    return (
        f"<svg class='bars spark' width='{SPARK_W}' height='{SPARK_H}' "
        f"viewBox='0 0 {SPARK_W} {SPARK_H}' role='img' aria-label='equity'>"
        f"<line x1='0' x2='{SPARK_W}' y1='{at(0.0):.1f}' y2='{at(0.0):.1f}' class='zero'/>"
        f"<polyline class='spark-line' points='{line}'/></svg>"
    )


def spark_table(block: SparkTableBlock) -> str:
    """The table of `block` with each row's equity line in its column."""
    names = [*block.columns[: block.spark_at], block.spark_name, *block.columns[block.spark_at :]]
    columns = [(fmt.text(name), "l" if k == 0 else "r") for k, name in enumerate(names)]
    rows = []
    for cells, spark in zip(block.rows, block.sparks, strict=True):
        shown = [
            f"<td class='{'l' if k == 0 else 'r nums'}'>{fmt.text(cell)}</td>"
            for k, cell in enumerate(cells)
        ]
        shown.insert(block.spark_at, f"<td class='r'>{sparkline(spark)}</td>")
        rows.append(f"<tr>{''.join(shown)}</tr>")
    return (
        f"<div class='scroll'><table class='tbl hover'>{head(*columns)}"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


__all__ = ["folds_chart", "spark_table", "sparkline"]
