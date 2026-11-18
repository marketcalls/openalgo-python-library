"""The charts of the bar checks as inline SVG, in the style of `_charts`:

    candles    bars as candles: a rising bar filled, a falling one hollow (so it reads
               without colour), its wick from low to high; x is the bar, y the price
    lines      returns over dates on one time axis: the run's own bold, another run
               thin, a buy and hold dotted; the lines of one group (a symbol) share a
               colour and end in the group's shape, so they read without colour, and the
               key names each line's style and shape
    paths      groups of paths at the run's sessions, each group in its own colour and
               dash, the run's own bold
    table      a table of text, its first column a label

Each mark carries a `<title>`, the browser's hover readout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..report import _format as fmt
from ..report._dashboard import scale
from ..report._dashboard.bars import AXIS_TEXT, frame
from ..report._dashboard.tables import head
from ._blocks import BOLD, DOTTED, CandlesBlock, Line, LinesBlock, PathsBlock, TableBlock
from ._charts import HEIGHT, MARGIN, WIDTH, domain, inner, points, value_axis

#: How the key names a line's style.
LINE_WORDS = {BOLD: "bold", DOTTED: "dotted"}
#: The groups of paths a chart tells apart by colour and dash, in order.
GROUP_CLASSES = 4
GROUP_DASHES = ("solid", "long dashes", "dots", "dash and dot")
#: The shape at the end of each line of a group of lines, in order, and its radius in
#: pixels.
SHAPES = ("circle", "square", "triangle", "diamond")
SHAPE_SIZE = 4.0
#: The width of a candle's body, as a share of the space of one bar.
BODY_SHARE = 0.6


def _date_ends(x: scale.Linear, first: pd.Timestamp, last: pd.Timestamp, at: tuple) -> str:
    _, height = inner()
    return "".join(
        f"<text x='{x(position):.1f}' y='{height + 16}' text-anchor='{anchor}' "
        f"style='{AXIS_TEXT}'>{day:%Y-%m-%d}</text>"
        for position, anchor, day in ((at[0], "start", first), (at[1], "end", last))
    )


def candles_chart(block: CandlesBlock) -> str:
    """The bars of `block` as candles, with their key."""
    bars = block.bars
    count = len(bars)
    if count < 2:
        return "<p class='empty'>No chart: fewer than two bars.</p>"
    width, height = inner()
    o, h, low, c = (bars[name].to_numpy(dtype=float) for name in ("open", "high", "low", "close"))
    bottom, top = domain(np.concatenate((low, h)), with_zero=False)
    x, y = scale.Linear(-0.5, count - 0.5, 0.0, width), scale.Linear(bottom, top, height, 0.0)
    body = max(1.0, BODY_SHARE * width / count)
    marks = [value_axis(y, bottom, top, False, width)]
    times = pd.to_datetime(bars["ts_open"].to_numpy(), utc=True).tz_convert("Asia/Kolkata")
    for k in range(count):
        css = "candle-up" if c[k] >= o[k] else "candle-down"
        upper, lower = y(max(o[k], c[k])), y(min(o[k], c[k]))
        tip = (
            f"{times[k]:%Y-%m-%d %H:%M}: open {fmt.number(o[k])}, high {fmt.number(h[k])}, "
            f"low {fmt.number(low[k])}, close {fmt.number(c[k])}"
        )
        marks.append(
            f"<g><title>{tip}</title><line class='wick' x1='{x(k):.1f}' x2='{x(k):.1f}' "
            f"y1='{y(h[k]):.1f}' y2='{y(low[k]):.1f}'/><rect class='{css}' "
            f"x='{x(k) - body / 2:.1f}' y='{upper:.1f}' width='{body:.1f}' "
            f"height='{max(0.8, lower - upper):.1f}'/></g>"
        )
    marks.append(_date_ends(x, times[0], times[-1], (0, count - 1)))
    svg = frame(WIDTH, HEIGHT, MARGIN, fmt.text(block.title), "".join(marks))
    key = "Filled: a bar that closed at or above its open; hollow: one that closed below it"
    return f"{svg}<p class='key'>{key}. Wicks run from the low to the high.</p>"


def _line_class(line: Line) -> str:
    """The class of a line: bold for the run, else its group's colour, dotted for a buy
    and hold."""
    if line.style == BOLD:
        return "line-bold"
    kind = "hold" if line.style == DOTTED else "series"
    return f"{kind}-{line.group % len(SHAPES)}"


def _shape(group: int, cx: float, cy: float) -> str:
    """The shape of `group` centred on (cx, cy), filled in the group's colour."""
    r, css = SHAPE_SIZE, f"end-{group % len(SHAPES)}"
    name = SHAPES[group % len(SHAPES)]
    if name == "circle":
        return f"<circle class='{css}' cx='{cx:.1f}' cy='{cy:.1f}' r='{r * 0.85:.1f}'/>"
    if name == "square":
        side = 1.5 * r
        return (
            f"<rect class='{css}' x='{cx - side / 2:.1f}' y='{cy - side / 2:.1f}' "
            f"width='{side:.1f}' height='{side:.1f}'/>"
        )
    if name == "triangle":
        corners = ((cx, cy - r), (cx + r, cy + 0.8 * r), (cx - r, cy + 0.8 * r))
    else:
        corners = ((cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy))
    path = " L ".join(f"{px:.1f} {py:.1f}" for px, py in corners)
    return f"<path class='{css}' d='M {path} Z'/>"


def lines_chart(block: LinesBlock) -> str:
    """The lines of `block` on one time axis, each ending in its group's shape, with
    their key."""
    lines = [line for line in block.lines if len(line.values)]
    if not lines:
        return "<p class='empty'>No chart: no line has a point.</p>"
    width, height = inner()
    days = [pd.DatetimeIndex(line.dates).tz_localize(None) for line in lines]
    first, last = min(d[0] for d in days), max(d[-1] for d in days)
    span = max((last - first).value, 1)
    x = scale.Linear(0.0, float(span), 0.0, width)
    low, high = domain(np.concatenate([line.values for line in lines]))
    y = scale.Linear(low, high, height, 0.0)
    marks = [value_axis(y, low, high, True, width)]
    words = []
    for line, dates in zip(lines, days, strict=True):
        xs = np.array([x(float((day - first).value)) for day in dates])
        ys = [y(v) for v in line.values]
        marks.append(
            f"<g><title>{fmt.text(line.label)}</title><polyline class='{_line_class(line)}' "
            f"points='{points(xs, ys, 1)}'/>{_shape(line.group, xs[-1], ys[-1])}</g>"
        )
        style = LINE_WORDS.get(line.style, "thin")
        shape = SHAPES[line.group % len(SHAPES)]
        words.append(f"{fmt.text(line.label)}: {style}, ending in a {shape}")
    marks.append(_date_ends(x, first, last, (0.0, float(span))))
    svg = frame(WIDTH, HEIGHT, MARGIN, fmt.text(block.title), "".join(marks))
    return f"{svg}<p class='key'>{'; '.join(words)}. {fmt.text(block.note)}</p>"


def paths_chart(block: PathsBlock) -> str:
    """The groups of paths of `block` and the run's own path bold, with their key."""
    width, height = inner()
    pool = np.concatenate([block.real, *(group.paths.ravel() for group in block.groups)])
    low, high = domain(pool)
    last = float(block.steps[-1]) if len(block.steps) else 1.0
    x, y = scale.Linear(0.0, max(last, 1.0), 0.0, width), scale.Linear(low, high, height, 0.0)
    xs = np.array([x(float(step)) for step in block.steps])
    marks = [value_axis(y, low, high, True, width)]
    words = []
    for k, group in enumerate(block.groups):
        css = f"group-{k % GROUP_CLASSES}"
        for path in group.paths:
            marks.append(
                f"<polyline class='{css}' points='{points(xs, [y(v) for v in path], 0)}'/>"
            )
        words.append(f"{GROUP_DASHES[k % GROUP_CLASSES]}: {fmt.text(group.label)}")
    real = points(xs, [y(v) for v in block.real], 1)
    marks.append(f"<polyline class='line-bold' points='{real}'><title>the run</title></polyline>")
    if len(block.dates):
        marks.append(_date_ends(x, block.dates[0], block.dates[-1], (0.0, max(last, 1.0))))
    svg = frame(WIDTH, HEIGHT, MARGIN, fmt.text(block.title), "".join(marks))
    groups = "; ".join(words) if words else "no candidate was kept"
    return f"{svg}<p class='key'>{groups}; bold: the run's own equity.</p>"


def table_html(block: TableBlock) -> str:
    """The table of `block`: its first column a label, the others figures."""
    columns = [(name, "l" if k == 0 else "r") for k, name in enumerate(block.columns)]
    rows = "".join(
        "<tr>"
        + "".join(
            f"<td class='{'l' if k == 0 else 'r nums'}'>{fmt.text(cell)}</td>"
            for k, cell in enumerate(row)
        )
        + "</tr>"
        for row in block.rows
    )
    return (
        f"<div class='scroll'><table class='tbl hover'>{head(*columns)}"
        f"<tbody>{rows}</tbody></table></div>"
    )


__all__ = ["candles_chart", "lines_chart", "paths_chart", "table_html"]
