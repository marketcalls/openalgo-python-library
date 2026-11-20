"""The scatter chart of the trades: each closed trade's excursion (its MAE or MFE) against
its return, as inline SVG on a fixed coordinate system that scales with the card, as the
bar charts are drawn (`bars`).

A winning trade is a filled circle and any other a cross, so the two read apart without
colour; the legend shows both marks. Each mark carries a `<title>`, the browser's hover
readout. Both axes are percent and both always hold zero, drawn dashed.

Beside it, `scatter_chart` is a general scatter for a custom tearsheet: any two columns
of numbers, in groups of colour.
"""

import math
from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np

from .. import _format as fmt
from . import scale
from .bars import AXIS_TEXT, MUTED_TEXT, Margin, frame, tick_texts, value_text

#: Ticks asked of each percent axis.
AXIS_TICKS = 5
#: Share of an axis's span left free beyond its outermost values, on each side.
AXIS_PADDING = 0.05
#: Radius of a winner's circle and half the width of a cross, pixels.
MARK_PX = 3.5
#: The chart's size and margins, pixels of its coordinate system.
WIDTH, HEIGHT = 520, 280
MARGIN = Margin(28, 14, 42, 56)


class Point(NamedTuple):
    """One closed trade: its number, the excursion and the return (fractions), and
    whether it won (net P&L above zero)."""

    trade: int
    excursion: float
    ret: float
    win: bool


def _axis(values: list[float]) -> tuple[float, float]:
    """An axis over `values` and zero, padded by `AXIS_PADDING` of its span on each side
    so no mark sits on the frame, then widened to round ends."""
    low, high = min(0.0, *values), max(0.0, *values)
    pad = (high - low) * AXIS_PADDING
    return scale.widen(low - pad, high + pad)


def _legend(inner_w: int) -> str:
    x = inner_w - 150
    return (
        f"<g transform='translate({x}, -18)' style='{MUTED_TEXT}'>"
        f"<circle class='win' cx='5' cy='5' r='{MARK_PX}'/>"
        "<text x='14' y='9'>Win</text>"
        f"<path class='loss' d='{_cross(60.0, 5.0)}'/>"
        "<text x='69' y='9'>Loss or flat</text></g>"
    )


def _cross(cx: float, cy: float) -> str:
    d = MARK_PX
    return (
        f"M{cx - d:.2f} {cy - d:.2f}L{cx + d:.2f} {cy + d:.2f}"
        f"M{cx - d:.2f} {cy + d:.2f}L{cx + d:.2f} {cy - d:.2f}"
    )


def _percent_ticks(low: float, high: float) -> list[tuple[float, str]]:
    decimals = scale.percent_decimals(low, high, AXIS_TICKS)
    return [(t, fmt.percent(t, decimals)) for t in scale.ticks(low, high, AXIS_TICKS)]


def _ticks(
    x: scale.Linear,
    y: scale.Linear,
    size: tuple[int, int],
    texts: Callable[[float, float], list[tuple[float, str]]],
) -> list[str]:
    """Grid lines and tick labels of both axes, labelled by `texts(low, high)`."""
    inner_w, inner_h = size
    marks = [
        f"<line x1='{x(t):.2f}' x2='{x(t):.2f}' y1='0' y2='{inner_h}' class='grid'/>"
        f"<text x='{x(t):.2f}' y='{inner_h + 16}' text-anchor='middle' style='{AXIS_TEXT}'>"
        f"{text}</text>"
        for t, text in texts(x.low, x.high)
    ]
    marks += [
        f"<line x1='0' x2='{inner_w}' y1='{y(t):.2f}' y2='{y(t):.2f}' class='grid'/>"
        f"<text x='-8' y='{y(t) + 3:.2f}' text-anchor='end' style='{AXIS_TEXT}'>"
        f"{text}</text>"
        for t, text in texts(y.low, y.high)
    ]
    return marks


def _zero_x(x: scale.Linear, inner_h: int) -> str:
    return f"<line x1='{x(0):.2f}' x2='{x(0):.2f}' y1='0' y2='{inner_h}' class='zero'/>"


def _zero_y(y: scale.Linear, inner_w: int) -> str:
    return f"<line x1='0' x2='{inner_w}' y1='{y(0):.2f}' y2='{y(0):.2f}' class='zero'/>"


def _grid(x: scale.Linear, y: scale.Linear, size: tuple[int, int]) -> list[str]:
    """Grid lines, percent tick labels and the dashed zero lines of both axes."""
    inner_w, inner_h = size
    return [*_ticks(x, y, size, _percent_ticks), _zero_x(x, inner_h), _zero_y(y, inner_w)]


def _mark(point: Point, cx: float, cy: float, name: str) -> str:
    readout = (
        f"<title>Trade {point.trade}: {name} {fmt.percent(point.excursion)}, "
        f"return {fmt.percent(point.ret)}</title>"
    )
    if point.win:
        return f"<circle class='win' cx='{cx:.2f}' cy='{cy:.2f}' r='{MARK_PX}'>{readout}</circle>"
    return f"<path class='loss' d='{_cross(cx, cy)}'>{readout}</path>"


def scatter(points: Sequence[Point], *, name: str) -> str:
    """The trades' excursion `name` ("MAE" or "MFE") on the horizontal axis against
    their return on the vertical one, both percent; "No data" when no point has both
    figures."""
    known = [p for p in points if math.isfinite(p.excursion) and math.isfinite(p.ret)]
    if not known:
        return "<p class='empty'>No data</p>"
    inner_w = WIDTH - MARGIN.left - MARGIN.right
    inner_h = HEIGHT - MARGIN.top - MARGIN.bottom
    x_low, x_high = _axis([p.excursion for p in known])
    y_low, y_high = _axis([p.ret for p in known])
    x = scale.Linear(x_low, x_high, 0, inner_w)
    y = scale.Linear(y_low, y_high, inner_h, 0)
    marks = [_legend(inner_w), *_grid(x, y, (inner_w, inner_h))]
    marks.append(
        f"<text x='{inner_w / 2:.2f}' y='{inner_h + 34}' text-anchor='middle' "
        f"style='{MUTED_TEXT}'>{name}, % of capital at work</text>"
        f"<text transform='translate({-44}, {inner_h / 2:.2f}) rotate(-90)' "
        f"text-anchor='middle' style='{MUTED_TEXT}'>Return, %</text>"
    )
    marks += [_mark(p, x(p.excursion), y(p.ret), name) for p in known]
    return frame(WIDTH, HEIGHT, MARGIN, f"{name} against return", "".join(marks))


#: A general scatter chart's radius of a point, pixels.
DOT_PX = 3.0
#: Pixels a character of the 10-pixel axis font takes (for the left margin), and the
#: room between the value labels and the rotated name of the vertical axis.
CHAR_PX, AXIS_NAME_PX = 6, 30


class Group(NamedTuple):
    """One group of a general scatter chart: its name ("" for the only group), the x and
    y of its points, and its CSS colour."""

    name: str
    x: np.ndarray
    y: np.ndarray
    colour: str


def _span(values: np.ndarray) -> tuple[float, float]:
    """An axis over `values` alone, padded by `AXIS_PADDING` of its span on each side
    (of its size, or of one, when the values are all equal), widened to round ends."""
    low, high = float(values.min()), float(values.max())
    pad = (high - low) * AXIS_PADDING or (abs(high) * AXIS_PADDING or 1.0)
    return scale.widen(low - pad, high + pad)


def scatter_chart(
    groups: Sequence[Group], *, x_name: str, y_name: str, label: str, percent: bool = False
) -> str:
    """Each group's points, `x_name` across against `y_name` up, as filled dots in the
    group's colour, each with its values as a hover readout. The axes span the values
    (not zero unless they reach it); a dashed zero line marks an axis that crosses zero.
    Values are fractions shown as percent with `percent`, else plain numbers. "No data"
    when no point has both values."""
    shown = [
        g._replace(x=g.x[known], y=g.y[known])
        for g in groups
        if (known := np.isfinite(g.x) & np.isfinite(g.y)).any()
    ]
    if not shown:
        return "<p class='empty'>No data</p>"
    xs = np.concatenate([g.x for g in shown])
    ys = np.concatenate([g.y for g in shown])
    x_low, x_high = _span(xs)
    y_low, y_high = _span(ys)

    def texts(low: float, high: float) -> list[tuple[float, str]]:
        return tick_texts(low, high, AXIS_TICKS, percent)

    widest = max(len(text) for _, text in texts(y_low, y_high))
    margin = MARGIN._replace(left=max(MARGIN.left, widest * CHAR_PX + AXIS_NAME_PX))
    inner_w, inner_h = WIDTH - margin.left - margin.right, HEIGHT - margin.top - margin.bottom
    x = scale.Linear(x_low, x_high, 0, inner_w)
    y = scale.Linear(y_low, y_high, inner_h, 0)
    marks = _ticks(x, y, (inner_w, inner_h), texts)
    if x_low < 0 < x_high:
        marks.append(_zero_x(x, inner_h))
    if y_low < 0 < y_high:
        marks.append(_zero_y(y, inner_w))
    marks.append(
        f"<text x='{inner_w / 2:.2f}' y='{inner_h + 34}' text-anchor='middle' "
        f"style='{MUTED_TEXT}'>{fmt.text(x_name)}</text>"
        f"<text transform='translate({12 - margin.left}, {inner_h / 2:.2f}) rotate(-90)' "
        f"text-anchor='middle' style='{MUTED_TEXT}'>{fmt.text(y_name)}</text>"
    )
    for g in shown:
        named = f"{fmt.text(g.name)}: " if g.name else ""
        for px, py in zip(g.x, g.y, strict=True):
            readout = (
                f"{named}{fmt.text(x_name)} {value_text(px, percent)}, "
                f"{fmt.text(y_name)} {value_text(py, percent)}"
            )
            marks.append(
                f"<circle cx='{x(px):.2f}' cy='{y(py):.2f}' r='{DOT_PX}' "
                f"style='fill:{g.colour};stroke:{g.colour}' fill-opacity='0.7'>"
                f"<title>{readout}</title></circle>"
            )
    return frame(WIDTH, HEIGHT, margin, fmt.text(label), "".join(marks))


__all__ = ["Group", "Point", "scatter", "scatter_chart"]
