"""The scatter chart of the trades: each closed trade's excursion (its MAE or MFE) against
its return, as inline SVG on a fixed coordinate system that scales with the card, as the
bar charts are drawn (`bars`).

A winning trade is a filled circle and any other a cross, so the two read apart without
colour; the legend shows both marks. Each mark carries a `<title>`, the browser's hover
readout. Both axes are percent and both always hold zero, drawn dashed.
"""

import math
from collections.abc import Sequence
from typing import NamedTuple

from .. import _format as fmt
from . import scale
from .bars import AXIS_TEXT, MUTED_TEXT, Margin, frame

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


def _pct(value: float, decimals: int) -> str:
    return f"{value * 100:.{decimals}f}%"


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


def _grid(x: scale.Linear, y: scale.Linear, size: tuple[int, int]) -> list[str]:
    """Grid lines, tick labels and the dashed zero lines of both axes."""
    inner_w, inner_h = size
    x_decimals = scale.percent_decimals(x.low, x.high, AXIS_TICKS)
    y_decimals = scale.percent_decimals(y.low, y.high, AXIS_TICKS)
    marks = [
        f"<line x1='{x(t):.2f}' x2='{x(t):.2f}' y1='0' y2='{inner_h}' class='grid'/>"
        f"<text x='{x(t):.2f}' y='{inner_h + 16}' text-anchor='middle' style='{AXIS_TEXT}'>"
        f"{_pct(t, x_decimals)}</text>"
        for t in scale.ticks(x.low, x.high, AXIS_TICKS)
    ]
    marks += [
        f"<line x1='0' x2='{inner_w}' y1='{y(t):.2f}' y2='{y(t):.2f}' class='grid'/>"
        f"<text x='-8' y='{y(t) + 3:.2f}' text-anchor='end' style='{AXIS_TEXT}'>"
        f"{_pct(t, y_decimals)}</text>"
        for t in scale.ticks(y.low, y.high, AXIS_TICKS)
    ]
    marks.append(f"<line x1='{x(0):.2f}' x2='{x(0):.2f}' y1='0' y2='{inner_h}' class='zero'/>")
    marks.append(f"<line x1='0' x2='{inner_w}' y1='{y(0):.2f}' y2='{y(0):.2f}' class='zero'/>")
    return marks


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


__all__ = ["Point", "scatter"]
