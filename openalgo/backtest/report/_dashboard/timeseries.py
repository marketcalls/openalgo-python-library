"""The dashboard's time-series charts: cumulative return, drawdown and the rolling
statistics, drawn as inline SVG over the session dates.

Each point is one session and the sessions are evenly spaced (no gaps for weekends or
holidays). The value axis is on the right; its domain leaves a fifth of the pane above
the highest value and a tenth below the lowest, as the reference dashboard does, and
its labels are round steps at least `MIN_LABEL_GAP_PX` apart. The plot is an SVG
stretched to the card's width with strokes that keep their width, the labels are HTML
placed by percent and pixel, so text never stretches. The values ride on the figure as
data attributes, which the page's script reads to draw the hover readout (`script`).
"""

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd

from .. import _format as fmt
from . import scale

#: Height of the time axis under the pane, pixels.
TIME_AXIS_PX = 28
#: Width of the value axis right of the pane, pixels.
VALUE_AXIS_PX = 56
#: Width of the SVG's coordinate system; the SVG is stretched to the card's width.
VIEW_WIDTH = 1000
#: Shares of the pane left empty above the highest and below the lowest value.
TOP_MARGIN, BOTTOM_MARGIN = 0.2, 0.1
#: The smallest gap between two value labels, and how close a label may come to the
#: pane's top or bottom edge, pixels.
MIN_LABEL_GAP_PX = 30
LABEL_EDGE_PX = 8
#: The smallest gap between two time labels, pixels, at the chart's nominal width.
MIN_TIME_GAP_PX = 64
#: Nominal widths of a chart in a full-width card, a half and a third, pixels: the
#: time labels are thinned for them (the script thins them again for the real width).
WIDE, HALF, THIRD = 1100, 520, 300
#: Spans, in calendar days, above which the time axis is labelled by year, then month.
YEARLY_SPAN_DAYS, MONTHLY_SPAN_DAYS = 730, 92
#: Opacity of an area's fill at the line and at the pane's bottom, and of the fill
#: between a baseline series and zero at zero and at the bottom.
AREA_OPACITY = (0.22, 0.02)
BASELINE_OPACITY = (0.06, 0.32)
#: Significant digits of the values carried for the hover readout.
DATA_DIGITS = 6
#: The class of a time label hidden because it would crowd the one before it.
HIDDEN = " class='off'"


class Line(NamedTuple):
    """One series of a chart: values aligned to the chart's dates (NaN where none), a
    CSS colour, "line", "area" (filled to the pane's bottom) or "baseline" (filled to
    zero), and the stroke width in pixels."""

    name: str
    values: np.ndarray
    color: str
    kind: str = "line"
    width: float = 2.0


class _Pane(NamedTuple):
    top: float  # the value at the pane's top edge (log of growth on a log scale)
    bottom: float
    height: float  # pixels
    log: bool

    def y(self, value: float) -> float:
        """Pixels from the top of the pane to `value` (a plain value, not its log)."""
        shown = math.log1p(value) if self.log else value
        return (self.top - shown) / (self.top - self.bottom) * self.height


def _transform(values: np.ndarray, log: bool) -> np.ndarray:
    if not log:
        return values
    with np.errstate(invalid="ignore", divide="ignore"):
        growth = np.where(values > -1.0, values, np.nan)
        return np.log1p(growth)


def _pane(lines: Sequence[Line], height: float, log: bool) -> _Pane:
    pool = np.concatenate([_transform(line.values, log) for line in lines])
    pool = pool[np.isfinite(pool)]
    low, high = float(pool.min()), float(pool.max())
    if high == low:
        pad = abs(high) * 0.01 or 0.01
        low, high = low - pad, high + pad
    span = (high - low) / (1 - TOP_MARGIN - BOTTOM_MARGIN)
    return _Pane(high + span * TOP_MARGIN, low - span * BOTTOM_MARGIN, height, log)


def _value_ticks(pane: _Pane) -> list[float]:
    """Round values on the axis, at least `MIN_LABEL_GAP_PX` apart, away from the
    edges. On a log scale the step is the smallest whose ticks keep that gap."""
    low = math.expm1(pane.bottom) if pane.log else pane.bottom
    high = math.expm1(pane.top) if pane.log else pane.top
    power = 10.0 ** math.floor(math.log10((high - low) / 100))
    for exponent in range(6):
        for multiple in scale.AXIS_MULTIPLES:
            step = multiple * power * 10**exponent
            first, last = math.ceil(low / step), math.floor(high / step)
            values = [round(i * step, scale.TICK_DECIMALS) for i in range(first, last + 1)]
            ys = [pane.y(v) for v in values]
            if all(b - a >= MIN_LABEL_GAP_PX for a, b in zip(ys[1:], ys, strict=False)):
                edge = pane.height - LABEL_EDGE_PX
                return [v for v, y in zip(values, ys, strict=True) if LABEL_EDGE_PX <= y <= edge]
    return []


def _value_label(value: float, percent: bool) -> str:
    return f"{value * 100:.1f}%" if percent else f"{value:.2f}"


def _x(position: int, count: int) -> float:
    return position / (count - 1) * VIEW_WIDTH


def _path(values: np.ndarray, pane: _Pane) -> list[tuple[float, float]]:
    count = len(values)
    return [
        (_x(i, count), pane.y(float(v)))
        for i, v in enumerate(values)
        if math.isfinite(v) and (not pane.log or v > -1.0)
    ]


def _d(points: list[tuple[float, float]]) -> str:
    return "M" + "L".join(f"{x:.2f} {y:.2f}" for x, y in points)


def _gradient(gid: str, color: str, opacities: tuple[float, float], y0: float, y1: float) -> str:
    stops = "".join(
        f"<stop offset='{offset}' style='stop-color:{color};stop-opacity:{opacity}'/>"
        for offset, opacity in zip((0, 1), opacities, strict=True)
    )
    return (
        f"<linearGradient id='{gid}' gradientUnits='userSpaceOnUse' x1='0' y1='{y0:.2f}' "
        f"x2='0' y2='{y1:.2f}'>{stops}</linearGradient>"
    )


def _series_svg(line: Line, pane: _Pane, gid: str) -> tuple[str, str]:
    """The gradient definition (or "") and the shapes of one series."""
    points = _path(line.values, pane)
    if not points:
        return "", ""
    stroke = (
        f"<path class='ts-line' d='{_d(points)}' style='stroke:{line.color}' "
        f"stroke-width='{line.width}'/>"
    )
    if line.kind == "area":
        first, last = points[0][0], points[-1][0]
        fill = f"{_d(points)}L{last:.2f} {pane.height:.2f}L{first:.2f} {pane.height:.2f}Z"
        defs = _gradient(gid, line.color, AREA_OPACITY, _top(points), pane.height)
        return defs, f"<path d='{fill}' fill='url(#{gid})'/>" + stroke
    if line.kind == "baseline":
        zero = pane.y(0.0)
        first, last = points[0][0], points[-1][0]
        fill = f"M{first:.2f} {zero:.2f}L{_d(points)[1:]}L{last:.2f} {zero:.2f}Z"
        defs = _gradient(gid, line.color, BASELINE_OPACITY, zero, pane.height)
        return defs, f"<path d='{fill}' fill='url(#{gid})'/>" + stroke
    return "", stroke


def _top(points: list[tuple[float, float]]) -> float:
    """The highest point of a path (the smallest y)."""
    return min(y for _, y in points)


def _time_labels(dates: pd.DatetimeIndex, nominal_width: int) -> list[tuple[float, str, bool]]:
    """(percent across the pane, text, shown) of the time labels: the first session of
    each year on a span over two years, of each month on a span over three months (the
    year in place of January), else of each day with the month on its first. A label
    closer than `MIN_TIME_GAP_PX` to the last one shown, at the nominal width, is not
    shown (the script decides again at the real width)."""
    count = len(dates)
    span = (dates[-1] - dates[0]).days
    if span > YEARLY_SPAN_DAYS:
        keys, texts = dates.year, [str(d.year) for d in dates]
    elif span > MONTHLY_SPAN_DAYS:
        keys = dates.year * 12 + dates.month
        texts = [str(d.year) if d.month == 1 else d.strftime("%b") for d in dates]
    else:
        keys = dates.year * 400 + dates.day_of_year
        texts = [d.strftime("%b") if d.day == 1 else str(d.day) for d in dates]
    gap = MIN_TIME_GAP_PX / nominal_width * 100
    labels: list[tuple[float, str, bool]] = []
    last_shown = -math.inf
    for i in range(1, count):
        if keys[i] == keys[i - 1]:
            continue
        where = i / (count - 1) * 100
        shown = where - last_shown >= gap
        last_shown = where if shown else last_shown
        labels.append((where, texts[i], shown))
    return labels


def _data_attributes(lines: Sequence[Line], pane: _Pane, first: int, count: int) -> str:
    def number(v: float) -> str:
        return f"{v:.{DATA_DIGITS}g}" if math.isfinite(v) else ""

    parts = [
        f"data-first='{first}'",
        f"data-n='{count}'",
        f"data-top='{pane.top!r}'",
        f"data-bottom='{pane.bottom!r}'",
        f"data-log='{int(pane.log)}'",
        f"data-lines='{len(lines)}'",
    ]
    for k, line in enumerate(lines):
        values = ",".join(number(float(v)) for v in line.values)
        parts.append(f"data-c{k}='{line.color}' data-v{k}='{values}'")
    return " ".join(parts)


def chart(
    dates: pd.DatetimeIndex,
    lines: Sequence[Line],
    *,
    height: int,
    percent: bool,
    label: str,
    ident: str,
    zero_line: bool = False,
    log_scale: bool = False,
    nominal_width: int = WIDE,
    empty: str = "No data",
) -> str:
    """A time-series chart of `lines` over the session `dates`, `height` pixels tall
    with its time axis, values as percentages (fractions in) or plain numbers, a
    dashed line at zero when `zero_line` and a line series is drawn, the values on a
    log scale of growth when `log_scale` (percent charts only). Sessions before the
    first and after the last value of any series are left out. With fewer than two
    sessions to draw, the `empty` sentence is returned instead."""
    known = np.zeros(len(dates), dtype=bool)
    for line in lines:
        known |= np.isfinite(_transform(line.values, log_scale))
    where = np.flatnonzero(known)
    if len(where) < 2:
        return f"<p class='empty'>{empty}</p>"
    first, last = int(where[0]), int(where[-1]) + 1
    shown = [line._replace(values=line.values[first:last]) for line in lines]
    days = dates[first:last]
    pane = _pane(shown, height - TIME_AXIS_PX, log_scale)
    ticks = _value_ticks(pane)
    grid = "".join(f"M0 {pane.y(t):.2f}H{VIEW_WIDTH}" for t in ticks)
    shapes = [f"<path class='ts-grid' d='{grid}'/>"] if grid else []
    if zero_line and any(line.kind == "line" for line in shown) and 0 <= pane.y(0.0) <= pane.height:
        shapes.append(f"<path class='ts-zero' d='M0 {pane.y(0.0):.2f}H{VIEW_WIDTH}'/>")
    defs = []
    for k, line in enumerate(shown):
        gid = f"{ident}-{k}"
        definition, drawn = _series_svg(line, pane, gid)
        defs.append(definition)
        shapes.append(drawn)
    y_labels = "".join(
        f"<span style='top:{pane.y(t):.1f}px'>{_value_label(t, percent)}</span>" for t in ticks
    )
    x_labels = "".join(
        f"<span{'' if shown else HIDDEN} style='left:{at:.3f}%'>{text}</span>"
        for at, text, shown in _time_labels(days, nominal_width)
    )
    fmt_name = "percent" if percent else "number"
    return (
        f"<figure class='ts' style='height:{height}px' data-fmt='{fmt_name}' "
        f"{_data_attributes(shown, pane, first, last - first)}>"
        f"<svg class='ts-pane' viewBox='0 0 {VIEW_WIDTH} {pane.height:.0f}' "
        f"preserveAspectRatio='none' role='img' aria-label='{fmt.text(label)}' "
        f"style='height:{pane.height:.0f}px'><defs>{''.join(defs)}</defs>{''.join(shapes)}</svg>"
        f"<div class='ts-y' style='height:{pane.height:.0f}px'>{y_labels}</div>"
        f"<div class='ts-x'>{x_labels}</div></figure>"
    )


__all__ = ["HALF", "THIRD", "WIDE", "Line", "chart"]
