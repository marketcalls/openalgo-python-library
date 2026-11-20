"""A general bar chart for a custom tearsheet: values by category, one bar per series side
by side, upright or in rows, as inline SVG in the look of the dashboard's bar charts
(`bars`), beside its yearly returns chart, which is unchanged.

Each bar carries a `<title>` with its category, series and value, the browser's hover
readout, so a label cut short or thinned out on a crowded axis still reads whole.
"""

import math
from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np

from .. import _format as fmt
from . import scale
from .bars import AXIS_TEXT, MUTED_TEXT, Margin, frame, tick_texts, value_text

#: Ticks asked of the value axis.
VALUE_TICKS = 6
#: Characters of a category label shown before it is cut.
LABEL_CHARS = 16
#: Pixels a character of the 10-pixel axis font takes, for margins and for thinning
#: crowded labels.
CHAR_PX = 6
#: The chart's width and an upright chart's height, pixels of its coordinate system.
WIDTH, HEIGHT = 680, 300
#: Height of a row of a chart in rows, pixels.
ROW_PX = 26
#: Share of a category's band left between bars, and between the bars of its series.
BAND_PADDING, SERIES_PADDING = 0.3, 0.12

#: A bar's (x, y, width, height) from its band start, band width and value.
Place = Callable[[float, float, float], tuple[float, float, float, float]]


class BarSeries(NamedTuple):
    """One series: its name, one value per category (NaN for none), its CSS colour."""

    name: str
    values: np.ndarray
    colour: str


def _cut(label: str) -> str:
    """A category label cut to `LABEL_CHARS` characters, as HTML-safe text."""
    short = label if len(label) <= LABEL_CHARS else label[: LABEL_CHARS - 3] + "..."
    return fmt.text(short)


def _bars(
    categories: Sequence[str],
    series: Sequence[BarSeries],
    bands: tuple[list[float], float],
    place: Place,
    percent: bool,
) -> list[str]:
    """The bars of every category, its series side by side within its band."""
    starts, band = bands
    offsets, sub = scale.bands(len(series), band, SERIES_PADDING)
    marks = []
    for i, (start, category) in enumerate(zip(starts, categories, strict=True)):
        for one, offset in zip(series, offsets, strict=True):
            value = float(one.values[i])
            if not np.isfinite(value):
                continue
            x, y, w, h = place(start + offset, sub, value)
            marks.append(
                f"<rect x='{x:.2f}' y='{y:.2f}' width='{w:.2f}' height='{h:.2f}' "
                f"style='fill:{one.colour}' rx='1'><title>{fmt.text(category)} &#183; "
                f"{fmt.text(one.name)}: {value_text(value, percent)}</title></rect>"
            )
    return marks


def _upright(categories, series, axis, percent, signed) -> tuple[Margin, int, list[str]]:
    """An upright chart's margin, height and marks: values up, categories across."""
    low, high, ticks = axis
    widest = max(len(text) for _, text in ticks)
    margin = Margin(14, 14, 34, max(44, widest * CHAR_PX + 14))
    inner_w, inner_h = WIDTH - margin.left - margin.right, HEIGHT - margin.top - margin.bottom
    y = scale.Linear(low, high, inner_h, 0)
    starts, band = scale.bands(len(categories), inner_w, BAND_PADDING)
    longest = min(LABEL_CHARS, max(len(c) for c in categories))
    every = max(1, math.ceil((longest + 1) * CHAR_PX / (inner_w / len(categories))))
    marks = [
        f"<line x1='0' x2='{inner_w}' y1='{y(t):.2f}' y2='{y(t):.2f}' class='grid'/>"
        f"<text x='-8' y='{y(t) + 3:.2f}' text-anchor='end' style='{AXIS_TEXT}'>{text}</text>"
        for t, text in ticks
    ]
    marks += [
        f"<text x='{start + band / 2:.2f}' y='{inner_h + 18}' text-anchor='middle' "
        f"style='{MUTED_TEXT}'>{_cut(category)}</text>"
        for i, (start, category) in enumerate(zip(starts, categories, strict=True))
        if i % every == 0
    ]

    def place(at: float, size: float, value: float) -> tuple[float, float, float, float]:
        return at, min(y(value), y(0)), size, abs(y(value) - y(0))

    marks += _bars(categories, series, (starts, band), place, percent)
    if signed:
        marks.append(f"<line x1='0' x2='{inner_w}' y1='{y(0):.2f}' y2='{y(0):.2f}' class='base'/>")
    return margin, HEIGHT, marks


def _in_rows(categories, series, axis, percent, signed) -> tuple[Margin, int, list[str]]:
    """A chart in rows' margin, height and marks: a row per category, values across."""
    low, high, ticks = axis
    longest = min(LABEL_CHARS, max(len(c) for c in categories))
    margin = Margin(10, 18, 26, longest * CHAR_PX + 16)
    height = margin.top + margin.bottom + ROW_PX * len(categories)
    inner_w, inner_h = WIDTH - margin.left - margin.right, height - margin.top - margin.bottom
    x = scale.Linear(low, high, 0, inner_w)
    starts, band = scale.bands(len(categories), inner_h, BAND_PADDING)
    marks = [
        f"<line x1='{x(t):.2f}' x2='{x(t):.2f}' y1='0' y2='{inner_h}' class='grid'/>"
        f"<text x='{x(t):.2f}' y='{inner_h + 16}' text-anchor='middle' style='{AXIS_TEXT}'>"
        f"{text}</text>"
        for t, text in ticks
    ]
    marks += [
        f"<text x='-8' y='{start + band / 2 + 3:.2f}' text-anchor='end' "
        f"style='{MUTED_TEXT}'>{_cut(category)}</text>"
        for start, category in zip(starts, categories, strict=True)
    ]

    def place(at: float, size: float, value: float) -> tuple[float, float, float, float]:
        return min(x(value), x(0)), at, abs(x(value) - x(0)), size

    marks += _bars(categories, series, (starts, band), place, percent)
    if signed:
        marks.append(f"<line x1='{x(0):.2f}' x2='{x(0):.2f}' y1='0' y2='{inner_h}' class='base'/>")
    return margin, height, marks


def bar_chart(
    categories: Sequence[str],
    series: Sequence[BarSeries],
    *,
    label: str,
    horizontal: bool = False,
    percent: bool = False,
) -> str:
    """Each category's value as a bar, the series side by side, on one value axis that
    holds zero, with a solid zero line when the values change sign. Values are fractions
    shown as percent with `percent`, else plain numbers. `categories` are plain text, in
    order; a label longer than `LABEL_CHARS` is cut and crowded labels are thinned (the
    hover readout has each whole). `horizontal` draws a row per category. "No data"
    when no value is known."""
    values = [float(v) for one in series for v in one.values if np.isfinite(v)]
    if not len(categories) or not values:
        return "<p class='empty'>No data</p>"
    low, high = scale.widen(min(0.0, *values), max(0.0, *values))
    axis = (low, high, tick_texts(low, high, VALUE_TICKS, percent))
    signed = min(values) < 0 < max(values)
    draw = _in_rows if horizontal else _upright
    margin, height, marks = draw(categories, series, axis, percent, signed)
    return frame(WIDTH, height, margin, fmt.text(label), "".join(marks))


__all__ = ["BarSeries", "bar_chart"]
