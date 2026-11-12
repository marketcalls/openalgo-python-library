"""The dashboard's bar charts, drawn as inline SVG on a fixed coordinate system that
scales with the card: the histogram of session returns, the box plot of session and
monthly returns, the yearly returns against the benchmark and the losing streaks.

Each mark carries a `<title>`, the browser's hover readout, so the figures read without
script. Colours are the page's theme variables.
"""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from .. import _format as fmt
from . import scale

#: Bins asked of the histogram of session returns (round edges make it about this many).
HISTOGRAM_BINS = 41
#: Ticks asked of a percent axis of the histogram and the box plot.
PERCENT_TICKS = 6
#: Font of axis text, as an SVG style.
AXIS_TEXT = "font-size:10px;font-family:var(--font-mono);fill:var(--text-faint)"
MUTED_TEXT = "font-size:10px;font-family:var(--font-mono);fill:var(--text-muted)"


class Margin(NamedTuple):
    top: int
    right: int
    bottom: int
    left: int


def _frame(width: int, height: int, margin: Margin, label: str, body: str) -> str:
    return (
        f"<svg class='bars' width='100%' viewBox='0 0 {width} {height}' role='img' "
        f"aria-label='{label}'><g transform='translate({margin.left},{margin.top})'>"
        f"{body}</g></svg>"
    )


def _pct(value: float, decimals: int) -> str:
    return f"{value * 100:.{decimals}f}%"


def histogram(values: np.ndarray, *, height: int = 220) -> str:
    """Session returns (fractions) in round-edged bins, gains green and losses red by
    the bin's centre, with a dashed line at their mean."""
    data = values[np.isfinite(values)]
    if not len(data):
        return "<p class='empty'>No data</p>"
    width, margin = 520, Margin(10, 12, 26, 12)
    inner_w, inner_h = width - margin.left - margin.right, height - margin.top - margin.bottom
    low, high = scale.widen(float(data.min()), float(data.max()))
    x = scale.Linear(low, high, 0, inner_w)
    counted = scale.bins(data, low, high, HISTOGRAM_BINS)
    y = scale.Linear(0, max(n for _, _, n in counted) or 1, inner_h, 0)
    marks = []
    for left, right, n in counted:
        centre = (left + right) / 2
        tone = "--pnl-pos" if centre >= 0 else "--pnl-neg"
        wide = max(1.0, x(right) - x(left) - 1)
        marks.append(
            f"<rect x='{x(left):.2f}' y='{y(n):.2f}' width='{wide:.2f}' "
            f"height='{inner_h - y(n):.2f}' style='fill:var({tone})' opacity='0.78'>"
            f"<title>{fmt.percent(centre)} &#183; {n}</title></rect>"
        )
    mean = float(data.mean())
    marks.append(
        f"<line x1='{x(mean):.2f}' x2='{x(mean):.2f}' y1='0' y2='{inner_h}' "
        "style='stroke:var(--accent)' stroke-width='1.5' stroke-dasharray='3 3'/>"
    )
    marks.append(f"<line x1='0' x2='{inner_w}' y1='{inner_h}' y2='{inner_h}' class='axis'/>")
    marks += [
        f"<text x='{x(t):.2f}' y='{inner_h + 16}' text-anchor='middle' style='{AXIS_TEXT}'>"
        f"{_pct(t, 1)}</text>"
        for t in scale.ticks(low, high, PERCENT_TICKS)
    ]
    return _frame(width, height, margin, "Session returns distribution", "".join(marks))


class Box(NamedTuple):
    """One row of the box plot: its label and its values (fractions)."""

    label: str
    values: np.ndarray


def box_plot(boxes: Sequence[Box], *, height: int = 200) -> str:
    """Lowest, quartiles, median and highest of each row, on one shared percent axis,
    the box green or red by the sign of its median, with a dashed line at zero."""
    rows = [(b.label, scale.quartiles(b.values)) for b in boxes if np.isfinite(b.values).any()]
    if not rows:
        return "<p class='empty'>No data</p>"
    width, margin = 520, Margin(12, 16, 26, 64)
    inner_w, inner_h = width - margin.left - margin.right, height - margin.top - margin.bottom
    low, high = scale.widen(min(q[0] for _, q in rows), max(q[4] for _, q in rows))
    x = scale.Linear(low, high, 0, inner_w)
    row_h = inner_h / len(rows)
    box_h = min(22.0, row_h * 0.5)
    marks = []
    for t in scale.ticks(low, high, PERCENT_TICKS):
        marks.append(
            f"<line x1='{x(t):.2f}' x2='{x(t):.2f}' y1='0' y2='{inner_h}' class='grid'/>"
            f"<text x='{x(t):.2f}' y='{inner_h + 16}' text-anchor='middle' "
            f"style='{AXIS_TEXT}'>{_pct(t, 1)}</text>"
        )
    if low < 0 < high:
        marks.append(f"<line x1='{x(0):.2f}' x2='{x(0):.2f}' y1='0' y2='{inner_h}' class='zero'/>")
    for i, (label, (lo, q1, med, q3, hi)) in enumerate(rows):
        cy = i * row_h + row_h / 2
        tone = "var(--pnl-pos)" if med >= 0 else "var(--pnl-neg)"
        top = cy - box_h / 2
        marks.append(
            f"<text x='-10' y='{cy + 3:.2f}' text-anchor='end' "
            f"style='font-size:11px;fill:var(--text-muted)'>{label}</text>"
            f"<line x1='{x(lo):.2f}' x2='{x(hi):.2f}' y1='{cy:.2f}' y2='{cy:.2f}' class='whisker'/>"
            f"<line x1='{x(lo):.2f}' x2='{x(lo):.2f}' y1='{cy - 5:.2f}' y2='{cy + 5:.2f}' "
            "class='whisker'/>"
            f"<line x1='{x(hi):.2f}' x2='{x(hi):.2f}' y1='{cy - 5:.2f}' y2='{cy + 5:.2f}' "
            "class='whisker'/>"
            f"<rect x='{x(q1):.2f}' y='{top:.2f}' width='{max(1.0, x(q3) - x(q1)):.2f}' "
            f"height='{box_h:.2f}' style='fill:{tone};stroke:{tone}' fill-opacity='0.22'/>"
            f"<line x1='{x(med):.2f}' x2='{x(med):.2f}' y1='{top:.2f}' y2='{top + box_h:.2f}' "
            f"style='stroke:{tone}' stroke-width='2'>"
            f"<title>median {fmt.percent(med)}</title></line>"
        )
    return _frame(width, height, margin, "Returns box plot", "".join(marks))


class YearRow(NamedTuple):
    """One year of the yearly returns: the strategy's and the benchmark's (NaN when
    unknown), fractions."""

    year: int
    strategy: float
    benchmark: float


def _value_axis(values: list[float], inner_h: float, count: int) -> tuple[scale.Linear, list]:
    low, high = min(0.0, *values), max(0.0, *values)
    low, high = scale.widen(low, high)
    return scale.Linear(low, high, inner_h, 0), scale.ticks(low, high, count)


def yearly_bars(rows: Sequence[YearRow], benchmark_name: str | None) -> str:
    """Each year's return as a bar, the benchmark's beside it (with a benchmark), with
    the zero line and the strategy's mean year as a dashed red line."""
    data = [r for r in rows if np.isfinite(r.strategy)]
    if not data:
        return "<p class='empty'>No annual data</p>"
    width, height, margin = 680, 300, Margin(26, 14, 34, 44)
    inner_w, inner_h = width - margin.left - margin.right, height - margin.top - margin.bottom
    keys = ["strategy", "benchmark"] if benchmark_name else ["strategy"]
    values = [r.strategy for r in data]
    if benchmark_name:
        values += [r.benchmark for r in data if np.isfinite(r.benchmark)]
    y, ticks = _value_axis(values, inner_h, 6)
    starts, band = scale.bands(len(data), inner_w, 0.3)
    offsets, sub = scale.bands(len(keys), band, 0.12)
    colours = {"strategy": "var(--text)", "benchmark": "var(--benchmark)"}
    legend_x = inner_w - (150 if benchmark_name else 70)
    marks = [
        f"<g transform='translate({legend_x}, -16)'>"
        "<rect width='10' height='10' style='fill:var(--text)' rx='1'/>"
        "<text x='14' y='9' style='font-size:10px;fill:var(--text-muted)'>Strategy</text>"
        + (
            "<rect x='78' width='10' height='10' style='fill:var(--benchmark)' rx='1'/>"
            f"<text x='92' y='9' style='font-size:10px;fill:var(--text-muted)'>"
            f"{fmt.text(benchmark_name)}</text>"
            if benchmark_name
            else ""
        )
        + "</g>"
    ]
    marks += [
        f"<line x1='0' x2='{inner_w}' y1='{y(t):.2f}' y2='{y(t):.2f}' class='grid'/>"
        f"<text x='-8' y='{y(t) + 3:.2f}' text-anchor='end' style='{AXIS_TEXT}'>{_pct(t, 0)}</text>"
        for t in ticks
    ]
    mean = float(np.mean(values[: len(data)]))
    marks.append(f"<line x1='0' x2='{inner_w}' y1='{y(0):.2f}' y2='{y(0):.2f}' class='base'/>")
    marks.append(
        f"<line x1='0' x2='{inner_w}' y1='{y(mean):.2f}' y2='{y(mean):.2f}' class='mean'>"
        f"<title>Strategy average: {fmt.percent(mean)}</title></line>"
    )
    names = {"strategy": "Strategy", "benchmark": fmt.text(benchmark_name or "")}
    for start, row in zip(starts, data, strict=True):
        for key, offset in zip(keys, offsets, strict=True):
            value = getattr(row, key)
            if not np.isfinite(value):
                continue
            marks.append(
                f"<rect x='{start + offset:.2f}' y='{min(y(value), y(0)):.2f}' width='{sub:.2f}' "
                f"height='{abs(y(value) - y(0)):.2f}' style='fill:{colours[key]}' rx='1'>"
                f"<title>{row.year} {names[key]}: {fmt.percent(value)}</title></rect>"
            )
        marks.append(
            f"<text x='{start + band / 2:.2f}' y='{inner_h + 18}' text-anchor='middle' "
            f"style='{MUTED_TEXT}'>{row.year}</text>"
        )
    return _frame(width, height, margin, "Yearly returns", "".join(marks))


def streak_bars(counts: Sequence[tuple[int, int]]) -> str:
    """How many losing streaks of each length, (length, count) pairs, as red bars."""
    width, height, margin = 680, 260, Margin(20, 14, 34, 40)
    inner_w, inner_h = width - margin.left - margin.right, height - margin.top - margin.bottom
    low, high = scale.widen(0.0, float(max(n for _, n in counts)), 10)
    y = scale.Linear(low, high, inner_h, 0)
    starts, band = scale.bands(len(counts), inner_w, 0.25)
    marks = [
        f"<line x1='0' x2='{inner_w}' y1='{y(t):.2f}' y2='{y(t):.2f}' class='grid'/>"
        f"<text x='-8' y='{y(t) + 3:.2f}' text-anchor='end' style='{AXIS_TEXT}'>{t:g}</text>"
        for t in scale.ticks(low, high, 5)
    ]
    for start, (length, n) in zip(starts, counts, strict=True):
        marks.append(
            f"<rect x='{start:.2f}' y='{y(n):.2f}' width='{band:.2f}' "
            f"height='{inner_h - y(n):.2f}' style='fill:var(--pnl-neg)' opacity='0.85' rx='1'>"
            f"<title>{length}-session streak: {n} times</title></rect>"
            f"<text x='{start + band / 2:.2f}' y='{inner_h + 18}' text-anchor='middle' "
            f"style='{MUTED_TEXT}'>{length}</text>"
        )
    return _frame(width, height, margin, "Losing streak lengths", "".join(marks))


__all__ = ["Box", "YearRow", "box_plot", "histogram", "streak_bars", "yearly_bars"]
