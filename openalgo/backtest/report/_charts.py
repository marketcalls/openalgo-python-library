"""Charts as inline SVG and plain HTML, with no script.

The plotted marks are SVG drawn in a fixed coordinate box and stretched to the width
of the page (`preserveAspectRatio="none"`, with strokes that do not scale), while the
axis labels are HTML placed by percentage. Text therefore keeps its size at phone
width. Native tooltips come from SVG `<title>` and HTML `title` attributes; every value
is also in a table elsewhere in the report.
"""

import math
from collections.abc import Callable

import numpy as np
import pandas as pd

from . import _format as fmt
from ._performance import Performance

VIEW_WIDTH = 1000.0  # SVG units across every plot
VIEW_HEIGHT = 300.0  # SVG units down every plot
MAX_PLOT_BUCKETS = 1200  # a long series keeps the lowest and highest point per bucket
HOVER_COLUMNS = 60  # tooltip columns across the equity chart
TARGET_TICKS = 5  # value gridlines per axis, about
TARGET_BINS = 20  # histogram bins, about
_NICE_STEPS = (1.0, 2.0, 2.5, 5.0, 10.0)


def nice_step(span: float, count: int) -> float:
    """A round step (1, 2, 2.5 or 5 times a power of ten) near `span / count`."""
    raw = span / max(count, 1)
    if not math.isfinite(raw) or raw <= 0:
        return 1.0
    magnitude = 10.0 ** math.floor(math.log10(raw))
    return next(m * magnitude for m in _NICE_STEPS if m * magnitude >= raw * (1 - 1e-9))


def value_axis(low: float, high: float) -> tuple[float, float, list[float]]:
    """(domain low, domain high, ticks) with round ticks on both ends of the domain."""
    if not (math.isfinite(low) and math.isfinite(high)):
        low, high = 0.0, 1.0
    if high - low <= abs(high) * 1e-9:
        pad = abs(high) * 0.01 or 1.0
        low, high = low - pad, high + pad
    step = nice_step(high - low, TARGET_TICKS)
    start, stop = math.floor(low / step) * step, math.ceil(high / step) * step
    ticks = [start + i * step for i in range(round((stop - start) / step) + 1)]
    return start, stop, ticks


def _thin(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """At most two points per bucket of x (its lowest and highest), in x order."""
    if len(x) <= 2 * MAX_PLOT_BUCKETS:
        return x, y
    bucket = np.minimum((x / VIEW_WIDTH * MAX_PLOT_BUCKETS).astype(np.int64), MAX_PLOT_BUCKETS - 1)
    frame = pd.DataFrame({"bucket": bucket, "y": y})
    groups = frame.groupby("bucket")["y"]
    keep = np.unique(np.concatenate([groups.idxmin().to_numpy(), groups.idxmax().to_numpy()]))
    keep = np.unique(np.concatenate([[0, len(x) - 1], keep]))
    return x[keep], y[keep]


def _path(x: np.ndarray, y: np.ndarray) -> str:
    """An SVG path through the points; a NaN breaks the line."""
    parts, pen_down = [], False
    for px, py in zip(x, y, strict=True):
        if not math.isfinite(py):
            pen_down = False
            continue
        parts.append(f"{'L' if pen_down else 'M'}{px:.2f} {py:.2f}")
        pen_down = True
    return "".join(parts)


def _y_position(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return VIEW_HEIGHT - (values - low) / (high - low) * VIEW_HEIGHT


def _time_labels(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, str]]:
    days = (end - start).days
    if days >= 3 * 365:
        stamps, pattern = pd.date_range(start, end, freq="YS"), "%Y"
    elif days >= 120:
        months = max(1, math.ceil(days / 30.44 / 6))
        stamps, pattern = pd.date_range(start, end, freq=f"{months}MS"), "%b %Y"
    else:
        stamps, pattern = (
            pd.date_range(start.normalize(), end, freq=f"{max(1, days // 6)}D"),
            "%d %b",
        )
    stamps = [s for s in stamps if start <= s <= end] or [start]
    return [(s, s.strftime(pattern)) for s in stamps]


def _x_axis(start: pd.Timestamp, end: pd.Timestamp) -> str:
    span = (end - start).total_seconds() or 1.0
    labels = []
    for stamp, label in _time_labels(start, end):
        left = (stamp - start).total_seconds() / span * 100
        edge = " class='first'" if left < 4 else " class='last'" if left > 96 else ""
        labels.append(f"<span{edge} style='left:{left:.2f}%'>{fmt.text(label)}</span>")
    return f"<div class='x' aria-hidden='true'>{''.join(labels)}</div>"


def _y_axis(ticks: list[float], low: float, high: float, labels: list[str]) -> str:
    spans = (
        f"<span style='top:{(high - t) / (high - low) * 100:.2f}%'>{label}</span>"
        for t, label in zip(ticks, labels, strict=True)
    )
    return f"<div class='y' aria-hidden='true'>{''.join(spans)}</div>"


def _gridlines(ticks: list[float], low: float, high: float) -> str:
    rows = _y_position(np.array(ticks, dtype=float), low, high)
    return "".join(
        f"<line class='grid' x1='0' x2='{VIEW_WIDTH:.0f}' y1='{y:.2f}' y2='{y:.2f}'/>" for y in rows
    )


def _hover(x: np.ndarray, tip: Callable[[int], str]) -> str:
    """Invisible columns, each with the tooltip `tip(point)` of the last point in it."""
    edges = np.linspace(0.0, VIEW_WIDTH, HOVER_COLUMNS + 1)
    last = np.searchsorted(x, edges[1:], side="right") - 1
    rects = []
    for left, right, point in zip(edges[:-1], edges[1:], last, strict=True):
        if point < 0 or x[point] < left:
            continue
        width = right - left
        rects.append(
            f"<rect x='{left:.2f}' y='0' width='{width:.2f}' height='{VIEW_HEIGHT:.0f}'>"
            f"<title>{tip(int(point))}</title></rect>"
        )
    return f"<g class='hover'>{''.join(rects)}</g>"


def _plot(label: str, height: str, axis: tuple, marks: str, x_axis: str) -> str:
    low, high, ticks, tick_labels = axis
    svg = (
        f"<svg viewBox='0 0 {VIEW_WIDTH:.0f} {VIEW_HEIGHT:.0f}' preserveAspectRatio='none' "
        f"role='img' aria-label='{fmt.text(label)}'>{_gridlines(ticks, low, high)}{marks}</svg>"
    )
    return (
        f"<div class='chart' style='--h:{height}'>{_y_axis(ticks, low, high, tick_labels)}"
        f"<div class='canvas'>{svg}</div>{x_axis}</div>"
    )


def equity_chart(performance: Performance) -> str:
    """The equity curve in rupees (with the benchmark, when given, on the same scale)
    above the drawdown of the time-weighted wealth index."""
    equity, drawdown = performance.equity, performance.drawdown
    start, end = equity.index[0], equity.index[-1]
    span = (end - start).total_seconds() or 1.0

    def to_x(index: pd.DatetimeIndex) -> np.ndarray:
        return (index - start).total_seconds().to_numpy() / span * VIEW_WIDTH

    x, values = to_x(equity.index), equity.to_numpy(dtype=float)
    bench = performance.benchmark_equity
    pool = values if bench is None else np.concatenate([values, bench.to_numpy(dtype=float)])
    low, high, ticks = value_axis(float(np.nanmin(pool)), float(np.nanmax(pool)))
    marks = []
    if bench is not None:
        bx, by = to_x(bench.index), _y_position(bench.to_numpy(dtype=float), low, high)
        marks.append(f"<path class='line benchmark' d='{_path(bx, by)}'/>")
    ex, ey = _thin(x, values)
    marks.append(f"<path class='line equity' d='{_path(ex, _y_position(ey, low, high))}'/>")
    depth = drawdown.to_numpy(dtype=float)

    def tip(point: int) -> str:
        when, value = equity.index[point], values[point]
        return f"{fmt.day(when)}: {fmt.rupees(value)}, drawdown {fmt.percent(depth[point])}"

    marks.append(_hover(x, tip))
    axis = (low, high, ticks, fmt.rupee_ticks(ticks))
    x_axis = _x_axis(start, end)
    top = _plot("Equity in rupees", "18rem", axis, "".join(marks), x_axis)
    return top + _drawdown_plot(x, depth, x_axis)


# The drawdown axis reaches at least this deep, so a curve that never falls is drawn
# flat at the top of a one-percent axis rather than in the middle of an empty one.
_SHALLOWEST_DRAWDOWN_AXIS = -0.01


def _drawdown_plot(x: np.ndarray, depth: np.ndarray, x_axis: str) -> str:
    low, high, ticks = value_axis(min(float(np.min(depth)), _SHALLOWEST_DRAWDOWN_AXIS), 0.0)
    dx, dy = _thin(x, depth)
    line = _path(dx, _y_position(dy, low, high))
    top = _y_position(np.array([0.0]), low, high)[0]
    area = f"{line}L{dx[-1]:.2f} {top:.2f}L{dx[0]:.2f} {top:.2f}Z"
    marks = f"<path class='wash' d='{area}'/><path class='line drawdown' d='{line}'/>"
    axis = (low, high, ticks, fmt.percent_ticks(ticks))
    title = "<p class='panel-title'>Drawdown from the running peak</p>"
    return title + _plot("Drawdown", "7rem", axis, marks, x_axis)


def pnl_histogram(net_pnl: np.ndarray) -> str:
    """Distribution of closed trades' net P&L as columns; red bins lie at or below
    zero, blue above. Bin edges are round multiples of one step, so zero is an edge."""
    pnl = net_pnl[np.isfinite(net_pnl)]
    low, high = float(pnl.min()), float(pnl.max())
    step = nice_step(high - low, TARGET_BINS) if high > low else max(abs(high), 1.0)
    first = math.floor(low / step) * step
    last = max(math.ceil(high / step) * step, first + step)
    edges = np.arange(round((last - first) / step) + 1) * step + first
    counts, _ = np.histogram(pnl, bins=edges)
    peak = max(int(counts.max()), 1)
    bins = []
    for left, right, count in zip(edges[:-1], edges[1:], counts, strict=True):
        side = "loss" if right <= 0 else "gain"
        tip = f"{fmt.rupees(left, 0)} to {fmt.rupees(right, 0)}: {count} trade(s)"
        bins.append(
            f"<div class='bin {side}' title='{tip}'><i style='height:{count / peak * 100:.2f}%'>"
            "</i></div>"
        )
    marked = [first, last] + ([0.0] if first < 0 < last else [])
    positions = [(v - first) / (last - first) * 100 for v in marked]
    labels = "".join(
        f"<span style='left:{left:.2f}%'>{fmt.rupees_compact(v)}</span>"
        for v, left in zip(marked, positions, strict=True)
    )
    return (
        f"<div class='hist' role='img' aria-label='Distribution of trade net P&amp;L'>"
        f"{''.join(bins)}</div><div class='hist-axis' aria-hidden='true'>{labels}</div>"
    )


__all__ = ["equity_chart", "nice_step", "pnl_histogram", "value_axis"]
