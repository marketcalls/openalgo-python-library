"""Time-series charts of the report: cumulative return against the benchmark (on a
linear or a log scale), session return bars, the rolling statistics and the histogram
of session returns. Drawn with the SVG frame of `_charts` (no script)."""

import math

import numpy as np
import pandas as pd

from . import _charts as charts
from . import _format as fmt
from ._performance import Performance

#: On a log scale, ticks follow the 1, 2, 5 sequence once the growth spans this factor;
#: below it they are the round steps of a linear axis.
LOG_SEQUENCE_SPAN = 10.0
#: Half the height, in decades, of the log axis of a run that never moved.
LOG_FLAT_PAD = 0.01
_LOG_MULTIPLES = (1.0, 2.0, 5.0)


def _log_ticks(low: float, high: float) -> list[float]:
    """Round growth factors for a log axis from `low` to `high` (both positive)."""
    if high / low < LOG_SEQUENCE_SPAN:
        return [t for t in charts.value_axis(low, high)[2] if t > 0]
    first, last = math.floor(math.log10(low)), math.ceil(math.log10(high))
    candidates = (m * 10.0**k for k in range(first, last + 1) for m in _LOG_MULTIPLES)
    return [t for t in candidates if low <= t <= high] or [low, high]


def _baseline(value: float, low: float, high: float) -> str:
    """A horizontal rule at `value` when it lies inside the axis."""
    if not low <= value <= high:
        return ""
    y = charts.y_position(np.array([value]), low, high)[0]
    return f"<line class='baseline' x1='0' x2='{charts.VIEW_WIDTH:.0f}' y1='{y:.2f}' y2='{y:.2f}'/>"


def _log(values: np.ndarray) -> np.ndarray:
    """log10 of positive values; NaN (a break in the line) for zero, after ruin."""
    positive = np.where(values > 0, values, np.nan)
    return np.log10(positive)


def _growth_axis(pool: np.ndarray, log_scale: bool) -> tuple:
    """(transform, low, high, tick positions, tick growth factors) of an axis of growth
    factors that holds `pool`."""
    known = pool[np.isfinite(pool) & (pool > 0)] if log_scale else pool[np.isfinite(pool)]
    low, high = (float(known.min()), float(known.max())) if len(known) else (1.0, 1.0)
    if not log_scale:
        lo, hi, ticks = charts.value_axis(low, high)
        return np.asarray, lo, hi, ticks, ticks
    ticks = _log_ticks(low, high) if low < high else [low]
    lo, hi = math.log10(min(ticks[0], low)), math.log10(max(ticks[-1], high))
    if lo == hi:
        lo, hi = lo - LOG_FLAT_PAD, hi + LOG_FLAT_PAD
    return _log, lo, hi, [math.log10(t) for t in ticks], ticks


def cumulative_chart(performance: Performance, log_scale: bool = False) -> str:
    """The time-weighted cumulative return at every equity point, with the benchmark's
    when given, on a linear axis or (`log_scale`) a log axis of growth. Points after
    ruin have no logarithm and are left out of the log chart."""
    wealth = performance.wealth
    growth = wealth.to_numpy(dtype=float)
    start, end = wealth.index[0], wealth.index[-1]
    to_x = charts.time_scale(start, end)
    lines = [("equity", to_x(wealth.index), growth)]
    bench = performance.benchmark_equity
    if bench is not None:
        bench_growth = bench.to_numpy(dtype=float) / performance.metrics["start_equity"]
        lines.insert(0, ("benchmark", to_x(bench.index), bench_growth))
    pool = np.concatenate([values for _, _, values in lines])
    transform, lo, hi, positions, ticks = _growth_axis(pool, log_scale)
    marks = [_baseline(float(transform(np.array([1.0]))[0]), lo, hi)]
    for name, x, values in lines:
        y = charts.y_position(transform(values), lo, hi)
        px, py = charts.thin(x, y) if name == "equity" else (x, y)
        marks.append(f"<path class='line {name}' d='{charts.line_path(px, py)}'/>")

    def tip(point: int) -> str:
        change = fmt.percent(growth[point] - 1.0, 2, signed=True)
        return f"{fmt.day(wealth.index[point])}: {change}"

    marks.append(charts.hover_columns(to_x(wealth.index), tip))
    axis = (lo, hi, positions, fmt.percent_ticks([t - 1.0 for t in ticks]))
    label = "Cumulative return, log scale" if log_scale else "Cumulative return"
    return charts.plot_frame(label, "16rem", axis, "".join(marks), charts.time_axis(start, end))


def session_bars(returns: pd.Series) -> str:
    """Each session's return as a bar up (blue) or down (red) from zero."""
    values = returns.to_numpy(dtype=float)
    start, end = returns.index[0], returns.index[-1]
    x = charts.time_scale(start, end)(returns.index)
    low, high, ticks = charts.value_axis(
        min(float(values.min()), 0.0), max(float(values.max()), 0.0)
    )
    zero = charts.y_position(np.array([0.0]), low, high)[0]
    tops = charts.y_position(values, low, high)

    def bars(keep: np.ndarray) -> str:
        # One vertical stroke per session: far smaller than a rectangle each.
        return "".join(
            f"M{px:.2f} {zero:.2f}V{py:.2f}" for px, py in zip(x[keep], tops[keep], strict=True)
        )

    marks = (
        f"<path class='bar gain' d='{bars(values > 0)}'/>"
        f"<path class='bar loss' d='{bars(values < 0)}'/>"
    )
    marks += _baseline(0.0, low, high)

    def tip(point: int) -> str:
        return f"{fmt.day(returns.index[point])}: {fmt.percent(values[point], 2, signed=True)}"

    marks += charts.hover_columns(x, tip)
    axis = (low, high, ticks, fmt.percent_ticks(ticks))
    return charts.plot_frame("Session returns", "10rem", axis, marks, charts.time_axis(start, end))


def rolling_chart(series: pd.Series, label: str, as_percent: bool, window: int) -> str:
    """One rolling statistic over session dates, NaN until a window is full."""
    values = series.to_numpy(dtype=float)
    known = values[np.isfinite(values)]
    if not len(known):
        return (
            f"<p class='note'>{fmt.text(label)}: the run has fewer sessions than one window "
            f"({window}).</p>"
        )
    start, end = series.index[0], series.index[-1]
    x = charts.time_scale(start, end)(series.index)
    low, high, ticks = charts.value_axis(float(known.min()), float(known.max()))
    line = charts.line_path(x, charts.y_position(values, low, high))
    marks = _baseline(0.0, low, high) + f"<path class='line rolling' d='{line}'/>"
    shown = fmt.percent if as_percent else fmt.number

    def tip(point: int) -> str:
        return f"{fmt.day(series.index[point])}: {shown(values[point])}"

    marks += charts.hover_columns(x, tip)
    labels = fmt.percent_ticks(ticks) if as_percent else [fmt.number(t) for t in ticks]
    title = f"<p class='panel-title'>{fmt.text(label)}</p>"
    return title + charts.plot_frame(
        label, "8rem", (low, high, ticks, labels), marks, charts.time_axis(start, end)
    )


def return_histogram(returns: pd.Series) -> str:
    """Distribution of session returns (see `_charts.histogram`)."""
    return charts.histogram(
        returns.to_numpy(dtype=float),
        label="Distribution of session returns",
        edge=lambda value: fmt.percent(value, 2),
        unit="session(s)",
        floor=0.0001,
    )


__all__ = ["cumulative_chart", "return_histogram", "rolling_chart", "session_bars"]
