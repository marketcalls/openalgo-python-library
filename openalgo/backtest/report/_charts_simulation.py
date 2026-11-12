"""Charts of distributions: the Monte Carlo band of cumulative return against the
number of draws, and box plots of returns per session, week, month, quarter and year.
Drawn with the SVG frame of `_charts` (no script)."""

import numpy as np

from . import _charts as charts
from . import _format as fmt
from ._kernel import call
from ._montecarlo import MonteCarlo

#: The five numbers of a box plot: minimum, quartiles and maximum.
BOX_LEVELS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
#: Share of a column's width taken by its box.
BOX_WIDTH_SHARE = 0.4


def _category_axis(names: list[str]) -> str:
    """Labels under evenly spaced columns, one per name."""
    count = len(names)
    spans = "".join(
        f"<span style='left:{(i + 0.5) / count * 100:.2f}%'>{fmt.text(name)}</span>"
        for i, name in enumerate(names)
    )
    return f"<div class='x categories' aria-hidden='true'>{spans}</div>"


def _count_axis(last: int, unit: str) -> str:
    """Labels of a count axis from 0 to `last` at round steps, the last one named."""
    step = max(1, round(charts.nice_step(last, charts.TARGET_TICKS)))
    values = list(range(0, last + 1, step))
    spans = []
    for value in values:
        left = value / max(last, 1) * 100
        edge = " class='first'" if left < 4 else " class='last'" if left > 96 else ""
        text = f"{value} {unit}" if value == values[-1] else str(value)
        spans.append(f"<span{edge} style='left:{left:.2f}%'>{text}</span>")
    return f"<div class='x' aria-hidden='true'>{''.join(spans)}</div>"


def _area(x: np.ndarray, upper: np.ndarray, lower: np.ndarray) -> str:
    """A closed path along `upper` and back along `lower`."""
    forward = charts.line_path(x, upper)
    back = "".join(f"L{px:.2f} {py:.2f}" for px, py in zip(x[::-1], lower[::-1], strict=True))
    return f"{forward}{back}Z"


def band_chart(study: MonteCarlo, unit: str) -> str:
    """The 5th to 95th and 25th to 75th percentile bands of cumulative return after each
    number of draws, the median path and the run's own path. `unit` names a draw
    ("sessions" or "trades")."""
    band, actual = study.band, study.actual_path
    last = int(band.index[-1])
    x = band.index.to_numpy(dtype=float) / max(last, 1) * charts.VIEW_WIDTH
    own_x = actual.index.to_numpy(dtype=float) / max(last, 1) * charts.VIEW_WIDTH
    values = band.to_numpy(dtype=float)
    pool = np.concatenate([values.ravel(), actual.to_numpy(dtype=float)])
    pool = pool[np.isfinite(pool)]
    low, high, ticks = charts.value_axis(float(pool.min()), float(pool.max()))

    def y(column: int) -> np.ndarray:
        return charts.y_position(values[:, column], low, high)

    marks = (
        f"<path class='band outer' d='{_area(x, y(4), y(0))}'/>"
        f"<path class='band inner' d='{_area(x, y(3), y(1))}'/>"
        f"<path class='line median' d='{charts.line_path(x, y(2))}'/>"
    )
    own_y = charts.y_position(actual.to_numpy(dtype=float), low, high)
    own_x, own_y = charts.thin(own_x, own_y)
    marks += f"<path class='line actual' d='{charts.line_path(own_x, own_y)}'/>"

    def tip(point: int) -> str:
        p5, p50, p95 = values[point, 0], values[point, 2], values[point, 4]
        return (
            f"{int(band.index[point])} {unit}: median {fmt.percent(p50, 1)}, 5th to 95th "
            f"percentile {fmt.percent(p5, 1)} to {fmt.percent(p95, 1)}"
        )

    marks += charts.hover_columns(x, tip)
    axis = (low, high, ticks, fmt.percent_ticks(ticks))
    return charts.plot_frame(
        "Monte Carlo cumulative return", "16rem", axis, marks, _count_axis(last, unit)
    )


def box_plot(groups: dict[str, np.ndarray]) -> str:
    """One box per group of returns: whiskers from the minimum to the maximum, a box
    from the first to the third quartile and a line at the median. A group with no
    value is an empty column."""
    names = list(groups)
    stats = {
        name: call("percentiles", values, BOX_LEVELS)
        for name, values in groups.items()
        if len(values)
    }
    pool = np.concatenate([[0.0], *stats.values()])
    low, high, ticks = charts.value_axis(float(pool.min()), float(pool.max()))
    column = charts.VIEW_WIDTH / len(names)
    marks = []
    for i, name in enumerate(names):
        if name not in stats:
            continue
        lo, q1, median, q3, hi = charts.y_position(stats[name], low, high)
        centre, half = (i + 0.5) * column, column * BOX_WIDTH_SHARE / 2
        values = stats[name]
        tip = (
            f"{fmt.text(name)}: lowest {fmt.percent(values[0])}, quartiles "
            f"{fmt.percent(values[1])} and {fmt.percent(values[3])}, median "
            f"{fmt.percent(values[2])}, highest {fmt.percent(values[4])}"
        )
        marks.append(
            f"<g class='box'><title>{tip}</title>"
            f"<line class='whisker' x1='{centre:.2f}' x2='{centre:.2f}' y1='{hi:.2f}' "
            f"y2='{lo:.2f}'/><rect x='{centre - half:.2f}' y='{q3:.2f}' "
            f"width='{2 * half:.2f}' height='{max(q1 - q3, 1.0):.2f}'/>"
            f"<line class='median' x1='{centre - half:.2f}' x2='{centre + half:.2f}' "
            f"y1='{median:.2f}' y2='{median:.2f}'/></g>"
        )
    axis = (low, high, ticks, fmt.percent_ticks(ticks))
    return charts.plot_frame(
        "Distribution of returns by period", "14rem", axis, "".join(marks), _category_axis(names)
    )


__all__ = ["BOX_LEVELS", "band_chart", "box_plot"]
