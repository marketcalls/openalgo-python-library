"""The user charts of a custom tearsheet (line, bar, histogram, scatter), drawn by the
dashboard's own chart code in its look: the time-series charts on the run's sessions
(`timeseries`), the general bar chart (`grouped_bars`), the histogram (`bars`) and the
general scatter (`scatter`).

A chart of one series takes the strategy's colour; a chart of several takes the series
colours (`--series-1` onward, which the page then writes) and a legend above it.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

from ..._messages import counted
from .. import _format as fmt
from .._dashboard import bars
from .._dashboard import timeseries as ts
from .._dashboard.grouped_bars import BarSeries, bar_chart
from .._dashboard.scatter import Group, scatter_chart
from .._dashboard.sections import STRATEGY_COLOUR
from .._periods import IST
from ._blocks import Block
from ._data import Points

#: A line chart's height by block width, pixels (the page's rolling charts are 200).
LINE_PX = {"full": 280, "half": 220, "third": 200}
#: The width a line chart's time labels are first thinned for, by block width.
NOMINAL_WIDTH = {"full": ts.WIDE, "half": ts.HALF, "third": ts.THIRD}
#: What a histogram's count axis is captioned.
COUNT_CAPTION = "count"
#: The start of what a chart returns when it has nothing to draw.
EMPTY = "<p class='empty'>"


class Drawn(NamedTuple):
    """A drawn block: its HTML, whether it uses the series colours, and a note for its
    subtitle (values left out), "" when none."""

    html: str
    several: bool = False
    note: str = ""


def series_colour(position: int) -> str:
    """The CSS colour of the series at `position` (from 0) of a chart of several."""
    return f"var(--series-{position + 1})"


def legend(names: list[str]) -> str:
    """A row of colour swatches naming the series, over a chart of several."""
    items = "".join(
        "<span style='display:inline-flex;align-items:center;gap:6px'>"
        f"<span style='width:10px;height:10px;border-radius:2px;background:{series_colour(k)}'>"
        f"</span>{fmt.text(name)}</span>"
        for k, name in enumerate(names)
    )
    return (
        "<div class='nums' style='display:flex;flex-wrap:wrap;gap:4px 16px;margin-bottom:8px;"
        f"font-size:.75rem;color:var(--text-muted)'>{items}</div>"
    )


def _with_legend(chart: str, names: list[str], several: bool) -> Drawn:
    if not several or chart.startswith(EMPTY):
        return Drawn(chart, several)
    return Drawn(legend(names) + chart, several)


def left_out(count: int, singular: str, plural: str) -> str:
    """The subtitle note of `count` values left out ("3 missing values left out"), ""
    for none."""
    return f"{counted(count, singular, plural)} left out" if count else ""


def on_sessions(series: pd.Series, sessions: pd.DatetimeIndex) -> np.ndarray:
    """`series` (indexed by dates or datetimes; naive ones are IST) on the run's
    sessions: for each session date, the last known value at or before the end of that
    IST date, NaN before the first."""
    index = series.index
    if index.tz is not None:
        index = index.tz_convert(IST).tz_localize(None)
    ordered = pd.Series(series.to_numpy(dtype=float), index=index).sort_index(kind="stable")
    ordered = ordered[np.isfinite(ordered.to_numpy())]
    by_day = ordered.set_axis(ordered.index.normalize())
    by_day = by_day[~by_day.index.duplicated(keep="last")]
    days = pd.DatetimeIndex(sessions).as_unit("ns")
    return by_day.set_axis(by_day.index.as_unit("ns")).reindex(days, method="ffill").to_numpy()


def number_labels(ticks: list[float]) -> list[str]:
    """A line chart's value labels for plain numbers, in Indian notation: whole numbers
    when every tick is one, else two decimals."""
    whole = all(float(t).is_integer() for t in ticks)
    return [fmt.number(t, 0 if whole else 2) for t in ticks]


def line(block: Block, data: tuple, sessions: pd.DatetimeIndex, anchor: str) -> Drawn:
    """A line block on the run's sessions."""
    frame, several = data
    kind = "area" if block.option("area") else "line"
    lines = [
        ts.Line(
            str(column),
            on_sessions(frame[column], sessions),
            series_colour(k) if several else STRATEGY_COLOUR,
            kind,
        )
        for k, column in enumerate(frame.columns)
    ]
    chart = ts.chart(
        sessions,
        lines,
        height=LINE_PX[block.width],
        percent=block.percent,
        label=block.title or "",
        ident=anchor,
        zero_line=block.option("zero_line"),
        log_scale=block.option("log_scale"),
        nominal_width=NOMINAL_WIDTH[block.width],
        value_labels=None if block.percent else number_labels,
    )
    return _with_legend(chart, [str(c) for c in frame.columns], several)


def category(value: object) -> str:
    """A bar's category as text: a date at midnight as "2026-11-12"."""
    if isinstance(value, pd.Timestamp):
        has_time = value != value.normalize()
        return value.strftime("%Y-%m-%d %H:%M" if has_time else "%Y-%m-%d")
    return str(value)


def bar(block: Block, data: tuple, sessions: pd.DatetimeIndex, anchor: str) -> Drawn:
    """A bar block: the index as the categories, one series per column."""
    frame, several = data
    series = [
        BarSeries(
            str(column),
            frame[column].to_numpy(dtype=float),
            series_colour(k) if several else STRATEGY_COLOUR,
        )
        for k, column in enumerate(frame.columns)
    ]
    chart = bar_chart(
        [category(value) for value in frame.index],
        series,
        label=block.title or "",
        horizontal=block.option("horizontal"),
        percent=block.percent,
    )
    return _with_legend(chart, [one.name for one in series], several)


def histogram(block: Block, values: np.ndarray, sessions: pd.DatetimeIndex, anchor: str) -> Drawn:
    """A histogram block; values that are not finite are left out and counted. The
    title is escaped here: `bars.frame` writes its label as given, the default page's
    labels being fixed text."""
    chart = bars.histogram(
        values,
        label=fmt.text(block.title or ""),
        decimals=None,
        counts=COUNT_CAPTION,
        bins=block.option("bins"),
        percent=block.percent,
    )
    return Drawn(
        chart, note=left_out(int((~np.isfinite(values)).sum()), "missing value", "missing values")
    )


def scatter(block: Block, points: Points, sessions: pd.DatetimeIndex, anchor: str) -> Drawn:
    """A scatter block, one colour per group in the order the groups first appear."""
    if points.group is None:
        groups = [Group("", points.x, points.y, STRATEGY_COLOUR)]
    else:
        names = list(pd.unique(points.group))
        groups = [
            Group(
                name,
                points.x[points.group == name],
                points.y[points.group == name],
                series_colour(k) if len(names) > 1 else STRATEGY_COLOUR,
            )
            for k, name in enumerate(names)
        ]
    chart = scatter_chart(
        groups,
        x_name=str(block.option("x")),
        y_name=str(block.option("y")),
        label=block.title or "",
        percent=block.percent,
    )
    several = len(groups) > 1
    missing = int((~(np.isfinite(points.x) & np.isfinite(points.y))).sum())
    drawn = _with_legend(chart, [g.name for g in groups], several)
    note = left_out(missing, "point without both values", "points without both values")
    return drawn._replace(note=note)


__all__ = [
    "Drawn",
    "bar",
    "category",
    "histogram",
    "left_out",
    "legend",
    "line",
    "number_labels",
    "on_sessions",
    "scatter",
    "series_colour",
]
