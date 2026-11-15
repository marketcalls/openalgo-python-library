"""What the dashboard draws, read from a `Performance`: series by session, the weekly
and yearly tables, the figures of the period tiles and the benchmark measured as the
strategy is.

Nothing here defines a new metric. The series are the run's own (session returns
compounded, the drawdown of the wealth index, the rolling statistics); the period
figures are cells of the run's tables (`monthly_returns`, `yearly_returns`,
`horizons`); and the benchmark's figures are `metrics.analyse` run on the benchmark's
path.
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .._analysis import analyse
from .._definitions import GROUPS
from .._performance import Performance
from .._tables_data import compound
from . import heatmaps
from .bars import YearRow

#: The metric groups measured on the return path alone, which a benchmark has as well
#: as the strategy; the trade groups and the benchmark group are not.
PATH_GROUPS = ("Run", "Returns and risk", "Drawdown", "Risk-adjusted ratios", "Session returns")
TAIL_GROUP = "Tail risk"
#: Keys of those groups that read the trades or cash flows, which a benchmark has none of.
NOT_ON_PATH = ("exposure", "risk_adjusted_return", "net_cash_flows", "missing_sessions")
#: The metrics a benchmark's figure is shown beside the strategy's for.
COMPARABLE = frozenset(
    key for group in (*PATH_GROUPS, TAIL_GROUP) for key in GROUPS[group] if key not in NOT_ON_PATH
)
#: The pandas period code of a week ending on Sunday.
WEEK = "W"


def cumulative(returns: pd.Series) -> pd.Series:
    """The compounded return since the start at each session, a fraction: the product
    of (1 + session return) so far, minus 1. A session without a return (NaN) has none
    and is left out of the product."""
    return (1.0 + returns).cumprod() - 1.0


def session_drawdown(performance: Performance) -> pd.Series:
    """The deepest drawdown of the wealth index in each session, on the session dates
    (a fraction, zero or negative): the session's lowest equity point against the
    running peak, so the chart reaches the maximum drawdown."""
    drawdown = performance.drawdown
    days = drawdown.index.tz_localize(None).normalize()
    deepest = drawdown.groupby(days).min()
    return deepest.reindex(performance.session_returns.index)


def weekly(returns: pd.Series) -> dict[int, list[heatmaps.Week]]:
    """Session returns compounded per week ending on Sunday, by the ISO year of that
    Sunday, each week labelled by its Sunday ("Jan 04") and numbered by its ISO week."""
    by_year: dict[int, list[heatmaps.Week]] = {}
    for period, value in compound(returns, WEEK).items():
        sunday = period.end_time.date()
        iso_year, iso_week, _ = sunday.isocalendar()
        week = heatmaps.Week(iso_week, sunday.strftime("%b %d"), float(value))
        by_year.setdefault(iso_year, []).append(week)
    return by_year


def yearly(performance: Performance) -> list[YearRow]:
    """Each calendar year's compounded return, with the benchmark's by the rule of
    `Performance.benchmark_comparison` (NaN without a benchmark)."""
    years = performance.yearly_returns
    if performance.benchmark_returns is None:
        bench = pd.Series(np.nan, index=years.index)
    else:
        bench = performance.benchmark_comparison("year")["benchmark"].reindex(years.index)
    return [
        YearRow(int(year), float(value), float(bench.loc[year])) for year, value in years.items()
    ]


def month_to_date(performance: Performance) -> float:
    """The return of the calendar month of the last session, from `monthly_returns`."""
    last = performance.session_returns.index[-1]
    return float(performance.monthly_returns.loc[last.year, last.month])


def year_to_date(performance: Performance) -> float:
    """The return of the calendar year of the last session, from `yearly_returns`."""
    return float(performance.yearly_returns.loc[performance.session_returns.index[-1].year])


def horizon(name: str, column: str) -> Callable[[Performance], float]:
    """A reader of one cell of `Performance.horizons` (NaN when the run does not cover
    the horizon)."""

    def read(performance: Performance) -> float:
        table = performance.horizons
        if name not in table.index:
            return math.nan
        return float(table.loc[name, column])

    return read


def session_end_times(performance: Performance) -> pd.Series:
    """The IST time of the last equity point of each session, by session date."""
    times = performance.equity.index.to_series()
    ends = times.groupby(times.index.tz_localize(None).normalize()).last()
    return ends.reindex(performance.session_returns.index)


def benchmark_performance(performance: Performance) -> Performance | None:
    """The benchmark measured as the strategy is: `metrics.analyse` on the path of the
    run's capital invested in the benchmark, which grows by each known benchmark session
    return (`Performance.benchmark_returns`) at the strategy's session ends and opens at
    the strategy's start. A session without a benchmark return has no point, as the
    rule of `benchmark_total_return` leaves it out, so the path's total return is that
    figure. None without a benchmark or without a known return. Uses the built-in
    calendar for the sessions per year."""
    returns = performance.benchmark_returns
    if returns is None:
        return None
    known = returns.dropna()
    if known.empty:
        return None
    m: Any = performance.metrics
    ends = session_end_times(performance).loc[known.index]
    path = pd.Series(
        m["start_equity"] * np.cumprod(1.0 + known.to_numpy(dtype=float)),
        index=pd.DatetimeIndex(ends),
        name="benchmark",
    )
    opening = m["start"] if m["start"] < path.index[0] else None
    return analyse(
        path,
        exchanges=performance.exchanges,
        opening_wealth=m["start_equity"],
        opening_time=opening,
        risk_free_rate=m["risk_free_rate"],
        rolling_window=int(m["rolling_window_sessions"]),
    )


__all__ = [
    "COMPARABLE",
    "benchmark_performance",
    "cumulative",
    "horizon",
    "month_to_date",
    "session_drawdown",
    "session_end_times",
    "weekly",
    "year_to_date",
    "yearly",
]
