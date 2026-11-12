"""The result of `metrics.analyse`: the metrics mapping and the series behind it."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Performance:
    """Performance of one strategy, or of a combined portfolio of strategies.

    metrics: read-only mapping of every figure; keys, units and definitions are in
        `openalgo.backtest.metrics.DEFINITIONS`. NaN marks a figure the data cannot
        define; benchmark keys are present only when a benchmark was given.
    equity: equity in rupees, indexed by IST time (the series that was analysed).
    drawdown: drawdown of the time-weighted wealth index at each equity point, a
        fraction from 0 down to -1, same index as `equity`.
    session_returns: return of each IST session date, a fraction, indexed by date.
    monthly_returns: compounded return per calendar month (rows: years, columns: 1 to
        12, NaN for a month with no session), fractions.
    yearly_returns: compounded return per calendar year, fractions, indexed by year.
    trades: the trades table as given (empty when none was given).
    cost_breakdown: rupees per cost component column found in the trades table,
        summed over closed trades.
    benchmark_equity: the benchmark scaled so that its first value in the run equals
        the starting equity, in rupees, at the start and at each session end (IST
        index), NaN before it has a value; None without a benchmark. With an opening
        wealth the start has no time of its own and is not a row.
    exchanges: the exchanges whose calendars annualise the ratios.
    """

    metrics: Mapping[str, Any]
    equity: pd.Series
    drawdown: pd.Series
    session_returns: pd.Series
    monthly_returns: pd.DataFrame
    yearly_returns: pd.Series
    trades: pd.DataFrame
    cost_breakdown: Mapping[str, float]
    benchmark_equity: pd.Series | None
    exchanges: tuple[str, ...]


__all__ = ["Performance"]
