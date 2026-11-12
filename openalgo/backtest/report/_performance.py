"""The result of `metrics.analyse`: the metrics mapping and the series and tables behind
it."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from . import _tables_data


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
        index), NaN where it has no close of its own (never a neighbouring day's); None
        without a benchmark. With an opening wealth the start has no time of its own and
        is not a row.
    exchanges: the exchanges whose calendars annualise the ratios.
    wealth: the time-weighted wealth index at each equity point (1 at the opening, 0
        from ruin on), same index as `equity`; the cumulative return is `wealth - 1`.
    drawdown_episodes: every drawdown episode in the order they start, one row each:
        start (the peak, or the start of the run for a fall from the opening), trough,
        end (the recovery; NaT when unrecovered), recovered, depth (a fraction, zero or
        negative), days (start to recovery, or to the last point when unrecovered),
        days_to_trough and days_to_recover (NaN when unrecovered); calendar days.
    rolling: rolling statistics over trailing windows of
        `metrics["rolling_window_sessions"]` sessions, indexed by session date:
        volatility, sharpe, sortino, win_rate and (with a benchmark) beta; NaN until a
        window is full.
    benchmark_returns: the benchmark's session returns on the strategy's session dates,
        NaN where the session or the one before it has no close of its own; None
        without a benchmark.
    horizons: the horizon summary, one row per horizon the run covers (1M, 3M, 6M, 1Y,
        3Y, 5Y) and "Since start": from (the IST date the window starts after),
        sessions, total_return, cagr, volatility, sharpe, max_drawdown (on session
        closes) and, with a benchmark, benchmark_return and benchmark_sessions (the
        sessions with both a strategy and a benchmark return).
    losing_streaks: how many losing streaks of each length the run had, indexed by
        length: sessions (runs of losing sessions) and trades (runs of losing closed
        trades in exit order).
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
    wealth: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    drawdown_episodes: pd.DataFrame = field(default_factory=pd.DataFrame)
    rolling: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark_returns: pd.Series | None = None
    horizons: pd.DataFrame = field(default_factory=pd.DataFrame)
    losing_streaks: pd.DataFrame = field(default_factory=pd.DataFrame)

    def drawdown_details(self, top: int = 5) -> pd.DataFrame:
        """The `top` deepest drawdown episodes, deepest first (the earlier one on a
        tie), with the columns of `drawdown_episodes`. Raises ConfigError for a `top`
        that is not a positive whole number."""
        return _tables_data.worst_drawdowns(self.drawdown_episodes, top)

    def benchmark_comparison(self, period: str = "year") -> pd.DataFrame:
        """Strategy against benchmark per calendar "year" or "month": the compounded
        returns of both (fractions), excess (strategy minus benchmark) and beat (True
        when the strategy did better). A period's benchmark return follows the rule of
        `benchmark_total_return`: compounded over its sessions that have a benchmark
        return; a period with no benchmark return has none, and its beat is NA. Years are
        indexed by their number, months by period. Raises ConfigError without a
        benchmark or for another period."""
        return _tables_data.benchmark_comparison(
            self.session_returns, self.benchmark_returns, period
        )


__all__ = ["Performance"]
