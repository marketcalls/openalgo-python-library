"""The analysis methods of `Result`, as a mixin: the trade list, the performance and its
metrics, the plain-text summary, the metrics table, Monte Carlo and the tearsheet.

They read the run's kernel output through `_result_report` (the inputs the metrics
package documents) and never change the result: `performance` is computed once, on first
use. The mixin holds no state of its own; `Result` sets the attributes it reads.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd

from . import _result_report
from ._summary import summary_text
from ._trade_list import trade_list as closed_trade_list
from .metrics import MonteCarlo, analyse, monte_carlo
from .report import _montecarlo, metrics_table
from .report._performance import Performance


class AnalysisMethods:
    """`trade_list`, `performance`, `metrics`, `summary`, `metrics_table`, `monte_carlo`
    and `tearsheet` of a `Result`."""

    _output: dict[str, Any]
    _symbols: list[str]
    _manifest: dict[str, Any]
    _assumptions: dict[str, Any]
    _warnings: list[str]
    _calendar: Any
    _benchmark: pd.Series | None
    _strategy_closes: list[np.ndarray]
    equity: pd.DataFrame

    def trade_list(self) -> pd.DataFrame:
        """The closed trades as a trader reads them, one row each, indexed by trade
        number: symbol, direction, product, entry and exit time (IST), quantity in
        units, entry and exit price, gross P&L, charges and net P&L in rupees,
        return_pct, bars_held (bars of the strategy's interval), days_held, exit_reason
        and r_multiple (`_trade_list.trade_list` defines each column). The tearsheet's
        trade table shows the same rows."""
        trades = _result_report.closed_trades(self._output, self._symbols, self._strategy_closes)
        return closed_trade_list(trades)

    @cached_property
    def performance(self) -> Performance:
        """Every metric of the run (`openalgo.backtest.metrics.analyse`) over its equity
        series and trades, annualised by the traded exchange's calendar. The capital is
        the opening wealth and the open of the first slice the opening time, so the P&L
        of the first slice, the return of the first session and the time in the first
        slice (elapsed time, CAGR, exposure) count. A result built without its run's
        calendar times the opening at the first point. A run with `benchmark=` measures
        the benchmark figures (alpha, beta, information ratio, the comparison tables and
        charts) against its closes. Its trades carry `bars_held` in bars of the
        strategy's interval. Raises ConfigError when the run has no equity point."""
        output, symbols = self._output, self._symbols
        trades = _result_report.trades_table(output, symbols, self._strategy_closes)
        equity = _result_report.equity_series(self.equity)
        arguments = self._manifest["arguments"]
        exchange, capital = arguments["exchange"], arguments["capital"]
        opening = None
        if len(equity):
            first = int(output["equity"]["ts"][0])
            opening = _result_report.opening_time(
                first, arguments["base"], exchange, self._calendar
            )
        return analyse(
            equity,
            exchanges=exchange,
            trades=trades,
            benchmark=self._benchmark,
            opening_wealth=capital,
            opening_time=opening,
        )

    @property
    def metrics(self) -> Any:
        """The metrics mapping of `performance`; keys, units and definitions are in
        `openalgo.backtest.metrics.DEFINITIONS`."""
        return self.performance.metrics

    def summary(self) -> str:
        """A short plain-text summary to print: net profit in rupees with Indian digit
        grouping ("Rs 1,23,456.78"), the return and the maximum drawdown in percent, the
        closed trades and their win rate ("no closed trades" when there are none, where
        the win rate is undefined) with the positions still open, and for a run with
        `benchmark=` the benchmark's return (`_summary`). `print(result)` is the full
        Assumptions block instead. Raises ConfigError when the run has no equity
        point, as `performance` does."""
        return summary_text(self.metrics, _result_report.benchmark_name(self._manifest))

    def metrics_table(self) -> pd.DataFrame:
        """Every metric as a DataFrame indexed by key: group, label, value, unit and
        definition (`openalgo.backtest.report.metrics_table`)."""
        return metrics_table(self.performance)

    def monte_carlo(
        self,
        *,
        simulations: int = _montecarlo.DEFAULT_SIMULATIONS,
        seed: int = _montecarlo.DEFAULT_SEED,
        resample: str = "sessions",
        bust: float | None = None,
        goal: float | None = None,
    ) -> MonteCarlo:
        """Monte Carlo by bootstrap of the run's session returns or closed trades
        (`openalgo.backtest.metrics.monte_carlo`, which documents the arguments)."""
        return monte_carlo(
            self.performance,
            simulations=simulations,
            seed=seed,
            resample=resample,
            bust=bust,
            goal=goal,
        )

    def tearsheet(
        self, path: str | None = None, *, mode: str = "full", log_scale: bool = False
    ) -> str:
        """The HTML tearsheet, a dashboard in `mode` ("full", "basic", "metrics" or
        "charts"; see `openalgo.backtest.report.tearsheet`): returned as a string, or
        written to `path` (and the path returned). `log_scale` draws the cumulative
        return on a log scale. Its trade table is `trade_list()`, its benchmark is named
        by the run's `benchmark=`, and its Notes section lists the run's assumptions and
        every warning, so a cost-free run says so on the report."""
        lines = [f"{key}: {value}" for key, value in self._assumptions.items()]
        lines += [f"warning: {text}" for text in self._warnings]
        name = self._manifest["strategy"]["name"]
        bench = _result_report.benchmark_name(self._manifest)
        report = _result_report.Report(self.performance, name, lines, bench)
        return _result_report.tearsheet(report, path, mode, log_scale)


__all__ = ["AnalysisMethods"]
