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
from ._result_excursion import instrument_multipliers
from ._summary import summary_text
from ._trade_list import trade_list as closed_trade_list
from .metrics import MonteCarlo, analyse, monte_carlo
from .report import _montecarlo, metrics_table
from .report._performance import Performance
from .report.custom import Layout


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
        and r_multiple, then mae, mfe, mae_pct, mfe_pct and slippage
        (`_trade_list.trade_list` defines each column). The tearsheet's trade table shows
        the same rows."""
        trades = _result_report.closed_trades(
            self._output, self._symbols, self._strategy_closes, self._multipliers()
        )
        return closed_trade_list(trades)

    def _multipliers(self) -> list[float]:
        """The contract multiplier of each instrument, by the kernel's index."""
        return instrument_multipliers(self._manifest)

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
        equity = _result_report.equity_series(self.equity)
        arguments = self._manifest["arguments"]
        return analyse(
            equity,
            exchanges=arguments["exchange"],
            trades=self._metrics_trades(),
            benchmark=self._benchmark,
            opening_wealth=arguments["capital"],
            opening_time=self._opening_time(),
        )

    def _metrics_trades(self) -> pd.DataFrame | None:
        """The run's closed and open trades in the metrics layout (`_result_report`), or
        None when it has none."""
        output, symbols = self._output, self._symbols
        closes, multipliers = self._strategy_closes, self._multipliers()
        return _result_report.trades_table(output, symbols, closes, multipliers)

    def _opening_time(self) -> int | None:
        """UTC nanoseconds of the open of the run's first slice, when it opened with its
        capital; None for a run with no equity point or a result without its calendar."""
        if not len(self._output["equity"]["ts"]):
            return None
        arguments = self._manifest["arguments"]
        first = int(self._output["equity"]["ts"][0])
        return _result_report.opening_time(
            first, arguments["base"], arguments["exchange"], self._calendar
        )

    @property
    def metrics(self) -> Any:
        """The metrics mapping of `performance`; keys, units and definitions are in
        `openalgo.backtest.metrics.DEFINITIONS`."""
        return self.performance.metrics

    @cached_property
    def _quick(self) -> Any:
        """The quick overfitting checks of the run's trade list (`_robust._quick`),
        computed on first use; None for a run with no equity point, which has no
        performance to check."""
        if not len(self._output["equity"]["ts"]):
            return None
        from ._robust._quick import quick_checks

        return quick_checks(self.performance)

    def _overfitting_warnings(self, pointer: str | None = None) -> list[str]:
        """A warning line per quick overfitting check that failed, ending with `pointer`,
        where the full set of checks is (None: the quick checks' own, which loads with
        them, `_robust._quick.POINTER`)."""
        if self._quick is None:
            return []
        return self._quick.warnings() if pointer is None else self._quick.warnings(pointer)

    def _all_warnings(self, pointer: str | None = None) -> list[str]:
        """The run's warnings, then a line per failed quick check ending with `pointer`
        (None: the result's own). The one composition of `Result.warnings`, the Notes of
        the tearsheet and, with `_messages.ALONE_POINTER` (a strategy of a portfolio run,
        whose own `robustness()` is refused), its lines in `PortfolioResult.warnings`."""
        return [*self._warnings, *self._overfitting_warnings(pointer)]

    def summary(self) -> str:
        """A short plain-text summary to print: net profit in rupees with Indian digit
        grouping ("Rs 1,23,456.78"), the return and the maximum drawdown in percent, the
        closed trades and their win rate ("no closed trades" when there are none, where
        the win rate is undefined) with the positions still open, the quick overfitting
        checks in one line (`_robust._quick`), and for a run with `benchmark=` the
        benchmark's return (`_summary`). `print(result)` is the full Assumptions block
        instead. Raises ConfigError when the run has no equity point, as `performance`
        does."""
        metrics = self.metrics
        bench = _result_report.benchmark_name(self._manifest)
        return summary_text(metrics, bench, self._quick.summary_line())

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
        self,
        path: str | None = None,
        *,
        mode: str = "full",
        log_scale: bool = False,
        layout: Layout | None = None,
    ) -> str:
        """The HTML tearsheet, a dashboard in `mode` ("full", "basic", "metrics" or
        "charts"; see `openalgo.backtest.report.tearsheet`): returned as a string, or
        written to `path` (and the path returned). `log_scale` draws the cumulative
        return on a log scale. Its trade table is `trade_list()`, its benchmark is named
        by the run's `benchmark=`, and its Notes section lists the run's assumptions and
        every warning, so a cost-free run says so on the report. `layout` (a
        `report.custom.Layout`) makes a custom tearsheet whose blocks' callables receive
        this result; `mode` then stays "full" (the layout's `base` is its mode)."""
        lines = [f"{key}: {value}" for key, value in self._assumptions.items()]
        lines += [f"warning: {text}" for text in self._all_warnings()]
        name = self._manifest["strategy"]["name"]
        bench = _result_report.benchmark_name(self._manifest)
        report = _result_report.Report(self.performance, name, lines, bench, self._quick)
        return _result_report.tearsheet(report, path, mode, log_scale, layout=layout, subject=self)


__all__ = ["AnalysisMethods"]
