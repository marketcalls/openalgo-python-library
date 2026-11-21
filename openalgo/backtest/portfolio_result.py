"""`PortfolioResult`: what `bt.backtest` returns for a list of strategies run on one account.

Every strategy keeps its own `Result` (`results`, `strategy(name)`), built from its engine
as a run of it alone is built, so its figures are measured on its own capital. The account
is the strategies together: its rows (`account`: equity, free cash, margin and daily loss
at every round of the run), its equity and returns on its funds, its `performance` and
metrics, every strategy's tables in one (a `strategy` column first), the orders the
account refused, the net position of each instrument (what a broker holds), the
correlation of the strategies' session returns and each one's contribution, and its
reports: `summary()` and the combined `tearsheet()` (`_portfolio_report`).

Conventions are `Result`'s: tz-aware IST times beside the kernel's UTC nanoseconds
(`*_ts` columns), rupees, quantities in units. Kernel ids restart in every strategy's
engine (each one's first order is 1), so a row of a combined table is identified by
(`strategy`, the id). Rows are in time order, those of one instant in call order.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd

from . import _portfolio_notes, _portfolio_report, _portfolio_tables
from ._native import native
from ._portfolio_figures import account_performance, contribution
from ._result_export import EXPORTED_TABLES, write_csv, write_excel
from ._result_report import slice_returns
from ._strategy_args import Account
from .metrics import MonteCarlo, monte_carlo, return_correlation
from .report import _montecarlo, metrics_table
from .report._performance import Performance
from .report.custom import Layout
from .result import Result

#: The account's tables `to_dataframes` returns first, in order: the properties of these
#: names; then every strategy's own tables (`EXPORTED_TABLES`), prefixed by its name.
PORTFOLIO_TABLES = (
    "account",
    "contribution",
    "refused",
    "net_positions",
    "trades",
    "fills",
    "orders",
    "events",
    "costs",
    "contract_notes",
    "final_positions",
)
#: The letter of a strategy's sheet prefix in a workbook (`to_excel`): S1, S2, ...
STRATEGY_SHEET = "S"


class PortfolioResult:
    """The result of a portfolio run (see the module docstring). `account` is the run's
    checked account, `results` each strategy's `Result` by class name in call order,
    `records` the native fund's records (`Fund.finish`), `benchmark` the closes the
    account is compared with (the first strategy's benchmark; None without one) and
    `run_warnings` the run's own warnings (a setting no strategy used)."""

    def __init__(
        self,
        account: Account,
        results: Mapping[str, Result],
        records: Mapping[str, Any],
        *,
        benchmark: pd.Series | None = None,
        run_warnings: Sequence[str] = (),
    ) -> None:
        self._account = account
        self._run_warnings = list(run_warnings)
        # The account's funds in rupees, as the fund kept them (the sum of the capitals to
        # the paisa when no account_capital was given).
        self._funds = float(records["account_capital"])
        self._results = dict(results)
        self._records = records
        self._benchmark = benchmark

    # ----- the strategies -------------------------------------------------------------

    @property
    def strategies(self) -> tuple[str, ...]:
        """The strategy names (class names), in call order."""
        return tuple(self._results)

    @property
    def results(self) -> Mapping[str, Result]:
        """Each strategy's own `Result` by name, in call order, measured on its own
        capital; its Assumptions block says the account it ran in."""
        return MappingProxyType(self._results)

    def strategy(self, name: str) -> Result:
        """The `Result` of the strategy `name`. Raises KeyError naming the strategies for
        a name the run does not have."""
        if name not in self._results:
            raise KeyError(f"no strategy {name!r} in this run; it has {list(self._results)}")
        return self._results[name]

    # ----- the account ----------------------------------------------------------------

    @cached_property
    def _rows(self) -> pd.DataFrame:
        return _portfolio_tables.account_frame(self._records)

    @property
    def account(self) -> pd.DataFrame:
        """The account's row of every round of the run (each instant at which a
        strategy's slice closed), indexed by its IST time: `equity_rupees` (the account's
        funds less the capitals plus every strategy's equity), `free_cash_rupees`,
        `margin_rupees` and `daily_loss_rupees` (the strategies' daily loss measures of
        the IST session day, added up; the figure `account_max_daily_loss` is held to)."""
        return self._rows.copy()

    @property
    def equity(self) -> pd.Series:
        """The account's equity in rupees, by the IST time of each round."""
        return self._rows["equity_rupees"].copy()

    @property
    def returns(self) -> pd.Series:
        """Simple returns of the account's equity, round to round, as `Result.returns` are
        slice to slice; the first round is measured from the account's funds, so the P&L
        made in it counts. The account's session returns are
        `performance.session_returns`."""
        return slice_returns(self._rows["equity_rupees"], self._funds)

    @cached_property
    def performance(self) -> Performance:
        """Every metric of the account (`openalgo.backtest.metrics.analyse`) over its
        equity, opened with its funds at the earliest open of a strategy's first slice,
        with every strategy's trades (a `strategy` column naming each), annualised by the
        strategies' exchanges' calendars and compared with the first strategy's
        benchmark. Raises ConfigError when the run has no equity point."""
        return account_performance(
            self._rows["equity_rupees"], self._funds, self._results, self._benchmark
        )

    @property
    def metrics(self) -> Any:
        """The metrics mapping of `performance` (`openalgo.backtest.metrics.DEFINITIONS`)."""
        return self.performance.metrics

    def metrics_table(self) -> pd.DataFrame:
        """Every metric of the account as a DataFrame indexed by key: group, label, value,
        unit and definition (`openalgo.backtest.report.metrics_table`)."""
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
        """Monte Carlo of the account by bootstrap of its session returns or of every
        strategy's closed trades (`openalgo.backtest.metrics.monte_carlo` documents the
        arguments and errors)."""
        return monte_carlo(
            self.performance,
            simulations=simulations,
            seed=seed,
            resample=resample,
            bust=bust,
            goal=goal,
        )

    def summary(self) -> str:
        """`Result.summary()` for the account: its net profit in rupees, return, maximum
        drawdown, closed trades with their win rate and the positions still open, its
        benchmark's return when it has one, and the quick overfitting checks of each
        strategy's own trades on a line each, none of the trades together
        (`_portfolio_report`). Raises ConfigError when the run has no equity point."""
        return _portfolio_report.summary(self)

    def tearsheet(
        self,
        path: str | os.PathLike | None = None,
        *,
        mode: str = "full",
        log_scale: bool = False,
        layout: Layout | None = None,
    ) -> str:
        """The combined HTML tearsheet: the account's page in `mode` (see
        `openalgo.backtest.report.tearsheet`) with the Strategies section, each strategy's
        quick overfitting checks in a row of their own, and the Portfolio section (each
        strategy's contribution and its P&L by month, the account's margin and free cash,
        the orders it refused and the net position of each instrument two strategies
        held; full mode). Returned as a string, or written to `path` and the path
        returned. `layout` (a `report.custom.Layout`) makes a custom page whose blocks'
        callables receive this portfolio. Each strategy's own page is
        `strategy(name).tearsheet()`. Raises ConfigError as `report.tearsheet` does."""
        return _portfolio_report.tearsheet(
            self, path, mode=mode, log_scale=log_scale, layout=layout
        )

    # ----- the strategies' tables together -----------------------------------------------

    @property
    def trades(self) -> pd.DataFrame:
        """Every strategy's closed trades (`Result.trades`), in the order they closed."""
        return _portfolio_tables.combined(self._results, "trades")

    @property
    def fills(self) -> pd.DataFrame:
        """Every strategy's fills (`Result.fills`), in time order: by slice, then by the
        step of the slice's sequence they filled at."""
        return _portfolio_tables.combined(self._results, "fills")

    @property
    def orders(self) -> pd.DataFrame:
        """Every strategy's orders (`Result.orders`), by the instant they were decided;
        an order the account refused has its `reject_code`."""
        return _portfolio_tables.combined(self._results, "orders")

    @property
    def events(self) -> pd.DataFrame:
        """Every strategy's event stream (`Result.events`), by slice and step."""
        return _portfolio_tables.combined(self._results, "events")

    @property
    def risk_events(self) -> pd.DataFrame:
        """The risk events of `events`: an account breach is `account_loss_hit` in every
        strategy's stream."""
        events = self.events
        return events[events["type"] == "risk"]

    @property
    def costs(self) -> pd.DataFrame:
        """What each fill of every strategy was charged (`Result.costs`), in time order."""
        return _portfolio_tables.combined(self._results, "costs")

    @property
    def contract_notes(self) -> pd.DataFrame:
        """Every strategy's contract notes (`Result.contract_notes`), by trade date: each
        strategy's fills are charged on notes of its own."""
        return _portfolio_tables.combined(self._results, "contract_notes")

    @property
    def final_positions(self) -> pd.DataFrame:
        """Every strategy's positions at the end of its run (`Result.final_positions`)."""
        return _portfolio_tables.combined(self._results, "final_positions")

    def trade_list(self) -> pd.DataFrame:
        """Every strategy's `trade_list()` in one, indexed by (strategy, trade), in the
        order the trades closed."""
        return _portfolio_tables.combined_trade_list(self._results)

    @cached_property
    def _net(self) -> tuple[pd.DataFrame, list[_portfolio_tables.OppositeSides]]:
        return _portfolio_tables.net_positions(self._results)

    @property
    def net_positions(self) -> pd.DataFrame:
        """The broker's view: per instrument and product, the strategies' units over time
        (`net_units`, their sum; `holders`; a `<strategy>_units` column each), a row for
        each slice in which a strategy's units changed (see
        `_portfolio_tables.net_positions`)."""
        return self._net[0].copy()

    @cached_property
    def _refused(self) -> pd.DataFrame:
        return _portfolio_tables.refused_frame(self._records, self._results)

    @property
    def refused(self) -> pd.DataFrame:
        """The orders the account refused or ended, in time order: strategy, order id (in
        that strategy's engine), symbol, side, time, slice, slice_ts, step and reason
        ("account_margin": the account's free cash did not cover the margin;
        "account_loss_gate": an entry after the account's daily loss limit was reached;
        "account_loss": a working entry the limit's breach cancelled;
        "account_profit_gate" and "account_profit": the same for the account's daily
        profit target; "account_trade_limit": an entry once the account's entries of the
        day reached its trade cap)."""
        return self._refused.copy()

    @property
    def correlation(self) -> pd.DataFrame:
        """The Pearson correlation of the strategies' session returns
        (`openalgo.backtest.metrics.return_correlation`). Raises ConfigError for a run of
        one strategy."""
        return return_correlation({name: r.performance for name, r in self._results.items()})

    @property
    def contribution(self) -> pd.DataFrame:
        """Each strategy's part in the account, indexed by strategy name: its net P&L, its
        share of the account's in percent, its P&L inside the account's maximum drawdown,
        its margin when the account's was highest and its orders the account refused
        (`_portfolio_figures.contribution` defines each column)."""
        return contribution(self._rows, self._funds, self._results, self._refused)

    # ----- context --------------------------------------------------------------------

    @property
    def assumptions(self) -> dict[str, Any]:
        """The run's settings in words: the account (pool, funds, what the pool lets one
        strategy do to another, the daily loss limit), the instruments several strategies
        trade, what a backtest does there that a broker does not and those held on
        opposite sides, and a line per strategy. Each strategy's own block is in
        `strategy(name).assumptions`."""
        account, opposite = self._account, self._net[1]
        return _portfolio_notes.assumptions(account, self._funds, self._results, opposite)

    @property
    def warnings(self) -> list[str]:
        """What the reader must know: the account's own warnings first (an instrument held
        on opposite sides; days a broker would have netted the charges of several
        strategies; a run setting no strategy used), then every strategy's
        (`Result.warnings`), each after its name."""
        notes = [_portfolio_notes.opposite_sides_warning(found) for found in self._net[1]]
        notes += _portfolio_notes.netting_warnings(self._results)
        notes += self._run_warnings
        for name, result in self._results.items():
            notes += [f"{name}: {text}" for text in result.warnings]
        return notes

    @property
    def manifest(self) -> dict[str, Any]:
        """The engine version, the account's settings and every strategy's manifest by
        name (`Result.manifest` without the engine version): JSON types."""
        account = self._account
        return {
            "engine_version": native.engine_version(),
            "account": {
                "pool": account.pool,
                "account_capital": self._funds,
                "account_max_daily_loss": account.account_max_daily_loss,
            },
            "strategies": {
                name: {k: v for k, v in result.manifest.items() if k != "engine_version"}
                for name, result in self._results.items()
            },
        }

    # ----- export ---------------------------------------------------------------------

    def _account_tables(self) -> dict[str, pd.DataFrame]:
        """The account's tables by name (`PORTFOLIO_TABLES`), `contribution` with its
        strategy names as a column."""
        tables = {name: getattr(self, name) for name in PORTFOLIO_TABLES}
        tables["contribution"] = tables["contribution"].reset_index()
        return tables

    def _strategy_tables(self, prefixes: Sequence[str]) -> dict[str, pd.DataFrame]:
        """Each strategy's own tables (`EXPORTED_TABLES`) named `<prefix>_<table>`, with
        `prefixes` one per strategy in call order."""
        return {
            f"{prefix}_{table}": getattr(result, table)
            for prefix, result in zip(prefixes, self._results.values(), strict=True)
            for table in EXPORTED_TABLES
        }

    def to_dataframes(self) -> dict[str, pd.DataFrame]:
        """Every table by name: the account's (`PORTFOLIO_TABLES`; `contribution` with its
        strategy names as a column), then each strategy's own tables (`Result
        .to_dataframes`) named `<strategy>_<table>`."""
        return self._account_tables() | self._strategy_tables(list(self._results))

    def to_csv(self, folder: str | os.PathLike) -> list[Path]:
        """Write every table of `to_dataframes()` to `folder` (created when missing), one
        CSV file each named after it, and return the paths. Times are plain IST
        date-times; the `*_ts` columns stay UTC nanoseconds."""
        return write_csv(self.to_dataframes(), folder)

    def to_excel(self, path: str | os.PathLike) -> Path:
        """Write every table of `to_dataframes()` to one Excel workbook at `path`, a sheet
        each, and return the path. A sheet name holds at most 31 characters, too few for
        a strategy's name and a table's, so the account's tables keep their names, a
        `strategies` sheet gives each strategy a prefix by its place in call order (S1 for
        the first) and its tables are the sheets `<prefix>_<table>` (`S1_fills`). Needs
        openpyxl, which openalgo does not install: raises MissingDependencyError (an
        ImportError) when it is missing."""
        prefixes = [f"{STRATEGY_SHEET}{place}" for place in range(1, len(self._results) + 1)]
        index = pd.DataFrame({"sheet_prefix": prefixes, "strategy": list(self._results)})
        sheets = self._account_tables() | {"strategies": index}
        return write_excel(sheets | self._strategy_tables(prefixes), path)

    def __repr__(self) -> str:
        account = self._account
        equity = self._rows["equity_rupees"]
        net = float(equity.iloc[-1]) - self._funds if len(equity) else 0.0
        lines = [
            f"PortfolioResult({', '.join(self._results)})",
            f"  pool: {account.pool}; account: {_portfolio_notes.rupees_text(self._funds)}",
            f"  net P&L: {net:,.2f}",
            f"  orders the account refused or ended: {len(self._refused)}",
        ]
        lines += [f"  warning: {text}" for text in self.warnings]
        return "\n".join(lines)


__all__ = ["PortfolioResult"]
