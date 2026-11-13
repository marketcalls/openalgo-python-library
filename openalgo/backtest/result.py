"""`Result`: what `bt.backtest` returns.

The tables are pandas DataFrames built from the kernel's result columns. Every time is a
tz-aware IST datetime (the `time` column of each table, and `entry_time`, `exit_time`,
`decided_at`, `good_until`) and every date an IST date (`date` columns); the kernel's
UTC nanoseconds are kept only in the `*_ts` columns beside them. Instruments are named
by symbol. Prices and money are rupees, quantities units. Every
table has the same columns whether or not it has rows.

Built: `fills`, `orders`, `trades`, `equity`, `returns`, `positions` (the position
history), `events`, `risk_events`, `stale_marks`, `final_positions`, `gross_pnl`,
`costs`, `contract_notes`, `settlements` and `rolls` (futures), `performance`,
`metrics`, `metrics_table()`, `monte_carlo()`, `tearsheet()` (full, basic, metrics or
charts), `r_summary` (the R-multiples, `_result_r`), `assumptions`,
`warnings`, `skipped`, `manifest` (see `_manifest`), `rerun()` (see `_rerun`),
`to_dataframes()`, `to_csv()` and `to_excel()` (`_result_export`). Not supported in this
version: the several-strategy views `strategy()`, `combined` and `correlation`, which
raise NotImplementedError.
"""

from __future__ import annotations

import math
import os
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

from . import _rejections, _result_costs, _result_export, _result_futures, _result_r, _result_report
from ._messages import SEVERAL_STRATEGIES, not_supported
from ._native import native
from ._series import IST
from .errors import BacktestError
from .metrics import MonteCarlo, analyse, monte_carlo
from .report import _montecarlo, metrics_table
from .report._performance import Performance

# The keys an event can carry, in order: those of every kind of event the binding writes
# (`rust/oa_py/src/backtest/engine/tables.rs`, `event`), so the events table has the
# same columns whatever events a run had.
EVENT_COLUMNS = (
    "slice",
    "slice_ts",
    "step",
    "type",
    "name",
    "order_id",
    "quantity_units",
    "price_rupees",
    "reject_code",
    "instrument",
    "product",
    "position_quantity_units",
    "scope",
    "level_rupees",
    "target_index",
    "observed_rupees",
    "previous_state",
    "trading_state",
    "cause",
    "gtt_id",
    "stop_trigger_rupees",
    "target_trigger_rupees",
    "order_trigger_rupees",
    "fired_leg",
    "child_limit_rupees",
)
# The keys of a final position (`rust/oa_py/src/backtest/engine/results.rs`, `finals`).
FINAL_POSITION_COLUMNS = (
    "instrument",
    "product",
    "quantity_units",
    "average_price_rupees",
    "mark_rupees",
    "unrealized_rupees",
    "stale",
)
# The `Instrument.kind` of a futures contract, whose final position has no market value.
FUTURE_KIND = "future"
# The columns of the orders table that hold a string or nothing. A missing one is NaN in
# an object column, as the events table marks a missing key, whatever the other rows
# hold: left to pandas, a column of None alone stays None while one with a string turns
# None into NaN, so the marker would depend on the run.
ORDER_TEXT_COLUMNS = ("reject_code", "tag", "sizer")
# The `good_until` of a GTT order in the kernel (`exec::GTT_GOOD_UNTIL`, the largest
# int64): no session close ends it.
GTT_GOOD_UNTIL = 2**63 - 1


def _text(values: list[str | None]) -> pd.Series:
    """An object column of strings with NaN, pandas' missing value, for each None."""
    return pd.Series([math.nan if value is None else value for value in values], dtype=object)


def _times(values: Any, open_ended: int | None = None) -> pd.DatetimeIndex:
    """IST times of int64 UTC nanoseconds, at nanosecond resolution even when there are
    none (pandas reads an empty column at second resolution); `open_ended`, the
    kernel's stand-in for "no end", becomes NaT."""
    nanos = pd.Series(values, dtype="int64")
    if open_ended is not None:
        nanos = nanos.astype("Int64").mask(nanos == open_ended)
    utc = pd.to_datetime(nanos, unit="ns", utc=True)
    return pd.DatetimeIndex(utc).as_unit("ns").tz_convert(IST)


def _fixed(rows: Any, columns: tuple[str, ...]) -> pd.DataFrame:
    """A table with `columns` first, in order, then any key the rows have beyond them."""
    frame = pd.DataFrame(rows)
    extra = [name for name in frame.columns if name not in columns]
    return frame.reindex(columns=[*columns, *extra])


class Result:
    """The result of one backtest. See the module docstring."""

    def __init__(
        self,
        output: dict[str, Any],
        symbols: list[str],
        assumptions: dict[str, Any],
        warnings: list[str],
        skipped: list[tuple],
        manifest: dict[str, Any],
        record: Any = None,
        calendar: Any = None,
        benchmark: pd.Series | None = None,
    ) -> None:
        self._output = output
        self._calendar = calendar
        # Reference closes for the reports (`_benchmark`); they never entered the run.
        self._benchmark = benchmark
        self._symbols = symbols
        self._assumptions = assumptions
        schedules = manifest.get("cost_schedules", [])
        self._warnings = [*warnings, *_result_costs.run_warnings(output, schedules)]
        self._skipped = skipped
        self._manifest = manifest
        self._record = record

    # ----- tables --------------------------------------------------------------------

    def _table(self, name: str, time_column: str | None) -> pd.DataFrame:
        frame = pd.DataFrame(self._output[name])
        if "instrument" in frame.columns:
            frame.insert(0, "symbol", [self._symbols[i] for i in frame["instrument"]])
        if time_column is not None:
            frame.insert(0, "time", _times(frame[time_column]))
        return frame

    @property
    def fills(self) -> pd.DataFrame:
        """One row per fill: slice, step of the sequence, side, units, price, where in the
        bar (`fill_at`), reason, order id, target index, flags, charges."""
        return self._table("fills", "slice_ts")

    @property
    def orders(self) -> pd.DataFrame:
        """Every order with its final state and reject code. `time` and `decided_at` are
        the IST instant it was decided, `good_until` the IST instant its DAY validity
        ends (NaT for a GTT order, which no session close ends). `sizer` names what set
        the quantity: a `bt.size` sizer ("qty", "lots", "value", "percent",
        "fixed_risk", "atr_risk"), "function" for a sizing function or "default" for the
        engine default. `reject_code`, `tag` and `sizer` are NaN for an order without
        one (a quantity given in units, in lots or as a fraction of the position, as
        `qty=`, `lots=` and `exit()` give it, has no sizer); `tag` is "rebalance" on
        every order of `rebalance()`."""
        frame = self._table("orders", "decided_at")
        frame["decided_at"] = frame["time"]
        frame["good_until"] = _times(frame["good_until"], open_ended=GTT_GOOD_UNTIL)
        for name in ORDER_TEXT_COLUMNS:
            frame[name] = _text(self._output["orders"][name])
        return frame

    @property
    def trades(self) -> pd.DataFrame:
        """Closed round trips; positions open at the end are in `final_positions`.
        `initial_risk_rupees` is what the opening fill would have lost at the stop its
        entry set, and `r_multiple` the net P&L over it (NaN for a trade opened without
        a stop; `_result_r`)."""
        frame = self._table("trades", None)
        frame.insert(1, "entry_time", _times(frame["entry_ts"]))
        frame.insert(2, "exit_time", _times(frame["exit_ts"]))
        frame["initial_risk_rupees"] = frame["initial_risk_rupees"].astype(float)
        risk = frame["initial_risk_rupees"]
        frame["r_multiple"] = _result_r.r_values(frame["net_pnl_rupees"], risk)
        return frame

    @property
    def positions(self) -> pd.DataFrame:
        """The position history: a row at the end of every slice in which a position or
        its levels changed (in force from the next slice)."""
        return self._table("levels", "slice_ts")

    @property
    def equity(self) -> pd.DataFrame:
        """The equity series, one row per slice, indexed by the IST close of the slice."""
        columns = dict(self._output["equity"])
        index = pd.DatetimeIndex(_times(columns.pop("ts")), name="time")
        return pd.DataFrame(columns, index=index)

    @property
    def returns(self) -> pd.Series:
        """Simple returns of the equity series, slice to slice; the first slice is
        measured from the capital, so the P&L made in it counts."""
        capital = self._manifest["arguments"]["capital"]
        return _result_report.slice_returns(self.equity["equity_rupees"], capital)

    @property
    def events(self) -> pd.DataFrame:
        """The event stream: order, position and risk events in order of occurrence. A
        key an event does not carry is empty in its row."""
        frame = _fixed(self._output["events"], EVENT_COLUMNS)
        frame.insert(0, "time", _times(frame["slice_ts"]))
        return frame

    @property
    def risk_events(self) -> pd.DataFrame:
        """The risk events of the stream."""
        events = self.events
        return events[events["type"] == "risk"]

    @property
    def stale_marks(self) -> pd.DataFrame:
        """Open positions marked at a last close for want of a bar; `time` is the IST
        open of the slice."""
        columns = ("slice", "slice_ts", "instrument", "mark_rupees")
        frame = pd.DataFrame(self._output["stale_marks"], columns=list(columns))
        frame.insert(0, "symbol", [self._symbols[i] for i in frame["instrument"]])
        frame.insert(0, "time", _times(frame["slice_ts"]))
        return frame

    @property
    def final_positions(self) -> pd.DataFrame:
        """Every position at the end of the run, open ones marked at the last close.

        `value_rupees` is the signed market value at the mark (units x mark, negative
        for a short) and `weight_pct` that value in percent of the run's final equity,
        so a portfolio's end weights read off directly. Both are NaN for a future: it is
        not bought with cash (the account holds its margin, and its open profit or loss
        is already in equity), so its contract value, often several times the equity,
        would read as a share of the portfolio; `equity["exposure_rupees"]` counts it."""
        frame = _fixed(self._output["final_positions"], FINAL_POSITION_COLUMNS)
        frame.insert(0, "symbol", [self._symbols[i] for i in frame["instrument"]])
        kinds = [self._manifest["instruments"][i]["kind"] for i in frame["instrument"]]
        held = frame["quantity_units"] * frame["mark_rupees"]
        frame["value_rupees"] = held.mask(pd.Series(kinds, index=frame.index) == FUTURE_KIND)
        equity = self.equity["equity_rupees"]
        final_equity = float(equity.iloc[-1]) if len(equity) else math.nan
        frame["weight_pct"] = frame["value_rupees"] / final_equity * 100
        return frame

    @property
    def gross_pnl(self) -> dict[str, float]:
        """Realized, unrealized and total P&L before charges, rupees."""
        realized, unrealized, total = self._output["gross_pnl"]
        return {"realized": realized, "unrealized": unrealized, "total": total}

    # ----- context -------------------------------------------------------------------

    @property
    def assumptions(self) -> dict[str, Any]:
        """The settings the run used, defaults resolved."""
        return dict(self._assumptions)

    @property
    def warnings(self) -> list[str]:
        """What the reader of the result must know, most important first."""
        return list(self._warnings)

    @property
    def skipped(self) -> pd.DataFrame:
        """Orders a sizer skipped: the IST decision instant, symbol and reason."""
        frame = pd.DataFrame(self._skipped, columns=["decided_at", "symbol", "reason"])
        frame["decided_at"] = _times(frame["decided_at"])
        return frame

    @property
    def r_summary(self) -> dict[str, Any]:
        """The R-multiples of the closed trades (`trades["r_multiple"]`): how many, how
        many had an initial risk, their mean (the expectancy in R), median, standard
        deviation, best, worst, total and share above zero, and the distribution in
        one-R bins (`_result_r.r_summary`)."""
        return _result_r.r_summary(self.trades["r_multiple"])

    @property
    def manifest(self) -> dict[str, Any]:
        """Engine version, strategy and its source hash, every resolved argument, data
        partition hashes, reference-table and cost-schedule versions, random seed and
        platform: JSON types (`_manifest.RunManifest.from_dict` reads it back)."""
        return {"engine_version": native.engine_version(), **self._manifest}

    def to_dataframes(self) -> dict[str, pd.DataFrame]:
        """Every built table, by name."""
        names = (
            "fills",
            "orders",
            "trades",
            "positions",
            "equity",
            "events",
            "stale_marks",
            "costs",
            "contract_notes",
        )
        return {name: getattr(self, name) for name in names}

    def to_csv(self, folder: str | os.PathLike) -> list[Path]:
        """Write every table of `to_dataframes()` to `folder` (created when missing), one
        CSV file per table named after it (`fills.csv`, ...), and return the paths. Times
        are written as plain IST date-times, without the zone; the `*_ts` columns stay
        UTC nanoseconds (`_result_export`)."""
        return _result_export.to_csv(self.to_dataframes(), folder)

    def to_excel(self, path: str | os.PathLike) -> Path:
        """Write every table of `to_dataframes()` to one Excel workbook at `path` (its
        folder is created when missing), one sheet per table named after it, and return
        the path. Times are written as plain
        IST date-times, without the zone (a cell holds none); the `*_ts` columns stay UTC
        nanoseconds. Needs openpyxl, which openalgo does not install: raises
        MissingDependencyError, an ImportError, saying `pip install openpyxl` when it is
        missing."""
        return _result_export.to_excel(self.to_dataframes(), path)

    # ----- charges and reports --------------------------------------------------------

    @property
    def costs(self) -> pd.DataFrame:
        """What each fill was charged, rupees: the fill (time, symbol, product, side,
        units, price, reason), its IST trade date and cost segment, the six components
        (unrounded accruals in contract-note rounding), `cost` (their reported total),
        `dp_charge` (on the first delivery sell of a scrip in a day), `total`, the cost
        schedule version, whether its rates are verified for the date, and the trade id
        once the trade has closed. Empty for a run with costs=None; `attrs["rounding"]`
        is the rounding mode."""
        schedules = self._manifest.get("cost_schedules", [])
        return _result_costs.cost_frame(self._output, self._symbols, schedules)

    @property
    def contract_notes(self) -> pd.DataFrame:
        """One row per contract note (trade date and note group; contract-note rounding
        only): its charged lines and total, the accrued total, what its fills posted and
        the rounding adjustment posted at the session close, or at the run's end when the
        run ends before it. `settled` is True once the adjustment is posted.
        `attrs["rule"]` states the assumed contract-note rule."""
        return _result_costs.notes_frame(self._output)

    @property
    def settlements(self) -> pd.DataFrame:
        """The futures mark to market, the cash each futures position gains or loses: a
        row per position at each session close (`kind` "daily") and for a contract
        settled on its last trading day ("expiry"), with the units held after the close,
        the settlement price, `last_close_used` (no settlement price in the data, so the
        last close) and the cash flow in rupees. The postings sum to the position's
        gross P&L."""
        return _result_futures.settlements_frame(self._output, self._symbols)

    @property
    def rolls(self) -> pd.DataFrame:
        """Every switch of a futures series from one contract to the next: when, the
        two contracts, the spread that back-adjusts the series and whether positions
        rolled there (False with roll=None, where they are settled at expiry)."""
        return _result_futures.switches_frame(self._output, self._symbols)

    @cached_property
    def performance(self) -> Performance:
        """Every metric of the run (`openalgo.backtest.metrics.analyse`) over its equity
        series and trades, annualised by the traded exchange's calendar. The capital is
        the opening wealth and the open of the first slice the opening time, so the P&L
        of the first slice, the return of the first session and the time in the first
        slice (elapsed time, CAGR, exposure) count. A result built without its run's
        calendar times the opening at the first point. A run with `benchmark=` measures
        the benchmark figures (alpha, beta, information ratio, the comparison tables and
        charts) against its closes. Raises ConfigError when the run has no equity
        point."""
        trades = _result_report.trades_table(self._output, self._symbols)
        equity = _result_report.equity_series(self.equity)
        arguments = self._manifest["arguments"]
        exchange, capital = arguments["exchange"], arguments["capital"]
        opening = None
        if len(equity):
            first = int(self._output["equity"]["ts"][0])
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
        """The HTML tearsheet in `mode` ("full", "basic", "metrics" or "charts"; see
        `openalgo.backtest.report.tearsheet`): returned as a string, or written to `path`
        (and the path returned). `log_scale` draws the cumulative return on a log scale.
        Its Assumptions section lists the run's assumptions and every warning, so a
        cost-free run says so on the report."""
        lines = [f"{key}: {value}" for key, value in self._assumptions.items()]
        lines += [f"warning: {text}" for text in self._warnings]
        name = self._manifest["strategy"]["name"]
        report = _result_report.Report(self.performance, name, lines)
        return _result_report.tearsheet(report, path, mode, log_scale)

    # ----- running again, and what is not built yet ----------------------------------

    def rerun(self, **overrides: Any) -> Result:
        """The same backtest again with run arguments or strategy parameters changed;
        with none, this result bit for bit. Raises ConfigError for an unknown name and
        BacktestError for a result that was not made by `bt.backtest`."""
        if self._record is None:
            raise BacktestError("this result was not made by bt.backtest and cannot rerun")
        return self._record.rerun(overrides, self.manifest)

    def strategy(self, name: str) -> Result:
        """One strategy's result in a run of several; not supported in this version."""
        raise not_supported(*SEVERAL_STRATEGIES)

    @property
    def combined(self) -> Result:
        """The combined result of a run of several strategies; not supported in this
        version."""
        raise not_supported(*SEVERAL_STRATEGIES)

    @property
    def correlation(self) -> pd.DataFrame:
        """The correlation of a run's strategies; not supported in this version."""
        raise not_supported(*SEVERAL_STRATEGIES)

    def __repr__(self) -> str:
        pnl = self.gross_pnl
        lines = [f"Result({', '.join(self._symbols)})"]
        lines += [f"  {key}: {value}" for key, value in self._assumptions.items()]
        lines.append(f"  gross P&L: {pnl['total']:.2f} (realized {pnl['realized']:.2f})")
        charged = self.equity["costs_rupees"].iloc[-1] if len(self.equity) else 0.0
        lines.append(f"  charges: {charged:.2f}; net P&L: {pnl['total'] - charged:.2f}")
        rejected = _rejections.rejected_summary(self._output["orders"])
        lines += [] if rejected is None else [f"  {rejected}"]
        lines += [f"  warning: {text}" for text in self._warnings]
        return "\n".join(lines)


__all__ = ["Result"]
