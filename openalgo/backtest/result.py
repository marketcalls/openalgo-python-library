"""`Result`: what `bt.backtest` returns (14-strategy-api section 5).

The tables are pandas DataFrames built from the kernel's result columns. Times are IST
(`time` columns, tz-aware) next to the kernel's UTC nanoseconds (`*_ts` columns);
instruments are named by symbol. Prices and money are rupees, quantities units.

Built: `fills`, `orders`, `trades`, `equity`, `returns`, `positions` (the position
history), `events`, `risk_events`, `stale_marks`, `final_positions`, `gross_pnl`,
`costs`, `contract_notes`, `settlements` and `rolls` (futures), `performance`,
`metrics`, `tearsheet()`, `assumptions`,
`warnings`, `skipped`, `manifest` (see `_manifest`), `rerun()` (see `_rerun`),
`to_dataframes()`. Not built yet: the several-strategy views `strategy()`, `combined`,
`correlation` (Phase 3).
"""

from __future__ import annotations

from functools import cached_property
from typing import Any

import pandas as pd

from . import _result_costs, _result_futures, _result_report
from ._native import native
from ._series import IST
from .errors import BacktestError
from .metrics import analyse
from .report._performance import Performance


def _times(values: Any) -> pd.DatetimeIndex:
    return pd.to_datetime(pd.Series(values, dtype="int64"), utc=True).dt.tz_convert(IST)


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
    ) -> None:
        self._output = output
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
        if time_column is not None and len(frame):
            frame.insert(0, "time", _times(frame[time_column]).to_numpy())
        return frame

    @property
    def fills(self) -> pd.DataFrame:
        """One row per fill: slice, step of the sequence, side, units, price, where in the
        bar (`fill_at`), reason, order id, target index, flags, charges."""
        return self._table("fills", "slice_ts")

    @property
    def orders(self) -> pd.DataFrame:
        """Every order with its final state and reject code."""
        return self._table("orders", "decided_at")

    @property
    def trades(self) -> pd.DataFrame:
        """Closed round trips; positions open at the end are in `final_positions`."""
        frame = self._table("trades", None)
        if len(frame):
            frame.insert(1, "entry_time", _times(frame["entry_ts"]).to_numpy())
            frame.insert(2, "exit_time", _times(frame["exit_ts"]).to_numpy())
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
        """Simple returns of the equity series, slice to slice."""
        return self.equity["equity_rupees"].pct_change().fillna(0.0).rename("returns")

    @property
    def events(self) -> pd.DataFrame:
        """The event stream: order, position and risk events in order of occurrence."""
        frame = pd.DataFrame(self._output["events"])
        if len(frame):
            frame.insert(0, "time", _times(frame["slice_ts"]).to_numpy())
        return frame

    @property
    def risk_events(self) -> pd.DataFrame:
        """The risk events of the stream."""
        events = self.events
        return events[events["type"] == "risk"] if len(events) else events

    @property
    def stale_marks(self) -> pd.DataFrame:
        """Open positions marked at a last close for want of a bar."""
        columns = ("slice", "slice_ts", "instrument", "mark_rupees")
        frame = pd.DataFrame(self._output["stale_marks"], columns=list(columns))
        frame.insert(0, "symbol", [self._symbols[i] for i in frame["instrument"]])
        return frame

    @property
    def final_positions(self) -> pd.DataFrame:
        """Every position at the end of the run, open ones marked at the last close."""
        frame = pd.DataFrame(self._output["final_positions"])
        if len(frame):
            frame.insert(0, "symbol", [self._symbols[i] for i in frame["instrument"]])
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
        """Orders a sizer skipped: decision instant, symbol and reason."""
        return pd.DataFrame(self._skipped, columns=["decided_at", "symbol", "reason"])

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
        the rounding adjustment posted at the session close. `settled` is False for a
        note whose session close the run did not reach. `attrs["rule"]` states the
        assumed contract-note rule."""
        return _result_costs.notes_frame(self._output)

    @property
    def settlements(self) -> pd.DataFrame:
        """The futures mark to market (04-execution 7): a row per futures position at
        each session close (`kind` "daily") and for a contract settled on its last
        trading day ("expiry"), with the units held after the close, the settlement
        price, `last_close_used` (no settlement price in the data, so the last close)
        and the cash flow in rupees. The postings sum to the position's gross P&L."""
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
        the opening wealth, so the P&L of the first slice and the return of the first
        session count. Raises ConfigError when the run has no equity point."""
        trades = _result_report.trades_table(self._output, self._symbols)
        equity = _result_report.equity_series(self.equity)
        exchange = self._manifest["arguments"]["exchange"]
        capital = self._manifest["arguments"]["capital"]
        return analyse(equity, exchanges=exchange, trades=trades, opening_wealth=capital)

    @property
    def metrics(self) -> Any:
        """The metrics mapping of `performance`; keys, units and definitions are in
        `openalgo.backtest.metrics.DEFINITIONS`."""
        return self.performance.metrics

    def tearsheet(self, path: str | None = None) -> str:
        """The HTML tearsheet: returned as a string, or written to `path` (and the path
        returned). Its Assumptions section lists the run's assumptions and every warning,
        so a cost-free run says so on the report."""
        lines = [f"{key}: {value}" for key, value in self._assumptions.items()]
        lines += [f"warning: {text}" for text in self._warnings]
        name = self._manifest["strategy"]["name"]
        report = _result_report.Report(self.performance, name, lines)
        return _result_report.tearsheet(report, path)

    # ----- running again, and what is not built yet ----------------------------------

    def rerun(self, **overrides: Any) -> Result:
        """The same backtest again with run arguments or strategy parameters changed;
        with none, this result bit for bit. Raises ConfigError for an unknown name and
        BacktestError for a result that was not made by `bt.backtest`."""
        if self._record is None:
            raise BacktestError("this result was not made by bt.backtest and cannot rerun")
        return self._record.rerun(overrides, self.manifest)

    def strategy(self, name: str) -> Result:
        raise NotImplementedError("several strategies in one run are planned for Phase 3")

    @property
    def combined(self) -> Result:
        raise NotImplementedError("several strategies in one run are planned for Phase 3")

    @property
    def correlation(self) -> pd.DataFrame:
        raise NotImplementedError("several strategies in one run are planned for Phase 3")

    def __repr__(self) -> str:
        pnl = self.gross_pnl
        lines = [f"Result({', '.join(self._symbols)})"]
        lines += [f"  {key}: {value}" for key, value in self._assumptions.items()]
        lines.append(f"  gross P&L: {pnl['total']:.2f} (realized {pnl['realized']:.2f})")
        charged = self.equity["costs_rupees"].iloc[-1] if len(self.equity) else 0.0
        lines.append(f"  charges: {charged:.2f}; net P&L: {pnl['total'] - charged:.2f}")
        lines += [f"  warning: {text}" for text in self._warnings]
        return "\n".join(lines)


__all__ = ["Result"]
