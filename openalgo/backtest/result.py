"""`Result`: what `bt.backtest` returns (14-strategy-api section 5).

The tables are pandas DataFrames built from the kernel's result columns. Times are IST
(`time` columns, tz-aware) next to the kernel's UTC nanoseconds (`*_ts` columns);
instruments are named by symbol. Prices and money are rupees, quantities units.

Built: `fills`, `orders`, `trades`, `equity`, `returns`, `positions` (the position
history), `events`, `risk_events`, `stale_marks`, `final_positions`, `gross_pnl`,
`assumptions`, `warnings`, `skipped`, `manifest` (see `_manifest`), `rerun()` (see
`_rerun`), `to_dataframes()`. Not built yet: `metrics`, `costs`, `tearsheet()` (Phase 1
work), and the several-strategy views `strategy()`, `combined`, `correlation` (Phase 3).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ._native import native
from ._series import IST
from .errors import BacktestError

NOT_BUILT = "is not implemented yet (Phase 1 work in progress)"


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
        self._warnings = warnings
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
        names = ("fills", "orders", "trades", "positions", "equity", "events", "stale_marks")
        return {name: getattr(self, name) for name in names}

    # ----- not built yet --------------------------------------------------------------

    @property
    def metrics(self) -> Any:
        raise NotImplementedError(f"Result.metrics {NOT_BUILT}")

    @property
    def costs(self) -> Any:
        raise NotImplementedError(f"Result.costs {NOT_BUILT}")

    def tearsheet(self, path: str | None = None) -> Any:
        raise NotImplementedError(f"Result.tearsheet {NOT_BUILT}")

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
        lines += [f"  warning: {text}" for text in self._warnings]
        return "\n".join(lines)


__all__ = ["Result"]
