"""What the trade-list checks read from a run: its closed trades and its sessions.

`TradeRun` is built from a `Performance` (`Result.performance`, or any
`metrics.analyse` output), so the checks need neither the bars nor a rerun:

    pnl             the closed trades' net P&L in rupees, in exit order (then entry
                    order), as `metrics.monte_carlo(resample="trades")` books them
    entry_session   the position in `sessions` of each closed trade's entry session: the
                    IST date of its entry, or, for a trade entered on a date without a
                    session of the run (only possible in a hand-built table), the last
                    session before it (the first session for one entered before it)
    sessions        the run's session dates (`Performance.session_returns`)
    capital         the start equity in rupees: the run's capital for a `Result`
    cumulative      the run's cumulative return at each session close, a fraction
    prices          (symbol, its close at each session close in rupees) of the run's
                    first symbol, the close of its last strategy bar of the session, when
                    the report has the run's bars (`_run_bars`); None otherwise

Positions still open at the end are not trades yet; `open_trades` counts them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .._series import IST
from ..report._dashboard.data import cumulative
from ..report._montecarlo import closed_by_exit
from ..report._performance import Performance


@dataclass(frozen=True, eq=False)
class TradeRun:
    """A run's closed trades and sessions (module docstring). `sessions_per_year` and
    `risk_free_rate` annualise as the run's metrics do; `elapsed_nanos` is the run's
    elapsed time, start to end."""

    pnl: np.ndarray
    entry_session: np.ndarray
    sessions: pd.DatetimeIndex
    capital: float
    sessions_per_year: float
    risk_free_rate: float
    elapsed_nanos: int
    open_trades: int
    cumulative: np.ndarray
    performance: Performance
    prices: tuple[str, np.ndarray] | None = None

    @property
    def closed(self) -> int:
        """The number of closed trades."""
        return len(self.pnl)

    @property
    def win_rate(self) -> float:
        """Winning trades (net P&L above zero) over closed trades; NaN without any."""
        return float(np.mean(self.pnl > 0)) if len(self.pnl) else float("nan")


def _entry_sessions(entries: pd.Series, sessions: pd.DatetimeIndex) -> np.ndarray:
    """The position in `sessions` of each entry's IST date (module docstring)."""
    if not len(entries):
        return np.empty(0, dtype=np.int64)
    days = pd.to_datetime(entries, utc=True).dt.tz_convert(IST).dt.tz_localize(None)
    dates = days.dt.normalize().to_numpy(dtype="datetime64[D]")
    known = sessions.to_numpy(dtype="datetime64[D]")
    found = np.searchsorted(known, dates, side="right") - 1
    return np.maximum(found, 0).astype(np.int64)


def trade_run(performance: Performance) -> TradeRun:
    """The `TradeRun` of `performance`."""
    trades = performance.trades
    closed = closed_by_exit(trades)
    sessions = performance.session_returns.index
    metrics = performance.metrics
    pnl = closed["net_pnl"].to_numpy(dtype=float) if len(closed) else np.empty(0)
    entries = closed["entry_time"] if len(closed) else pd.Series([], dtype="int64")
    open_count = int(pd.isna(trades["exit_time"]).sum()) if len(trades) else 0
    return TradeRun(
        pnl=pnl,
        entry_session=_entry_sessions(entries, sessions),
        sessions=sessions,
        capital=float(metrics["start_equity"]),
        sessions_per_year=float(metrics["sessions_per_year"]),
        risk_free_rate=float(metrics["risk_free_rate"]),
        elapsed_nanos=int((metrics["end"] - metrics["start"]).value),
        open_trades=open_count,
        cumulative=cumulative(performance.session_returns).to_numpy(dtype=float),
        performance=performance,
    )


__all__ = ["TradeRun", "trade_run"]
