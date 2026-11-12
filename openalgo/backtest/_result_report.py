"""The reports of a `Result`: the metrics package's `Performance` over the run's equity
and trades, and the HTML tearsheet.

The kernel's tables are turned into the inputs `metrics.analyse` documents:

    equity   the equity series in rupees, indexed by the IST close of each slice. Its
             first point, the close of the first slice, can already hold P&L (an order
             a timer places at the first open fills there), so every point is a mark
             and the run's capital is passed as the opening wealth, with the open of
             the first slice as the opening time
    trades   one row per round trip: closed trades from the kernel's trades table, and
             each position still open at the end as an open trade (exit_time NaT) priced
             at its average price, so that exposure and turnover count it. Side is the
             side of the entry ("BUY" long, "SELL" short), quantity the units bought (or
             sold) to build it, net_pnl = gross_pnl - costs. The charges of each closed
             trade are broken down into the cost components and the DP charge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ._calendar_convert import interval_nanos
from ._result_costs import open_charges, trade_charges
from ._series import IST
from .errors import BacktestError
from .report import tearsheet as render_tearsheet
from .report._performance import Performance

_SIDES = {"long": "BUY", "short": "SELL"}


def _times(values: Any) -> pd.Series:
    nanos = pd.Series(values, dtype="int64")
    return pd.to_datetime(nanos, utc=True).dt.tz_convert(IST)


def closed_trades(output: dict[str, Any], symbols: list[str]) -> pd.DataFrame:
    """The kernel's closed trades in the metrics layout, with their cost breakdown."""
    raw = pd.DataFrame(output["trades"])
    if not len(raw):
        return pd.DataFrame()
    frame = pd.DataFrame(
        {
            "trade_id": raw["trade_id"],
            "symbol": [symbols[i] for i in raw["instrument"]],
            "product": raw["product"],
            "entry_time": _times(raw["entry_ts"]),
            "exit_time": _times(raw["exit_ts"]),
            "entry_price": raw["entry_price_rupees"],
            "exit_price": raw["exit_price_rupees"],
            "quantity": raw["entry_units"],
            "side": raw["direction"].map(_SIDES),
            "gross_pnl": raw["gross_pnl_rupees"],
            "costs": raw["costs_rupees"],
            "net_pnl": raw["net_pnl_rupees"],
            "exit_reason": raw["exit_reason"],
        }
    )
    charges = trade_charges(output)
    if len(charges):
        frame = frame.join(charges, on="trade_id")
    return frame


def _entry_time(output: dict[str, Any], instrument: int, product: str) -> int:
    """UTC nanoseconds of the fill that opened the position still held at the end."""
    levels = pd.DataFrame(output["levels"])
    rows = levels[(levels["instrument"] == instrument) & (levels["product"] == product)]
    flat = np.flatnonzero(rows["quantity_units"].to_numpy() == 0.0)
    start = flat[-1] + 1 if len(flat) else 0
    return int(rows["slice_ts"].iloc[start])


def open_trades(output: dict[str, Any], symbols: list[str]) -> pd.DataFrame:
    """Positions still open at the end, as open trades in the metrics layout."""
    rows = []
    for position in output["final_positions"]:
        units = position["quantity_units"]
        if units == 0.0:
            continue
        instrument, product = position["instrument"], position["product"]
        costs = open_charges(output, instrument, product)
        gross = position["unrealized_rupees"]
        rows.append(
            {
                "symbol": symbols[instrument],
                "product": product,
                "entry_time": _times([_entry_time(output, instrument, product)]).iloc[0],
                "exit_time": pd.NaT,
                "entry_price": position["average_price_rupees"],
                "exit_price": np.nan,
                "quantity": abs(units),
                "side": "BUY" if units > 0 else "SELL",
                "gross_pnl": gross,
                "costs": costs,
                "net_pnl": gross - costs,
                "exit_reason": None,
            }
        )
    return pd.DataFrame(rows)


def trades_table(output: dict[str, Any], symbols: list[str]) -> pd.DataFrame | None:
    """Closed and open trades in the metrics layout, or None when there are none."""
    parts = [
        part for part in (closed_trades(output, symbols), open_trades(output, symbols)) if len(part)
    ]
    if not parts:
        return None
    frame = pd.concat(parts, ignore_index=True)
    frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True).dt.tz_convert(IST)
    return frame


def equity_series(equity: pd.DataFrame) -> pd.Series:
    """The equity column of `Result.equity`, named for the metrics package."""
    return equity["equity_rupees"].rename("equity")


def slice_returns(equity: pd.Series, capital: float) -> pd.Series:
    """Simple returns of an equity series slice to slice, the first slice measured
    from the capital (the equity before it)."""
    before = equity.shift(1).fillna(capital)
    return (equity / before - 1.0).rename("returns")


def opening_time(first_close_ns: int, base: str, exchange: str, calendar: Any) -> int | None:
    """UTC nanoseconds of the open of the run's first slice, the instant the run opened
    with its capital; None without the run's calendar.

    A slice is one base bar on the session grid: with the "D" base the whole session,
    so it opens at the session's open; with an intraday base of step `s` it is the bar
    whose slot holds its close, so it opens at `open + floor((close - 1 - open) / s) s`
    (a session's last bar may be shorter than a step). Raises BacktestError when the
    first close is not in a session of the calendar, which a run never produces.
    """
    if calendar is None:
        return None
    day = pd.Timestamp(first_close_ns, tz="UTC").tz_convert(IST).date()
    session = calendar.session(exchange, day)
    if session is None:
        raise BacktestError(
            f"the run's first slice closes on {day}, which is not a {exchange} session of "
            "its calendar: the result does not match the calendar it was run on"
        )
    session_open = int(session[0])
    step = interval_nanos(base)
    if step is None:
        return session_open
    return session_open + ((first_close_ns - 1 - session_open) // step) * step


@dataclass(frozen=True)
class Report:
    """What the tearsheet reads: the performance, a title and the assumption lines."""

    performance: Performance
    name: str
    assumptions: list[str]


def tearsheet(report: Report, path: str | None, mode: str, log_scale: bool) -> str:
    """The HTML tearsheet of a run in `mode` (returned, or written to `path`)."""
    return render_tearsheet(report, path, mode=mode, log_scale=log_scale)


__all__ = [
    "Report",
    "closed_trades",
    "equity_series",
    "open_trades",
    "opening_time",
    "slice_returns",
    "tearsheet",
    "trades_table",
]
