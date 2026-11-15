"""`Result.trade_list()`: the closed trades as a trader reads them, one row each.

Built from a trades table in the layout of the metrics package (`metrics.analyse`
documents it; `_result_report.closed_trades` makes it from a run, and
`Performance.trades` holds it), so the trade list of a `Result` and the trade table of
the tearsheet are the same table. Open trades (no exit time) are left out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: The columns of the trade list, in order (see `trade_list`).
TRADE_LIST_COLUMNS = (
    "symbol",
    "direction",
    "product",
    "entry_time",
    "exit_time",
    "quantity",
    "entry_price",
    "exit_price",
    "gross_pnl",
    "charges",
    "net_pnl",
    "return_pct",
    "bars_held",
    "days_held",
    "exit_reason",
    "r_multiple",
)
#: The side of a trade's entry, as the metrics layout writes it, and its direction.
DIRECTIONS = {"BUY": "long", "SELL": "short"}
SECONDS_PER_DAY = 86_400
PERCENT = 100.0


def _optional(frame: pd.DataFrame, name: str, missing: object = np.nan) -> pd.Series:
    """Column `name` of `frame`, or `missing` in every row when the table has none."""
    if name in frame.columns:
        return frame[name]
    return pd.Series(missing, index=frame.index, dtype=object if missing is None else float)


def trade_list(trades: pd.DataFrame | None) -> pd.DataFrame:
    """The closed trades of `trades` (the metrics layout), one row each, in the table's
    order (a run's trades are numbered in the order they closed), indexed by trade
    number ("trade": the run's trade id, else 1, 2, ... in table order).

    Columns:
        symbol        the instrument (NaN when the table names none)
        direction     "long" or "short"
        product       CNC, MIS or NRML (NaN when the table names none)
        entry_time    IST time of the fill that opened the trade
        exit_time     IST time of the fill that closed it
        quantity      units bought (long) or sold (short) to build the position, adds
                      included; never lots
        entry_price   quantity-weighted average price of the opening fills, rupees
        exit_price    quantity-weighted average price of the closing fills, rupees
        gross_pnl     P&L before charges, rupees
        charges       charges of every fill of the trade, rupees
        net_pnl       gross_pnl - charges, rupees
        return_pct    net_pnl over the value of the entry (entry_price x quantity), in
                      percent: 10.0 is ten percent; for a short, of the value sold
        bars_held     bars of the strategy's interval that closed while the trade was
                      open, after entry_time and up to exit_time (a daily strategy:
                      sessions; `_result_report.strategy_bars_held`); NA when the
                      table does not record bars
        days_held     calendar days from entry_time to exit_time, fractional
        exit_reason   why the closing fill happened ("signal", "stop", "target", ...)
        r_multiple    net_pnl over the trade's initial risk; NaN without a stop

    The columns are the same, with no rows, when there are no closed trades or no
    trades table (None).
    """
    if trades is None or trades.empty:
        empty = pd.DataFrame(columns=list(TRADE_LIST_COLUMNS))
        empty.index.name = "trade"
        return empty
    closed = trades[trades["exit_time"].notna()]
    quantity = closed["quantity"].astype(float)
    entry_price = closed["entry_price"].astype(float)
    held = closed["exit_time"] - closed["entry_time"]
    frame = pd.DataFrame(
        {
            "symbol": _optional(closed, "symbol"),
            "direction": closed["side"].map(DIRECTIONS),
            "product": _optional(closed, "product"),
            "entry_time": closed["entry_time"],
            "exit_time": closed["exit_time"],
            "quantity": quantity,
            "entry_price": entry_price,
            "exit_price": closed["exit_price"].astype(float),
            "gross_pnl": closed["gross_pnl"].astype(float),
            "charges": closed["costs"].astype(float),
            "net_pnl": closed["net_pnl"].astype(float),
            "return_pct": closed["net_pnl"].astype(float) / (entry_price * quantity) * PERCENT,
            "bars_held": _optional(closed, "bars_held").astype("Int64"),
            "days_held": held.dt.total_seconds() / SECONDS_PER_DAY,
            "exit_reason": _optional(closed, "exit_reason"),
            "r_multiple": _optional(closed, "r_multiple").astype(float),
        },
        columns=list(TRADE_LIST_COLUMNS),
    )
    if "trade_id" in closed.columns:
        frame.index = pd.Index(closed["trade_id"].astype("int64"), name="trade")
    else:
        frame.index = pd.RangeIndex(1, len(frame) + 1, name="trade")
    return frame


__all__ = ["DIRECTIONS", "TRADE_LIST_COLUMNS", "trade_list"]
