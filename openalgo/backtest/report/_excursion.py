"""The excursion and trading-cost figures of the closed trades: average, worst and best
MAE and MFE, the edge ratio, the profit winners kept, their turnover, the slippage and
the trading cost (definitions in `DEFINITIONS`). The trading cost counts the closed
trades only, so its basis points are of their turnover, which leaves out the entries of
trades still open that the turnover of every trade counts.

They read optional columns of the trades table (`metrics.analyse` documents them): the
excursions `mae` and `mfe` in rupees and `mae_pct` and `mfe_pct` in percent of each
trade's largest capital at work, and `slippage` in rupees. A table without the
excursion columns gives NaN excursion figures, and one without `slippage` NaN slippage
and trading cost, with no error; so does a slippage that is NaN on a closed trade (none
was modelled for a fill a broker priced), never a 0.
"""

import math
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from ._extended import ratio
from ._inputs import TradeTable, _floats

#: The excursion columns of a trades table: rupees, then percent of capital at work.
EXCURSION_COLUMNS = ("mae", "mfe", "mae_pct", "mfe_pct")
#: The slippage column of a trades table, rupees.
SLIPPAGE_COLUMN = "slippage"
#: Percent per unit fraction (the `_pct` columns are percent, the metrics fractions).
PERCENT = 100.0
#: Basis points per unit fraction: 1 basis point is 0.01 percent.
BASIS_POINTS = 10_000.0
#: The excursion figures, NaN together when the table has no excursion columns.
EXCURSION_KEYS = (
    "average_mae",
    "average_mfe",
    "worst_mae",
    "best_mfe",
    "edge_ratio",
    "profit_kept",
)


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else math.nan


def _extreme(values: np.ndarray, pick: Callable[[np.ndarray], float]) -> float:
    return float(pick(values)) if len(values) else math.nan


def _column(closed: pd.DataFrame, name: str) -> np.ndarray:
    """Column `name` of the closed trades as floats. Raises ConfigError for a value
    that is not a number."""
    return _floats(closed[name], what=f"trades {name}")


def excursion_figures(table: TradeTable) -> dict[str, float]:
    """The MAE and MFE figures of the closed trades of `table` (module docs)."""
    closed = table.frame.loc[~table.is_open]
    if not all(name in closed.columns for name in EXCURSION_COLUMNS):
        return dict.fromkeys(EXCURSION_KEYS, math.nan)
    mae, mfe = _column(closed, "mae"), _column(closed, "mfe")
    mae_pct, mfe_pct = _column(closed, "mae_pct"), _column(closed, "mfe_pct")
    winners = table.closed(table.net_pnl) > 0
    return {
        "average_mae": _mean(mae_pct) / PERCENT,
        "average_mfe": _mean(mfe_pct) / PERCENT,
        "worst_mae": _extreme(mae, np.min),
        "best_mfe": _extreme(mfe, np.max),
        "edge_ratio": ratio(_mean(mfe_pct), _mean(np.abs(mae_pct))),
        "profit_kept": ratio(
            float(np.sum(table.closed(table.gross_pnl)[winners])), float(np.sum(mfe[winners]))
        ),
    }


def _closed_turnover(table: TradeTable) -> float:
    """The traded value of the closed trades, rupees: |price| x quantity x multiplier of
    each entry and each exit."""
    quantity, multiplier = table.closed(table.quantity), table.closed(table.multiplier)
    entries = np.abs(table.closed(table.entry_price)) * quantity * multiplier
    exits = np.abs(table.closed(table.exit_price)) * quantity * multiplier
    return float(np.sum(entries) + np.sum(exits))


def trading_cost_figures(table: TradeTable, figures: Mapping[str, float]) -> dict[str, float]:
    """The turnover, slippage and trading cost of the closed trades of `table` (module
    docs); `figures` holds their `total_costs` and `gross_pnl`, rupees."""
    closed = table.frame.loc[~table.is_open]
    slippage = (
        float(np.sum(_column(closed, SLIPPAGE_COLUMN)))
        if SLIPPAGE_COLUMN in closed.columns
        else math.nan
    )
    cost = figures["total_costs"] + slippage
    gross = figures["gross_pnl"]
    turnover = _closed_turnover(table)
    return {
        "closed_turnover": turnover,
        "total_slippage": slippage,
        "trading_cost": cost,
        "trading_cost_share_of_gross_pnl": cost / gross if gross > 0 else math.nan,
        "trading_cost_bps": cost / turnover * BASIS_POINTS if turnover > 0 else math.nan,
    }


__all__ = [
    "BASIS_POINTS",
    "EXCURSION_COLUMNS",
    "EXCURSION_KEYS",
    "SLIPPAGE_COLUMN",
    "excursion_figures",
    "trading_cost_figures",
]
