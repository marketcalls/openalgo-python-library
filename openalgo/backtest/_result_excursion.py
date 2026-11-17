"""The MAE and MFE of a run's trades in percent, for `Result.trades` and the trades table
of the reports (`_result_report`).

The kernel records each closed trade's maximum adverse and favourable excursions in
rupees (`mae_rupees`, at most 0, and `mfe_rupees`, at least 0: the lowest and highest
running P&L the trade showed before charges, at prices known to have traded while it
was open; `oa_bt::engine::excursion` defines them) and what slippage cost its fills
(`slippage_rupees`, NaN when a fill had none modelled). Here the excursions are put in
percent of the trade's largest capital at work: its average entry price x the largest
quantity it held x the contract multiplier (rupees per one-rupee move per unit).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

#: Percent per unit fraction: 10.0 is ten percent.
PERCENT = 100.0


def instrument_multipliers(manifest: dict[str, Any]) -> list[float]:
    """The contract multiplier of each instrument of a run, by the kernel's instrument
    index (the manifest lists the instruments in registration order)."""
    return [float(instrument["multiplier"]) for instrument in manifest["instruments"]]


def trade_multipliers(instruments: pd.Series, multipliers: list[float]) -> pd.Series:
    """The multiplier of each trade's instrument, from its kernel instrument index."""
    return pd.Series(
        [multipliers[int(i)] for i in instruments], index=instruments.index, dtype=float
    )


def capital_at_work(
    entry_price: pd.Series, max_units: pd.Series, multiplier: pd.Series
) -> pd.Series:
    """Each trade's largest capital at work, rupees: average entry price x the largest
    quantity held (units) x the multiplier."""
    return entry_price.astype(float).abs() * max_units.astype(float) * multiplier


def excursion_pct(rupees: pd.Series, capital: pd.Series) -> pd.Series:
    """An excursion in rupees as percent of the capital at work; NaN where the capital
    is not a positive amount."""
    usable = capital.where(capital > 0)
    return (rupees.astype(float) / usable * PERCENT).astype(float)


def excursion_columns(raw: pd.DataFrame, multipliers: list[float]) -> dict[str, pd.Series]:
    """`mae_pct` and `mfe_pct` of the kernel's trades table `raw` (its `mae_rupees`,
    `mfe_rupees`, `entry_price_rupees`, `max_units` and `instrument` columns)."""
    if not len(raw):
        empty = pd.Series(np.empty(0), index=raw.index, dtype=float)
        return {"mae_pct": empty, "mfe_pct": empty.copy()}
    multiplier = trade_multipliers(raw["instrument"], multipliers)
    capital = capital_at_work(raw["entry_price_rupees"], raw["max_units"], multiplier)
    return {
        "mae_pct": excursion_pct(raw["mae_rupees"], capital),
        "mfe_pct": excursion_pct(raw["mfe_rupees"], capital),
    }


__all__ = [
    "PERCENT",
    "capital_at_work",
    "excursion_columns",
    "excursion_pct",
    "instrument_multipliers",
    "trade_multipliers",
]
