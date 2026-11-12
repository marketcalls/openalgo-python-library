"""The charge tables of a `Result`: what each fill was charged, the contract notes, and
the warnings for unverified market-table rows a run used.

The kernel's `charges` output (`rust/oa_py/src/backtest/engine/charges.rs`) holds one
record per charged fill and one per contract note. Money is rupees; dates are IST.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd

from ._series import IST
from .costs import COMPONENTS, CONTRACT_NOTE_RULE

#: Columns of `Result.costs`, in order.
COST_COLUMNS = (
    "time",
    "symbol",
    "product",
    "side",
    "quantity_units",
    "price_rupees",
    "reason",
    "date",
    "segment",
    *COMPONENTS,
    "cost",
    "dp_charge",
    "total",
    "schedule",
    "verified",
    "trade_id",
)
#: Columns of `Result.contract_notes`, in order.
NOTE_COLUMNS = (
    "date",
    "note_group",
    "fills",
    *COMPONENTS,
    "charged_total",
    "accrued_total",
    "posted",
    "adjustment",
    "settled",
)
_EPOCH = dt.date(1970, 1, 1)
_VERIFIED = "verified"
_UNVERIFIED = "unverified"


def _dates(days: Any) -> list[dt.date]:
    return [_EPOCH + dt.timedelta(days=int(day)) for day in days]


def cost_frame(output: dict[str, Any], symbols: list[str], schedules: list[str]) -> pd.DataFrame:
    """One row per charged fill (see `Result.costs`)."""
    charges = output["charges"]
    rows = pd.DataFrame(charges["fills"])
    frame = pd.DataFrame(columns=list(COST_COLUMNS))
    if len(rows):
        fills = pd.DataFrame(output["fills"]).iloc[rows["fill"].to_numpy()].reset_index(drop=True)
        times = pd.to_datetime(fills["slice_ts"].to_numpy(dtype=np.int64), utc=True)
        frame = pd.DataFrame(
            {
                "time": times.tz_convert(IST),
                "symbol": [symbols[i] for i in fills["instrument"]],
                **{name: fills[name] for name in ("product", "side", "quantity_units")},
                "price_rupees": fills["price_rupees"],
                "reason": fills["reason"],
                "date": _dates(rows["day"]),
                **{name: rows[name] for name in ("segment", *COMPONENTS)},
                **{name: rows[name] for name in ("cost", "dp_charge", "total")},
                "schedule": [schedules[i] for i in rows["schedule"]],
                "verified": rows["trust"] == _VERIFIED,
                "trade_id": rows["trade"].astype("Int64"),
            }
        )
    frame.attrs["rounding"] = charges["rounding"]
    return frame


def notes_frame(output: dict[str, Any]) -> pd.DataFrame:
    """One row per contract note (see `Result.contract_notes`)."""
    notes = pd.DataFrame(output["charges"]["notes"])
    frame = pd.DataFrame(columns=list(NOTE_COLUMNS))
    if len(notes):
        frame = notes.rename(columns={"group": "note_group", "day": "date"})
        frame["date"] = _dates(frame["date"])
        frame = frame[list(NOTE_COLUMNS)]
    frame.attrs["rule"] = CONTRACT_NOTE_RULE
    return frame


def run_warnings(output: dict[str, Any], schedules: list[str]) -> list[str]:
    """One warning per unverified slippage or margin row the run used, and one per cost
    schedule and segment whose rates were not verified for some fill's date. A row of
    status "default" (the futures margin of decision D23) is stated in the Assumptions
    block instead."""
    lines = []
    for table, row_id, status, first_day, source, notes in output["charges"]["tables"]:
        if status == _UNVERIFIED:
            first = _dates([first_day])[0]
            lines.append(
                f"{table} row {row_id}, used from {first}, is unverified: {notes} "
                f"(source: {source})"
            )
    rows = pd.DataFrame(output["charges"]["fills"])
    if len(rows):
        flagged = rows[rows["trust"] != _VERIFIED]
        for (schedule, segment, trust), group in flagged.groupby(["schedule", "segment", "trust"]):
            first = _dates([group["day"].min()])[0]
            lines.append(
                f"charges: the {segment} rates of cost schedule {schedules[schedule]} are "
                f"{trust.replace('_', ' ')} for {len(group)} fill(s), from {first}"
            )
    return lines


def trade_charges(output: dict[str, Any]) -> pd.DataFrame:
    """The components and the DP charge summed per closed trade, indexed by trade id."""
    rows = pd.DataFrame(output["charges"]["fills"])
    columns = [*COMPONENTS, "dp_charge"]
    if not len(rows):
        return pd.DataFrame(columns=columns, dtype=float)
    closed = rows[rows["trade"].notna()]
    return closed.groupby(closed["trade"].astype(np.int64))[columns].sum()


def open_charges(output: dict[str, Any], instrument: int, product: str) -> float:
    """Rupees charged on the fills of the trade still open on an instrument and product."""
    rows = pd.DataFrame(output["charges"]["fills"])
    if not len(rows):
        return 0.0
    fills = pd.DataFrame(output["fills"]).iloc[rows["fill"].to_numpy()].reset_index(drop=True)
    mask = (
        rows["trade"].isna().to_numpy()
        & (fills["instrument"] == instrument).to_numpy()
        & (fills["product"] == product).to_numpy()
    )
    return float(rows.loc[mask, "total"].sum())


__all__ = [
    "COST_COLUMNS",
    "NOTE_COLUMNS",
    "cost_frame",
    "notes_frame",
    "open_charges",
    "run_warnings",
    "trade_charges",
]
