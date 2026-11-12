"""The futures tables of a `Result`: the switches of each futures series (`rolls`) and
the mark-to-market postings (`settlements`), from the kernel's `futures` output
(`rust/oa_bt/src/engine/futures.rs`, `settlement.rs`).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ._calendar_convert import from_day
from ._series import IST

SETTLEMENT_COLUMNS = (
    "day",
    "slice_ts",
    "instrument",
    "product",
    "quantity_units",
    "price_rupees",
    "last_close_used",
    "amount_rupees",
    "kind",
)


def _ist(values: Any) -> Any:
    return pd.to_datetime(pd.Series(values, dtype="int64"), utc=True).dt.tz_convert(IST)


def settlements_frame(output: dict[str, Any], symbols: list[str]) -> pd.DataFrame:
    """One row per posting: `time` (IST open of the slice it was posted in), IST
    `date`, `symbol`, product, units held after the session close, the settlement
    price, `last_close_used` (the data gave no settlement price), the cash flow
    `amount_rupees` and `kind` ("daily" or "expiry")."""
    frame = pd.DataFrame(output["futures"]["settlements"], columns=list(SETTLEMENT_COLUMNS))
    frame.insert(0, "symbol", [symbols[i] for i in frame["instrument"]])
    frame.insert(0, "date", [from_day(day) for day in frame["day"]])
    frame.insert(0, "time", _ist(frame["slice_ts"]))
    return frame.drop(columns=["day"])


def switches_frame(output: dict[str, Any], symbols: list[str]) -> pd.DataFrame:
    """One row per switch of a futures series: `time` (IST open of the slice from which
    the next contract is traded), its IST `date`, `from_symbol`, `to_symbol`, the
    `spread_rupees` that back-adjusts the series and `rolls_positions` (False when
    positions are settled at expiry instead)."""
    columns = [
        "series",
        "from",
        "to",
        "slice",
        "slice_ts",
        "day",
        "spread_rupees",
        "rolls_positions",
    ]
    frame = pd.DataFrame(output["futures"]["switches"], columns=columns)
    frame.insert(0, "to_symbol", [symbols[i] for i in frame["to"]])
    frame.insert(0, "from_symbol", [symbols[i] for i in frame["from"]])
    frame.insert(0, "date", [from_day(day) for day in frame["day"]])
    frame.insert(0, "time", _ist(frame["slice_ts"]))
    return frame.drop(columns=["day"])


__all__ = ["settlements_frame", "switches_frame"]
