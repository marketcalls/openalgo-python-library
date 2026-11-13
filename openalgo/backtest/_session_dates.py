"""Own daily bars dated by naive midnight timestamps, as the Assumptions block states them.

`client.history(...)` returns daily bars indexed by naive midnight dates. Such a list of
session dates is not ambiguous, so the data layer reads it as IST dates without `tz=`
(`data._stamps`, for daily and longer bars only), and the run says so in one line of its
Assumptions block, under "timestamps", naming the symbols read that way.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .data._stamps import SESSION_DATES

ASSUMPTION = "naive midnight timestamps read as IST session dates (daily bars need no tz=)"


def session_dates_note(symbols: Sequence[str], loaded: Sequence[Any]) -> dict[str, str]:
    """{"timestamps": the Assumptions line} when the bars of any symbol were read as IST
    session dates, for example "SBIN: naive midnight timestamps read as IST session dates
    (daily bars need no tz=)", else {}. `loaded` is the data layer's `BarsResult` of each
    symbol, in the order of `symbols`."""
    dated = [
        symbol
        for symbol, result in zip(symbols, loaded, strict=True)
        if result.meta.timestamp_convention.startswith(SESSION_DATES)
    ]
    return {"timestamps": f"{', '.join(dated)}: {ASSUMPTION}"} if dated else {}


__all__ = ["session_dates_note"]
