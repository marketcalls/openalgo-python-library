"""The ATR a `bt.atr` distance stands for at a decision (05-risk-management 3).

`bt.atr(mult, period)` as a stop, target or breakeven distance, and `bt.trail_atr(mult,
period)`, are resolved when the order is placed: the ATR of `period` strategy bars from
`openalgo.ta` at the latest closed bar, which is the close before the entry (SPEC-25).
The kernel receives the value in rupees and never computes an indicator itself.
"""

from __future__ import annotations

import math

import numpy as np

from openalgo.indicators import ta

from ._series import BarStore
from .errors import ConfigError


def decision_atr(store: BarStore, period: int) -> float:
    """The ATR of `period` strategy bars at the latest closed bar of `store`, in rupees
    (Wilder's smoothing, as `openalgo.ta.atr`).

    Raises ConfigError while fewer than `period` bars have closed, or when the ATR is
    not a positive number (a series that has not moved)."""
    closed = len(store)
    if closed < period:
        raise ConfigError(
            f"bt.atr(period={period}) on {store.symbol} needs {period} closed strategy bars "
            f"at the decision and {closed} have closed: place the order later or use a "
            "shorter period"
        )
    high, low, close = (
        np.asarray(store.columns[name], dtype=float) for name in ("high", "low", "close")
    )
    value = float(ta.atr(high, low, close, period)[-1])
    if not (math.isfinite(value) and value > 0):
        raise ConfigError(
            f"bt.atr(period={period}) on {store.symbol} is {value} at this decision; an ATR "
            "distance needs a positive ATR"
        )
    return value


__all__ = ["decision_atr"]
