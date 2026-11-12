"""The ATR a `bt.atr` distance or `bt.size.atr_risk` stands for at a decision
(05-risk-management 3, SPEC-25).

`bt.atr(mult, period)` as a stop, target or breakeven distance, `bt.trail_atr(mult,
period)`, and `bt.size.atr_risk(..., period)` are resolved when the order is placed: the
ATR of `period` strategy bars from `openalgo.ta` at the latest closed bar, which is the
close before the entry. The kernel receives the value in rupees and never computes an
indicator itself.

Both evaluation paths read the same series (`atr_values`): an `on_bar` order takes its
last value over the bars revealed so far (`decision_atr`); a condition strategy takes
the value at each strategy bar over the whole run (`_condition_atr`). The ATR is causal
(the causality registry says so, and it is checked here), so the value at a bar is the
same either way.
"""

from __future__ import annotations

import math

import numpy as np

from openalgo.indicators import ta

from ._messages import counted
from ._series import SeriesStore
from .errors import ConfigError

# The bar columns the ATR is computed from.
HLC = ("high", "low", "close")


def atr_values(store: SeriesStore, period: int, whole_run: bool = False) -> np.ndarray:
    """The `openalgo.ta` ATR of `period` bars at every bar of `store`, in rupees (Wilder's
    smoothing): over the bars revealed so far, or over the whole run when `whole_run`.
    NaN while fewer than `period` bars have closed.

    Raises ConfigError if the causality registry does not clear `ta.atr`."""
    # The registry loads with the first ATR, not with the run functions: a strategy that
    # names no ATR never reads it.
    from openalgo.indicators import causality

    if not causality.is_causal(ta.atr):
        raise ConfigError("ta.atr looks ahead by the causality registry: it cannot size an order")
    columns = {name: store.column(name) for name in HLC} if whole_run else store.columns
    high, low, close = (np.asarray(columns[name], dtype=float) for name in HLC)
    if len(close) < period:
        return np.full(len(close), math.nan)
    return np.asarray(ta.atr(high, low, close, period), dtype=float)


def decision_atr(store: SeriesStore, period: int) -> float:
    """The ATR of `period` strategy bars at the latest closed bar of `store`, in rupees.

    Raises ConfigError while fewer than `period` bars have closed, or when the ATR is
    not a positive number (a series that has not moved)."""
    closed = len(store)
    if closed < period:
        raise ConfigError(
            f"bt.atr(period={period}) on {store.symbol} needs "
            f"{counted(period, 'closed strategy bar')} at the decision and {closed} "
            f"{'has' if closed == 1 else 'have'} closed: place the order later or use a "
            "shorter period"
        )
    value = float(atr_values(store, period)[-1])
    if not (math.isfinite(value) and value > 0):
        raise ConfigError(
            f"bt.atr(period={period}) on {store.symbol} is {value} at this decision; an ATR "
            "distance needs a positive ATR"
        )
    return value


def sizer_atr(store: SeriesStore, period: int) -> float:
    """The ATR an `atr_risk` sizer takes at the latest closed bar, rupees; NaN while
    fewer than `period` bars have closed (the kernel then refuses the sizer with a
    message)."""
    values = atr_values(store, period)
    return float(values[-1]) if len(values) else math.nan


__all__ = ["HLC", "atr_values", "decision_atr", "sizer_atr"]
