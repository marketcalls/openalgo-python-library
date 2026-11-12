"""Sizers: `bt.size.qty`, `lots`, `value`, `percent`, `fixed_risk` and `atr_risk`
(12-position-sizing section 5).

A sizer is passed as `size=` on an order or set as the strategy's default. The kernel
turns it into a quantity in units at the decision price, rounds it down to the venue
rule (whole lots; whole shares unless the run is fractional), caps it at the freeze
quantity (whole lots) and skips the order with a logged reason when it rounds to zero.
Percent arguments are percent units (`percent(50)` is 50 percent). Raises ConfigError
for a value that is not positive.

An order with no `qty`, `lots` or `size` (and no strategy `size`) takes the engine
default (03-architecture 7): one lot for futures and options; for cash equity 1x the
strategy's equity (never more than its free cash, split equally across its
instruments) net of the order's estimated costs, sized at the fill price in fractional
mode and at the decision price with a 2 percent cash buffer in venue mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..indicators import ta
from ._series import BarStore
from .errors import ConfigError

BASES = ("equity", "capital", "cash")
# The bar columns the ATR is computed from.
HLC = ("high", "low", "close")


def _positive(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
        raise ConfigError(f"{what} must be a positive number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class Sizer:
    """A sizer: its kind and arguments, as the kernel reads them."""

    kind: str
    amount: float | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def to_native(self) -> dict[str, Any]:
        """The quantity dict of an order request."""
        return {"kind": "sizer", "sizer": self.kind, "amount": self.amount, **self.options}


def qty(units: float) -> Sizer:
    """A fixed quantity in units, rounded down to the venue rule."""
    return Sizer("qty", _positive(units, "qty"))


def lots(count: int) -> Sizer:
    """A fixed number of lots."""
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ConfigError(f"lots must be a positive whole number, got {count!r}")
    # The kernel reads a lot count as a whole number, so it crosses as an int.
    return Sizer("lots", count)


def value(rupees: float) -> Sizer:
    """Rupees of notional: rupees / (price x multiplier)."""
    return Sizer("value", _positive(rupees, "value"))


def percent(p: float, basis: str = "equity") -> Sizer:
    """p percent of equity (default), initial capital ("capital") or free cash ("cash")."""
    if basis not in BASES:
        raise ConfigError(f"unknown sizing basis {basis!r}: use one of {BASES}")
    return Sizer("percent", _positive(p, "percent"), {"basis": basis})


def _whole(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{what} must be a whole number, 1 or more, got {value!r}")
    return value


def fixed_risk(risk_pct: float, units: int = 1, hard_limit: float | None = None) -> Sizer:
    """Risk `risk_pct` percent of equity, less the estimated costs of the round trip, to
    the order's stop (needs `stop=`): quantity = budget / (stop distance x multiplier),
    capped at `hard_limit` units and split into `units` equal tranches (the order is
    one of them)."""
    limit = None if hard_limit is None else _positive(hard_limit, "hard_limit")
    options = {"units": _whole(units, "units"), "hard_limit": limit}
    return Sizer("fixed_risk", _positive(risk_pct, "risk_pct"), options)


def atr_risk(risk_pct: float, mult: float, period: int = 14) -> Sizer:
    """Fixed risk with the stop distance taken as `mult` x the ATR of `period` strategy
    bars (`openalgo.ta.atr`) at the decision."""
    options = {"multiple": _positive(mult, "mult"), "period": _whole(period, "period")}
    return Sizer("atr_risk", _positive(risk_pct, "risk_pct"), options)


def resolve_atr(quantity: dict[str, Any], store: BarStore) -> dict[str, Any]:
    """The quantity dict of a request with the ATR of an `atr_risk` sizer filled in from
    the strategy bars revealed so far (`openalgo.ta.atr`, rupees). Any other quantity is
    returned as it is. With fewer bars than the period the ATR is NaN and the kernel
    refuses the sizer with a message."""
    if quantity.get("sizer") != "atr_risk" or quantity.get("atr") is not None:
        return quantity
    high, low, close = (np.asarray(store.columns[name], dtype=float) for name in HLC)
    period = int(quantity["period"])
    atr = ta.atr(high, low, close, period)[-1] if len(close) >= period else math.nan
    return {**quantity, "atr": float(atr)}


__all__ = [
    "BASES",
    "Sizer",
    "atr_risk",
    "fixed_risk",
    "lots",
    "percent",
    "qty",
    "resolve_atr",
    "value",
]
