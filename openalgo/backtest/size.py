"""Sizers: `bt.size.qty`, `lots`, `value`, `percent`, `fixed_risk` and `atr_risk`
(12-position-sizing section 5).

A sizer is passed as `size=` on an order or set as the strategy's default. The kernel
turns it into a quantity in units at the decision price, rounds it down to the venue
rule (whole lots; whole shares unless the run is fractional) and skips the order with a
logged reason when it rounds to zero. Percent arguments are percent units
(`percent(50)` is 50 percent). Raises ConfigError for a value that is not positive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import ConfigError

BASES = ("equity", "capital", "cash")


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
    return Sizer("lots", float(count))


def value(rupees: float) -> Sizer:
    """Rupees of notional: rupees / (price x multiplier)."""
    return Sizer("value", _positive(rupees, "value"))


def percent(p: float, basis: str = "equity") -> Sizer:
    """p percent of equity (default), initial capital ("capital") or free cash ("cash")."""
    if basis not in BASES:
        raise ConfigError(f"unknown sizing basis {basis!r}: use one of {BASES}")
    return Sizer("percent", _positive(p, "percent"), {"basis": basis})


def fixed_risk(risk_pct: float, units: int = 1, hard_limit: float | None = None) -> Sizer:
    """Risk `risk_pct` percent of equity to the order's stop (needs a stop)."""
    options = {"units": units, "hard_limit": hard_limit}
    return Sizer("fixed_risk", _positive(risk_pct, "risk_pct"), options)


def atr_risk(risk_pct: float, mult: float, period: int = 14) -> Sizer:
    """Fixed risk with the stop distance taken as ATR x `mult`."""
    options = {"multiple": _positive(mult, "mult"), "period": period}
    return Sizer("atr_risk", _positive(risk_pct, "risk_pct"), options)


__all__ = ["BASES", "Sizer", "atr_risk", "fixed_risk", "lots", "percent", "qty", "value"]
