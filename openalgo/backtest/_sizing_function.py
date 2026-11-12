"""A strategy's own sizing function: `size=` a callable `size(ctx, order) -> units`
(12-position-sizing 5, FR-SIZE-06).

The function runs at the decision, in Python, when the order is placed. It receives
`ctx`, a `SizingInputs` holding what the built-in sizers see for that order (the
kernel's sizing context, `Engine.sizing_inputs`), and `order`, a `SizingOrder` naming
the order. It returns a quantity in units (never lots), which the kernel then treats as
any sizer's: cut to the strategy's caps, rounded down to the venue rule, capped at the
freeze quantity, and skipped with reason "rounded_to_zero" when it is 0. Anything else
(a negative number, NaN, a bool, a string, None) is a ConfigError naming the function.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ._engine import call
from .errors import ConfigError

# The sizer name of a sizing function's quantity in a request (`request.rs`).
FUNCTION = "function"


@dataclass(frozen=True)
class SizingInputs:
    """What a sizing function sees: money in rupees, quantities in units.

    equity: the sizing equity (the current equity, or the initial capital when the
        strategy sets `reinvest = False`). capital: the capital allocated at the start.
    cash: equity not blocked as margin. price: the decision price, the instrument's last
        close. stop_distance: the distance from that price to the order's stop, rupees
        of price, or None without a stop. lot_size: units in one lot. multiplier: rupees
        per one-rupee move per unit. fractional: True when fractional quantities are
        allowed. position: the units held now, negative when short.
    """

    equity: float
    capital: float
    cash: float
    price: float
    stop_distance: float | None
    lot_size: int
    multiplier: float
    fractional: bool
    position: float


@dataclass(frozen=True)
class SizingOrder:
    """The order a sizing function sizes: symbol, side ("BUY" or "SELL"), product
    ("CNC", "MIS" or "NRML") and the strategy's tag (or None)."""

    symbol: str
    side: str
    product: str
    tag: str | None


def quantity(function: Callable[..., Any]) -> dict[str, Any]:
    """The quantity dict of a request sized by `function`, resolved at submission."""
    return {"kind": "sizer", "sizer": FUNCTION, "amount": None, "function": function}


def _units(returned: Any, function: Callable[..., Any]) -> float:
    """The quantity a sizing function returned, checked. Raises ConfigError."""
    usable = isinstance(returned, numbers.Real) and not isinstance(returned, bool)
    if usable and math.isfinite(returned) and returned >= 0:
        return float(returned)
    name = getattr(function, "__name__", repr(function))
    raise ConfigError(
        f"the sizing function {name} returned {returned!r}: return a quantity in units, 0 "
        "or more (0 skips the order)"
    )


def resolve(request: dict[str, Any], engine: Any, symbol: str) -> dict[str, Any]:
    """`request` with its sizing function called and its quantity resolved; any other
    request is returned as it is. `symbol` names the order's instrument. Raises
    ConfigError for a returned value that is not a quantity, and lets an exception of
    the function itself propagate (it stops the run, as one raised in `on_bar`)."""
    function = request["quantity"].get("function")
    if function is None:
        return request
    native = {**request, "quantity": {"kind": "sizer", "sizer": FUNCTION, "amount": 0.0}}
    inputs = SizingInputs(**call(engine.sizing_inputs, native))
    order = SizingOrder(symbol, request["side"], request["product"], request["tag"])
    units = _units(function(inputs, order), function)
    return {**native, "quantity": {**native["quantity"], "amount": units}}


__all__ = ["FUNCTION", "SizingInputs", "SizingOrder", "quantity", "resolve"]
