"""Positions and gross P&L replayed from a case's expected fills.

This is bookkeeping, not a kernel: it nets the fills in the order they are listed and
values reductions at average cost (12-position-sizing section 1). The consistency checks
use it to confirm that the fills, the level timeline, the events, the final positions
and the stated P&L of a fixture agree with each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .bars import bars_by_slice, rupees
from .timegrid import Case, parse_ist

Fill = dict[str, Any]
ZERO = Decimal(0)


@dataclass(frozen=True)
class Position:
    """A net position: signed quantity in units (long positive), average price in rupees."""

    quantity_units: int = 0
    average_price: Decimal | None = None
    realized: Decimal = ZERO

    @property
    def side(self) -> str:
        """flat, long or short."""
        if self.quantity_units == 0:
            return "flat"
        return "long" if self.quantity_units > 0 else "short"


def signed_quantity(fill: Fill) -> int:
    """Quantity of a fill in units, positive for BUY and negative for SELL."""
    return fill["quantity_units"] if fill["side"] == "BUY" else -fill["quantity_units"]


def apply_fill(position: Position, fill: Fill) -> Position:
    """The position after one fill, at average cost.

    Raises ValueError when a fill would take the position through zero: a reversal is
    two fills (12-position-sizing section 1).
    """
    change = signed_quantity(fill)
    price = rupees(fill["price_rupees"])
    held = position.quantity_units
    if held == 0 or (held > 0) == (change > 0):
        cost = abs(held) * (position.average_price or ZERO) + abs(change) * price
        return Position(held + change, cost / abs(held + change), position.realized)
    if abs(change) > abs(held):
        raise ValueError(f"fill at {fill['slice_ist']} takes the position through zero")
    direction = 1 if held > 0 else -1
    realized = position.realized + direction * abs(change) * (price - position.average_price)
    remaining = held + change
    return Position(remaining, position.average_price if remaining else None, realized)


def position_history(case: Case) -> list[tuple[Fill, Position]]:
    """Each expected fill with its instrument's position just after it, in fill order."""
    positions: dict[str, Position] = {}
    history = []
    for fill in case["expected"]["fills"]:
        after = apply_fill(positions.get(fill["instrument"], Position()), fill)
        positions[fill["instrument"]] = after
        history.append((fill, after))
    return history


def position_after(case: Case, instrument: str, slice_ist: str, last_step: int = 8) -> Position:
    """Position of an instrument after the fills up to a step of a slice (inclusive)."""
    moment = parse_ist(slice_ist)
    position = Position()
    for fill, after in position_history(case):
        if fill["instrument"] != instrument:
            continue
        fill_moment = parse_ist(fill["slice_ist"])
        if fill_moment < moment or (fill_moment == moment and fill["step"] <= last_step):
            position = after
    return position


def final_positions(case: Case) -> dict[str, Position]:
    """Position of every declared instrument after all expected fills."""
    positions = {instrument["symbol"]: Position() for instrument in case["instruments"]}
    for fill, after in position_history(case):
        positions[fill["instrument"]] = after
    return positions


def realized_pnl(case: Case) -> Decimal:
    """Gross realized P&L in rupees over all instruments."""
    return sum((position.realized for position in final_positions(case).values()), ZERO)


def unrealized_pnl(case: Case) -> Decimal:
    """Gross unrealized P&L in rupees: open positions marked at their last close."""
    total = ZERO
    bars = bars_by_slice(case)
    for instrument, position in final_positions(case).items():
        if position.quantity_units:
            last_close = list(bars[instrument].values())[-1].close
            total += position.quantity_units * (last_close - position.average_price)
    return total
