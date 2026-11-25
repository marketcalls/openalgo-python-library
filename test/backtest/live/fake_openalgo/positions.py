"""Net positions built from the fake broker's fills, as a broker's position book keeps
them: one row per instrument and product, the signed quantity in units, the average
price of the open quantity, and the P&L realized by closing it (rupees, gross of
charges: the broker's books carry none).
"""

from __future__ import annotations

from dataclasses import dataclass

from .orders import BUY, Fill


@dataclass
class Position:
    """The net position in one instrument and product. `quantity` is signed units (long
    above 0); `average_price` is that of the open quantity, 0 when flat."""

    symbol: str
    exchange: str
    product: str
    quantity: int = 0
    average_price: float = 0.0
    realized: float = 0.0

    def take(self, fill: Fill) -> None:
        """Net one fill into the position."""
        signed = fill.quantity if fill.action == BUY else -fill.quantity
        if self.quantity == 0 or (self.quantity > 0) == (signed > 0):
            held = abs(self.quantity)
            total = held + fill.quantity
            self.average_price = (self.average_price * held + fill.price * fill.quantity) / total
            self.quantity += signed
            return
        closing = min(fill.quantity, abs(self.quantity))
        direction = 1 if self.quantity > 0 else -1
        self.realized += closing * (fill.price - self.average_price) * direction
        self.quantity += signed
        if self.quantity == 0:
            self.average_price = 0.0
        elif (self.quantity > 0) != (direction > 0):
            # The fill went through flat: what is left was opened at the fill's price.
            self.average_price = fill.price

    def unrealized(self, last_price: float | None) -> float:
        """The open quantity's P&L at `last_price`, rupees (0 when flat or unpriced)."""
        if not self.quantity or last_price is None:
            return 0.0
        return (last_price - self.average_price) * self.quantity


class Positions:
    """Every position of one book, in the order they were opened."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str, str], Position] = {}

    def take(self, fill: Fill) -> None:
        """Net one fill into its instrument's position."""
        key = (fill.symbol, fill.exchange, fill.product)
        if key not in self._rows:
            self._rows[key] = Position(fill.symbol, fill.exchange, fill.product)
        self._rows[key].take(fill)

    def carry(self, position: Position) -> None:
        """Open the day with `position` carried over from the day before: its units at its
        average price, nothing realized yet."""
        key = (position.symbol, position.exchange, position.product)
        self._rows[key] = Position(
            position.symbol,
            position.exchange,
            position.product,
            position.quantity,
            position.average_price,
        )

    def rows(self) -> list[Position]:
        """Every position, flat ones included (a broker lists the day's closed rows)."""
        return list(self._rows.values())

    def net(self, symbol: str, exchange: str) -> int:
        """The net quantity in units over every product of one instrument."""
        return sum(
            row.quantity
            for row in self._rows.values()
            if row.symbol == symbol and row.exchange == exchange
        )

    def realized(self) -> float:
        """The P&L realized over every position, rupees."""
        return sum(row.realized for row in self._rows.values())


__all__ = ["Position", "Positions"]
