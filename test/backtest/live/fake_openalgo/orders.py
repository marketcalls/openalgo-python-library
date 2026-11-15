"""The fake broker's orders and fills, and the price rules that work them.

The rules are the sandbox's (sandbox/execution_engine.py, `_process_order` and
`_process_trigger_pending_order`), which follow the exchange's:

- MARKET fills at once at the last price (the sandbox's bid or ask is not modelled).
- LIMIT fills at its limit price once the last price reaches it.
- SL-M and SL rest as "trigger pending" until the last price reaches the trigger (a
  BUY at or above it, a SELL at or below it). SL-M then fills at the last price. SL
  becomes a limit order: it fills at the last price when that is within the limit, and
  otherwise rests "open", unfilled, until the price comes back (a stop-limit the market
  ran through).

Prices are rupees, quantities units, times UTC nanoseconds.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The order statuses of the server's books (sandbox/order_manager.py).
OPEN = "open"
TRIGGER_PENDING = "trigger pending"
COMPLETE = "complete"
REJECTED = "rejected"
CANCELLED = "cancelled"
WORKING = frozenset({OPEN, TRIGGER_PENDING})

#: The price types OpenAlgo accepts (utils/constants.py, VALID_PRICE_TYPES).
MARKET, LIMIT, STOP_LIMIT, STOP_MARKET = "MARKET", "LIMIT", "SL", "SL-M"
PRICE_TYPES = frozenset({MARKET, LIMIT, STOP_LIMIT, STOP_MARKET})
#: The price types that rest as "trigger pending" until their trigger is reached.
STOP_TYPES = frozenset({STOP_LIMIT, STOP_MARKET})
#: The price types that carry a limit price.
LIMIT_TYPES = frozenset({LIMIT, STOP_LIMIT})
BUY, SELL = "BUY", "SELL"


@dataclass
class FakeOrder:
    """An order at the fake broker. `value` is the rupees filled so far (units times
    price, summed), from which the average fill price comes."""

    order_id: str
    strategy: str
    symbol: str
    exchange: str
    action: str
    quantity: int
    price_type: str
    product: str
    price: float
    trigger_price: float
    placed_ns: int
    status: str = OPEN
    filled: int = 0
    value: float = 0.0
    rejection_reason: str = ""

    @property
    def working(self) -> bool:
        """True while the order can still fill, be modified or be cancelled."""
        return self.status in WORKING

    @property
    def pending(self) -> int:
        """The units still to fill: none once the order is complete, rejected or
        cancelled."""
        return self.quantity - self.filled if self.working else 0

    @property
    def average_price(self) -> float:
        """The average fill price, rupees; 0 before any fill (as OpenAlgo writes it)."""
        return self.value / self.filled if self.filled else 0.0


@dataclass(frozen=True)
class Fill:
    """One execution of an order: `quantity` units at `price` rupees at `ts_ns`."""

    trade_id: str
    order_id: str
    strategy: str
    symbol: str
    exchange: str
    product: str
    action: str
    quantity: int
    price: float
    ts_ns: int


def reached(action: str, level: float, price: float) -> bool:
    """Whether `price` reaches a BUY or SELL order's `level` from its side: a stop's
    trigger (a BUY at or above it, a SELL at or below it)."""
    return price >= level if action == BUY else price <= level


def within_limit(action: str, limit: float, price: float) -> bool:
    """Whether `price` is within a BUY or SELL order's limit: a BUY at or below it, a SELL
    at or above it."""
    return price <= limit if action == BUY else price >= limit


def released_by(order: FakeOrder, price: float) -> bool:
    """Whether a trigger-pending stop is triggered by the last price `price`."""
    return order.status == TRIGGER_PENDING and reached(order.action, order.trigger_price, price)


def fill_price(order: FakeOrder, price: float) -> float | None:
    """The price an open (not trigger-pending) order fills at when the last price is
    `price`, or None when it does not fill now."""
    if order.status != OPEN:
        return None
    if order.price_type in (MARKET, STOP_MARKET):
        return price
    if order.price_type == LIMIT:
        return order.price if within_limit(order.action, order.price, price) else None
    return price if within_limit(order.action, order.price, price) else None


__all__ = [
    "BUY",
    "CANCELLED",
    "COMPLETE",
    "LIMIT",
    "LIMIT_TYPES",
    "MARKET",
    "OPEN",
    "PRICE_TYPES",
    "REJECTED",
    "SELL",
    "STOP_LIMIT",
    "STOP_MARKET",
    "STOP_TYPES",
    "TRIGGER_PENDING",
    "WORKING",
    "FakeOrder",
    "Fill",
    "fill_price",
    "reached",
    "released_by",
    "within_limit",
]
