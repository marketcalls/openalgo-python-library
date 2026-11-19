"""The fake broker's orders and fills, and the price rules that work them.

The rules are the sandbox's (sandbox/execution_engine.py, `_process_order` and
`_process_trigger_pending_order`; sandbox/order_manager.py, place_order), which follow
the exchange's. "The market" is the best ask for a BUY and the best bid for a SELL when
the test gave them with the last trade, else the last price (execution_engine.py:520-529).

- MARKET fills at once at the market, at the sandbox and at a broker alike.
- LIMIT that is marketable when it arrives (placed, or modified at a broker) fills at
  once at the market, never at its limit: at a broker at the market, in the sandbox at the
  last price (order_manager.py:769-773). A resting LIMIT the price reaches later fills at
  its limit (execution_engine.py:532-540).
- SL-M and SL rest as "trigger pending" until the last price reaches the trigger (a
  BUY at or above it, a SELL at or below it). SL-M then fills at the last price in the
  sandbox, at the market at a broker (it becomes a market order). SL becomes a limit
  order: it fills at that price when it is within the limit, and otherwise rests "open",
  unfilled, until the price comes back (a stop-limit the market ran through).

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
    price, summed), from which the average fill price comes; `client` names the client
    of the fake that placed it (`FakeOpenAlgo.client`), as the broker knows which API
    session sent an order ("" for an order no client placed)."""

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
    client: str = ""

    @property
    def working(self) -> bool:
        """True while the order can still fill, be modified or be cancelled."""
        return self.status in WORKING

    @property
    def pending(self) -> int:
        """The units still to fill: none once the order is complete, rejected or
        cancelled."""
        return self.remaining if self.working else 0

    @property
    def remaining(self) -> int:
        """The units not filled, whatever the status (a cancelled order keeps them)."""
        return self.quantity - self.filled

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


@dataclass(frozen=True)
class Quote:
    """The last trade's price and the best bid and ask given with it (None when not
    given), rupees."""

    last: float
    bid: float | None = None
    ask: float | None = None


def market_price(action: str, quote: Quote) -> float:
    """The price a market order takes: the best ask for a BUY, the best bid for a SELL,
    the last price when that side was not given (sandbox/execution_engine.py:520-529)."""
    best = quote.ask if action == BUY else quote.bid
    return best if best is not None and best > 0 else quote.last


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


def fill_price(order: FakeOrder, quote: Quote, *, arriving: bool, at_broker: bool) -> float | None:
    """The price an open (not trigger-pending) order fills at now, or None when it does
    not fill now (see the module's rules). `arriving`: the order has just reached the
    book; `at_broker`: a broker's book, which fills at the market where the sandbox takes
    the last price."""
    if order.status != OPEN:
        return None
    at_market = at_broker or order.price_type == MARKET
    taker = market_price(order.action, quote) if at_market else quote.last
    if order.price_type in (MARKET, STOP_MARKET):
        return taker
    if order.price_type == LIMIT and not arriving:
        return order.price if within_limit(order.action, order.price, quote.last) else None
    return taker if within_limit(order.action, order.price, taker) else None


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
    "Quote",
    "fill_price",
    "market_price",
    "reached",
    "released_by",
    "within_limit",
]
