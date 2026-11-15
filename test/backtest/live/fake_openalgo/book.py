"""One account's books at the fake broker: the orders, the fills and the positions of
either the live broker or the OpenAlgo sandbox. The analyzer switch decides which book a
REST call reaches; the two never share an order (services/place_order_service.py routes
every call by the switch at the moment of the call).

Each operation returns the `Change`s it made, oldest first: the order-updates stream
reports them. Prices are rupees, quantities units, times UTC nanoseconds.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Protocol

from openalgo.backtest._live._types import Mode

from .market import MarketState
from .orders import (
    CANCELLED,
    COMPLETE,
    OPEN,
    REJECTED,
    STOP_TYPES,
    TRIGGER_PENDING,
    FakeOrder,
    Fill,
    Quote,
    fill_price,
    reached,
    released_by,
)
from .positions import Positions

#: What happened to an order, in a `Change`.
PLACED, TRIGGERED, FILLED, MODIFIED, CANCELLED_EVENT, REJECTED_EVENT = (
    "placed",
    "triggered",
    "filled",
    "modified",
    "cancelled",
    "rejected",
)


class Ids(Protocol):
    """Issues order and trade IDs (`ids.SandboxIds`, `ids.BrokerIds`)."""

    def order_id(self, now_ns: int) -> str:
        """A new order ID at `now_ns`."""

    def trade_id(self, now_ns: int) -> str:
        """A new trade ID at `now_ns`."""


@dataclass(frozen=True)
class Change:
    """An order as it stood right after `event` (one of the event names above), and the
    fill that event made, if any."""

    order: FakeOrder
    event: str
    fill: Fill | None = None


@dataclass(frozen=True)
class OrderTerms:
    """What a placement asks for: quantity in units, prices in rupees (0 where they do
    not apply)."""

    strategy: str
    symbol: str
    exchange: str
    action: str
    quantity: int
    price_type: str
    product: str
    price: float
    trigger_price: float


class Book:
    """The orders, fills and positions of one mode. `cash` is the available cash the
    book's funds report, rupees."""

    def __init__(self, mode: Mode, ids: Ids, cash: float) -> None:
        self.mode = mode
        self.cash = cash
        self._ids = ids
        self.orders: dict[str, FakeOrder] = {}
        self.fills: list[Fill] = []
        self.positions = Positions()

    def place(self, terms: OrderTerms, now_ns: int, last_price: float | None) -> Change:
        """Accept an order. A stop whose trigger the last price already reached starts
        open, as the sandbox does (sandbox/order_manager.py, place_order); `last_price`
        is None when no price could be had (none traded, or the broker's login is gone),
        and a stop then rests trigger pending."""
        order = FakeOrder(
            order_id=self._ids.order_id(now_ns),
            placed_ns=now_ns,
            **dataclasses.asdict(terms),
        )
        triggered_now = last_price is not None and reached(
            order.action, order.trigger_price, last_price
        )
        if order.price_type in STOP_TYPES and not triggered_now:
            order.status = TRIGGER_PENDING
        self.orders[order.order_id] = order
        return self._change(order, PLACED)

    def reject(self, order: FakeOrder, reason: str) -> Change:
        """The broker refuses an accepted order."""
        order.status, order.rejection_reason = REJECTED, reason
        return self._change(order, REJECTED_EVENT)

    def cancel(self, order: FakeOrder) -> Change:
        """Cancel a working order. Raises ValueError for one that is not working."""
        self._require_working(order)
        order.status = CANCELLED
        return self._change(order, CANCELLED_EVENT)

    def modify(
        self, order: FakeOrder, *, quantity: int, price_type: str, price: float, trigger: float
    ) -> Change:
        """Change a working order's terms. Raises ValueError for one that is not working
        or a quantity below what has filled."""
        self._require_working(order)
        if quantity <= order.filled:
            raise ValueError(f"order {order.order_id}: {quantity} units is not above its fills")
        was_stop = order.price_type in STOP_TYPES
        order.quantity, order.price_type = quantity, price_type
        order.price, order.trigger_price = price, trigger
        if price_type not in STOP_TYPES and order.status == TRIGGER_PENDING:
            order.status = OPEN
        elif price_type in STOP_TYPES and not was_stop:
            # A new stop rests until its trigger; `work` releases it at once when the
            # last price has reached it already.
            order.status = TRIGGER_PENDING
        return self._change(order, MODIFIED)

    def work(
        self,
        order: FakeOrder,
        market: MarketState,
        units_cap: int | None,
        now_ns: int,
        *,
        arriving: bool,
    ) -> list[Change]:
        """Work one order against the market at `now_ns`: release a triggered stop,
        then fill what the price allows (no more than `units_cap` units when given).
        `arriving`: the order has just reached the book (`orders.fill_price`)."""
        price = market.last_price
        if not order.working or price is None:
            return []
        released = released_by(order, price)
        if released:
            order.status = OPEN
        quote = Quote(price, market.bid, market.ask)
        at = fill_price(order, quote, arriving=arriving, at_broker=self.mode is Mode.LIVE)
        if at is None:
            # A stop-limit the price ran through: now resting open, unfilled.
            return [self._change(order, TRIGGERED)] if released else []
        units = order.pending if units_cap is None else min(units_cap, order.pending)
        return [self.fill(order, units, at, now_ns)]

    def fill(self, order: FakeOrder, units: int, price: float, now_ns: int) -> Change:
        """Fill `units` of a working order at `price`. Raises ValueError for a fill the
        order cannot take."""
        self._require_working(order)
        if not 0 < units <= order.pending or price <= 0:
            raise ValueError(
                f"order {order.order_id} can fill 1 to {order.pending} units at a price "
                f"above 0, not {units} units at {price} rupees"
            )
        fill = Fill(
            trade_id=self._ids.trade_id(now_ns),
            order_id=order.order_id,
            strategy=order.strategy,
            symbol=order.symbol,
            exchange=order.exchange,
            product=order.product,
            action=order.action,
            quantity=units,
            price=price,
            ts_ns=now_ns,
        )
        order.filled += units
        order.value += units * price
        if order.filled == order.quantity:
            order.status = COMPLETE
        else:
            order.status = OPEN
        self.fills.append(fill)
        self.positions.take(fill)
        return self._change(order, FILLED, fill)

    def working_orders(self, symbol: str, exchange: str) -> list[FakeOrder]:
        """The working orders of one instrument, oldest first."""
        return [
            order
            for order in self.orders.values()
            if order.working and order.symbol == symbol and order.exchange == exchange
        ]

    @staticmethod
    def _require_working(order: FakeOrder) -> None:
        if not order.working:
            raise ValueError(f"order {order.order_id} is {order.status}, not working")

    @staticmethod
    def _change(order: FakeOrder, event: str, fill: Fill | None = None) -> Change:
        return Change(dataclasses.replace(order), event, fill)


__all__ = [
    "CANCELLED_EVENT",
    "FILLED",
    "MODIFIED",
    "PLACED",
    "REJECTED_EVENT",
    "TRIGGERED",
    "Book",
    "Change",
    "Ids",
    "OrderTerms",
]
