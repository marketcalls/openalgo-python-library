"""Which order update tells more, for the order path (`orders`): updates arrive twice (the
stream and the polls), late and out of order, and some without the price of their fills.

Every rule compares two updates of one order. Quantities are units, prices rupees.
"""

from __future__ import annotations

from types import MappingProxyType

from ._types import OrderStatus, OrderUpdate

#: How far a working order has come: a stop waiting for its trigger, then an open order.
_PROGRESS = MappingProxyType({OrderStatus.TRIGGER_PENDING: 0, OrderStatus.OPEN: 1})


def is_news(seen: OrderUpdate | None, update: OrderUpdate) -> bool:
    """Whether `update` tells more of its order than `seen`, the last update taken.

    More fills are news and fewer are stale. With the same fills, a terminal status, a
    changed quantity (a modify) or a stop the stream reports triggered (trigger pending,
    then open) is news; a status never moves back. A poll's "open" after "trigger
    pending" is not news: brokers' order books may write a waiting stop as open (the
    server's broker/*/mapping/order_data.py). Once terminal, nothing is news.
    """
    if seen is None:
        return True
    if seen.status.terminal:
        return False
    if update.filled_quantity != seen.filled_quantity:
        return update.filled_quantity > seen.filled_quantity
    if update.status.terminal:
        return True
    if update.status is seen.status:
        return update.quantity != seen.quantity
    moves_on = _PROGRESS[update.status] > _PROGRESS[seen.status]
    return moves_on and update.source == "stream"


def is_priceless(update: OrderUpdate) -> bool:
    """Whether an update reports fills without their price: OpenAlgo writes 0 when the
    order book or a polled stream has no average price (services/orderstatus_service.py,
    websocket_proxy/order_adapter.py on the server)."""
    return update.filled_quantity > 0 and update.average_price <= 0


def is_newer(kept: OrderUpdate, update: OrderUpdate) -> bool:
    """Whether `update` replaces `kept` as an order's latest word: it is news, or it is
    the same state with the price `kept` lacked."""
    same_state = (update.status, update.filled_quantity, update.quantity) == (
        kept.status,
        kept.filled_quantity,
        kept.quantity,
    )
    priced_now = is_priceless(kept) and not is_priceless(update)
    return is_news(kept, update) or (same_state and priced_now)


__all__ = ["is_news", "is_newer", "is_priceless"]
