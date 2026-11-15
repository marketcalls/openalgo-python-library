"""Which order update tells more, for the order path (`orders`): updates arrive twice (the
stream and the polls), late and out of order, and some without the price of their fills.

A status is taken as reported, never read into: brokers report a modify, a cancel and a
stop's trigger with the same words, so only fills and terminal statuses drive actions.
Every rule compares two updates of one order. Quantities are units, prices rupees.
"""

from __future__ import annotations

from ._types import OrderUpdate


def is_news(seen: OrderUpdate | None, update: OrderUpdate) -> bool:
    """Whether `update` tells more of its order than `seen`, the last update taken.

    More units filled is news, fewer is stale. With the same fills, a terminal status or
    any change of the reported status, order quantity or average price is news; a word
    with no price for fills already known at a price changes nothing, and a duplicate is
    not news. Once terminal, nothing is.
    """
    if seen is None:
        return True
    if seen.status.terminal:
        return False
    if update.filled_quantity != seen.filled_quantity:
        return update.filled_quantity > seen.filled_quantity
    if update.status.terminal:
        return True
    price = seen.average_price if _loses_the_price(seen, update) else update.average_price
    return (update.status, update.quantity, price) != (
        seen.status,
        seen.quantity,
        seen.average_price,
    )


def is_priceless(update: OrderUpdate) -> bool:
    """Whether an update reports fills without their price: OpenAlgo writes 0 when the
    order book or a polled stream has no average price (services/orderstatus_service.py,
    websocket_proxy/order_adapter.py on the server)."""
    return update.filled_quantity > 0 and update.average_price <= 0


def is_newer(kept: OrderUpdate, update: OrderUpdate) -> bool:
    """Whether `update` replaces `kept` as an order's latest word: it is news, or it is
    the same state with the price `kept` lacked (a terminal word included)."""
    same_state = (update.status, update.filled_quantity, update.quantity) == (
        kept.status,
        kept.filled_quantity,
        kept.quantity,
    )
    priced_now = is_priceless(kept) and not is_priceless(update)
    return is_news(kept, update) or (same_state and priced_now)


def repeats(seen: OrderUpdate | None, update: OrderUpdate) -> bool:
    """Whether a word that is not news says exactly what is known of a working order:
    worth keeping in the journal as the stream's own word, unlike a stale word, one that
    only lacks the known price, or anything after a terminal status."""
    if seen is None or seen.status.terminal:
        return False
    return (update.status, update.filled_quantity, update.quantity, update.average_price) == (
        seen.status,
        seen.filled_quantity,
        seen.quantity,
        seen.average_price,
    )


def _loses_the_price(seen: OrderUpdate, update: OrderUpdate) -> bool:
    return is_priceless(update) and seen.average_price > 0


__all__ = ["is_news", "is_newer", "is_priceless", "repeats"]
