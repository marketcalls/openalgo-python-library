"""Which order update tells more, for the order path (`orders`): updates arrive twice (the
stream and the polls), late and out of order, and some without the price of their fills.

A status is taken as reported, never read into: brokers report a modify, a cancel and a
stop's trigger with the same words, so only fills and terminal statuses drive actions.
Every rule compares two updates of one order. Quantities are units, prices rupees.
"""

from __future__ import annotations

import dataclasses

from ._types import OrderUpdate


def is_news(seen: OrderUpdate | None, update: OrderUpdate) -> bool:
    """Whether `update` tells more of its order than `seen`, the last update taken.

    More units filled is news, fewer is stale. With the same fills, a terminal status or
    a change of the reported status or of the order quantity is news; a change of the
    average price alone is not (`with_known_price`), and a duplicate is not news. Once
    terminal, nothing is.
    """
    if seen is None:
        return True
    if seen.status.terminal:
        return False
    if update.filled_quantity != seen.filled_quantity:
        return update.filled_quantity > seen.filled_quantity
    if update.status.terminal:
        return True
    return (update.status, update.quantity) != (seen.status, seen.quantity)


def with_known_price(seen: OrderUpdate | None, update: OrderUpdate) -> OrderUpdate:
    """`update` at the price already known for its filled quantity: once those units have
    a positive price, it stands for them, so neither a word without a price nor a
    tradebook mean that differs from the broker's own average re-prices a fill the
    engine has."""
    if (
        seen is None
        or seen.average_price <= 0
        or seen.filled_quantity != update.filled_quantity
        or seen.average_price == update.average_price
    ):
        return update
    return dataclasses.replace(update, average_price=seen.average_price)


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


__all__ = ["is_news", "is_newer", "is_priceless", "repeats", "with_known_price"]
