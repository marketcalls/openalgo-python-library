"""An order's fills read from OpenAlgo's tradebook, for the order path (`orders`).

The tradebook is the one source that gives every trade of an order: `orderstatus` prices
a complete order at its first trade (services/orderstatus_service.py on the server), a
broker's book often has no filled quantity, and on the brokers the server polls itself
(groww and samco; `_POLLING_BROKERS` in services/order_update_service.py) every stream
message says 0 filled. So a polled fill is priced, and a cancel's fills confirmed, from
the tradebook. A broker whose tradebook names orders by another ID cannot be served
(`UNSUPPORTED_BROKERS`). Quantities are units, prices rupees.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from ..errors import LiveError
from ._answers import server_answer
from ._order_answers import ServerWatch
from ._order_news import is_priceless
from ._order_updates import normalize_update, number, units
from ._types import LOGGER_NAME, GuardLike, OrderStatus, OrderUpdate

logger = logging.getLogger(LOGGER_NAME)

#: An order's word from a poll: its update, or the `orderstatus` answer whose fills are
#: to be read from the tradebook because the broker's book does not give them.
Polled = OrderUpdate | Mapping[str, Any]
#: An order's units filled and their quantity-weighted average price in the tradebook.
Fills = tuple[int, float]
#: The brokers the order path cannot trade through, by OpenAlgo's broker name, with why:
#: the preflight refuses them.
UNSUPPORTED_BROKERS = MappingProxyType(
    {
        "fivepaisa": (
            "its tradebook names each trade by the exchange's order ID (ExchOrderID) while "
            "its orders, order book and order stream use the broker's (BrokerOrderId), so "
            "a fill cannot be matched to its order (broker/fivepaisa/mapping/order_data.py "
            "on the OpenAlgo server)"
        ),
    }
)


def needs_trades(update: OrderUpdate) -> bool:
    """Whether an update is handed over only once the tradebook has confirmed it: a fill
    with no price, a fill found by polling (`orderstatus` prices it at its first trade),
    or a cancel that reports no fill (it may have filled in part: a polled stream always
    says 0 filled)."""
    polled_fill = update.source == "poll" and update.filled_quantity > 0
    unfilled_cancel = update.status is OrderStatus.CANCELLED and update.filled_quantity == 0
    return is_priceless(update) or polled_fill or unfilled_cancel


def wants_trades(word: Polled) -> bool:
    """Whether a polled word needs the tradebook: its fills are not given, or it is an
    update the tradebook must confirm (`needs_trades`)."""
    return not isinstance(word, OrderUpdate) or needs_trades(word)


def read_fills(trades: list[Mapping[str, Any]], order_ids: set[str]) -> dict[str, Fills]:
    """Each order's fills in the tradebook's rows (no trade: 0 units at 0.0). An order
    whose trade rows cannot be read is logged and left out."""
    fills = {}
    for order_id in order_ids:
        own = [trade for trade in trades if str(trade.get("orderid")) == order_id]
        try:
            filled = sum(units(trade, "quantity", order_id) for trade in own)
            value = sum(
                units(trade, "quantity", order_id) * number(trade, "average_price", order_id)
                for trade in own
            )
        except LiveError as error:
            logger.error("the trades of order %s cannot be read: %s", order_id, error)
            continue
        fills[order_id] = (filled, value / filled if filled else 0.0)
    return fills


def read_tradebook(
    client: Any, *, guard: GuardLike, watch: ServerWatch, order_ids: set[str], now: int
) -> dict[str, Fills] | None:
    """The fills of `order_ids` in one `tradebook` read at `now`, its answer noted by
    `watch` and checked by `guard`: none when it cannot be read (logged), None when its
    answer locked the guard (it came from the other mode's book)."""
    answer = server_answer(client.tradebook())
    watch.note(answer, now)
    guard.check_response(answer)
    if guard.locked:
        return None
    trades = answer.get("data")
    if answer.get("status") != "success" or not isinstance(trades, list):
        logger.warning("the tradebook could not be read: %s", answer.get("message"))
        return {}
    return read_fills(trades, order_ids)


def with_fills(word: Polled, fills: Fills | None, now: int) -> tuple[OrderUpdate | None, bool]:
    """A word completed from the order's fills, and whether the tradebook confirmed it.

    A polled answer without its fills takes the tradebook's. A cancel takes the
    tradebook's fills when they are as many as it reports or more (fewer: the tradebook
    is behind). A fill takes the tradebook's average price when the units agree. With no
    fills read, or when they do not agree, the word is returned unconfirmed; an answer
    without its fills, or one that cannot be read, gives None.
    """
    if not isinstance(word, OrderUpdate):
        if fills is None:
            return None, False
        data = {**word["data"], "filled_quantity": fills[0], "average_price": fills[1]}
        try:
            return normalize_update({**word, "data": data}, ts_recv_ns=now, source="poll"), True
        except LiveError as error:
            logger.error("an orderstatus answer cannot be read: %s", error)
            return None, False
    if fills is None:
        return word, False
    filled, average_price = fills
    if word.status is OrderStatus.CANCELLED:
        if filled < word.filled_quantity:
            return word, False
        confirmed = dataclasses.replace(
            word, filled_quantity=filled, pending_quantity=0, average_price=average_price
        )
        return confirmed, True
    if word.filled_quantity == 0:
        return word, True
    if filled != word.filled_quantity:
        return word, False
    return dataclasses.replace(word, average_price=average_price), True


__all__ = [
    "UNSUPPORTED_BROKERS",
    "Fills",
    "Polled",
    "needs_trades",
    "read_fills",
    "read_tradebook",
    "wants_trades",
    "with_fills",
]
