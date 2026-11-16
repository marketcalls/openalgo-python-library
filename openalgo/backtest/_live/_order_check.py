"""One `orderstatus` read of an order of the session, for the order path (`orders`): the
call itself (through the server watch and the mode guard), the polled word it gives the
follower (`_order_follow`), and what the row says of the order now (`OrderCheck`), which
settles a modify or a cancel whose answer was lost or refused.

The row is the broker's order-book row (the sandbox's own in SANDBOX mode,
services/orderstatus_service.py on the server): a broker's carries no filled quantity,
and the books write the price type as "pricetype", the sandbox as "price_type".
Quantities are units, prices rupees, times UTC nanoseconds.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..errors import LiveError
from ._answers import server_answer
from ._order_answers import ServerWatch
from ._order_trades import Polled
from ._order_updates import fills_given, given, normalize_update, number, status_of, units
from ._types import LOGGER_NAME, GuardLike, OrderStatus, OrderUpdate

logger = logging.getLogger(LOGGER_NAME)


@dataclass(frozen=True)
class OrderCheck:
    """What one `orderstatus` read found of an order of the session.

    When the row was read, `status` is its status as reported, `quantity` its units,
    `filled_quantity` the units filled when the row gives them (None when it does not: a
    broker's book has no such field), `price` and `trigger_price` in rupees (0.0 where
    they do not apply) and `price_type` ("MARKET", "LIMIT", "SL", "SL-M"). When it was
    not (an error, a timeout, a row that cannot be read), every one of them is None and
    `reason` says why. `news` is the update the read handed to the engine, as a poll's
    (journaled and checked by the guard), or None: nothing new, or a fill or cancel held
    until the tradebook confirms it (the polls hand it over then).
    """

    order_id: str
    status: OrderStatus | None = None
    quantity: int | None = None
    filled_quantity: int | None = None
    price: float | None = None
    trigger_price: float | None = None
    price_type: str | None = None
    news: OrderUpdate | None = None
    reason: str = ""

    @property
    def readable(self) -> bool:
        """True when the row was read: its status and terms are given."""
        return self.status is not None


def read_status(
    client: Any, *, guard: GuardLike, watch: ServerWatch, order_id: str, strategy: str, now: int
) -> Mapping[str, Any]:
    """One `orderstatus` call for an order sent for `strategy`: the server's answer
    (`server_answer`), noted by `watch` at `now` and checked by `guard` (the analyzer
    switch routes the read, so an answer of the other mode locks it)."""
    answer = server_answer(client.orderstatus(order_id=order_id, strategy=strategy))
    watch.note(answer, now)
    guard.check_response(answer)
    return answer


def polled_word(answer: Mapping[str, Any], order_id: str, now: int) -> Polled | None:
    """The word an `orderstatus` answer received at `now` gives the follower: the order's
    update, or the answer itself when an open row's fills are to be read from the
    tradebook. A cancelled row without its fills is a cancel that reports none, which the
    follower holds until the tradebook confirms its fills, so a tradebook that cannot be
    read leaves the cancel known and held, not lost. None (logged) for a failed answer or
    a row that cannot be read."""
    data = answer.get("data")
    if answer.get("status") != "success" or not isinstance(data, Mapping):
        logger.warning("orderstatus of %s failed: %s", order_id, answer.get("message"))
        return None
    try:
        if fills_given(data, order_id):
            return normalize_update(answer, ts_recv_ns=now, source="poll")
        if status_of(data, order_id) is OrderStatus.CANCELLED:
            unfilled = {**data, "filled_quantity": 0, "average_price": 0.0}
            return normalize_update({**answer, "data": unfilled}, ts_recv_ns=now, source="poll")
        return answer
    except LiveError as error:
        logger.error("orderstatus of %s cannot be read: %s", order_id, error)
        return None


def check_of(answer: Mapping[str, Any], order_id: str, news: OrderUpdate | None) -> OrderCheck:
    """What an `orderstatus` answer says of order `order_id`, with the `news` it gave."""
    data = answer.get("data")
    if answer.get("status") != "success" or not isinstance(data, Mapping):
        reason = f"orderstatus failed: {answer.get('message') or 'no message'}"
        return OrderCheck(order_id, news=news, reason=reason)
    try:
        filled = given(data.get("filled_quantity"))
        return OrderCheck(
            order_id,
            status=status_of(data, order_id),
            quantity=units(data, "quantity", order_id),
            filled_quantity=units(data, "filled_quantity", order_id) if filled else None,
            price=number(data, "price", order_id),
            trigger_price=number(data, "trigger_price", order_id),
            price_type=_price_type(data, order_id),
            news=news,
        )
    except LiveError as error:
        return OrderCheck(order_id, news=news, reason=str(error))


def _price_type(row: Mapping[str, Any], order_id: str) -> str:
    """The row's price type, upper case. Raises LiveError when it gives none."""
    price_type = str(row.get("pricetype") or row.get("price_type") or "").strip().upper()
    if not price_type:
        raise LiveError(f"order {order_id}: the row gives no price type")
    return price_type


__all__ = ["OrderCheck", "check_of", "polled_word", "read_status"]
