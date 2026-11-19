"""Unbound intents matched against OpenAlgo's order book, for the order path (`orders`).

An intent is unbound when the answer to its `placeorder` was lost (a 502, a timeout, a
lost connection): its order may or may not be at the broker. The order book settles it:
the one order of the intent's shape placed at or after it is the intent's, and no such
order, read twice, means it was never placed. Times are UTC nanoseconds, quantities units.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import enum
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._layout import IST_OFFSET_NANOS, NANOS_PER_SECOND
from ..errors import LiveError
from ._order_updates import units
from ._types import OrderIntent, OrderUpdate

#: The preflight refuses a local clock more than 2 seconds off (08-live-trading section
#: 4), so an order-book time up to this much before an intent can be the intent's order.
CLOCK_SKEW_LIMIT_NS = 2 * NANOS_PER_SECOND
#: Order-book times brokers write that ISO 8601 does not cover (the `timestamp` of the
#: server's broker/*/mapping/order_data.py), read as IST wall clock.
_BOOK_TIME_FORMATS = ("%d-%b-%Y %H:%M:%S", "%d-%m-%Y %H:%M:%S", "%H:%M:%S %d-%m-%Y")


class Resolved(enum.Enum):
    """What the order book said of an unbound intent."""

    #: One order matched: the intent is bound to it.
    BOUND = "bound"
    #: Two reads found no order: the intent never became an order.
    NOT_PLACED = "not_placed"
    #: Orders matched that the path cannot tell apart: the intent stays unbound and the
    #: operator settles it.
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class Resolution:
    """An unbound intent settled against the order book: `order_id` when BOUND, with
    `update`, the order's latest update when one arrived before it was bound (None
    otherwise, or while its fill's price is not known); the order IDs that may be the
    intent's in `candidates` when AMBIGUOUS."""

    intent: OrderIntent
    kind: Resolved
    order_id: str = ""
    update: OrderUpdate | None = None
    candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class BookMatch:
    """The order-book rows that may be an unbound intent's order: `orders` placed at or
    after it, and `unsure` ones matching on every field but a quantity or a time that
    cannot be read."""

    orders: tuple[str, ...] = ()
    unsure: tuple[str, ...] = ()


def book_rows(answer: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The order rows of a successful `orderbook` answer (`data.orders`). Raises
    LiveError when it holds no list of orders."""
    data = answer.get("data")
    rows = data.get("orders") if isinstance(data, Mapping) else data
    if not isinstance(rows, list):
        raise LiveError(f"the order book answer holds no list of orders: {dict(answer)!r}")
    return rows


def match_unbound(intent: OrderIntent, rows: list[Mapping[str, Any]], taken: set[str]) -> BookMatch:
    """The rows that may be an unbound intent's order, leaving out the orders in `taken`
    (owned by other intents).

    A row matches on the intent's strategy (when the row names one: brokers' books do
    not), symbol, exchange, action, price type, product and quantity, placed no earlier
    than the intent's second less `CLOCK_SKEW_LIMIT_NS`.
    """
    earliest_ns = intent.created_ns - intent.created_ns % NANOS_PER_SECOND - CLOCK_SKEW_LIMIT_NS
    orders: list[str] = []
    unsure: list[str] = []
    for row in rows:
        order_id = str(row.get("orderid") or "")
        if not order_id or order_id in taken or not _same_shape(intent, row):
            continue
        try:
            quantity = units(row, "quantity", order_id)
        except LiveError:
            # It may be the intent's order: the operator is told, never guessed for.
            unsure.append(order_id)
            continue
        if quantity != intent.quantity:
            continue
        placed_ns = book_time_ns(row.get("timestamp"))
        if placed_ns is None:
            unsure.append(order_id)
        elif placed_ns >= earliest_ns:
            orders.append(order_id)
    return BookMatch(tuple(orders), tuple(unsure))


def book_time_ns(text: Any) -> int | None:
    """An order-book time as UTC nanoseconds, to the second; None when it cannot be read.
    A time without a zone is IST wall clock, as OpenAlgo and the brokers write it."""
    moment = _parse_book_time(str(text or "").strip())
    if moment is None:
        return None
    if moment.tzinfo is not None:
        return math.floor(moment.timestamp()) * NANOS_PER_SECOND
    wall_ns = math.floor(moment.replace(tzinfo=dt.UTC).timestamp()) * NANOS_PER_SECOND
    return wall_ns - IST_OFFSET_NANOS


def _same_shape(intent: OrderIntent, row: Mapping[str, Any]) -> bool:
    strategy = row.get("strategy")
    # The sandbox's book says "price_type" in `orderstatus`, the books "pricetype".
    price_type = row.get("pricetype") or row.get("price_type") or ""
    return (
        (not strategy or strategy == intent.strategy)
        and row.get("symbol") == intent.symbol
        and row.get("exchange") == intent.exchange
        and str(row.get("action") or "").upper() == intent.action
        and str(price_type).upper() == intent.price_type
        and str(row.get("product") or "").upper() == intent.product
    )


def _parse_book_time(text: str) -> dt.datetime | None:
    with contextlib.suppress(ValueError):
        return dt.datetime.fromisoformat(text)
    for fmt in _BOOK_TIME_FORMATS:
        with contextlib.suppress(ValueError):
            return dt.datetime.strptime(text, fmt)
    return None


__all__ = [
    "CLOCK_SKEW_LIMIT_NS",
    "BookMatch",
    "Resolution",
    "Resolved",
    "book_rows",
    "book_time_ns",
    "match_unbound",
]
