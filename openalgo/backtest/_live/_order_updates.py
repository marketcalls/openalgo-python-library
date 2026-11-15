"""Order updates read from OpenAlgo's answers, for the order path (`orders`): a message of
the order-updates stream or an `orderstatus` answer, in live or sandbox mode, as an
`OrderUpdate`.

Quantities are units, prices rupees, times UTC nanoseconds. Nothing is guessed: a status
word or a field the order path cannot read raises LiveError naming it.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from ..errors import LiveError
from ._types import OrderStatus, OrderUpdate, mode_of

#: Order-status words of the stream, `orderstatus` and the order book, live and sandbox.
#: Brokers' books and the sandbox write "trigger pending" (sandbox/order_manager.py on the
#: server); the stream's vocabulary has an underscore (websocket_proxy/order_adapter.py).
STATUS_WORDS = MappingProxyType(
    {
        "open": OrderStatus.OPEN,
        "trigger pending": OrderStatus.TRIGGER_PENDING,
        "trigger_pending": OrderStatus.TRIGGER_PENDING,
        "complete": OrderStatus.COMPLETE,
        "rejected": OrderStatus.REJECTED,
        "cancelled": OrderStatus.CANCELLED,
    }
)
#: Statuses whose filled quantity the status itself gives when an answer omits it: a
#: rejected order and a stop not yet triggered have none (a complete one has it all).
_UNFILLED_STATUSES = frozenset({OrderStatus.REJECTED, OrderStatus.TRIGGER_PENDING})
#: The `type` of an order-updates stream message (subscribers/wsproxy_subscriber.py).
_STREAM_TYPE = "order_update"
_SOURCES = frozenset({"stream", "poll"})


def normalize_update(message: Mapping[str, Any], *, ts_recv_ns: int, source: str) -> OrderUpdate:
    """An `OrderUpdate` from an order-updates stream message (`"type": "order_update"`)
    or a successful `orderstatus` answer, in live or sandbox mode.

    `ts_recv_ns` is when it was received (UTC nanoseconds) and `source` "stream" or
    "poll". A complete order with no filled quantity (the server's polled stream sends 0
    for a book row without one) has filled all of it; a rejected one, or a stop not yet
    triggered, without one has filled none. Raises LiveError for any other shape, an
    unknown status word (named), a missing or unreadable field, and an open or cancelled
    order without its filled quantity (read its fills from the tradebook).
    """
    if source not in _SOURCES:
        raise LiveError(f"an order update's source is 'stream' or 'poll', not {source!r}")
    if message.get("type") == _STREAM_TYPE:
        row: Mapping[str, Any] = message
    elif message.get("status") == "success" and isinstance(message.get("data"), Mapping):
        row = message["data"]
    else:
        raise LiveError(
            f"not an order update or a successful orderstatus answer: {dict(message)!r}"
        )
    order_id = str(row.get("orderid") or "")
    if not order_id:
        raise LiveError(f"an order update without an order ID: {dict(row)!r}")
    status = status_of(row, order_id)
    quantity = units(row, "quantity", order_id)
    filled = _filled_of(row, status, quantity, order_id)
    if given(row.get("pending_quantity")):
        pending = units(row, "pending_quantity", order_id)
    else:
        pending = 0 if status.terminal else quantity - filled
    return OrderUpdate(
        order_id=order_id,
        symbol=str(row.get("symbol") or ""),
        exchange=str(row.get("exchange") or ""),
        action=str(row.get("action") or "").upper(),
        quantity=quantity,
        filled_quantity=filled,
        pending_quantity=pending,
        # An order with no fill has no average price: OpenAlgo writes 0 or leaves it out.
        average_price=number(row, "average_price", order_id) if row.get("average_price") else 0.0,
        status=status,
        rejection_reason=str(row.get("rejection_reason") or ""),
        mode=mode_of(message),
        ts_recv_ns=ts_recv_ns,
        source=source,
    )


def status_of(row: Mapping[str, Any], order_id: str) -> OrderStatus:
    """The status of an order-update or order-book row. Raises LiveError for a word the
    order path does not know, naming it."""
    word = str(row.get("order_status") or "").strip().lower()
    status = STATUS_WORDS.get(word)
    if status is None:
        raise LiveError(
            f"order {order_id} has status {row.get('order_status')!r}, which the order path "
            f"does not know (it knows {sorted(STATUS_WORDS)})"
        )
    return status


def fills_given(row: Mapping[str, Any], order_id: str) -> bool:
    """Whether a row gives its filled quantity, or its status settles it; when not, the
    fills are read from the tradebook. Raises LiveError for an unknown status."""
    if given(row.get("filled_quantity")):
        return True
    return status_of(row, order_id) not in (OrderStatus.OPEN, OrderStatus.CANCELLED)


def given(value: Any) -> bool:
    """Whether a field holds a value: OpenAlgo leaves out, nulls or blanks one it lacks."""
    return value is not None and value != ""


def number(row: Mapping[str, Any], field: str, order_id: str) -> float:
    """A numeric field of a row. Raises LiveError when it is missing or not a number."""
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        raise LiveError(f"order {order_id}: {field} {row.get(field)!r} is not a number") from None


def units(row: Mapping[str, Any], field: str, order_id: str) -> int:
    """A quantity field in whole units. Raises LiveError when it is missing, not a number
    or not whole."""
    value = number(row, field, order_id)
    if not value.is_integer():
        raise LiveError(f"order {order_id}: {field} {value} is not a whole number of units")
    return int(value)


def _filled_of(row: Mapping[str, Any], status: OrderStatus, quantity: int, order_id: str) -> int:
    filled = units(row, "filled_quantity", order_id) if given(row.get("filled_quantity")) else None
    if status is OrderStatus.COMPLETE and not filled:
        return quantity
    if filled is not None:
        return filled
    if status in _UNFILLED_STATUSES:
        return 0
    raise LiveError(
        f"order {order_id} is {status.value} but its filled quantity is not given: read its "
        "fills from the tradebook"
    )


__all__ = [
    "STATUS_WORDS",
    "fills_given",
    "given",
    "normalize_update",
    "number",
    "status_of",
    "units",
]
