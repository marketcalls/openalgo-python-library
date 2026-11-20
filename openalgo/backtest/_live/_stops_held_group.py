"""A held cancel on an instrument several strategies of the session share, decided by the
group's two-book read (18-multi-strategy 4.5, the lead's decision M1; unit M7).

A held cancel is a member's order the broker cancelled with fills the trade book has not
confirmed (16-session-design 4.11). The position book shows the net of every member, so
what that one order filled can be read from it only when the net moved by that order
alone. The targeted read takes the position book first and the order book after it
(`ReadPositions(orders=True)`), and the cancel is decided only when nothing of another
member can be at the broker unheard (`unheard`):

- no other member's order in flight (handed to the order worker and unanswered, held
  back, unbound or ambiguous: `_stops_orders.IN_FLIGHT`), and no working order of another
  member but its confirmed resting stop with nothing pending on it: any other order may be
  filling now, and a live order book does not count fills;
- in the order book read after the position book, every other member's resting stop
  working with no fill its engine lacks: not ended (complete, cancelled or rejected) while
  its engine works it, and no more units filled than its engine holds where the book counts
  them (a sandbox book does); a stop the book does not show at all is not known either.

The other held words of the group (a fill held for its price) the caller rules out first
(`_stops_held.HeldWords`). Reading the position book first means a fill landing between
the two reads shows in the order book and stops the decision, never the reverse.

Then the broker's net is weighed against the sum of the members' units (`decide_shared`):
equal, the order filled nothing more; fewer on the order's closing side by `d`, at most its
units left, it filled `d` before its cancel; anything else, the caller waits, reads again
and alerts.

Quantities are signed units (long above 0). Used by the loop only.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..errors import LiveError
from ._order_book import book_rows
from ._order_updates import status_of
from ._session_trader import Trader
from ._stops_orders import IN_FLIGHT, Sent
from ._types import OrderStatus
from ._venue_requests import LIVE_PRODUCT

#: An order-book row as the decision reads it: its status, and its units filled when the
#: book counts them (None when it does not).
BookRow = tuple[OrderStatus, int | None]


def unheard(session: Any, trader: Trader, orderbook: Mapping[str, Any] | None) -> str | None:
    """Why the held cancel of `trader`, a member of a shared instrument, cannot be decided
    from a read whose order book, read after its position book, is `orderbook` (module
    docstring), in words; None when nothing of another member can be at the broker
    unheard. `session` is the `_session_loop.Session`."""
    rows = None if orderbook is None else _rows(orderbook, trader.symbol, trader.exchange)
    if rows is None:
        return "the order book could not be read after the position book"
    for member in session.groups.members(trader):
        if member is trader or member.finished:
            continue
        why = _member_unheard(session, member, rows)
        if why is not None:
            return why
    return None


def decide_shared(
    net: int | None, group_units: int, order: Mapping[str, Any], action: str
) -> int | None:
    """The units a member's cancelled order filled in all (module docstring): `net` the
    broker's net units of the instrument (None for a book that could not be read),
    `group_units` the sum of the members' units, `order` the kernel's order
    (`filled_units`, `quantity_units`) and `action` its side ("BUY" or "SELL"); None when
    the books do not tell."""
    if net is None:
        return None
    known = round(order["filled_units"])
    left = round(order["quantity_units"]) - known
    if net == group_units:
        return known
    filled = group_units - net if action == "SELL" else net - group_units
    return known + filled if 0 < filled <= left else None


def _member_unheard(session: Any, member: Trader, rows: Mapping[str, BookRow]) -> str | None:
    """Why `member`'s orders may hold a fill its engine has not taken (module docstring),
    in words; None when none can."""
    leg = session.stops.leg(member)
    if any(watched.state in IN_FLIGHT for watched in leg.orders.values()):
        return f"an order of {member.name} is in flight"
    for kernel_order in member.working():
        watched = leg.orders.get(kernel_order)
        resting = kernel_order == leg.stop and watched is not None
        if not (resting and watched.state is Sent.CONFIRMED and not watched.cancel_sent):
            return f"order o{kernel_order} of {member.name} works and may be filling"
        if member.venue.changing(kernel_order):
            return f"the stop of {member.name} has a change pending"
        order_id = member.venue.order_id_of(kernel_order)
        row = rows.get(order_id)
        if row is None:
            return f"the order book does not show the stop {order_id} of {member.name}"
        status, filled = row
        if status.terminal or (filled is not None and filled > member.venue.filled(kernel_order)):
            return f"the stop {order_id} of {member.name} shows a fill its engine has not taken"
    return None


def _rows(answer: Mapping[str, Any], symbol: str, exchange: str) -> dict[str, BookRow] | None:
    """The order book's rows of `exchange:symbol` in the live product, by order ID; None
    when the answer or a row of the instrument cannot be read (a guess is never made)."""
    if answer.get("status") != "success":
        return None
    try:
        rows = book_rows(answer)
    except LiveError:
        return None
    read: dict[str, BookRow] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            return None
        same = (row.get("symbol"), row.get("exchange")) == (symbol, exchange)
        if not same or row.get("product", LIVE_PRODUCT) != LIVE_PRODUCT:
            continue
        order_id = str(row.get("orderid") or "")
        try:
            status = status_of(row, order_id)
            counted = row.get("filled_quantity")
            filled = None if counted is None else round(float(counted))
        except (LiveError, TypeError, ValueError):
            return None
        read[order_id] = (status, filled)
    return read


__all__ = ["BookRow", "decide_shared", "unheard"]
