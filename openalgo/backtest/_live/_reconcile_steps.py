"""What the reconciler (`reconcile`) judges a leg by (16-session-design 5.1 and 5.2;
unit 12, split from `_reconcile_act` in unit M6): the per-leg in-flight rule, the
session's own orders, and a comparison's steps on the trade and order books (`attribute`,
`order_findings`). What a finding seen twice in a row does is `_reconcile_act`'s.

**In flight** (`in_flight`): an order of the leg handed to the order worker and not
answered, unbound, held back or ambiguous; an add whose answer was lost; a word the order
path holds for the trade book; a modify or cancel waiting for its result or its bind; any
order of the leg working other than its confirmed, unchanged stop, and other than those
the exit guard holds (they are not at the broker), and, while the guard holds the stop's
cancel, an exit never sent that waits for it.

Quantities are signed units, prices rupees, times UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

import math
from typing import Any

from .._engine import call
from ._reconcile_rules import LegBooks
from ._session_trader import Trader
from ._stops_orders import IN_FLIGHT, Sent
from ._types import OrderStatus, OrderUpdate
from ._venue_requests import SAME_PRICE_RUPEES, STOP_TYPES

#: A finding: its kind first (a `_reconcile_rules.Kind`, or "orphan", "foreign",
#: "stop_terms"), then what tells one from another.
Finding = tuple[Any, ...]


def in_flight(session: Any, trader: Trader) -> bool:
    """Whether `trader`'s leg is in flight (module docstring)."""
    stops, guard = session.stops, session.reconcile.guard
    leg = stops.leg(trader)
    if leg.unbound_add_units or stops.held.holds(trader):
        return True
    if any(watched.state in IN_FLIGHT for watched in leg.orders.values()):
        return True
    held, waits = guard.held_orders(trader.name), bool(guard.held_cancels(trader.name))
    for kernel_order in trader.working():
        watched = leg.orders.get(kernel_order)
        if kernel_order in held or (waits and watched is None):
            continue
        if trader.venue.changing(kernel_order):
            return True
        resting = kernel_order == leg.stop and watched is not None
        if not (resting and watched.state is Sent.CONFIRMED and not watched.cancel_sent):
            return True
    return False


def attribute(session: Any, trader: Trader, books: LegBooks, at_ns: int) -> bool:
    """A comparison's first step (5.2): each own order's trades in the trade book, beyond
    what its fill ledger has given the kernel, through the ledger as news read at `at_ns`
    (complete once they come to the order's units). True when any gave the kernel units:
    the read took the position book before the trade book, so its position book may
    predate them (16-session-design 5.4, round 9)."""
    applied = False
    for order_id, kernel_order in trader.venue.bound().items():
        units, average = books.fills(order_id)
        if units <= trader.venue.filled(kernel_order):
            continue
        order = call(trader.engine.order, kernel_order)
        intent = trader.venue.intent_of(kernel_order)
        quantity = intent.quantity if order is None else round(order["quantity_units"])
        update = OrderUpdate(
            order_id=order_id,
            symbol=intent.symbol,
            exchange=intent.exchange,
            action=intent.action,
            quantity=quantity,
            filled_quantity=units,
            pending_quantity=max(0, quantity - units),
            average_price=average,
            status=OrderStatus.COMPLETE if units >= quantity else OrderStatus.OPEN,
            rejection_reason="",
            mode=session.parts.mode,
            ts_recv_ns=at_ns,
            source="reconcile",
        )
        session.event("attributed", strategy=trader.name, order_id=order_id, filled=units)
        applied |= session.news.apply_update(trader, update).increment is not None
    return applied


def order_findings(
    session: Any,
    trader: Trader,
    books: LegBooks,
    own: frozenset[str],
    *,
    others: frozenset[str] = frozenset(),
    orphans_of_earlier: bool = True,
) -> tuple[set[Finding], bool]:
    """A comparison's second and fourth steps (5.2): the order book's findings (an own
    order working that the engine does not hold, a working order naming the strategy that
    the session did not send, the leg's stop with other units or trigger than the
    engine's), and whether a row has ended while the engine works its order (a poll is
    wanted). On a shared instrument (18-multi-strategy 4.2) the rows of `others`, the
    orders bound to its other strategies, are theirs, and an order of an earlier start no
    strategy holds is found only where `orphans_of_earlier` (the group's first member)."""
    findings: set[Finding] = set()
    if books.rows is None:
        return findings, False
    bound, working = trader.venue.bound(), set(trader.working())
    polls = False
    for row in books.rows:
        kernel_order = bound.get(row.order_id)
        if row.order_id in others:
            continue
        if kernel_order in working and not row.working:
            polls = True
        elif row.working and row.order_id in own and kernel_order not in working:
            if kernel_order is not None or orphans_of_earlier:
                findings.add(("orphan", row.order_id))
        elif row.working and row.order_id not in own and row.strategy == trader.name:
            findings.add(("foreign", row.order_id))
    stop = session.stops.leg(trader).stop
    order_id = trader.venue.order_id_of(stop) if stop is not None else ""
    row = books.row(order_id) if order_id else None
    if stop is None or row is None or not row.working or row.price_type not in STOP_TYPES:
        return findings, polls
    kernel = call(trader.engine.order, stop)
    trigger = float(kernel["trigger_rupees"] or 0.0)
    same = math.isclose(row.trigger_price, trigger, rel_tol=0.0, abs_tol=SAME_PRICE_RUPEES)
    if row.quantity != round(kernel["quantity_units"]) or not same:
        findings.add(("stop_terms", order_id, row.quantity, row.trigger_price))
    return findings, polls


def own_orders(trader: Trader, earlier: frozenset[str]) -> frozenset[str]:
    """The broker order IDs of the session's own orders a leg's trades may be of: those
    its venue map bound, and those of the day's earlier starts (the journal's)."""
    return frozenset(trader.venue.bound()) | earlier


__all__ = ["Finding", "attribute", "in_flight", "order_findings", "own_orders"]
