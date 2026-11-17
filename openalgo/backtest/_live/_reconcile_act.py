"""What the reconciler (`reconcile`) judges a leg by and does about a mismatch
(16-session-design 5.1 to 5.3): the per-leg in-flight rule, the session's own orders, a
comparison's steps on the trade and order books (`attribute`, `order_findings`), and the
safe corrections a finding seen twice in a row allows.

**In flight** (`in_flight`): an order of the leg handed to the order worker and not
answered, unbound, held back or ambiguous; an add whose answer was lost; a word the order
path holds for the trade book; a modify or cancel waiting for its result or its bind; any
order of the leg working other than its confirmed, unchanged stop, and other than those
the exit guard holds (they are not at the broker), and, while the guard holds the stop's
cancel, an exit never sent that waits for it.

**The safe corrections** (`Corrections.act`, decided 2026-11-16): the strategy's entries
locked for the rest of the session ("reconciliation") and a RECONCILE_MISMATCH alert
naming the leg, both figures and the orders, then only what can never open or add
exposure: an explained difference adopted (`apply_external_fill` at the other orders'
price, the kernel fitting the leg's stop to the adopted units in the same call, and a
cancel of it the exit guard held sent when the stop is larger than what is left, or,
after the session's last lines with units left, withdrawn first so that the fit cuts the
stop instead: 16-session-design 3.3 step 5 "As built", round 7), the leg's own stop
cancelled when the broker holds the other side, the session's own orphan cancelled.
Never an exit, never the kill switch. A finding on a leg in flight past the in-flight
limit is alerted, with nothing done; so is a restart's difference whose journal units
the restored engine no longer holds (`stale`), with entries locked.

Quantities are signed units, prices rupees, times UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from .._engine import call
from ..errors import VenueRefusalError
from ._plan_catch_up import INSTRUMENT
from ._reconcile_rules import FEWER_STOP_CANCELLED, RULES, Kind, LegBooks, adoption_price
from ._session_news import RECONCILE_LOCK
from ._session_trader import Trader
from ._stops_orders import IN_FLIGHT, Sent
from ._types import LOGGER_NAME, AlertKind, OrderStatus, OrderUpdate
from ._venue_requests import LIVE_PRODUCT, SAME_PRICE_RUPEES, STOP_TYPES
from ._worker_queue import CancelOrphan

logger = logging.getLogger(LOGGER_NAME)

#: The exit reason the kernel records for units adopted from other orders' trades.
ADOPTED_REASON = "manual"
#: What an adoption did when it also sent the cancel of the leg's stop that the exit guard
#: held (`ExitGuard.adopted`): the stop is cancelled rather than cut down.
ADOPTED_STOP_CANCELLED = "adopted, its held stop cancel sent"
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


def attribute(session: Any, trader: Trader, books: LegBooks, at_ns: int) -> None:
    """A comparison's first step (5.2): each own order's trades in the trade book, beyond
    what its fill ledger has given the kernel, through the ledger as news read at `at_ns`
    (complete once they come to the order's units)."""
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
        session.news.apply_update(trader, update)


def order_findings(
    session: Any, trader: Trader, books: LegBooks, own: frozenset[str]
) -> tuple[set[Finding], bool]:
    """A comparison's second and fourth steps (5.2): the order book's findings (an own
    order working that the engine does not hold, a working order naming the strategy that
    the session did not send, the leg's stop with other units or trigger than the
    engine's), and whether a row has ended while the engine works its order (a poll is
    wanted)."""
    findings: set[Finding] = set()
    if books.rows is None:
        return findings, False
    bound, working = trader.venue.bound(), set(trader.working())
    polls = False
    for row in books.rows:
        kernel_order = bound.get(row.order_id)
        if kernel_order in working and not row.working:
            polls = True
        elif row.working and row.order_id in own and kernel_order not in working:
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


class Corrections:
    """The safe corrections of one session (module docstring); `session` is the
    `_session_loop.Session`, `adopted` the reconciler's signed units adopted by
    strategy, `orphans` the order IDs it asked to cancel."""

    def __init__(self, session: Any, adopted: dict[str, int], orphans: set[str]) -> None:
        self._session = session
        self._adopted = adopted
        self._orphans = orphans

    def act(self, trader: Trader, finding: Finding, books: LegBooks, own: frozenset[str]) -> str:
        """Act on a finding seen twice in a row; what was done."""
        session, kind = self._session, finding[0]
        session.mismatches += 1
        if trader.lock(RECONCILE_LOCK):
            session.event("lock", strategy=trader.name, reason=RECONCILE_LOCK)
        action = "locked"
        if kind == "orphan":
            self._orphans.add(finding[1])
            session.hand_over([CancelOrphan(finding[1])])
            action = "orphan cancelled"
            words = f"order {finding[1]} works at the broker, the engine does not hold it: "
            words += "it is cancelled"
        elif kind == "foreign":
            words = f"order {finding[1]} names the strategy and the session did not send it: "
            words += "nothing is sent for it. Check it in OpenAlgo"
        elif kind == "stop_terms":
            words = f"its stop {finding[1]} shows {finding[2]} units at {finding[3]:.2f} at the "
            words += "broker, not the engine's terms. Check it in OpenAlgo"
        else:
            rule = RULES[kind]
            words = rule.words
            if rule.adopt:
                action = self._adopt(trader, finding, books, own)
            if action == ADOPTED_STOP_CANCELLED and kind is Kind.EXPLAINED_FEWER:
                words = FEWER_STOP_CANCELLED
            if rule.cancel_stop:
                action = self._cancel_stop(trader)
        session.alert(AlertKind.RECONCILE_MISMATCH, self._words(trader, finding, books, own, words))
        if kind is Kind.OTHER_SIDE:
            session.alert(
                AlertKind.ERROR,
                f"{trader.name}: OpenAlgo's books show {trader.symbol} {trader.exchange} on "
                f"the other side of the engine's position: close it by hand in OpenAlgo",
            )
        return action

    def alert_only(
        self, trader: Trader, finding: Finding, books: LegBooks, own: frozenset[str]
    ) -> str:
        """A finding on a leg in flight past the in-flight limit: alerted, nothing done."""
        words = (
            "an order of the leg has been in flight too long, so nothing is done. Check the "
            "account in OpenAlgo"
        )
        text = self._words(trader, finding, books, own, words, locked=False)
        self._session.alert(AlertKind.ERROR, text)
        return "alerted"

    def stale(self, trader: Trader, finding: Finding, books: LegBooks, own: frozenset[str]) -> str:
        """A restart's difference judged on the journal's units when the restored engine
        holds other units: entries locked and alerted, nothing adopted or cancelled, since
        the books read at the start no longer describe the leg."""
        session = self._session
        session.mismatches += 1
        if trader.lock(RECONCILE_LOCK):
            session.event("lock", strategy=trader.name, reason=RECONCILE_LOCK)
        words = (
            f"the engine holds {round(trader.open_units())} units once restored, since a fill "
            "reached it after these books were read, so nothing is done until a fresh read "
            "of the books judges the leg"
        )
        session.alert(AlertKind.RECONCILE_MISMATCH, self._words(trader, finding, books, own, words))
        return "locked"

    def _adopt(self, trader: Trader, finding: Finding, books: LegBooks, own: frozenset[str]) -> str:
        engine, broker = finding[1], finding[2]
        units, side = abs(engine - broker), "SELL" if broker < engine else "BUY"
        session = self._session
        try:
            price = adoption_price(books.external(own), units, side)
            # After the last lines a held cancel of the stop is withdrawn before the fill,
            # so the kernel's fit with it cuts the stop to what is left (module docstring).
            session.reconcile.guard.withdraw_for_cut(trader.name, broker)
            call(
                trader.engine.apply_external_fill,
                INSTRUMENT,
                LIVE_PRODUCT,
                side,
                float(units),
                price,
                session.now(),
                ADOPTED_REASON,
            )
        except (ValueError, VenueRefusalError) as error:
            logger.error("%s: the broker's units not adopted: %s", trader.name, error)
            return f"not adopted ({error})"
        signed = units if side == "BUY" else -units
        self._adopted[trader.name] = self._adopted.get(trader.name, 0) + signed
        session.event("adopted", strategy=trader.name, side=side, units=units, price=price)
        # An exit's cancel of the stop the exit guard held: the kernel waits for it and
        # fits nothing more, so the stop would stay larger than what is left.
        if session.reconcile.guard.adopted(trader.name):
            return ADOPTED_STOP_CANCELLED
        return "adopted"

    def _cancel_stop(self, trader: Trader) -> str:
        stop = self._session.stops.leg(trader).stop
        if stop is None or stop not in trader.working():
            return "no stop to cancel"
        # The exit guard holds the stop's cancel of a leg whose broker holds fewer units:
        # this one it lets go (a stop on the other side's position could add to it).
        self._session.reconcile.guard.allow_cancel(trader.name, stop)
        trader.cancel(stop)
        return "stop cancelled"

    def _words(
        self,
        trader: Trader,
        finding: Finding,
        books: LegBooks,
        own: frozenset[str],
        what: str,
        *,
        locked: bool = True,
    ) -> str:
        """The alert's words: the leg, the engine's units (as compared), the broker's, the
        other orders on the instrument, what is done, and the lock."""
        units = finding[1] if isinstance(finding[0], Kind) else round(trader.open_units())
        others = sorted({trade.order_id for trade in books.external(own)})
        orders = f"; other orders {', '.join(others)}" if others else ""
        lock = " Entries of the strategy are locked for the rest of the session" if locked else ""
        return (
            f"{trader.name} {trader.symbol} {trader.exchange}: the engine holds {units} units, "
            f"OpenAlgo's books show {books.units}{orders}: {what}.{lock}"
        )


__all__ = [
    "ADOPTED_REASON",
    "ADOPTED_STOP_CANCELLED",
    "Corrections",
    "Finding",
    "attribute",
    "in_flight",
    "order_findings",
    "own_orders",
]
