"""What the reconciler (`reconcile`) judges a leg by and does about a mismatch
(16-session-design 5.1 to 5.3): the per-leg in-flight rule, the session's own orders, and
the safe corrections a finding seen twice in a row allows.

**In flight** (`in_flight`): an order of the leg handed to the order worker and not
answered, unbound, held back or ambiguous; an add whose answer was lost; a word the order
path holds for the trade book; a modify or cancel waiting for its result or its bind; any
order of the leg working other than its confirmed, unchanged stop, and other than those
the exit guard holds (they are not at the broker).

**The safe corrections** (`Corrections.act`, decided 2026-11-16): the strategy's entries
locked for the rest of the session ("reconciliation") and a RECONCILE_MISMATCH alert
naming the leg, both figures and the orders, then only what can never open or add
exposure: an explained difference adopted (`apply_external_fill` at the other orders'
price, the kernel fitting the leg's stop to the adopted units in the same call), the
leg's own stop cancelled when the broker holds the other side, the session's own orphan
cancelled. Never an exit, never the kill switch. A finding on a leg in flight past the
in-flight limit is alerted, with nothing done.

Quantities are signed units, prices rupees, times UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

import logging
from typing import Any

from .._engine import call
from ..errors import VenueRefusalError
from ._plan_catch_up import INSTRUMENT
from ._reconcile_rules import RULES, Kind, LegBooks, adoption_price
from ._session_news import RECONCILE_LOCK
from ._session_trader import Trader
from ._stops_orders import IN_FLIGHT, Sent
from ._types import LOGGER_NAME, AlertKind
from ._venue_requests import LIVE_PRODUCT
from ._worker_queue import CancelOrphan

logger = logging.getLogger(LOGGER_NAME)

#: The exit reason the kernel records for units adopted from other orders' trades.
ADOPTED_REASON = "manual"
#: A finding: its kind first (a `_reconcile_rules.Kind`, or "orphan", "foreign",
#: "stop_terms"), then what tells one from another.
Finding = tuple[Any, ...]


def in_flight(session: Any, trader: Trader, held: frozenset[int]) -> bool:
    """Whether `trader`'s leg is in flight (module docstring); `held` are its kernel
    orders whose requests the exit guard holds."""
    stops = session.stops
    leg = stops.leg(trader)
    if leg.unbound_add_units or stops.held.holds(trader):
        return True
    if any(watched.state in IN_FLIGHT for watched in leg.orders.values()):
        return True
    for kernel_order in trader.working():
        if kernel_order in held:
            continue
        if trader.venue.changing(kernel_order):
            return True
        watched = leg.orders.get(kernel_order)
        resting = kernel_order == leg.stop and watched is not None
        if not (resting and watched.state is Sent.CONFIRMED and not watched.cancel_sent):
            return True
    return False


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

    def _adopt(self, trader: Trader, finding: Finding, books: LegBooks, own: frozenset[str]) -> str:
        engine, broker = finding[1], finding[2]
        units, side = abs(engine - broker), "SELL" if broker < engine else "BUY"
        session = self._session
        try:
            price = adoption_price(books.external(own), units, side)
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
        return "adopted"

    def _cancel_stop(self, trader: Trader) -> str:
        stop = self._session.stops.leg(trader).stop
        if stop is None or stop not in trader.working():
            return "no stop to cancel"
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


__all__ = ["ADOPTED_REASON", "Corrections", "Finding", "in_flight", "own_orders"]
