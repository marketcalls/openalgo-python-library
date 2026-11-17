"""No exit beyond what the broker holds (16-session-design 5.4 "As built", 5.3; unit 12).

**Guarded** (`open`): from the first comparison that shows a leg's broker holding fewer
units on the engine's side than the engine counts, none, or the other side, until one
agrees (`close`), the guard holds back from the order worker:

- a placement of a closing order of any owner (a strategy's exit, the safe stop's
  flatten, the square-off, the kill switch's exit, a stop placed again), or a modify that
  raises a closing order's units, beyond the room (`_reconcile_rules.exit_room`): the
  engine exits a whole position and sizes a stop for its whole position, so such an order
  would take the broker through zero;
- a placement of an entry against the engine's position (one that reverses it closes it
  first);
- the cancel of the leg's stop while the engine holds a position. The engine cancels the
  stop before it sends an exit, and that exit would be held: the stop stays at the broker
  protecting what it holds, and the exit waits in the engine for the cancel. The
  reconciler's own cancel of the stop (the broker on the other side) goes
  (`allow_cancel`).

One ERROR alert per held order says so (`_reconcile_held`).

**Awaiting a read** (`await_read`): the safe stop's flatten and each square-off ask for a
book read first, since a sale by hand after the last read is not known yet (and so does a
restart whose start books no longer describe a leg: `Reconciler.differs_at_start`). Until a
comparison of a read begun after the request has come (`read`), the leg's exit
placements, entries against its position and the cancel of its stop are held; a stop
placed again is not (it rests until triggered). The comparison then guards the leg, or
lets them go.

What no read knows yet, the limit 16-session-design 5.4 "As built" accepts: a sale by
hand made after the read began; for a square-off, whose read is asked
`reconcile.SQUARE_OFF_READ_BEFORE_S` ahead so the square-off is not delayed, one made in
those seconds; and for a strategy's own exit or a target's exit, which ask for no read,
one made since the last comparison (at most `reconcile.RECONCILE_EVERY_S` before, unless
a fill brought one sooner). Such an exit may sell more than the broker holds; the
comparisons after it find the broker holding units the engine does not (`Kind.MORE`), and
two in a row alert it and lock the strategy's entries.

Once a leg is neither guarded nor awaiting a read, each held request that still fits what
the engine holds goes to the worker in its order; any other is refused as not sent: an
entry also once the strategy's entries are locked. The cancel of the stop of a leg
protect by state holds stays held while the stop is for no more units than the engine
holds: the stop protects it, and the session asks for no exit of it any more. A modify
that keeps or lowers the units, and any other cancel, always go: they never add
exposure. A held order is not at the broker, so it does not keep its leg in flight
(`held_orders`, `held_cancels`).

**A sale by hand adopted** (`adopted`, `withdraw_for_cut`, 3.3 step 5 "As built"): the
stop is fitted to what is left at once. A held cancel of an order that may still fill
more units at the broker, by the terms the broker confirmed, than the engine holds once
the reconciler adopted the sale goes at once (`allow_cancel`): with nothing left the stop
is cancelled; with some left before the last lines it is cancelled too, and protect by
state places a stop for what is left. After the last lines, with some left, the held
cancel of the stop is withdrawn instead, never sent (lead, round 7): the venue map tells
the kernel it was refused (`VenueMap.withdraw_cancel`), so the kernel's fit with the
adoption cuts the stop to what is left by a modify that only lowers its units, which the
guard admits, and the units left keep a stop; the exit that asked for the cancel waits in
the engine and is never sent. A stop whose cancel was not held is cut by the kernel's own
fit too.

**At the close** (`end`): once the session has printed its last lines, it sends nothing
they did not name. Nothing held goes to the worker, whatever the close's final read shows,
but the cancel a sale by hand of all the position frees (`adopted`; with units left it is
withdrawn for the stop's cut, `withdraw_for_cut`): the lines and alerts have named each
held order as not sent, and an operator may already be closing the position by hand as
they say. Every new order request is held too (an exit, a stop placed again, a trail, an
entry, the cancel of an exit or of the stop of a position), but what only takes exposure
away, the wider set the lead ratified on 2026-11-17 (16-session-design 3.3 step 5 "As
built", round 4): the reconciler's own cancel of a stop on the other side
(`allow_cancel`), the cancel of a stop once the engine holds the leg flat and a modify
that only lowers a stop's units (the stop fitted to a sale by hand the reconciler adopts),
and the cancel of an entry or an add (the safe stop asked for it before the lines; it
waited for the order's bind). The worker's own requests (a book read, an orphan's cancel)
go. Nothing is alerted as going later; the close names every request still held (`left`).

Quantities are units, times UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from .._engine import call
from ._order_state import ENTRY_PURPOSES
from ._reconcile_held import held_words
from ._reconcile_rules import exit_room
from ._session_exits import EXIT_PURPOSES
from ._types import LOGGER_NAME, AlertKind, OrderIntent, Purpose
from ._venue_requests import CancelRequest, ModifyRequest, PlaceRequest
from .inbox import NotSent, NotSentReason

logger = logging.getLogger(LOGGER_NAME)

#: The purposes of the orders that close a position, a leg's stop among them.
CLOSING = frozenset({Purpose.STOP, *EXIT_PURPOSES})


class ExitGuard:
    """The guard of one session (module docstring); `session` is the
    `_session_loop.Session`."""

    def __init__(self, session: Any) -> None:
        self._session = session
        # The broker's units of each guarded leg, by strategy, as the last read showed.
        self._broker: dict[str, int] = {}
        # When each leg's awaited read was asked for, by strategy.
        self._awaiting: dict[str, int] = {}
        self._held: dict[str, list[Any]] = {}
        self._allowed: set[tuple[str, int]] = set()
        self._alerted: set[tuple[str, int]] = set()
        self._ended = False

    def guards(self, name: str) -> bool:
        """True while strategy `name`'s leg is guarded (the broker holds fewer units)."""
        return name in self._broker

    def broker_units(self, name: str) -> int | None:
        """The broker's signed units of strategy `name`'s leg as the last read showed them
        while the leg is guarded, else None."""
        return self._broker.get(name)

    def holds_back(self, name: str) -> bool:
        """True while the guard holds a request of strategy `name`."""
        return bool(self._held.get(name))

    def open(self, name: str, broker_units: int) -> None:
        """Guard strategy `name`'s leg: the broker holds `broker_units` (signed). A request
        held while the leg awaited a read is alerted now."""
        if self._broker.get(name) != broker_units:
            self._session.event("exit_guard", strategy=name, broker_units=broker_units)
        self._broker[name] = broker_units
        for request in self._held.get(name, []):
            self._alert(request)

    def close(self, name: str) -> None:
        """The leg agrees with the broker again: its held requests go once it awaits no
        read (module docstring)."""
        if self._broker.pop(name, None) is None:
            return
        self._session.event("exit_guard_lifted", strategy=name)
        if name not in self._awaiting:
            self._release(name)

    def await_read(self, name: str, asked_ns: int) -> None:
        """Strategy `name`'s exits wait for a comparison of a read begun at or after
        `asked_ns` (module docstring)."""
        self._awaiting[name] = max(asked_ns, self._awaiting.get(name, asked_ns))
        self._session.event("exit_awaits_read", strategy=name)

    def awaited_since(self) -> list[int]:
        """When each read still awaited was asked for."""
        return list(self._awaiting.values())

    def read(self, name: str, started_ns: int) -> None:
        """A comparison of strategy `name`'s leg from a read begun at `started_ns`: the
        read it awaited, when it began late enough; its held requests then go unless the
        leg is guarded."""
        asked_ns = self._awaiting.get(name)
        if asked_ns is None or started_ns < asked_ns:
            return
        del self._awaiting[name]
        if name not in self._broker:
            self._release(name)

    def allow_cancel(self, name: str, kernel_order: int) -> None:
        """The reconciler cancels strategy `name`'s stop itself (the broker holds the other
        side): its cancel goes, a held one now."""
        self._allowed.add((name, kernel_order))
        held = self._held.get(name, [])
        cancels = [
            request
            for request in held
            if isinstance(request, CancelRequest) and request.kernel_order == kernel_order
        ]
        if cancels:
            self._held[name] = [request for request in held if request not in cancels]
            self._session.hand_over(cancels)

    def withdraw_for_cut(self, name: str, left_units: int) -> bool:
        """Before the reconciler adopts a sale by hand that leaves strategy `name`'s leg
        `left_units` (signed): after the end, with units left, each held cancel of the
        leg's stop that may still fill more than that is withdrawn, never sent, so the
        kernel's fit with the adoption cuts the stop (module docstring, "A sale by hand
        adopted"). True when one was withdrawn."""
        if not self._ended or left_units == 0:
            return False
        trader = self._session.trader_named(name)
        held = self._held.get(name, [])
        withdrawn = [
            request
            for request in held
            if isinstance(request, CancelRequest)
            and request.intent.purpose is Purpose.STOP
            and _left_to_fill(trader, request.kernel_order) > abs(left_units)
        ]
        self._held[name] = [request for request in held if request not in withdrawn]
        for request in withdrawn:
            trader.venue.withdraw_cancel(request)
            self._session.event(
                "cancel_withdrawn", strategy=name, kernel_order=request.kernel_order
            )
        return bool(withdrawn)

    def adopted(self, name: str) -> bool:
        """The reconciler adopted a sale by hand on strategy `name`'s leg: each held cancel
        of an order larger than what the engine holds now goes at once, before and after
        the end; after the end with units left, the stop's was withdrawn before the
        adoption (`withdraw_for_cut`). Module docstring, "A sale by hand adopted". True
        when one went."""
        held = self.held_cancels(name)
        if not held:
            return False
        trader = self._session.trader_named(name)
        outgrown = [kernel_order for kernel_order in held if _outgrown(trader, kernel_order)]
        for kernel_order in sorted(outgrown):
            self.allow_cancel(name, kernel_order)
        return bool(outgrown)

    def admit(self, requests: Sequence[Any]) -> list[Any]:
        """`requests` but those the guard holds (module docstring), in order."""
        admitted = []
        for request in requests:
            if self._holds(request):
                self._hold(request)
            else:
                admitted.append(request)
        return admitted

    def held_orders(self, name: str) -> frozenset[int]:
        """The kernel orders of strategy `name` whose requests the guard holds."""
        return frozenset(request.kernel_order for request in self._held.get(name, []))

    def held_cancels(self, name: str) -> frozenset[int]:
        """The kernel orders of strategy `name` whose cancel the guard holds: they work at
        the broker."""
        held = self._held.get(name, [])
        return frozenset(r.kernel_order for r in held if isinstance(r, CancelRequest))

    def end(self) -> None:
        """The session has printed its last lines: from now nothing held goes, and no new
        order request but what only takes exposure away (module docstring, "At the
        close")."""
        self._ended = True

    def left(self) -> list[Any]:
        """Every request still held (the session's close reports them)."""
        return [request for held in self._held.values() for request in held]

    def _holds(self, request: Any) -> bool:
        if not isinstance(request, PlaceRequest | ModifyRequest | CancelRequest):
            return False
        intent: OrderIntent = request.intent
        name = intent.strategy
        allowed = (name, request.kernel_order) in self._allowed
        if self._ended:
            own = allowed and isinstance(request, CancelRequest)
            return not (own or self._takes_away(request))
        guarded, awaiting = name in self._broker, name in self._awaiting
        if not (guarded or awaiting):
            return False
        engine = round(self._session.trader_named(name).open_units())
        if isinstance(request, CancelRequest):
            return intent.purpose is Purpose.STOP and engine != 0 and not allowed
        if intent.purpose in ENTRY_PURPOSES:
            return isinstance(request, PlaceRequest) and _against(intent.action, engine)
        if intent.purpose not in CLOSING:
            return False
        if not guarded:
            return isinstance(request, PlaceRequest) and intent.purpose in EXIT_PURPOSES
        units = intent.quantity
        if isinstance(request, ModifyRequest):
            units = request.terms.quantity
            if units <= request.replaces.quantity:
                return False
        return units > exit_room(engine, self._broker[name])

    def _takes_away(self, request: Any) -> bool:
        """Whether an order request only takes exposure away, so it goes after the end
        (module docstring, "At the close"): the cancel of an entry or an add, the cancel of
        a stop of a leg the engine holds flat, a modify of a stop that lowers its units and
        keeps its type and prices."""
        purpose = request.intent.purpose
        if isinstance(request, CancelRequest):
            if purpose in ENTRY_PURPOSES:
                return True
            trader = self._session.trader_named(request.intent.strategy)
            return purpose is Purpose.STOP and round(trader.open_units()) == 0
        if isinstance(request, ModifyRequest) and purpose is Purpose.STOP:
            new, old = request.terms, request.replaces
            same = (new.price_type, new.price, new.trigger_price) == (
                old.price_type,
                old.price,
                old.trigger_price,
            )
            return same and new.quantity < old.quantity
        return False

    def _hold(self, request: Any) -> None:
        name = request.intent.strategy
        self._held.setdefault(name, []).append(request)
        kind = "cancel" if isinstance(request, CancelRequest) else "request"
        self._session.event(
            "exit_held", strategy=name, kernel_order=request.kernel_order, held=kind
        )
        if name in self._broker:
            self._alert(request)

    def _alert(self, request: Any) -> None:
        """One ERROR alert per held order of a guarded leg; none once the session has
        ended, since nothing held goes then ("At the close")."""
        intent = request.intent
        key = (intent.strategy, request.kernel_order)
        if key in self._alerted or self._ended:
            return
        self._alerted.add(key)
        self._session.alert(AlertKind.ERROR, held_words(request, self._broker[intent.strategy]))

    def _release(self, name: str) -> None:
        """The leg is neither guarded nor awaiting a read: each held request that fits goes
        to the worker, in order; the stop's cancel of a leg protect by state holds stays
        held unless the stop has outgrown the position; any other is refused as not sent
        (module docstring). Once the session has ended every one stays held, as its last
        lines said ("At the close")."""
        if self._ended:
            return
        session = self._session
        trader = session.trader_named(name)
        protected = session.stops.protection.holds(trader)
        fits, kept = [], []
        for request in self._held.pop(name, []):
            cancel = isinstance(request, CancelRequest)
            if protected and cancel and not _outgrown(trader, request.kernel_order):
                kept.append(request)
            elif self._fits(trader, request):
                fits.append(request)
            else:
                why = "held back while the broker held fewer units, and no longer fits the position"
                session.news.take(NotSent(request, NotSentReason.REFUSED, why))
        if kept:
            self._held[name] = kept
        if fits:
            session.hand_over(fits)

    def _fits(self, trader: Any, request: Any) -> bool:
        """Whether a held request still fits: its order works in the engine with no change
        of it waiting, an entry while entries are not locked, and an exit or a stop within
        what the engine holds."""
        kernel_order = request.kernel_order
        if kernel_order not in trader.working():
            return False
        if isinstance(request, CancelRequest):
            return True
        units = request.terms.quantity if isinstance(request, ModifyRequest) else None
        if isinstance(request, PlaceRequest):
            if trader.venue.changing(kernel_order):
                return False
            if request.intent.purpose in ENTRY_PURPOSES:
                return not call(trader.engine.entry_locks)
            units = request.intent.quantity
        return units is not None and units <= abs(round(trader.open_units()))


def _outgrown(trader: Any, kernel_order: int) -> bool:
    """Whether order `kernel_order` may still fill more units at the broker, by the terms
    the broker confirmed, than the engine's position holds (a trigger of such a stop
    would take the broker's position through zero)."""
    return _left_to_fill(trader, kernel_order) > abs(round(trader.open_units()))


def _left_to_fill(trader: Any, kernel_order: int) -> int:
    """The units order `kernel_order` may still fill at the broker, by the terms the
    broker confirmed."""
    venue = trader.venue
    return venue.terms(kernel_order).quantity - venue.filled(kernel_order)


def _against(action: str, engine_units: int) -> bool:
    """Whether an order on `action`'s side reduces the engine's position (signed units)."""
    return (engine_units > 0 and action == "SELL") or (engine_units < 0 and action == "BUY")


__all__ = ["CLOSING", "ExitGuard"]
