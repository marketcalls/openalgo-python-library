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

One ERROR alert per held order says so.

**Awaiting a read** (`await_read`): the safe stop's flatten and each square-off ask for a
book read first, since a sale by hand after the last read is not known yet (and so does a
restart whose start books no longer describe a leg: `Reconciler.differs_at_start`). Until a
comparison of a read begun after the request has come (`read`), the leg's exit
placements, entries against its position and the cancel of its stop are held; a stop
placed again is not (it rests until triggered). The comparison then guards the leg, or
lets them go.

Once a leg is neither guarded nor awaiting a read, each held request that still fits what
the engine holds goes to the worker in its order; any other is refused as not sent: an
entry also once the strategy's entries are locked. The cancel of the stop of a leg
protect by state holds stays held: the stop protects it, and the session asks for no exit
of it any more. A modify that keeps or lowers the units, and any other cancel, always go:
they never add exposure. A held order is not at the broker, so it does not keep its leg
in flight (`held_orders`, `held_cancels`).

Quantities are units, times UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from .._engine import call
from ._order_state import ENTRY_PURPOSES
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

    def guards(self, name: str) -> bool:
        """True while strategy `name`'s leg is guarded (the broker holds fewer units)."""
        return name in self._broker

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

    def left(self) -> list[Any]:
        """Every request still held (the session's close reports them)."""
        return [request for held in self._held.values() for request in held]

    def _holds(self, request: Any) -> bool:
        if not isinstance(request, PlaceRequest | ModifyRequest | CancelRequest):
            return False
        intent: OrderIntent = request.intent
        name = intent.strategy
        guarded, awaiting = name in self._broker, name in self._awaiting
        if not (guarded or awaiting):
            return False
        engine = round(self._session.trader_named(name).open_units())
        if isinstance(request, CancelRequest):
            allowed = (name, request.kernel_order) in self._allowed
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
        """One ERROR alert per held order of a guarded leg."""
        intent = request.intent
        key = (intent.strategy, request.kernel_order)
        if key in self._alerted:
            return
        self._alerted.add(key)
        self._session.alert(AlertKind.ERROR, _held_words(request, self._broker[intent.strategy]))

    def _release(self, name: str) -> None:
        """The leg is neither guarded nor awaiting a read: each held request that fits goes
        to the worker, in order; the stop's cancel of a leg protect by state holds stays
        held; any other is refused as not sent (module docstring)."""
        session = self._session
        trader = session.trader_named(name)
        protected = session.stops.protection.holds(trader)
        fits, kept = [], []
        for request in self._held.pop(name, []):
            if protected and isinstance(request, CancelRequest):
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


def _against(action: str, engine_units: int) -> bool:
    """Whether an order on `action`'s side reduces the engine's position (signed units)."""
    return (engine_units > 0 and action == "SELL") or (engine_units < 0 and action == "BUY")


def _held_words(request: Any, broker_units: int) -> str:
    """The ERROR alert of a held order of a guarded leg."""
    intent = request.intent
    leg = f"{intent.symbol} {intent.exchange} ({intent.action} {intent.quantity})"
    book = f"OpenAlgo's position book shows {broker_units} units"
    if isinstance(request, CancelRequest):
        return (
            f"{intent.strategy}: the cancel of the stop of {leg} is held back, so the stop "
            f"stays at the broker: {book}, and the exit it makes way for would take the "
            "broker's position through zero. Both go once the books agree; check the "
            "position in OpenAlgo"
        )
    if intent.purpose in ENTRY_PURPOSES:
        return (
            f"{intent.strategy}: the entry of {leg} is held back: {book}, and it would close "
            "more than the broker holds. It is sent once the books agree, unless entries "
            "are locked by then; check the position in OpenAlgo"
        )
    what = "stop" if intent.purpose is Purpose.STOP else intent.purpose.value.replace("_", " ")
    return (
        f"{intent.strategy}: the {what} of {leg} is held back: {book}, and it would take "
        "the broker's position through zero. It is sent once the books agree; check the "
        "position in OpenAlgo"
    )


__all__ = ["CLOSING", "ExitGuard"]
