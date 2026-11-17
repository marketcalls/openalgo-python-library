"""What the stops supervisor knows of one leg's orders (16-session-design 4.1, 4.2, 4.9):
each order the session handed the order worker, how far it got (handed, held back,
unanswered, ambiguous, accepted, confirmed working, ended), whether a cancel of it was
sent, and which order is the leg's broker-held stop.

A stop is **confirmed** once it was accepted with an order ID and a word after that
(stream or poll: open, trigger pending, or a fill) says it works (4.2). The leg is
**guarded** from the placement of its first stop until it is flat again: a leg that never
had a stop (no stop rule, or a trail or breakeven not armed yet) is not counted as
unprotected (open decision 4).

Quantities are whole units, times UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from ._order_answers import Outcome
from ._order_book import Resolved
from ._order_state import ENTRY_PURPOSES
from ._types import OrderStatus, OrderUpdate, Purpose
from ._venue_requests import CancelRequest, ModifyRequest, OrderRequest, PlaceRequest
from .inbox import NotSentReason

#: The worker's reasons for a request it put back to send later.
_HELD = frozenset({NotSentReason.HELD, NotSentReason.SILENT})


class Sent(enum.Enum):
    """How far a placement got."""

    #: Handed to the worker; no answer yet.
    HANDED = "handed"
    #: Put back by the worker: OpenAlgo silent, or an intent of the instrument unbound.
    HELD = "held"
    #: Sent with no answer: unbound until the order book shows whether it was placed.
    UNANSWERED = "unanswered"
    #: The order book cannot tell it apart from another order: unbound for the day.
    AMBIGUOUS = "ambiguous"
    #: Accepted with an order ID.
    BOUND = "bound"
    #: Accepted, and a word since says it works.
    CONFIRMED = "confirmed"
    #: Ended: filled, cancelled, rejected or not placed.
    ENDED = "ended"


#: The states of an order that may be at the broker without the session knowing its fate.
IN_FLIGHT = frozenset({Sent.HANDED, Sent.HELD, Sent.UNANSWERED, Sent.AMBIGUOUS})


@dataclass
class Watched:
    """One order of the leg: kernel order `kernel_order` placed as `purpose` for `units`,
    how far it got (`state`), its broker order ID once bound, when it was accepted
    (`bound_ns`), whether a cancel of it waits for its word (`cancel_sent`), whether a
    modify of it is held back (`modify_held`), how it ended, and its intent's ID."""

    kernel_order: int
    purpose: Purpose
    units: int
    intent_id: str
    action: str
    state: Sent = Sent.HANDED
    order_id: str = ""
    bound_ns: int | None = None
    cancel_sent: bool = False
    modify_held: bool = False
    ended: OrderStatus | None = None

    @property
    def is_exit(self) -> bool:
        """True for an order that closes the position: an exit, a risk exit, a square-off."""
        return self.purpose not in ENTRY_PURPOSES and self.purpose is not Purpose.STOP

    def name(self) -> str:
        """How an alert names the order: its broker order ID once bound, else what it is
        and its intent ID."""
        if self.order_id:
            return self.order_id
        return f"{self.action} {self.units} ({self.purpose.value}, intent {self.intent_id})"


@dataclass
class LegOrders:
    """The orders of one strategy's leg (module docstring): every one handed over, by
    kernel order, the leg's current stop (`stop`, the latest one placed and not ended),
    whether the leg is `guarded`, and the units of adds whose answer was lost
    (`unbound_add_units`)."""

    orders: dict[int, Watched] = field(default_factory=dict)
    stop: int | None = None
    guarded: bool = False
    unbound_add_units: int = 0

    def handed(self, request: OrderRequest) -> None:
        """A request of the leg went to the worker."""
        if isinstance(request, PlaceRequest):
            intent = request.intent
            self.orders[request.kernel_order] = Watched(
                request.kernel_order,
                intent.purpose,
                intent.quantity,
                intent.intent_id,
                intent.action,
            )
            if intent.purpose is Purpose.STOP:
                self.stop, self.guarded = request.kernel_order, True
            return
        watched = self.orders.get(request.kernel_order)
        if watched is not None and isinstance(request, CancelRequest):
            watched.cancel_sent = True
        elif watched is not None and isinstance(request, ModifyRequest):
            watched.modify_held = False

    def placed(self, kernel_order: int, outcome: Outcome, order_id: str, at_ns: int) -> None:
        """The answer to a placement: accepted, refused, unanswered or queued."""
        watched = self.orders.get(kernel_order)
        if watched is None:
            return
        if outcome is Outcome.ACCEPTED:
            self._bind(watched, order_id, at_ns)
        elif outcome is Outcome.REJECTED:
            self._end(watched, OrderStatus.REJECTED)
        elif outcome is Outcome.UNANSWERED:
            watched.state = Sent.UNANSWERED
            self._unbound_add(watched, +1)

    def not_sent(self, request: OrderRequest, reason: NotSentReason) -> None:
        """A request the worker did not send: a placement put back is held; one refused or
        stale ended (one the guard locked out stays as it was: nothing more is sent); a
        modify put back is held; a cancel refused waits for its read."""
        watched = self.orders.get(request.kernel_order)
        if watched is None:
            return
        if isinstance(request, PlaceRequest):
            if reason in _HELD:
                watched.state = Sent.HELD
            elif reason is not NotSentReason.LOCKED:
                self._end(watched, OrderStatus.REJECTED)
        elif isinstance(request, ModifyRequest):
            watched.modify_held = reason in _HELD

    def resolved(self, kernel_order: int, kind: Resolved, order_id: str, at_ns: int) -> None:
        """The order book settled an unanswered placement."""
        watched = self.orders.get(kernel_order)
        if watched is None:
            return
        if kind is Resolved.BOUND:
            self._unbound_add(watched, -1)
            self._bind(watched, order_id, at_ns)
        elif kind is Resolved.NOT_PLACED:
            self._end(watched, OrderStatus.REJECTED)
        else:
            watched.state = Sent.AMBIGUOUS

    def heard(self, kernel_order: int, update: OrderUpdate) -> Watched | None:
        """A word of a bound order: a working word or a fill confirms it, a terminal one
        ends it. Returns the order, or None for one the leg does not hold."""
        watched = self.orders.get(kernel_order)
        if watched is None:
            return None
        if update.status.terminal:
            self._end(watched, update.status)
        elif watched.state in (Sent.BOUND, Sent.UNANSWERED, Sent.HANDED):
            watched.state = Sent.CONFIRMED
        return watched

    def cancel_refused(self, kernel_order: int) -> None:
        """The venue refused a cancel and the order works on: no cancel waits for it."""
        watched = self.orders.get(kernel_order)
        if watched is not None:
            watched.cancel_sent = False

    def flat(self) -> None:
        """The leg is flat with no stop working: it is no longer guarded."""
        self.guarded = False
        self.unbound_add_units = 0

    def exits(self) -> list[Watched]:
        """The leg's exits that have not ended, oldest first."""
        return [w for w in self.orders.values() if w.is_exit and w.state is not Sent.ENDED]

    def _bind(self, watched: Watched, order_id: str, at_ns: int) -> None:
        if watched.state in (Sent.CONFIRMED, Sent.ENDED):
            return
        watched.state, watched.order_id, watched.bound_ns = Sent.BOUND, order_id, at_ns

    def _end(self, watched: Watched, status: OrderStatus) -> None:
        if watched.state is Sent.UNANSWERED:
            self._unbound_add(watched, -1)
        watched.state, watched.ended, watched.cancel_sent = Sent.ENDED, status, False
        if self.stop == watched.kernel_order:
            self.stop = None

    def _unbound_add(self, watched: Watched, sign: int) -> None:
        if watched.purpose is Purpose.ADD and self.guarded and watched.state is Sent.UNANSWERED:
            self.unbound_add_units = max(0, self.unbound_add_units + sign * watched.units)


__all__ = ["IN_FLIGHT", "LegOrders", "Sent", "Watched"]
