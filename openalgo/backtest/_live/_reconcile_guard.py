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

A member of an instrument several strategies of the session share is guarded with no
room at all (`open(..., hold_all=True)`): while its group differs from the broker's net,
every closing placement, unit-raising modify and exit-making stop cancel of every member
is held (18-multi-strategy 4.8), and the leg's figure is still the broker's net, which
the close and the alerts name. One ERROR alert per held order says so (`_reconcile_held`).
What becomes of a held request (its release once the books agree, a sale by hand adopted,
the close) is `_reconcile_guard_held`'s.

**Awaiting a read** (`await_read`): the safe stop's flatten and each square-off ask for a
book read first, since a sale by hand after the last read is not known yet (and so does a
restart whose start books no longer describe a leg: `Reconciler.differs_at_start`). Until a
comparison of a read begun after the request has come (`read`), the leg's exit
placements, entries against its position and the cancel of its stop are held; a stop
placed again is not (it rests until triggered). The comparison then guards the leg, or
lets them go.

**A leg's stop GTT** (a CNC or NRML leg, 20-positional-live 7.2): the same rules hold its
GTT calls, the cancel of its stop GTT as the cancel of a stop, a stop GTT placed again or
raised beyond the room as a stop placed again; `_reconcile_guard_gtt` has them.

What no read knows yet, the limit 16-session-design 5.4 "As built" accepts: a sale by
hand made after the read began; for a square-off, whose read is asked
`reconcile.SQUARE_OFF_READ_BEFORE_S` ahead so the square-off is not delayed, one made in
those seconds; and for a strategy's own exit or a target's exit, which ask for no read,
one made since the last comparison (at most `reconcile.RECONCILE_EVERY_S` before, unless
a fill brought one sooner). Such an exit may sell more than the broker holds; the
comparisons after it find the broker holding units the engine does not (`Kind.MORE`), and
two in a row alert it and lock the strategy's entries.

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

from ._gtt_types import GttRequest
from ._order_state import ENTRY_PURPOSES
from ._reconcile_guard_gtt import gtt_holds, gtt_takes_away
from ._reconcile_guard_held import HeldRequests
from ._reconcile_rules import exit_room, reduces
from ._session_exits import EXIT_PURPOSES
from ._types import LOGGER_NAME, OrderIntent, Purpose
from ._venue_requests import CancelRequest, ModifyRequest, PlaceRequest

logger = logging.getLogger(LOGGER_NAME)

#: The purposes of the orders that close a position, a leg's stop among them.
CLOSING = frozenset({Purpose.STOP, *EXIT_PURPOSES})


class ExitGuard(HeldRequests):
    """The guard of one session (module docstring); `session` is the
    `_session_loop.Session`. What it holds is its `HeldRequests`'."""

    def __init__(self, session: Any) -> None:
        super().__init__()
        self._session = session
        # The broker's units of each guarded leg, by strategy, as the last read showed.
        self._broker: dict[str, int] = {}
        # The guarded legs whose every closing order is held (members of a shared group).
        self._hold_all: set[str] = set()
        # When each leg's awaited read was asked for, by strategy.
        self._awaiting: dict[str, int] = {}
        self._ended = False

    def guards(self, name: str) -> bool:
        """True while strategy `name`'s leg is guarded (the broker holds fewer units)."""
        return name in self._broker

    def broker_units(self, name: str) -> int | None:
        """The broker's signed units of strategy `name`'s instrument (the net of every
        strategy of the session that trades it) as the last read showed them while the leg
        is guarded, else None."""
        return self._broker.get(name)

    def open(self, name: str, broker_units: int, *, hold_all: bool = False) -> None:
        """Guard strategy `name`'s leg: the broker holds `broker_units` (signed, the net of
        the instrument). `hold_all`: every closing order of the leg is held, whatever its
        units (a member of a shared instrument, module docstring); else those beyond what
        the broker holds on the engine's side. A request held while the leg awaited a read
        is alerted now."""
        if self._broker.get(name) != broker_units:
            self._session.event("exit_guard", strategy=name, broker_units=broker_units)
        self._broker[name] = broker_units
        if hold_all:
            self._hold_all.add(name)
        self._alert_held(name)

    def close(self, name: str) -> None:
        """The leg agrees with the broker again: its held requests go once it awaits no
        read (module docstring)."""
        if self._broker.pop(name, None) is None:
            return
        self._hold_all.discard(name)
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

    def admit(self, requests: Sequence[Any]) -> list[Any]:
        """`requests` but those the guard holds (module docstring), in order."""
        admitted = []
        for request in requests:
            if self._holds(request):
                self._hold(request)
            else:
                admitted.append(request)
        return admitted

    def end(self) -> None:
        """The session has printed its last lines: from now nothing held goes, and no new
        order request but what only takes exposure away (module docstring, "At the
        close")."""
        self._ended = True

    def _holds(self, request: Any) -> bool:
        if isinstance(request, GttRequest):
            return self._holds_gtt(request)
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
            return isinstance(request, PlaceRequest) and reduces(intent.action, engine)
        if intent.purpose not in CLOSING:
            return False
        if not guarded:
            return isinstance(request, PlaceRequest) and intent.purpose in EXIT_PURPOSES
        units = intent.quantity
        if isinstance(request, ModifyRequest):
            units = request.terms.quantity
            if units <= request.replaces.quantity:
                return False
        return name in self._hold_all or units > exit_room(engine, self._broker[name])

    def _holds_gtt(self, request: GttRequest) -> bool:
        """Whether the guard holds a GTT call of a CNC or NRML leg (`_reconcile_guard_gtt`,
        module docstring "A leg's stop GTT")."""
        name = request.strategy
        trader = self._session.trader_named(name)
        if self._ended:
            return not gtt_takes_away(trader, request)
        return gtt_holds(
            trader,
            request,
            awaiting=name in self._awaiting,
            broker=self._broker.get(name),
            hold_all=name in self._hold_all,
        )

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


__all__ = ["CLOSING", "ExitGuard"]
