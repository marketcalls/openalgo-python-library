"""The requests the exit guard (`_reconcile_guard`) holds back, and what becomes of them
(16-session-design 5.4 "As built", 3.3 step 5; unit 12, split from the guard in unit M6).

One ERROR alert per held order of a guarded leg (`_reconcile_held`); none once the session
has ended.

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
the engine and is never sent. Should the kernel refuse the adoption, the withdrawn cancel
is held again (`put_back`) and the stop stays as it was. A stop whose cancel was not held
is cut by the kernel's own fit too.

A held GTT call of a CNC or NRML leg (`_reconcile_guard_gtt`) is kept apart from the
orders, named by its GTT in the alert and the event stream, and released in order once the
leg is neither guarded nor awaiting a read, when its kernel GTT still waits for it.

Quantities are units, times UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .._engine import call
from ._gtt_types import GttAction, GttRequest
from ._order_state import ENTRY_PURPOSES
from ._reconcile_guard_gtt import gtt_awaited
from ._reconcile_held import held_gtt_words, held_words
from ._types import AlertKind, Purpose
from ._venue_requests import CancelRequest, ModifyRequest, PlaceRequest
from .inbox import NotSent, NotSentReason


class HeldRequests:
    """The requests an `ExitGuard` holds, by strategy, and their alerts (module
    docstring). The guard holds the session (`_session`), the broker's units of each
    guarded leg (`_broker`), the legs whose every closing order is held (`_hold_all`)
    and whether the session has ended (`_ended`)."""

    _session: Any
    _broker: dict[str, int]
    _hold_all: set[str]
    _ended: bool

    def __init__(self) -> None:
        self._held: dict[str, list[Any]] = {}
        # The GTT calls held, by strategy, apart from the orders' requests.
        self._held_gtts: dict[str, list[GttRequest]] = {}
        self._alerted: set[tuple[str, int]] = set()
        self._alerted_gtts: set[tuple[str, int]] = set()
        # The stop cancels the reconciler sends itself, by strategy and kernel order.
        self._allowed: set[tuple[str, int]] = set()

    def holds_back(self, name: str) -> bool:
        """True while the guard holds a request of strategy `name`."""
        return bool(self._held.get(name)) or bool(self._held_gtts.get(name))

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

    def withdraw_for_cut(self, name: str, left_units: int) -> list[CancelRequest]:
        """Before the reconciler adopts a sale by hand that leaves strategy `name`'s leg
        `left_units` (signed): after the end, with units left, each held cancel of the
        leg's stop that may still fill more than that is withdrawn, never sent, so the
        kernel's fit with the adoption cuts the stop (module docstring, "A sale by hand
        adopted"). Returns those withdrawn, for `put_back` should the adoption fail."""
        if not self._ended or left_units == 0:
            return []
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
        return withdrawn

    def put_back(self, name: str, withdrawn: Sequence[CancelRequest]) -> None:
        """The kernel refused the adoption `withdrawn` (from `withdraw_for_cut`) made way
        for, so no cut comes: each is held again and waits for its answer at the venue map
        again (`VenueMap.restore_cancel`), and the close names it as not sent (lead,
        round 8). The kernel took it as refused, which leaves the stop working there as at
        the broker."""
        if not withdrawn:
            return
        trader = self._session.trader_named(name)
        for request in withdrawn:
            trader.venue.restore_cancel(request)
            self._session.event("cancel_put_back", strategy=name, kernel_order=request.kernel_order)
        self._held[name] = [*withdrawn, *self._held.get(name, [])]

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

    def held_orders(self, name: str) -> frozenset[int]:
        """The kernel orders of strategy `name` whose requests the guard holds."""
        return frozenset(request.kernel_order for request in self._held.get(name, []))

    def held_cancels(self, name: str) -> frozenset[int]:
        """The kernel orders of strategy `name` whose cancel the guard holds: they work at
        the broker."""
        held = self._held.get(name, [])
        return frozenset(r.kernel_order for r in held if isinstance(r, CancelRequest))

    def holds_gtt_cancel(self, name: str) -> bool:
        """True while the guard holds the cancel of a stop GTT of strategy `name`: it rests
        at the broker, and the exit that waits for its answer was never sent."""
        return any(request.action is GttAction.CANCEL for request in self._held_gtts.get(name, []))

    def left(self) -> list[Any]:
        """Every request still held (the session's close reports them)."""
        orders = [request for held in self._held.values() for request in held]
        return orders + [request for held in self._held_gtts.values() for request in held]

    def _alert_held(self, name: str) -> None:
        """Alert each request held of strategy `name`, now that its leg is guarded."""
        for request in [*self._held.get(name, []), *self._held_gtts.get(name, [])]:
            self._alert(request)

    def _hold(self, request: Any) -> None:
        if isinstance(request, GttRequest):
            self._hold_gtt(request)
            return
        name = request.intent.strategy
        self._held.setdefault(name, []).append(request)
        kind = "cancel" if isinstance(request, CancelRequest) else "request"
        self._session.event(
            "exit_held", strategy=name, kernel_order=request.kernel_order, held=kind
        )
        if name in self._broker:
            self._alert(request)

    def _hold_gtt(self, request: GttRequest) -> None:
        name = request.strategy
        self._held_gtts.setdefault(name, []).append(request)
        kind = "cancel" if request.action is GttAction.CANCEL else "request"
        self._session.event("exit_held", strategy=name, gtt=request.gtt_id, held=kind)
        if name in self._broker:
            self._alert(request)

    def _alert(self, request: Any) -> None:
        """One ERROR alert per held order or GTT call of a guarded leg; none once the
        session has ended, since nothing held goes then ("At the close")."""
        if isinstance(request, GttRequest):
            self._alert_gtt(request)
            return
        intent = request.intent
        key = (intent.strategy, request.kernel_order)
        if key in self._alerted or self._ended:
            return
        self._alerted.add(key)
        name = intent.strategy
        words = held_words(request, self._broker[name], shared=name in self._hold_all)
        self._session.alert(AlertKind.ERROR, words)

    def _alert_gtt(self, request: GttRequest) -> None:
        name = request.strategy
        key = (name, request.gtt_id)
        if key in self._alerted_gtts or self._ended:
            return
        self._alerted_gtts.add(key)
        known = self._session.trader_named(name).gtts
        seen = None if known is None else known.gtt(request.gtt_id)
        trigger_id = "" if seen is None else seen.trigger_id
        shared = name in self._hold_all
        words = held_gtt_words(request, trigger_id, self._broker[name], shared=shared)
        self._session.alert(AlertKind.ERROR, words)

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
        fits += self._release_gtts(trader)
        if fits:
            session.hand_over(fits)

    def _release_gtts(self, trader: Any) -> list[GttRequest]:
        """The held GTT calls of `trader` whose kernel GTT still waits for them, in order;
        the others are dropped (module docstring)."""
        released = []
        for request in self._held_gtts.pop(trader.name, []):
            if gtt_awaited(trader, request):
                released.append(request)
            else:
                self._session.event(
                    "gtt_dropped",
                    strategy=trader.name,
                    gtt=request.gtt_id,
                    held=request.action.value,
                )
        return released

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


__all__ = ["HeldRequests"]
