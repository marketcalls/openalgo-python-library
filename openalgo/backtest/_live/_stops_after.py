"""What becomes of a leg and its broker-held stop after the session's last lines
(16-session-design 3.3 step 5 "As built", unit 12).

Once the close has printed its last lines (`AfterLines.end`), the session sends nothing
those lines did not name. The close's final book read takes in the books and the order
news only: no price reaches the engine or the price rule, and the stops supervisor's
duties no longer run (`stops.StopsSupervisor`), so no target, trail or price rule acts;
the exit guard sends only what takes exposure away (`_reconcile_guard`), and the read
waits for its answers before the order worker closes.

A word in that window that ends a leg's stop (filled, cancelled or rejected at the
broker) is therefore not acted on: what the engine makes of it (the stop placed again, an
exit at market) is not sent. It is noted (`heard`), and so is a leg whose units changed
in that window (a fill whose answer came then, a sale by hand the reconciler adopted), as
the engine held them when the lines were printed (`end`). So is the cut of a leg's stop to
what a sale by hand adopted then leaves (a modify that only lowers its units, `handed`):
until the broker confirmed it no note says the stop protects the leg, and one the broker
refused (`refused`), or that had no answer when the session closed, is named. The result's
note of a position
left open first says what happened, then the leg's state as `_stops_report.left_open`
gives it (`note`), and one ERROR alert per leg says the same (`alert`): the last line is
never left claiming a position or a protection the leg no longer has.

A leg whose last comparison saw a difference with OpenAlgo's books for the first time
(fewer units, none, more or the other side: `Reconciler.unconfirmed`) has no later read to
confirm it once the session has ended (16-session-design 5.4 "As built", round 7). It is
reported whatever the engine holds, flat included: the note, and one ERROR alert, name
OpenAlgo's units and the engine's at that read, the engine's own fills since and what they
leave (`_reconcile_close.Differing.words`), that nothing was sent for the difference, then
what works toward the leg and its stop with the order book to check before closing it by
hand (`_stops_report.left_open`); the result lists it at OpenAlgo's units as those fills
leave them (`_session_close`). When they leave OpenAlgo's books flat, the alert names what
of the leg still works and that nothing is to be closed by hand (`NOTHING_TO_CLOSE`), and
the result lists no position (round 8). A leg left open that no read judged after a fill
of its own orders landed during a read, with no such difference, has the engine's units
named as not confirmed in its note (`NOT_CONFIRMED`, round 9).

A leg left open that the exit guard still holds after a difference seen twice
(`CLOSED_DIFFERING`) is alerted too: no comparison agreed after it, so a line printed
before the final read may have said its stop protects it; the alert and the note name the
units OpenAlgo's books showed and the stop's own. A leg left flat is alerted when anything
happened to it, or when an order of it still works in the engine
(`_stops_report.still_working`), since that order may work at the broker; no line named a
flat leg. Each alert names the changes of the leg's orders the exit guard held back (a
held placement's order is named by the note as not sent). Nothing is sent for any of it.

Quantities are signed units. Used by the loop only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from ._gtt_types import GttRequest
from ._reconcile_close import NOT_CONFIRMED, NOTHING_TO_CLOSE, Differing, differing
from ._session_trader import Trader
from ._stops_orders import LegOrders
from ._stops_report import left_open, leg_words, still_working
from ._types import AlertKind, OrderStatus, OrderUpdate, Purpose
from ._venue_requests import CancelRequest, ModifyRequest

#: How a note says a stop ended, by the status of its last word.
_ENDED_WORDS: Mapping[OrderStatus, str] = MappingProxyType(
    {
        OrderStatus.COMPLETE: "filled",
        OrderStatus.CANCELLED: "cancelled",
        OrderStatus.REJECTED: "rejected",
    }
)
#: What the alert of a leg left flat asks of the operator.
FLAT_NOW = (
    "the engine holds no position now: check the order book in OpenAlgo, cancelling what "
    "still works"
)
#: What the alert of a leg the exit guard still holds at the close says first.
CLOSED_DIFFERING = "the session closed before OpenAlgo's books agreed with the engine"
#: When it happened, in a note's words.
_AFTER_THE_LINES = "after the session's last lines"
#: What became of a stop's cut the broker did not confirm, in a note's words: refused as
#: the venue map takes a modify to be (an error answer, a request not sent, a row with the
#: old terms), or no answer by the close.
_CUT_REFUSED = "was refused"
_CUT_UNANSWERED = "got no answer before the session closed"


class AfterLines:
    """What happened to each leg and its stop after the session's last lines (module
    docstring)."""

    def __init__(self) -> None:
        self._ended = False
        # By strategy: the engine's signed units when the last lines were printed.
        self._units: dict[str, float] = {}
        # By strategy: its stop's name and how it ended after them.
        self._stops: dict[str, tuple[str, OrderStatus]] = {}
        # By strategy: the last cut of its stop handed over after them, and the
        # strategies one of whose cuts was refused.
        self._cuts: dict[str, ModifyRequest] = {}
        self._refused: set[str] = set()

    @property
    def ended(self) -> bool:
        """True once the close has printed its last lines."""
        return self._ended

    def end(self, traders: Sequence[Trader]) -> None:
        """The close has printed its last lines: the units each of `traders` holds now are
        kept, and from now a word that ends a leg's stop is noted."""
        self._ended = True
        self._units = {trader.name: trader.open_units() for trader in traders}

    def heard(self, trader: Trader, leg: LegOrders, kernel_order: int, update: OrderUpdate) -> bool:
        """A word of `trader`'s order `kernel_order`, before `leg` takes it: True when it
        ends the leg's stop after the last lines, which is then noted."""
        if not self._ended or kernel_order != leg.stop or not update.status.terminal:
            return False
        self._stops[trader.name] = (leg.orders[kernel_order].name(), update.status)
        return True

    def handed(self, request: Any) -> None:
        """A request handed to the order worker: after the last lines, a modify of a stop
        that lowers its units is that stop's cut, noted (module docstring)."""
        cut = (
            self._ended
            and isinstance(request, ModifyRequest)
            and request.intent.purpose is Purpose.STOP
            and request.terms.quantity < request.replaces.quantity
        )
        if cut:
            self._cuts[request.intent.strategy] = request

    def refused(self, request: Any) -> None:
        """A modify the broker or OpenAlgo refused, or that was not sent: when it is a
        stop's cut, the cut is named as refused."""
        name = request.intent.strategy
        known = self._cuts.get(name)
        if known is not None and known.request_id == request.request_id:
            self._refused.add(name)

    def note(self, session: Any, trader: Trader) -> str:
        """The note of `trader`'s position left open: what happened to the leg after the
        last lines, when anything did, how OpenAlgo's books differed from the engine when
        no later read confirms it (`Differing.words`), or that no read confirmed it after a
        fill, then its state (`left_open`); `session` is the `_session_loop.Session`."""
        differs = differing(session.reconcile, trader)
        if differs is None:
            unsure = [NOT_CONFIRMED] if session.reconcile.unjudged(trader.name) else []
            state = left_open(session.stops, trader)
            return "; ".join([*self._happened(trader), *unsure, state])
        state = left_open(session.stops, trader, books=False)
        return "; ".join([*self._happened(trader), differs.words(), state])

    def alert(self, session: Any) -> None:
        """One ERROR alert per leg whose difference with OpenAlgo's books no later read
        confirms, per leg left open that anything happened to after the last lines or that
        the exit guard still holds, and per leg left flat that anything happened to or that
        still has an order working (module docstring); `session` is the
        `_session_loop.Session`."""
        held = session.reconcile.guard.left()
        for trader in session.traders:
            differs = differing(session.reconcile, trader)
            text = self._alert_text(session.stops, trader, held, differs)
            if text is not None:
                session.alert(AlertKind.ERROR, text)

    def _alert_text(
        self, stops: Any, trader: Trader, held: Sequence[Any], differs: Differing | None
    ) -> str | None:
        """The ERROR alert of `trader`'s leg, or None when it needs none (`alert`)."""
        happened, not_sent = self._happened(trader), _held_changes(trader, held)
        if differs is not None and differs.flat_at_openalgo:
            working = still_working(stops, trader)
            parts = [*happened, differs.words(), *working, *not_sent, NOTHING_TO_CLOSE]
            return f"{trader.name} {trader.symbol} {trader.exchange}: {'; '.join(parts)}"
        if differs is not None:
            state = left_open(stops, trader, books=False)
            parts = [*happened, differs.words(), *not_sent, state]
            return f"{trader.name} {trader.symbol} {trader.exchange}: {'; '.join(parts)}"
        units = trader.open_units()
        if units:
            differs = [CLOSED_DIFFERING] if stops.broker_units(trader) is not None else []
            if not (happened or differs):
                return None
            parts = [*happened, *differs, *not_sent, left_open(stops, trader)]
            return f"{leg_words(trader, units)}: {'; '.join(parts)}"
        working = still_working(stops, trader)
        if not (happened or working):
            return None
        parts = [*happened, *working, *not_sent, FLAT_NOW]
        return f"{trader.name} {trader.symbol} {trader.exchange}: {'; '.join(parts)}"

    def _happened(self, trader: Trader) -> list[str]:
        """What happened to the leg after the last lines, in a note's words: its units
        changed (but by its stop's own fill, which says it), its stop ended, or both; then
        a cut of its stop the broker did not confirm."""
        now = trader.open_units()
        before = self._units.get(trader.name, now)
        stop = self._stops.get(trader.name)
        if stop is None:
            return ([] if before == now else [_moved(before, now)]) + self._cut(trader)
        name, status = stop
        ended = f"its stop {name} {_ENDED_WORDS[status]} at the broker"
        if before == now or status is OrderStatus.COMPLETE:
            return [f"{ended} {_AFTER_THE_LINES}", *self._cut(trader)]
        return [f"{_moved(before, now)}, and {ended}", *self._cut(trader)]

    def _cut(self, trader: Trader) -> list[str]:
        """The cut of the leg's stop after the last lines, in a note's words, unless the
        broker confirmed it (the terms the venue map holds are the cut's)."""
        cut = self._cuts.get(trader.name)
        if cut is None or trader.venue.terms(cut.kernel_order).quantity == cut.terms.quantity:
            return []
        how = _CUT_REFUSED if trader.name in self._refused else _CUT_UNANSWERED
        return [f"the cut of its stop {cut.order_id} to {cut.terms.quantity} units {how}"]


def _moved(before: float, now: float) -> str:
    """A change of a leg's signed units after the last lines, in a note's words."""
    return f"its position went from {before:g} to {now:g} units {_AFTER_THE_LINES}"


def _held_changes(trader: Trader, held: Sequence[Any]) -> list[str]:
    """The modifies and cancels of `trader`'s orders, and its GTT calls, the exit guard held
    back (`held` is every request it still holds), in an alert's words."""
    names = [
        f"the {'cancel' if isinstance(request, CancelRequest) else 'modify'} of "
        f"{request.intent.purpose.value.replace('_', ' ')} order "
        f"{trader.order_name(request.kernel_order)}"
        for request in held
        if isinstance(request, ModifyRequest | CancelRequest)
        and request.intent.strategy == trader.name
    ]
    names += [
        f"the {request.action.value} of {_gtt_name(trader, request)}"
        for request in held
        if isinstance(request, GttRequest) and request.strategy == trader.name
    ]
    return [f"not sent: {', '.join(names)}"] if names else []


def _gtt_name(trader: Trader, request: GttRequest) -> str:
    """A held GTT call's GTT in an alert's words: its trigger ID once its place answered."""
    kind = "stop GTT" if request.purpose is Purpose.STOP else "GTT order"
    known = None if trader.gtts is None else trader.gtts.gtt(request.gtt_id)
    return f"{kind} {known.trigger_id}" if known is not None and known.trigger_id else kind


__all__ = ["CLOSED_DIFFERING", "FLAT_NOW", "AfterLines"]
