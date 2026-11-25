"""The stops supervisor of a live session (16-session-design 4 and 3.3; unit 10): what the
session does in wall-clock time about the stops the broker holds (D31).

The kernel holds the leg rules (R2 to R7): it places a leg's stop with the fill that opens
the leg, keeps the stop's units and trigger, exits on a target or the daily loss, places
again a stop that ended by itself and exits at market on a rejected one. The supervisor
runs, on the loop's thread, the rules about the broker's behaviour in time:

- **Prices** (ticks, and quotes when the feed is silent) reach the kernel (tick mode,
  R5) and the price rule (`_stops_price`, 4.6).
- **Confirmation and unprotected windows** (4.2): a leg's stop is confirmed once accepted
  and heard working (`_stops_orders`); a guarded leg whose position holds more units
  than its confirmed stop covers is unprotected, in windows with their reason
  (`_stops_cover`). A stop accepted and not confirmed within `UNPROTECTED_EXIT_AFTER_S`
  exits the leg (`_stops_deadline`). A stop whose cancel was sent covers nothing: the
  cancel may be in effect at the broker.
- **Alerts** (`_stops_alerts`): an ERROR alert at once for a stop, an exit or a
  protective modify held back (OpenAlgo silent, an order of the instrument unresolved)
  and for an add whose answer was lost on a guarded leg (its units may be filled and
  uncovered); a REJECT alert for a stop the broker cancelled with no request from the
  session (the kernel places it again once, 4.10); a STOP_MOVED alert for a trailing
  modify the broker accepted (4.3).
- Held fills and held cancels (`_stops_held`, 4.11), refused cancels sent again
  (`_stops_cancels`, 4.5), a guarded leg left bare given its stop again (`_stops_bare`),
  a silent OpenAlgo probed and alerted while a stop or an exit waits (`_stops_silent`,
  4.9), protect by state (`_stops_protect`, 3.3), and the square-off of an instrument
  held for the day, whose stop's cancel is kept back so the stop works on
  (`_stops_held_day`, 2.4.0 item P3-3).

The loop calls the hooks as it takes each message, `duties` in its step D, and `review`
after the commands are handed over (E), so a window opens at the receive time of the
news that left units uncovered. Nothing is done once the mode guard has locked: nothing
can be sent. Once the close has printed its last lines no price is taken and no duty
runs, and a word that ends a leg's stop, and the cut of a stop and its refusal, are
noted for the result (`after`, 3.3 step 5).
Times are UTC nanoseconds, prices rupees, quantities units.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from ._fill_ledger import FillResult
from ._group_rules import and_join
from ._gtt_types import GttRequest
from ._order_answers import Outcome
from ._order_book import Resolution, Resolved
from ._session_trader import Trader
from ._stops_after import AfterLines
from ._stops_alerts import held_back, instrument_held_by, stop_moved, unbound_add
from ._stops_bare import BareLegs
from ._stops_cancels import CancelRetries
from ._stops_cover import Windows, reason_of
from ._stops_deadline import UNPROTECTED, UNPROTECTED_EXIT_AFTER_S, Deadlines
from ._stops_gtt import gtt_cover, gtt_guarded, overnight
from ._stops_held import HeldWords
from ._stops_held_day import DayHolds
from ._stops_kernel import cancelling, push_price
from ._stops_orders import LegOrders, Sent
from ._stops_price import PriceRule
from ._stops_protect import Protection
from ._stops_silent import SilentServer
from ._types import LOGGER_NAME, AlertKind, OrderIntent, OrderStatus, OrderUpdate, Purpose
from ._venue_requests import (
    STOP_TYPES,
    CancelRequest,
    ModifyRequest,
    OrderRequest,
    PlaceRequest,
    Priority,
)
from .inbox import NotSent, NotSentReason, PositionRows, SendResult, Settled
from .venue_map import HELD_REASONS, Applied, Handled

logger = logging.getLogger(LOGGER_NAME)


class StopsSupervisor:
    """The stops supervisor of `session`, a `_session_loop.Session` (module docstring).
    `windows`, `prices`, `held`, `cancels`, `silence`, `protection` and `after` are its
    parts."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._legs: dict[str, LegOrders] = {}
        self._requests: dict[int, OrderRequest] = {}
        self._changed: dict[str, tuple[int, int]] = {}
        self._noticed: set[int] = set()
        self._unguarded: set[str] = set()
        self._failed: set[str] = set()
        self.windows = Windows()
        self.prices = PriceRule(session)
        self.held = HeldWords(session)
        self.cancels = CancelRetries(session)
        self.protection = Protection(session, self)
        self.bare = BareLegs(session, self)
        self.deadlines = Deadlines(session, self)
        self.silence = SilentServer(session, self)
        self.after = AfterLines()
        self.day = DayHolds(session, self)

    def leg(self, trader: Trader) -> LegOrders:
        """What the supervisor knows of `trader`'s leg."""
        return self._legs.setdefault(trader.name, LegOrders())

    # ----- the loop's messages ----------------------------------------------------------

    def price(self, trader: Trader, price: float, at_ns: int) -> None:
        """A price of `trader`'s instrument, rupees at `at_ns` (a tick, a quote): to the
        kernel, then the price rule; to neither after the close's last lines."""
        if self.after.ended:
            return
        if push_price(trader, price, at_ns):
            self._touch(trader, at_ns)
        self.prices.observe(trader, self.leg(trader).stop, price, at_ns)

    def handed(self, requests: Sequence[Any]) -> None:
        """Requests handed to the order worker (a GTT's told to its strategy's GTT map)."""
        for request in requests:
            if isinstance(request, GttRequest):
                gtts = self._session.trader_named(request.strategy).gtts
                if gtts is not None:
                    gtts.handed(request)
            if isinstance(request, PlaceRequest | ModifyRequest | CancelRequest):
                self._requests[request.request_id] = request
                self._legs.setdefault(request.intent.strategy, LegOrders()).handed(request)
                self.after.handed(request)

    def updated(self, trader: Trader, update: OrderUpdate, result: FillResult) -> None:
        """An order word the kernel was given, with what it did."""
        kernel_order, leg = result.kernel_order, self.leg(trader)
        watched = leg.orders.get(kernel_order)
        cancel_sent = watched is not None and watched.cancel_sent
        after_lines = self.after.heard(trader, leg, kernel_order, update)
        leg.heard(kernel_order, update)
        self._touch(trader, update.ts_recv_ns)
        if result.increment is not None:
            self.prices.filled(trader, kernel_order, update.ts_recv_ns)
        if not update.status.terminal:
            return
        self.prices.ended(trader, kernel_order)
        self.cancels.ended(trader, kernel_order)
        # After the last lines the stop's end is noted for the result (`after`), not alerted.
        by_itself = update.status is OrderStatus.CANCELLED and not (cancel_sent or after_lines)
        if watched is not None and watched.purpose is Purpose.STOP and by_itself:
            self._session.alert(
                AlertKind.REJECT,
                f"{trader.name}: stop order {update.order_id} of {trader.symbol} "
                f"{trader.exchange} was cancelled at the broker with no request from the "
                "session: the engine places it again once, then exits at market",
            )

    def sent(self, trader: Trader, message: SendResult) -> None:
        """The worker's answer to a request it sent."""
        request = self._requests.pop(message.request_id, None)
        self.silence.done(message.request_id)
        if request is None:
            return
        now_ns, outcome = self._session.now(), message.outcome
        self._touch(trader, now_ns)
        leg, kernel_order = self.leg(trader), request.kernel_order
        if isinstance(request, PlaceRequest):
            leg.placed(kernel_order, outcome.kind, outcome.order_id, now_ns)
            if outcome.kind is Outcome.ACCEPTED:
                self._limit_exit(trader, request.intent, kernel_order, now_ns)
            elif outcome.kind is Outcome.UNANSWERED and request.intent.purpose is Purpose.ADD:
                unbound_add(self._session, self, trader, request)
        elif isinstance(request, CancelRequest):
            if outcome.kind in (Outcome.REJECTED, Outcome.UNANSWERED):
                self.cancels.reason(trader, kernel_order, outcome.reason)
        elif outcome.kind is Outcome.ACCEPTED:
            stop_moved(self._session, trader, request)
        elif outcome.kind is Outcome.REJECTED:
            self._modify_refused(trader, request)

    def not_sent(self, trader: Trader, message: NotSent) -> None:
        """A request the worker did not send (or put back to send later)."""
        request = message.request
        if not isinstance(request, PlaceRequest | ModifyRequest | CancelRequest):
            return
        held = message.reason in HELD_REASONS
        if held:
            self.silence.held(request)
        else:
            self._requests.pop(request.request_id, None)
            self.silence.done(request.request_id)
        self.leg(trader).not_sent(request, message.reason)
        self._touch(trader, self._session.now())
        if (
            held
            and request.priority is Priority.PROTECT
            and request.request_id not in self._noticed
        ):
            self._noticed.add(request.request_id)
            held_back(self._session, self, trader, request, message)
        elif isinstance(request, ModifyRequest):
            self._modify_refused(trader, request)
        elif isinstance(request, CancelRequest) and message.reason is NotSentReason.REFUSED:
            self.cancels.reason(trader, request.kernel_order, message.error)

    def settled(self, trader: Trader, message: Settled, handled: Handled) -> None:
        """A modify or cancel the worker settled by reading its order."""
        request, now_ns = message.request, self._session.now()
        self._touch(trader, now_ns)
        if isinstance(request, CancelRequest) and handled.applied is Applied.CANCEL_REFUSED:
            self.leg(trader).cancel_refused(request.kernel_order)
            if message.check is None:
                self.cancels.for_good(trader, request.kernel_order)
            else:
                self.cancels.refused(trader, request.kernel_order, request.order_id, now_ns)
        elif isinstance(request, ModifyRequest) and handled.applied is Applied.MODIFIED:
            stop_moved(self._session, trader, request)
        elif isinstance(request, ModifyRequest) and handled.applied is Applied.MODIFY_REFUSED:
            self._modify_refused(trader, request)

    def resolved(self, trader: Trader, resolution: Resolution) -> None:
        """An unanswered placement the order book settled."""
        kernel_order = trader.venue.kernel_order_of_intent(resolution.intent.intent_id)
        if kernel_order is None:
            return
        now_ns = self._session.now()
        self.leg(trader).resolved(kernel_order, resolution.kind, resolution.order_id, now_ns)
        self._touch(trader, now_ns)
        if resolution.kind is Resolved.BOUND:
            self._limit_exit(trader, resolution.intent, kernel_order, now_ns)

    def positions(self, message: PositionRows) -> None:
        """A position-book read for a held cancel."""
        self.held.positions(message)

    # ----- duties and review ------------------------------------------------------------

    def duties(self, now_ns: int, *, protecting: bool = False) -> None:
        """Step D: the unprotected deadline, the price rule, held words, cancels sent
        again, bare legs, a silent OpenAlgo, protect by state. With `protecting` (the
        emergency stop's turns) a part that fails is logged and skipped from then on, so
        protect by state, the last, still runs. None runs after the close's last lines."""
        if self._session.locked or self.after.ended:
            return
        parts = (
            self.deadlines.due,
            self.prices.due,
            self.held.due,
            self.cancels.due,
            self.bare.due,
            self.silence.due,
        )
        for part in parts:
            if not protecting:
                part(now_ns)
            elif part.__qualname__ not in self._failed:
                self._guarded(part, now_ns)
        self.protection.step(now_ns)

    def _guarded(self, part: Any, now_ns: int) -> None:
        """Run one part in the emergency stop. The session already stops on an error:
        a part that fails too must not keep the legs from being protected, so its error
        is logged with its traceback and the part is not run again."""
        try:
            part(now_ns)
        except Exception:
            self._failed.add(part.__qualname__)
            logger.exception(
                "the stops supervisor's %s failed in the emergency stop; protect by state "
                "goes on without it",
                part.__qualname__,
            )

    def review(self, now_ns: int) -> None:
        """After the commands are handed over: each leg's unprotected window."""
        for trader in self._session.traders:
            leg = self.leg(trader)
            span = self._changed.pop(trader.name, (now_ns, now_ns))
            units = round(abs(trader.open_units()))
            # An entry the order book cannot tell apart: its units count whatever the rules.
            unsure = abs(leg.ambiguous_entry_units())
            key = (trader.name, trader.symbol, trader.exchange)
            guarded = leg.guarded or (overnight(trader) and gtt_guarded(trader))
            if units == 0 and not leg.unbound_add_units and not unsure:
                leg.flat()
            elif not guarded and not unsure:
                self._warn_unguarded(trader, units)
            else:
                uncovered = units + leg.unbound_add_units + unsure - self.covered(trader)
                stop = leg.orders.get(leg.stop) if leg.stop is not None else None
                held = stop is not None and self.held.held_cancel(trader, stop.kernel_order)
                retries = self._session.exits.owns(trader.name)
                reason = reason_of(
                    leg,
                    held_cancel=held,
                    retries=retries,
                    held_by_other=instrument_held_by(self._session, self, trader) is not None,
                    closing_rejected=self._session.margin.unprotected_reason(trader),
                )
                opened = self.windows.note(key, uncovered, reason, span)
                if opened is not None:
                    self._session.event(
                        "unprotected",
                        strategy=trader.name,
                        units=opened.units,
                        reason=opened.reason,
                    )
                continue
            self.windows.note(key, 0, "", span)
        self.day.review(now_ns)

    # ----- what the alerts and the close read -------------------------------------------

    def covered(self, trader: Trader) -> int:
        """The units the leg's confirmed stop may still fill at the broker now, by the terms
        the broker confirmed (the venue map's, whatever modify or cancel of it waits), so
        more than the position until a sale by hand is fitted; none once a cancel of it
        was sent (the broker may have cancelled it already). A CNC or NRML leg's are its
        confirmed stop GTTs' (`_stops_gtt`)."""
        if overnight(trader):
            return gtt_cover(trader)
        leg = self.leg(trader)
        kernel_order = leg.stop
        watched = leg.orders.get(kernel_order) if kernel_order is not None else None
        if watched is None or watched.state is not Sent.CONFIRMED or watched.cancel_sent:
            return 0
        if kernel_order not in trader.working() or self.held.held_cancel(trader, kernel_order):
            return 0
        terms = trader.venue.terms(kernel_order)
        if terms.price_type not in STOP_TYPES:
            return 0
        return max(0, terms.quantity - trader.venue.filled(kernel_order))

    def broker_units(self, trader: Trader) -> int | None:
        """The broker's signed units of `trader`'s instrument (its net, of every strategy
        of the session that trades it) as the last read showed them while they differ from
        the engine's (`Reconciler.broker_units`), else None."""
        return self._session.reconcile.broker_units(trader.name)

    def books_words(self, trader: Trader, units: int) -> str:
        """OpenAlgo's `units` of `trader`'s instrument in a note's words: the net of every
        strategy of the session that trades it, named when there are several
        (18-multi-strategy 4.11)."""
        members = self._session.groups.members(trader)
        if len(members) <= 1:
            return f"{units} units"
        names = and_join([member.name for member in members])
        return f"{units} units of {trader.symbol} for {names} together"

    def cancelling(self, trader: Trader) -> frozenset[int]:
        """The kernel orders of `trader` whose cancel the kernel decided and the venue has
        not answered, but those whose cancel the reconciler's exit guard or the square-off
        of an instrument held for the day holds back: never sent, so they work at the
        broker."""
        held = self._session.reconcile.guard.held_cancels(trader.name)
        return cancelling(trader) - held - self.day.held_cancels(trader.name)

    # ----- helpers ------------------------------------------------------------------

    def _touch(self, trader: Trader, at_ns: int) -> None:
        """Note the receive time of news of the leg in this loop step: the first and the
        last (its window's instants, `Windows.note`)."""
        known = self._changed.get(trader.name)
        span = (at_ns, at_ns) if known is None else (min(known[0], at_ns), max(known[1], at_ns))
        self._changed[trader.name] = span

    def _limit_exit(
        self, trader: Trader, intent: OrderIntent, kernel_order: int, now_ns: int
    ) -> None:
        """An order was accepted: when it is the price rule's exit at a limit (the only
        risk exit the kernel sends as a LIMIT), the rule watches it."""
        if intent.purpose is Purpose.RISK_EXIT and intent.price_type == "LIMIT":
            self.prices.track_exit(trader, kernel_order, now_ns)

    def _modify_refused(self, trader: Trader, request: ModifyRequest) -> None:
        self.after.refused(request)
        if request.terms.price_type == "LIMIT" and self.prices.watching(
            trader, request.kernel_order
        ):
            self.prices.refused(trader, request.kernel_order)

    def _warn_unguarded(self, trader: Trader, units: int) -> None:
        if trader.name in self._unguarded:
            return
        self._unguarded.add(trader.name)
        self._session.warn(
            f"{trader.name}: a position of {units} {trader.symbol} {trader.exchange} was "
            "opened with no stop at the broker (no stop=, or a trail or breakeven not "
            "armed yet)"
        )


__all__ = ["UNPROTECTED", "UNPROTECTED_EXIT_AFTER_S", "StopsSupervisor"]
