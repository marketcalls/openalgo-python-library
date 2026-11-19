"""The venue map of one strategy's kernel session (16-session-design 2.5 E, 2.7, 3.6, 4.3,
4.5): the kernel's commands turned into the order worker's requests, the kernel order,
intent and broker order IDs tied together, and every answer the worker posts turned into
the one kernel call it calls for.

- **Commands to requests.** A "place" becomes a `PlaceRequest` with its `OrderIntent`
  (`intent_fields`); a "modify" a `ModifyRequest` with the `modifyorder` fields; a
  "cancel" a `CancelRequest`. A modify or cancel of an order whose placement is not bound
  yet waits here until it is, and is dropped when the placement ends rejected or not
  placed (the kernel ends the order then). Each request carries its priority
  (`_venue_requests`).
- **One result per Modify** (R8). A modify is pending from its command until exactly one
  of: `apply_modify` (an accepted answer, a row that shows its terms, or an order update
  that shows its quantity, which the kernel needs before fills beyond the old units),
  `apply_modify_refused` (a refusal, a request not sent for any reason but a hold, a row
  that shows the old terms, no row within the worker's deadline), or the order's
  terminal news. Later answers of the same request change nothing.
- **Cancels.** An accepted cancel changes nothing in the kernel: the CANCELLED news ends
  the order, since a fill may come first. A cancel the row shows still working is
  `apply_cancel_refused`, after which the session sends it again (4.5); so is one the
  exit guard held and withdraws, never sent (`withdraw_cancel`), which waits for its
  answer again if the guard holds it again (`restore_cancel`).
- **Fills** go through each order's `FillLedger`, the one door (`news`).

The kernel is always waiting when these run (2.5), so a refusal of "order_ended" or
"not_pending" is the race of an order that ended meanwhile, logged at debug level.
Quantities are whole units, prices rupees, times UTC nanoseconds. Used by the loop only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

from ..errors import LiveError, VenueRefusalError
from ._fill_ledger import VENUE_REJECT_CODE, FillLedger, FillResult
from ._journal_reads import JournalSession
from ._order_answers import Outcome
from ._order_book import Resolution, Resolved
from ._order_state import ENTRY_PURPOSES
from ._types import LOGGER_NAME, AlertsLike, Now, OrderIntent, OrderUpdate, Purpose
from ._venue_requests import (
    Applied,
    CancelRequest,
    Handled,
    ModifyRequest,
    OrderRequest,
    OrderTerms,
    PlaceRequest,
    Verdict,
    cancel_priority,
    cancel_verdict,
    intent_fields,
    modify_priority,
    modify_verdict,
    place_priority,
    terms_of,
)
from ._venue_saved import RestoredOrders, VenueOrder
from .inbox import NotSent, NotSentReason, SendResult, Settled

logger = logging.getLogger(LOGGER_NAME)

#: The reasons the worker gives for a request it put back to send later (16-session-design
#: 2.4): it has no result yet.
HELD_REASONS = frozenset({NotSentReason.HELD, NotSentReason.SILENT})
#: Refusals of a modify's or a cancel's result that mean the order ended meanwhile.
RACES = frozenset({"order_ended", "not_pending"})


class VenueMap(RestoredOrders):
    """The venue map of `engine` (an `Engine.session`), the session of strategy `strategy`
    (module docstring); a restart saves and rebuilds it (`_venue_saved.RestoredOrders`).

    `instruments` are the engine's in registration order, each with `symbol`, `exchange`
    and `tick_size` (rupees: the preflight's facts); `session` is the journal session
    (day, mode and starts: the intent IDs); `request_ids` numbers the requests, shared by
    every strategy's map so the worker's answers name one request; `alerts` the session's;
    `now` gives UTC nanoseconds. Raises LiveError for an instrument without a tick.
    """

    def __init__(
        self,
        engine: Any,
        *,
        strategy: str,
        instruments: Sequence[Any],
        session: JournalSession,
        request_ids: Iterator[int],
        alerts: AlertsLike,
        now: Now,
    ) -> None:
        self._engine = engine
        self._strategy = strategy
        self._markets = tuple((item.symbol, item.exchange) for item in instruments)
        self._ticks = tuple(item.tick_size for item in instruments)
        for (symbol, exchange), tick in zip(self._markets, self._ticks, strict=True):
            if tick is None:
                raise LiveError(
                    f"{exchange}:{symbol} has no tick size: a live session takes it from "
                    "the preflight's facts of the instrument"
                )
        self._session = session
        self._request_ids = request_ids
        self._alerts = alerts
        self._now = now
        self._orders: dict[int, VenueOrder] = {}
        self._by_order_id: dict[str, int] = {}
        self._by_intent: dict[str, int] = {}
        # Requests handed over whose answer (SendResult or NotSent) has not come.
        self._issued: dict[int, OrderRequest] = {}

    @property
    def strategy(self) -> str:
        """The strategy whose engine this maps."""
        return self._strategy

    def owns_request(self, request_id: int) -> bool:
        """True while a request this map made waits for its answer."""
        return request_id in self._issued

    def kernel_order_of(self, order_id: str) -> int | None:
        """The kernel order bound to broker order `order_id`, or None."""
        return self._by_order_id.get(order_id)

    def kernel_order_of_intent(self, intent_id: str) -> int | None:
        """The kernel order of intent `intent_id`, or None."""
        return self._by_intent.get(intent_id)

    def order_id_of(self, kernel_order: int) -> str:
        """The broker's order ID of a kernel order, "" while it is not bound. Raises
        LiveError for an order this map did not place."""
        return self._order(kernel_order).order_id

    def intent_of(self, kernel_order: int) -> OrderIntent:
        """The intent of a kernel order. Raises LiveError for one this map did not place."""
        return self._order(kernel_order).intent

    def placed(self) -> list[tuple[int, VenueOrder]]:
        """Every order this map placed, a restart's restored ones included, by kernel
        order: its intent, the broker's order ID once bound ("" before), its terms."""
        return sorted(self._orders.items())

    def terms(self, kernel_order: int) -> OrderTerms:
        """The terms the broker holds for a kernel order as far as the session knows: those
        it was placed with, or the last modify the broker confirmed; a modify or cancel
        held back or unanswered changes nothing here. Raises LiveError for an order this
        map did not place."""
        return self._order(kernel_order).terms

    # ----- commands to requests --------------------------------------------------------

    def requests(self, commands: Iterable[Mapping[str, Any]]) -> list[OrderRequest]:
        """The requests for the worker of `engine.take_commands()`'s dicts, in order; a
        modify or cancel of an order not bound yet waits for the bind. Raises LiveError
        for a command this session cannot send (`intent_fields`), a modify or cancel of an
        order this map did not place, or an unknown command."""
        made: list[OrderRequest] = []
        for command in commands:
            kind = command.get("command")
            if kind == "place":
                made.append(self._place(command))
            elif kind in ("modify", "cancel"):
                made.extend(self._change(command))
            else:
                raise LiveError(f"the kernel handed an unknown command: {dict(command)!r}")
        return made

    def _place(self, command: Mapping[str, Any]) -> PlaceRequest:
        intent = intent_fields(
            command,
            strategy=self._strategy,
            markets=self._markets,
            session=self._session,
            created_ns=self._now(),
        )
        kernel_order, tick = int(command["order_id"]), self._ticks[int(command["instrument"])]
        if kernel_order in self._orders:
            raise LiveError(f"kernel order o{kernel_order} of {self._strategy} is placed twice")
        ledger = FillLedger(self._engine, kernel_order, tick=tick, alerts=self._alerts)
        self._orders[kernel_order] = VenueOrder(intent, terms_of(command), ledger)
        self._by_intent[intent.intent_id] = kernel_order
        number, priority = next(self._request_ids), place_priority(intent)
        request = PlaceRequest(number, kernel_order, intent, priority)
        self._issued[request.request_id] = request
        return request

    def _change(self, command: Mapping[str, Any]) -> list[OrderRequest]:
        order = self._order(int(command["order_id"]))
        if order.ended:
            logger.debug("a %s of ended order %s is dropped", command["command"], order.intent)
            return []
        if not order.order_id:
            order.deferred.append(command)
            return []
        return [self._request(order, command)]

    def _request(self, order: VenueOrder, command: Mapping[str, Any]) -> OrderRequest:
        kernel_order, number = int(command["order_id"]), next(self._request_ids)
        request: OrderRequest
        if command["command"] == "modify":
            terms = terms_of(command)
            priority = modify_priority(order.intent, order.terms, terms)
            request = ModifyRequest(
                number, kernel_order, order.intent, order.order_id, terms, order.terms, priority
            )
            order.modify = request
        else:
            priority = cancel_priority(order.intent)
            request = CancelRequest(number, kernel_order, order.intent, order.order_id, priority)
            order.cancel = request
        self._issued[request.request_id] = request
        return request

    # ----- the worker's answers --------------------------------------------------------

    def sent(self, result: SendResult) -> Handled:
        """Take the outcome of a request the worker sent (2.5 A.3). Raises LiveError for a
        request this map did not make or that was answered already."""
        request = self._issued.pop(result.request_id, None)
        if request is None:
            raise LiveError(f"request {result.request_id} is not one this venue map waits for")
        order = self._order(request.kernel_order)
        kind = result.outcome.kind
        if isinstance(request, PlaceRequest):
            if kind is Outcome.ACCEPTED:
                return self._bind(order, result.outcome.order_id)
            if kind is Outcome.REJECTED:
                return self._reject(order, f"refused: {result.outcome.reason}")
            return Handled(Applied.NOTHING)
        if isinstance(request, ModifyRequest) and kind in (Outcome.ACCEPTED, Outcome.REJECTED):
            return self._modify_result(order, request, confirmed=kind is Outcome.ACCEPTED)
        # An unanswered or refused modify or cancel is settled by the worker; an accepted
        # cancel counts when the order's CANCELLED news comes, since a fill may come first.
        return Handled(Applied.NOTHING)

    def not_sent(self, message: NotSent) -> Handled:
        """Take a request the worker did not send (2.5 A.4): one put back waits; an entry
        or add is rejected; a modify gets `apply_modify_refused`; a stop or exit after the
        guard locked, and a cancel, change nothing (a refused cancel is settled by the
        worker's read); any other stop not sent is placed again once (R3), an exit
        rejected."""
        request = message.request
        if not isinstance(request, PlaceRequest | ModifyRequest | CancelRequest):
            raise LiveError(f"the venue map makes order requests only, not {request!r}")
        if message.reason in HELD_REASONS:
            return Handled(Applied.NOTHING)
        self._issued.pop(request.request_id, None)
        order = self._order(request.kernel_order)
        if isinstance(request, ModifyRequest):
            return self._modify_result(order, request, confirmed=False)
        if isinstance(request, PlaceRequest):
            entry = request.intent.purpose in ENTRY_PURPOSES
            if entry or message.reason is not NotSentReason.LOCKED:
                why = f"not sent ({message.reason.value}): {message.error}"
                return self._reject(order, why, placed=False)
        return Handled(Applied.NOTHING)

    def settled(self, message: Settled) -> Handled:
        """Take a modify or cancel the worker settled by reading its order (2.5 A.5, 4.3,
        4.5): the row's verdict applied once. A cancel settled with no row (no read can
        ever settle it: the order path does not follow the order) is refused, so the stop
        keeps protecting and the exit waits."""
        request = message.request
        if not isinstance(request, ModifyRequest | CancelRequest):
            raise LiveError(f"only a modify or a cancel is settled, not {request!r}")
        order, check = self._order(request.kernel_order), message.check
        if isinstance(request, ModifyRequest):
            # No row within the worker's deadline: the old terms are taken to stand (4.3).
            verdict = Verdict.REFUSED if check is None else modify_verdict(request, check)
            if verdict not in (Verdict.CONFIRMED, Verdict.REFUSED):
                return Handled(Applied.NOTHING)
            return self._modify_result(order, request, confirmed=verdict is Verdict.CONFIRMED)
        verdict = Verdict.REFUSED if check is None else cancel_verdict(check)
        waiting = order.cancel is not None and order.cancel.request_id == request.request_id
        if not waiting or verdict is not Verdict.REFUSED:
            return Handled(Applied.NOTHING)
        order.cancel = None
        if self._answer(self._engine.apply_cancel_refused, request.kernel_order):
            return Handled(Applied.CANCEL_REFUSED)
        return Handled(Applied.NOTHING)

    def withdraw_cancel(self, request: CancelRequest) -> bool:
        """A cancel this map made that the session will never send (the exit guard held it
        and withdraws it: 16-session-design 3.3 step 5 "As built", round 7): it waits for
        no answer, and the kernel takes it as refused (`apply_cancel_refused`), so the
        order works on and the kernel may fit it to the position. True when the kernel
        took it. Raises LiveError for an order this map did not place."""
        self._issued.pop(request.request_id, None)
        order = self._order(request.kernel_order)
        if order.cancel is not None and order.cancel.request_id == request.request_id:
            order.cancel = None
        return self._answer(self._engine.apply_cancel_refused, request.kernel_order)

    def restore_cancel(self, request: CancelRequest) -> None:
        """A cancel `withdraw_cancel` withdrew that the session holds again (the exit
        guard puts it back when the adoption it made way for is refused): it waits for its
        answer again, should it ever be sent. The kernel took it as refused, so the order
        works on there as at the broker, where it was never sent; a later refusal of it is
        the "not_pending" race. Raises LiveError for an order this map did not place."""
        order = self._order(request.kernel_order)
        self._issued[request.request_id] = request
        if order.cancel is None:
            order.cancel = request

    def resolved(self, resolution: Resolution) -> Handled:
        """Take an unbound placement the order book settled (2.5 A.6): BOUND binds it (the
        order's update, when the resolution carries one, is the loop's to pass to
        `news`), NOT_PLACED rejects it (a leg stop is placed again once, R3), AMBIGUOUS
        leaves it unbound."""
        kernel_order = self._by_intent.get(resolution.intent.intent_id)
        if kernel_order is None:
            raise LiveError(f"intent {resolution.intent.intent_id} is not this venue map's")
        order = self._orders[kernel_order]
        if resolution.kind is Resolved.BOUND:
            return self._bind(order, resolution.order_id)
        if resolution.kind is Resolved.NOT_PLACED:
            return self._reject(order, "not placed: the order book does not hold it", placed=False)
        return Handled(Applied.NOTHING)

    def news(self, update: OrderUpdate) -> FillResult:
        """Take an order's news (2.5 A.2): a quantity that shows the pending modify
        confirms it first, then the fills and the terminal status go through the order's
        ledger; a terminal word the kernel took ends what waited for a result (one it
        refused leaves the order working there, a mismatch in the result). Raises
        LiveError for an order this map has not bound."""
        kernel_order = self._by_order_id.get(update.order_id)
        if kernel_order is None:
            raise LiveError(f"order {update.order_id} is not bound in this venue map")
        order = self._orders[kernel_order]
        pending, units = order.modify, update.quantity
        if pending is not None and units == pending.terms.quantity != order.terms.quantity:
            self._modify_result(order, pending, confirmed=True)
        result = order.ledger.apply(update)
        # Ended only once the kernel has ended it: after a refused terminal result the
        # kernel still works the order, and its modify and cancel still need results.
        if update.status.terminal and kernel_order not in self._engine.working_orders():
            order.ended, order.modify, order.cancel = True, None, None
        return result

    # ----- the kernel calls ------------------------------------------------------------

    def _bind(self, order: VenueOrder, order_id: str) -> Handled:
        order.order_id = order_id
        self._by_order_id[order_id] = self._by_intent[order.intent.intent_id]
        deferred, order.deferred = order.deferred, []
        released = tuple(self._request(order, command) for command in deferred)
        return Handled(Applied.BOUND, released)

    def _reject(self, order: VenueOrder, why: str, *, placed: bool = True) -> Handled:
        kernel_order = self._by_intent[order.intent.intent_id]
        logger.warning("intent %s %s", order.intent.intent_id, why)
        order.ended, order.deferred = True, []
        # A leg stop that never reached the broker ended with no request of the session's,
        # which the kernel places again once (R3); a rejection would exit the leg at market.
        if not placed and order.intent.purpose is Purpose.STOP:
            ended = self._answer(self._engine.apply_cancel, kernel_order)
        else:
            ended = self._answer(self._engine.apply_reject, kernel_order, VENUE_REJECT_CODE)
        return Handled(Applied.REJECTED if ended else Applied.NOTHING)

    def _modify_result(
        self, order: VenueOrder, request: ModifyRequest, *, confirmed: bool
    ) -> Handled:
        if order.modify is None or order.modify.request_id != request.request_id:
            logger.debug("modify %d was answered already", request.request_id)
            return Handled(Applied.NOTHING)
        order.modify = None
        if confirmed and self._answer(self._engine.apply_modify, request.kernel_order):
            order.terms = request.terms
            return Handled(Applied.MODIFIED)
        # A confirmation the kernel cannot take (a modify below the units filled, say)
        # still owes it a result: the old terms stand.
        if self._answer(self._engine.apply_modify_refused, request.kernel_order):
            return Handled(Applied.MODIFY_REFUSED)
        return Handled(Applied.NOTHING)

    def _answer(self, call: Callable[..., Any], kernel_order: int, *args: Any) -> bool:
        """Make a result call on `kernel_order`; False when the kernel refused it, logged:
        a race (`RACES`) at debug level, anything else at error level."""
        try:
            call(kernel_order, *args)
        except VenueRefusalError as error:
            level = logging.DEBUG if error.kind in RACES else logging.ERROR
            logger.log(level, "the engine refused a result of o%d: %s", kernel_order, error)
            return False
        return True

    def _order(self, kernel_order: int) -> VenueOrder:
        order = self._orders.get(kernel_order)
        if order is None:
            raise LiveError(
                f"kernel order o{kernel_order} of {self._strategy} was not placed through "
                "this venue map (after a restart, its saved map is restored first)"
            )
        return order


__all__ = ["HELD_REASONS", "RACES", "Applied", "Handled", "VenueMap"]
