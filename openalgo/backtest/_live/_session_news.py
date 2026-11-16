"""What the session loop does with each message it takes in (16-session-design 2.5 A; unit
8): ticks to the bar builders, the order worker's answers to the venue maps and the
kernel, broker candles to the strategies, the feed's events to the feed link.

- `FeedTick`: adapted, its receive time stamped now, and handed to its exchange's bar
  builder.
- `OrderNews`: through the order's venue map and fill ledger to the kernel; ENTRY, EXIT
  and REJECT alerts with the order IDs; a fill the kernel refuses is a reconciliation
  mismatch: the strategy's entries are locked and a RECONCILE_MISMATCH alert sent.
- `SendResult`: the venue map binds, rejects or waits; a refused placement is a REJECT
  alert, a queued one (semi-auto) an ERROR alert, an exit with no answer an ERROR alert.
- `NotSent`: the venue map rejects an entry or refuses a modify; a stale or refused entry
  is a REJECT alert.
- `Settled`: the venue map applies the read's verdict.
- `Resolutions`: bound (with the order's update, if any), not placed, or ambiguous (the
  strategy's entries locked, an ERROR alert).
- `HeldFills`: kept for the stops supervisor.
- `GuardLocked`: the session's guard lock (`Session.guard_locked`).
- `Candles`: to the strategy of the instrument (`Trader.candles`).
- `FeedEvent`: to the feed link; a recovery resumes every bar builder over the outage and
  asks the worker to poll the orders the stream may have missed.
- `WorkerFailed`: raised as `WorkerFailedError`, which stops the session (3.5).

Not handled yet, logged: `Books` and `PositionRows` (the reconciler's), `Quote` (the risk
watch's), `DataFailed`. Times are UTC nanoseconds, prices rupees, quantities units. Used
by the loop only.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from ._fill_ledger import FillOutcome, FillResult, Increment
from ._order_answers import Outcome
from ._order_book import Resolved
from ._order_state import ENTRY_PURPOSES
from ._session_trader import Trader
from ._types import LOGGER_NAME, AlertKind, OrderIntent, OrderStatus, OrderUpdate
from ._venue_requests import CancelRequest, ModifyRequest, PlaceRequest
from ._worker_queue import StreamGap
from .inbox import (
    Candles,
    DataFailed,
    FeedEvent,
    FeedTick,
    GuardLocked,
    HeldFills,
    Message,
    NotSent,
    NotSentReason,
    OrderNews,
    Resolutions,
    SendResult,
    Settled,
    WorkerFailed,
)
from .ticks import tick_from_feed
from .venue_map import Applied

logger = logging.getLogger(LOGGER_NAME)

#: The entry lock a fill the kernel refused sets on its strategy (16-session-design 6.2,
#: decision 2: lock and alert, nothing sent to fix it).
RECONCILE_LOCK = "reconciliation"
#: The entry lock of an order the order book cannot tell apart from another (6.2).
AMBIGUOUS_LOCK = "ambiguous_order"
#: Why a request was not sent, for the REJECT alert of an entry.
_NOT_SENT_ENTRY = frozenset({NotSentReason.STALE, NotSentReason.REFUSED})


class WorkerFailedError(Exception):
    """The order worker stopped on `error` (a journal failure or an unexpected one)."""

    def __init__(self, error: BaseException) -> None:
        super().__init__(f"the order worker stopped: {type(error).__name__}: {error}")
        self.error = error


class News:
    """The handlers of the loop's messages (module docstring) for `session`, the
    `_session_loop.Session` that owns them."""

    def __init__(self, session: Any) -> None:
        self._session = session
        #: The fills the order path holds, as last posted (the stops supervisor's).
        self.held: tuple[Any, ...] = ()

    def take(self, message: Message) -> None:
        """Handle one message. Raises WorkerFailedError for `WorkerFailed`."""
        handler = _HANDLERS.get(type(message))
        if handler is None:
            logger.debug("message left to a later unit of the session: %r", message)
            return
        handler(self, message)

    # ----- the feed ------------------------------------------------------------------

    def _tick(self, message: FeedTick) -> None:
        session = self._session
        tick = tick_from_feed(message.message, recv_ns=session.now(), tally=session.tally)
        if tick is None:
            return
        builder = session.builders.get(tick.exchange)
        if builder is not None:
            builder.on_tick(tick)
        session.last_tick[(tick.symbol, tick.exchange)] = tick

    def _feed_event(self, message: FeedEvent) -> None:
        session = self._session
        recovery = None if session.link is None else session.link.note(message)
        if recovery is None:
            return
        if recovery.outage is not None:
            for builder in session.builders.values():
                builder.resume(recovery.outage.start_ns, session.now())
            session.event("outage", start_ns=recovery.outage.start_ns, end_ns=session.now())
        if recovery.stream_gap_since_ns is not None:
            session.hand_over([StreamGap(recovery.stream_gap_since_ns)])

    # ----- the order worker's answers ----------------------------------------------------

    def _news(self, message: OrderNews) -> None:
        trader = self._owner(message.update.order_id, message.intent_id)
        if trader is None:
            logger.error(
                "order news of %s (intent %s) for no order of this session's venue maps",
                message.update.order_id,
                message.intent_id,
            )
            return
        self.apply_update(trader, message.update)

    def apply_update(self, trader: Trader, update: OrderUpdate) -> FillResult:
        """Give one order update to the kernel through `trader`'s venue map, then send the
        alerts it calls for and note it in the event stream."""
        session = self._session
        kernel_order = trader.venue.kernel_order_of(update.order_id)
        result = trader.venue.news(update)
        intent = trader.venue.intent_of(result.kernel_order)
        session.event(
            "news",
            strategy=trader.name,
            order_id=update.order_id,
            kernel_order=kernel_order,
            status=update.status.value,
            filled=update.filled_quantity,
            average_price=update.average_price,
            source=update.source,
            outcome=result.outcome.value,
        )
        if result.increment is not None:
            self._fill_alert(intent, update, result.increment)
        if result.ended is OrderStatus.REJECTED:
            why = update.rejection_reason or "no reason given"
            session.rejected += 1
            session.alert(
                AlertKind.REJECT, f"{_order_words(intent)} order {update.order_id} rejected: {why}"
            )
            session.exits.rejected(trader, result.kernel_order, intent, why)
        if result.outcome is FillOutcome.REFUSED or result.mismatch:
            self._mismatch(trader, update)
        return result

    def _fill_alert(self, intent: OrderIntent, update: OrderUpdate, increment: Increment) -> None:
        kind = AlertKind.ENTRY if intent.purpose in ENTRY_PURPOSES else AlertKind.EXIT
        self._session.event(
            "fill",
            strategy=intent.strategy,
            order_id=update.order_id,
            units=increment.units,
            price=increment.price,
            source=update.source,
            estimated=increment.derived_from_average,
        )
        self._session.alert(
            kind,
            f"{intent.action} {increment.units} {intent.symbol} {intent.exchange} at "
            f"{increment.price:.2f} ({intent.purpose.value}, order {update.order_id})",
        )

    def _mismatch(self, trader: Trader, update: OrderUpdate) -> None:
        session = self._session
        session.mismatches += 1
        if trader.lock(RECONCILE_LOCK):
            session.event("lock", strategy=trader.name, reason=RECONCILE_LOCK)
        session.alert(
            AlertKind.RECONCILE_MISMATCH,
            f"{trader.name}: the engine and OpenAlgo differ on order {update.order_id} "
            f"({update.symbol} {update.exchange}, {update.filled_quantity} filled): entries "
            "of the strategy are locked for the rest of the session and nothing is sent to "
            "fix it. Check the position and the order in OpenAlgo",
        )

    def _sent(self, message: SendResult) -> None:
        session = self._session
        trader = self._requester(message.request_id)
        if trader is None:
            logger.error("send result of request %d for no venue map", message.request_id)
            return
        handled = trader.venue.sent(message)
        outcome = message.outcome
        session.event(
            "sent",
            strategy=trader.name,
            request_id=message.request_id,
            intent_id=message.intent_id,
            order_id=message.order_id or outcome.order_id,
            outcome=outcome.kind.value,
            reason=outcome.reason,
            applied=handled.applied.value,
        )
        session.hand_over(list(handled.requests))
        if not message.intent_id:
            return
        kernel_order = trader.venue.kernel_order_of_intent(message.intent_id)
        intent = trader.venue.intent_of(kernel_order)
        if outcome.kind is Outcome.REJECTED:
            session.rejected += 1
            session.alert(
                AlertKind.REJECT,
                f"{_order_words(intent)} order refused by OpenAlgo: {outcome.reason} "
                f"(intent {intent.intent_id})",
            )
            session.exits.rejected(trader, kernel_order, intent, outcome.reason)
        elif outcome.kind is Outcome.QUEUED:
            session.alert(
                AlertKind.ERROR,
                f"{_order_words(intent)} order was queued in OpenAlgo's Action Center as "
                f"{outcome.pending_order_id or 'a pending order'} (semi-auto mode), not "
                "placed: reject it there. The session sends nothing more",
            )
        elif outcome.kind is Outcome.UNANSWERED and intent.purpose not in ENTRY_PURPOSES:
            session.alert(
                AlertKind.ERROR,
                f"{_order_words(intent)} order got no answer from OpenAlgo (intent "
                f"{intent.intent_id}): nothing more is sent for {intent.symbol} until the "
                "order book shows whether it was placed",
            )

    def _not_sent(self, message: NotSent) -> None:
        session = self._session
        request = message.request
        if not isinstance(request, PlaceRequest | ModifyRequest | CancelRequest):
            logger.info("worker request %r not sent (%s)", request, message.reason.value)
            return
        trader = session.trader_named(request.intent.strategy)
        handled = trader.venue.not_sent(message)
        session.event(
            "not_sent",
            strategy=trader.name,
            request_id=request.request_id,
            reason=message.reason.value,
            error=message.error,
            applied=handled.applied.value,
        )
        entry = request.intent.purpose in ENTRY_PURPOSES
        if isinstance(request, PlaceRequest) and handled.applied is Applied.REJECTED:
            why = f"not sent ({message.reason.value})"
            session.exits.rejected(trader, request.kernel_order, request.intent, why)
        if isinstance(request, PlaceRequest) and entry and message.reason in _NOT_SENT_ENTRY:
            session.rejected += 1
            session.alert(
                AlertKind.REJECT,
                f"{_order_words(request.intent)} order not sent ({message.reason.value}): "
                f"{message.error}",
            )

    def _settled(self, message: Settled) -> None:
        request = message.request
        trader = self._session.trader_named(request.intent.strategy)
        handled = trader.venue.settled(message)
        self._session.event(
            "settled",
            strategy=trader.name,
            request_id=request.request_id,
            expired=message.expired,
            applied=handled.applied.value,
        )

    def _resolutions(self, message: Resolutions) -> None:
        session = self._session
        for resolution in message.resolutions:
            intent = resolution.intent
            trader = session.trader_named(intent.strategy)
            handled = trader.venue.resolved(resolution)
            session.event(
                "resolved",
                strategy=trader.name,
                intent_id=intent.intent_id,
                resolution=resolution.kind.value,
                order_id=resolution.order_id,
                applied=handled.applied.value,
            )
            session.hand_over(list(handled.requests))
            if handled.applied is Applied.BOUND and resolution.update is not None:
                self.apply_update(trader, resolution.update)
            elif handled.applied is Applied.REJECTED:
                session.rejected += 1
                session.alert(
                    AlertKind.REJECT,
                    f"{_order_words(intent)} order was not placed: the order book does not "
                    f"hold it (intent {intent.intent_id})",
                )
                kernel_order = trader.venue.kernel_order_of_intent(intent.intent_id)
                session.exits.rejected(trader, kernel_order, intent, "not placed")
            elif resolution.kind is Resolved.AMBIGUOUS:
                self._ambiguous(trader, intent, resolution.candidates)

    def _ambiguous(self, trader: Trader, intent: OrderIntent, candidates: tuple[str, ...]) -> None:
        session = self._session
        if trader.lock(AMBIGUOUS_LOCK):
            session.event("lock", strategy=trader.name, reason=AMBIGUOUS_LOCK)
        session.alert(
            AlertKind.ERROR,
            f"{_order_words(intent)} order (intent {intent.intent_id}) cannot be told apart "
            f"in the order book from {', '.join(candidates) or 'another order'}: nothing more "
            f"is sent for {intent.symbol} and its entries are locked. Check the order book in "
            "OpenAlgo",
        )

    def _held_fills(self, message: HeldFills) -> None:
        self.held = message.held

    def _guard_locked(self, message: GuardLocked) -> None:
        self._session.guard_locked(message.breach)

    def _candles(self, message: Candles) -> None:
        session = self._session
        trader = session.trader_of(message.symbol, message.exchange)
        if trader is None:
            logger.error(
                "candles of %s:%s, which no strategy trades", message.exchange, message.symbol
            )
            return
        for note in trader.candles.take(message.rows, message.read_ns):
            session.warn(note)

    def _data_failed(self, message: DataFailed) -> None:
        logger.warning("a data read failed (%r): %s", message.request, message.error)

    def _worker_failed(self, message: WorkerFailed) -> None:
        raise WorkerFailedError(message.error)

    # ----- lookups -------------------------------------------------------------------

    def _owner(self, order_id: str, intent_id: str) -> Trader | None:
        for trader in self._session.traders:
            if trader.venue.kernel_order_of(order_id) is not None:
                return trader
        for trader in self._session.traders:
            if trader.venue.kernel_order_of_intent(intent_id) is not None:
                logger.error("order %s of intent %s is not bound yet", order_id, intent_id)
        return None

    def _requester(self, request_id: int) -> Trader | None:
        for trader in self._session.traders:
            if trader.venue.owns_request(request_id):
                return trader
        return None


def _order_words(intent: OrderIntent) -> str:
    """How an alert names an order: "ema_cross: BUY 10 SBIN NSE (entry)"."""
    return (
        f"{intent.strategy}: {intent.action} {intent.quantity} {intent.symbol} "
        f"{intent.exchange} ({intent.purpose.value})"
    )


_HANDLERS: Mapping[type, Any] = MappingProxyType(
    {
        FeedTick: News._tick,
        FeedEvent: News._feed_event,
        OrderNews: News._news,
        SendResult: News._sent,
        NotSent: News._not_sent,
        Settled: News._settled,
        Resolutions: News._resolutions,
        HeldFills: News._held_fills,
        GuardLocked: News._guard_locked,
        Candles: News._candles,
        DataFailed: News._data_failed,
        WorkerFailed: News._worker_failed,
    }
)


__all__ = ["AMBIGUOUS_LOCK", "RECONCILE_LOCK", "News", "WorkerFailedError"]
