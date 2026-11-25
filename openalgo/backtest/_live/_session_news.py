"""What the session loop does with each message it takes in (16-session-design 2.5 A; units
8 and 10): the order worker's answers to the venue maps, the kernel and the stops
supervisor (`stops`); the market's messages (ticks, quotes, candles, the feed's events,
other clients' fills) go to `_session_feed`.

- `OrderNews`: through the order's venue map and fill ledger to the kernel, with its
  alerts (`_session_fills`); news of an orphan the reconciler cancels is expected.
- `SendResult`: the venue map binds, rejects or waits; a refused placement is a REJECT
  alert, a queued one (semi-auto) an ERROR alert, an exit with no answer an ERROR alert.
- `NotSent`: the venue map rejects an entry or refuses a modify; a stale or refused entry,
  or one the order rate withdrew, is a REJECT alert.
- `Throttled`: a request the order rate held back, the event `order_throttled` and an
  ERROR alert once a session (`_order_throttle.throttled_news`).
- `Settled`: the venue map applies the read's verdict. A modify it applied, by this or by
  its `SendResult`, notes the trigger it leaves and when (`_session_slippage.StopTriggers`).
- `Resolutions`: bound (with the order's update, if any), not placed, or ambiguous (the
  strategy's entries locked, an ERROR alert): `_session_resolved`.
- `HeldFills`: to the stops supervisor (held fills, 4.11).
- `PositionRows`: to the stops supervisor (held cancels, 4.11) and the risk watch (the
  kill switch's position reads, `watch`).
- `Books`: to the reconciler (`reconcile`); a read the worker did not make (the guard
  locked) stops its reads. Each fill the kernel takes is told to it too, and the news of
  an orphan it cancels is expected.
- `GttNews`: to the reconciler's record of the legs a GTT holds in flight
  (`_gtt_flight`), an event, then to its strategy's kernel and the session's duties
  around it (`_session_gtts`).
- `GuardLocked`: the session's guard lock (`Session.guard_locked`).
- `WorkerFailed`: raised as `WorkerFailedError`, which stops the session (3.5).

Each order answer reaches the stops supervisor after the venue map has applied it. A
closing order rejected on an instrument several strategies trade, by its placement's
answer or its REJECTED word (`_session_fills`), is taken by `_margin_reject` first: the
venue map then takes the answer, whose own rejection the kernel refuses as the race of an
order ended already. Times are UTC nanoseconds, prices rupees, quantities units. Used by
the loop only.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from ._gtt_types import GttOutcome
from ._order_answers import Outcome
from ._order_state import ENTRY_PURPOSES
from ._order_throttle import throttled_news
from ._session_feed import MARKET_DATA_HANDLERS
from ._session_fills import RECONCILE_LOCK, FillNews
from ._session_gtts import GttNewsHandler
from ._session_resolved import order_words
from ._session_trader import Trader
from ._types import LOGGER_NAME, AlertKind
from ._venue_requests import Applied, CancelRequest, Handled, ModifyRequest, PlaceRequest
from ._worker_queue import ReadBooks
from .inbox import (
    Books,
    GttNews,
    GuardLocked,
    HeldFills,
    Message,
    NotSent,
    NotSentReason,
    OrderNews,
    PositionRows,
    Resolutions,
    SendResult,
    Settled,
    Throttled,
    WorkerFailed,
)

logger = logging.getLogger(LOGGER_NAME)

#: Why a request was not sent, for the REJECT alert of an entry.
_NOT_SENT_ENTRY = frozenset({NotSentReason.STALE, NotSentReason.REFUSED, NotSentReason.THROTTLED})


class WorkerFailedError(Exception):
    """The order worker stopped on `error` (a journal failure or an unexpected one)."""

    def __init__(self, error: BaseException) -> None:
        super().__init__(f"the order worker stopped: {type(error).__name__}: {error}")
        self.error = error


class News(FillNews):
    """The handlers of the loop's messages (module docstring) for `session`, the
    `_session_loop.Session` that owns them."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._ambiguous_seen: set[str] = set()
        self._throttle_alerted = False
        self._gtts = GttNewsHandler(session)

    def take(self, message: Message) -> None:
        """Handle one message. Raises WorkerFailedError for `WorkerFailed`."""
        market = MARKET_DATA_HANDLERS.get(type(message))
        if market is not None:
            market(self._session, message)
            return
        handler = _HANDLERS.get(type(message))
        if handler is None:
            logger.debug("message left to a later unit of the session: %r", message)
            return
        handler(self, message)

    # ----- the order worker's answers ----------------------------------------------------

    def _news(self, message: OrderNews) -> None:
        trader = self._owner(message.update.order_id, message.intent_id)
        if trader is None and self._session.reconcile.orphan(message.update.order_id):
            logger.info("news of orphan %s: %s", message.update.order_id, message.update.status)
            return
        if trader is None:
            logger.error(
                "order news of %s (intent %s) for no order of this session's venue maps",
                message.update.order_id,
                message.intent_id,
            )
            return
        self.apply_update(trader, message.update)

    def _sent(self, message: SendResult) -> None:
        session = self._session
        trader = self._requester(message.request_id)
        if trader is None:
            logger.error("send result of request %d for no venue map", message.request_id)
            return
        if message.outcome.kind is Outcome.REJECTED and message.intent_id:
            refused = trader.venue.kernel_order_of_intent(message.intent_id)
            intent = trader.venue.intent_of(refused) if refused is not None else None
            if intent is not None and session.margin.takes(trader, intent):
                session.margin.reject(trader, refused, intent, message.outcome.reason)
        handled = trader.venue.sent(message)
        if message.order_id:
            self._modified(trader, trader.venue.kernel_order_of(message.order_id), handled)
        session.stops.sent(trader, message)
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
                f"{order_words(intent)} order refused by OpenAlgo: {outcome.reason} "
                f"(intent {intent.intent_id})",
            )
            session.exits.rejected(trader, kernel_order, intent, outcome.reason)
        elif outcome.kind is Outcome.QUEUED:
            session.alert(
                AlertKind.ERROR,
                f"{order_words(intent)} order was queued in OpenAlgo's Action Center as "
                f"{outcome.pending_order_id or 'a pending order'} (semi-auto mode), not "
                "placed: reject it there. The session sends nothing more",
            )
        elif outcome.kind is Outcome.UNANSWERED and intent.purpose not in ENTRY_PURPOSES:
            session.alert(
                AlertKind.ERROR,
                f"{order_words(intent)} order got no answer from OpenAlgo (intent "
                f"{intent.intent_id}): nothing more is sent for {intent.symbol} until the "
                "order book shows whether it was placed",
            )

    def _not_sent(self, message: NotSent) -> None:
        session = self._session
        request = message.request
        if not isinstance(request, PlaceRequest | ModifyRequest | CancelRequest):
            logger.info("worker request %r not sent (%s)", request, message.reason.value)
            if isinstance(request, ReadBooks):
                session.reconcile.not_read()
            return
        trader = session.trader_named(request.intent.strategy)
        handled = trader.venue.not_sent(message)
        session.stops.not_sent(trader, message)
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
                f"{order_words(request.intent)} order not sent ({message.reason.value}): "
                f"{message.error}",
            )

    def _throttled(self, message: Throttled) -> None:
        alerted = self._throttle_alerted
        self._throttle_alerted = throttled_news(self._session, message, alerted=alerted)

    def _settled(self, message: Settled) -> None:
        request = message.request
        trader = self._session.trader_named(request.intent.strategy)
        handled = trader.venue.settled(message)
        self._modified(trader, request.kernel_order, handled)
        self._session.stops.settled(trader, message, handled)
        self._session.event(
            "settled",
            strategy=trader.name,
            request_id=request.request_id,
            expired=message.expired,
            applied=handled.applied.value,
        )

    def _modified(self, trader: Trader, kernel_order: int | None, handled: Handled) -> None:
        """Note the trigger a modify the venue map applied leaves its order holding, and
        when: the fill comparison looks for a stop's fill from then (`StopTriggers`)."""
        if handled.applied is Applied.MODIFIED and kernel_order is not None:
            trigger = trader.venue.terms(kernel_order).trigger_price
            self._session.triggers.modified(trader.name, kernel_order, trigger, self._session.now())

    def _held_fills(self, message: HeldFills) -> None:
        self._session.stops.held.update(message.held)

    def _positions(self, message: PositionRows) -> None:
        self._session.stops.positions(message)
        self._session.risk.positions(message)

    def _books(self, message: Books) -> None:
        self._session.reconcile.books(message)

    def _gtt_news(self, message: GttNews) -> None:
        news = message.news
        self._session.reconcile.gtts.take(news)
        if isinstance(news, GttOutcome):
            fields = {"strategy": news.strategy, "seq": news.seq, "found": news.event.value}
        else:
            fields = {"strategy": news.request.strategy, "seq": news.seq, "answer": news.kind.value}
        self._session.event("gtt", **fields)
        self._gtts.take(news)

    def _guard_locked(self, message: GuardLocked) -> None:
        self._session.guard_locked(message.breach)

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


_HANDLERS: Mapping[type, Any] = MappingProxyType(
    {
        OrderNews: News._news,
        SendResult: News._sent,
        NotSent: News._not_sent,
        Throttled: News._throttled,
        Settled: News._settled,
        Resolutions: News._resolutions,
        HeldFills: News._held_fills,
        PositionRows: News._positions,
        Books: News._books,
        GttNews: News._gtt_news,
        GuardLocked: News._guard_locked,
        WorkerFailed: News._worker_failed,
    }
)


__all__ = ["RECONCILE_LOCK", "News", "WorkerFailedError"]
