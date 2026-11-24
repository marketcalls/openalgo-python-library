"""What the order worker does with one request it takes (16-session-design 2.4, "Executing
a request"), split from `worker` so each module keeps one job.

- A Place is `OrderPath.send`, posted as `SendResult`; a bound STOP then records its
  leg's journal stop (`_worker_stops`). An entry or add decided more than
  `ENTRY_STALE_AFTER_S` before is `NotSent(stale)` instead, and one of a strategy whose
  entries the loop has locked since it was queued is `NotSent(entries_locked)`, read just
  before the send (`_locked_entries`, 2.4.0 item P3-1).
- A Modify or Cancel of an order the order path knows terminal is `NotSent(order_ended)`;
  else `modify` or `cancel`, and an unanswered or refused answer leaves the order
  unsettled, its further requests held until a read settles it (`_worker_settle`).
- `ModeChangedError` (caught before `LiveError`, its base) is `NotSent(locked)`. Any other
  `LiveError` refuses the request (`NotSent(refused)`: an entry's moment has passed; a
  cancel is then settled by a read, 4.5), but for a protective request (priority 0) the
  order path holds back for now: OpenAlgo silent, or an intent of the instrument unbound.
  That one is put back for `REQUEUE_AFTER_S`, noticed as `NotSent(silent)` or
  `NotSent(held)` at once and then at most every `HELD_NOTICE_EVERY_S`. A protective
  request refused for any other reason (an order the session does not follow, a field the
  order path refuses) is a defect that no wait cures: an ERROR alert names it. A
  `JournalError` is raised: nothing may be sent once an intent cannot be written first.
- Settling (`settle_due`, 2.4 step 7): each unsettled order due a read is read with
  `check_order`, and the row's verdict posted as `Settled`; a modify past its deadline is
  settled with no row and an ERROR alert, and a cancel no read settles is alerted once.
  An order the order path cannot read at all (it does not follow it, so no read will
  ever settle it) is settled with no row at once, taken as refused: its one ERROR alert
  was its refusal's.
- A release of a held fill or cancel hands its news over (`_worker_news`, counted in hand
  from before the word leaves the order path's hold); the book reads go to the guard
  answer by answer and are posted as `Books` or `PositionRows`, or `NotSent(locked)` once
  the guard locks; a probe asks OpenAlgo whether it answers; a stream gap asks for polls.
  A session with a CNC strategy reads the holdings with every book read, right after the
  position book (`_broker_units`: a CNC leg's units are in both).
- A restart's `Settle` makes a restored order's modify or cancel unsettled, so a read
  settles it as an unanswered one (3.6); `CancelOrphan` cancels an order of the journal
  the engine does not hold (5.3), with an ERROR alert when it cannot be cancelled.
- A `GttRequest` goes to the GTT sender (`_gtt_send`); `ReadGtts` asks the GTT follower
  for a read of the GTT book, and `BindGttChild` binds a fired GTT's child
  (`_gtt_follow`; 20-positional-live 3).

Times are UTC nanoseconds. Used by the worker's thread only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

from .._layout import NANOS_PER_SECOND
from ..errors import JournalError, LiveError, ModeChangedError
from ._answers import server_answer
from ._gtt_send import GttSender
from ._gtt_types import BindGttChild, GttRequest, ReadGtts
from ._locked_entries import LockedEntries, locked_words
from ._order_answers import Outcome
from ._order_check import OrderCheck
from ._order_resolve import same_leg
from ._order_state import ENTRY_PURPOSES
from ._types import LOGGER_NAME, AlertKind, AlertsLike, OrderIntent, Purpose
from ._venue_requests import CancelRequest, ModifyRequest, PlaceRequest, Priority
from ._worker_news import NewsHandover
from ._worker_queue import (
    CancelOrphan,
    Probe,
    ReadBooks,
    ReadPositions,
    ReleaseCancel,
    ReleaseFill,
    Request,
    Settle,
    StreamGap,
    WorkerQueue,
    is_stale,
    order_key,
)
from ._worker_settle import (
    CANCEL_READS_ALERT,
    Change,
    Settlements,
    Unsettled,
    expired_text,
    unsettled_cancel_text,
)
from ._worker_stops import StopBook
from .guard import ModeGuard
from .inbox import Books, Message, NotSent, NotSentReason, PositionRows, SendResult, Settled
from .journal import Journal
from .orders import OrderPath

logger = logging.getLogger(LOGGER_NAME)

#: Seconds a protective request held back waits before it is tried again (2.4).
REQUEUE_AFTER_S = 0.5
#: Seconds between two notices of one protective request held back, so the loop can alert
#: without being flooded (4.8).
HELD_NOTICE_EVERY_S = 5.0
_REQUEUE_NS = round(REQUEUE_AFTER_S * NANOS_PER_SECOND)
_NOTICE_NS = round(HELD_NOTICE_EVERY_S * NANOS_PER_SECOND)
#: How an alert names a protective request that is not a placement.
_CHANGE_WORDS = {ModifyRequest: "modify", CancelRequest: "cancel"}
#: The books a reconciliation reads, as `inbox.Books` names them (5.2).
_BOOKS = ("positionbook", "orderbook", "tradebook", "funds")
#: The books of a held cancel's read, in order: the position book, then the order book
#: when asked for (`ReadPositions.orders`).
_HELD_CANCEL_BOOKS = ("positionbook", "orderbook")
#: The holdings, read right after the position book when the session reads them: before
#: the trade book, so a fill that lands during a read shows in the trade book as it would
#: in the position book (16-session-design 5.4, round 9).
_HOLDINGS = "holdings"


class Sender:
    """Executes the requests the worker takes (module docstring), through `path` and, for
    the book reads, `client` (the orders client); `journal` is handle W (its unbound
    intents tell a hold), `alerts` the session's; `guard`, `queue`, `settle`, `stops` and
    `news` are the worker's; `post` puts a message on the loop's inbox, `calling` marks the
    worker busy for a call that may reach the server, `now` gives UTC nanoseconds, and
    `locked` is the record of the strategies whose entries the loop has locked;
    `read_holdings` is True for a session with a CNC strategy; `gtts` sends the GTT
    requests and holds the GTT follower."""

    def __init__(
        self,
        client: Any,
        path: OrderPath,
        *,
        journal: Journal,
        alerts: AlertsLike,
        guard: ModeGuard,
        queue: WorkerQueue,
        settle: Settlements,
        stops: StopBook,
        news: NewsHandover,
        post: Callable[[Message], None],
        calling: Callable[[], AbstractContextManager[None]],
        now: Callable[[], int],
        locked: LockedEntries,
        read_holdings: bool,
        gtts: GttSender,
    ) -> None:
        self._client = client
        self._read_holdings = read_holdings
        self._gtts = gtts
        self._path = path
        self._journal = journal
        self._alerts = alerts
        self._guard = guard
        self._queue = queue
        self._settle = settle
        self._stops = stops
        self._news = news
        self._post = post
        self._calling = calling
        self._now = now
        self._locked = locked
        # When each protective request put back was last noticed (UTC nanoseconds).
        self._notices: dict[int, int] = {}

    def execute(self, request: Request) -> None:
        """Execute one request taken off the queue. Raises JournalError when the journal
        fails, and LiveError for an object that is not a request."""
        if isinstance(request, PlaceRequest):
            self._place(request)
        elif isinstance(request, ModifyRequest | CancelRequest):
            self._change(request)
        elif isinstance(request, ReleaseFill | ReleaseCancel):
            self._release(request)
        elif isinstance(request, ReadBooks | ReadPositions):
            self._read(request)
        elif isinstance(request, Probe):
            with self._calling():
                self._path.probe()
        elif isinstance(request, StreamGap):
            self._path.stream_gap(request.since_ns)
        elif isinstance(request, Settle):
            self.unsettle(request.change)
        elif isinstance(request, CancelOrphan):
            self._cancel_orphan(request.order_id)
        elif isinstance(request, GttRequest):
            self._gtts.execute(request)
        elif isinstance(request, ReadGtts):
            self._gtts.follower.read_now()
        elif isinstance(request, BindGttChild):
            self._gtts.follower.bind_child(request.seq, request.order_id, request.kernel_order)
        else:
            raise LiveError(f"the order worker takes requests, not {request!r}")

    def settle_due(self) -> None:
        """Settle what is due (module docstring). Stops at a guard lock, which the worker
        notices next. Raises JournalError when the journal fails."""
        now = self._now()
        for waiting in self._settle.expired(now):
            text = expired_text(waiting, now)
            logger.error(text)
            self._alerts.send(AlertKind.ERROR, text)
            self._queue.release(order_key(waiting.request))
            self._post(Settled(waiting.request, None))
        for waiting in self._settle.due(now):
            order_id = waiting.request.order_id
            try:
                with self._calling():
                    check = self._path.check_order(order_id)
                    self._news.taken(int(check.news is not None))
            except ModeChangedError:
                return
            except JournalError:
                raise
            except LiveError as error:
                logger.error("order %s cannot be read, taken as refused: %s", order_id, error)
                self._settle.ended(order_id)
                self._queue.release(order_key(waiting.request))
                self._post(Settled(waiting.request, None))
                continue
            self._judge(waiting, check)

    def _judge(self, waiting: Unsettled, check: OrderCheck) -> None:
        settled = self._settle.judge(waiting, check, self._now())
        cancel = isinstance(waiting.request, CancelRequest)
        if settled is None and cancel and waiting.reads == CANCEL_READS_ALERT:
            text = unsettled_cancel_text(waiting)
            logger.error(text)
            self._alerts.send(AlertKind.ERROR, text)
        if check.news is not None:
            self._news.hand_over(check.news, self._stops)
        if settled is not None:
            self._queue.release(order_key(waiting.request))
            self._post(settled)

    def unsettle(self, request: Change) -> None:
        """Hold `request`'s order until a read settles its modify or cancel."""
        self._settle.add(request, self._now())
        self._queue.hold(order_key(request))

    def _place(self, request: PlaceRequest) -> None:
        now = self._now()
        if is_stale(request, now):
            age = (now - request.intent.created_ns) / NANOS_PER_SECOND
            self._post(NotSent(request, NotSentReason.STALE, f"decided {age:.1f} s before"))
            return
        if request.intent.purpose in ENTRY_PURPOSES:
            # Read last, just before the send: the loop may have locked the strategy's
            # entries since the entry was queued (`_locked_entries`).
            locks = self._locked.reasons(request.intent.strategy)
            if locks:
                self._post(NotSent(request, NotSentReason.ENTRIES_LOCKED, locked_words(locks)))
                return
        try:
            with self._calling():
                outcome = self._path.send(request.intent)
        except ModeChangedError as error:
            self._post(NotSent(request, NotSentReason.LOCKED, str(error)))
            return
        except JournalError:
            raise
        except LiveError as error:
            self._not_sent(request, error)
            return
        self._notices.pop(request.request_id, None)
        # Posted first, so the loop knows the order even if the journal fails next.
        self._post(SendResult(request.request_id, outcome, intent_id=request.intent.intent_id))
        if outcome.kind is Outcome.ACCEPTED and request.intent.purpose is Purpose.STOP:
            self._stops.bound(request.intent, outcome.order_id)

    def _change(self, request: Change) -> None:
        last = self._path.last_known(request.order_id)
        if last is not None and last.status.terminal:
            words = f"order {request.order_id} is {last.status.value}"
            self._post(NotSent(request, NotSentReason.ORDER_ENDED, words))
            return
        try:
            with self._calling():
                if isinstance(request, ModifyRequest):
                    outcome = self._path.modify(request.order_id, **request.fields())
                else:
                    outcome = self._path.cancel(request.order_id)
        except ModeChangedError as error:
            self._post(NotSent(request, NotSentReason.LOCKED, str(error)))
            return
        except JournalError:
            raise
        except LiveError as error:
            self._not_sent(request, error)
            return
        self._notices.pop(request.request_id, None)
        self._post(SendResult(request.request_id, outcome, order_id=request.order_id))
        if outcome.kind in (Outcome.UNANSWERED, Outcome.REJECTED):
            self.unsettle(request)

    def _not_sent(self, request: PlaceRequest | Change, error: LiveError) -> None:
        """A request the order path would not send: a protective one it holds back for
        now is put back and noticed at a pace; any other is refused, a protective one
        with an ERROR alert (a cancel is then settled by a read)."""
        now = self._now()
        protective = request.priority is Priority.PROTECT
        if not (protective and self._held(request.intent)):
            if protective:
                self._refused_alert(request, error)
            self._notices.pop(request.request_id, None)
            self._post(NotSent(request, NotSentReason.REFUSED, str(error)))
            if isinstance(request, CancelRequest):
                self.unsettle(request)
            return
        self._queue.put_back(request, now + _REQUEUE_NS)
        noticed = self._notices.get(request.request_id)
        if noticed is None or now - noticed >= _NOTICE_NS:
            self._notices[request.request_id] = now
            silent = self._path.silent_since_ns is not None
            reason = NotSentReason.SILENT if silent else NotSentReason.HELD
            self._post(NotSent(request, reason, str(error)))

    def _held(self, intent: OrderIntent) -> bool:
        """Whether the order path holds back the sends of `intent`'s instrument for now:
        OpenAlgo has not answered since a timeout, or an intent of the same symbol,
        exchange and product is unbound (`OrderPath.send`)."""
        if self._path.silent_since_ns is not None:
            return True
        return any(same_leg(unbound, intent) for unbound in self._journal.unbound())

    def _refused_alert(self, request: PlaceRequest | Change, error: LiveError) -> None:
        intent = request.intent
        what = _CHANGE_WORDS.get(type(request), "placement")
        text = (
            f"a protective {what} of {intent.strategy} on "
            f"{intent.exchange}:{intent.symbol} ({intent.intent_id}) cannot be sent: {error}. "
            "The engine is told it was refused; check the leg in OpenAlgo"
        )
        logger.error(text)
        self._alerts.send(AlertKind.ERROR, text)

    def _release(self, request: ReleaseFill | ReleaseCancel) -> None:
        # In hand from before the word leaves the order path's hold until it is posted.
        self._news.taken(1)
        try:
            if isinstance(request, ReleaseFill):
                news = self._path.release_fill(request.order_id, request.average_price)
            else:
                news = self._path.release_cancel(
                    request.order_id, request.filled_quantity, request.average_price
                )
        except JournalError:
            raise
        except LiveError as error:
            # The word was handed over by other means meanwhile (a poll, the tradebook).
            logger.warning("held word of order %s not released: %s", request.order_id, error)
            news = None
        self._news.hand_over(news, self._stops)

    def _cancel_orphan(self, order_id: str) -> None:
        """Cancel an order of the journal that the engine does not hold; one the order path
        cannot cancel (it does not follow it, OpenAlgo refuses) is an ERROR alert to cancel
        it by hand. Its terminal news comes as any order's."""
        try:
            with self._calling():
                outcome = self._path.cancel(order_id)
        except ModeChangedError:
            return
        except JournalError:
            raise
        except LiveError as error:
            reason = str(error)
        else:
            if outcome.kind is Outcome.ACCEPTED:
                return
            reason = outcome.reason
        text = f"order {order_id}, which the engine does not hold, could not be cancelled "
        text += f"({reason}): cancel it by hand in OpenAlgo"
        logger.error(text)
        self._alerts.send(AlertKind.ERROR, text)

    def _read(self, request: ReadBooks | ReadPositions) -> None:
        """The books of a reconciliation, or the position book for a held cancel (and the
        order book after it when asked for), with the holdings after the position book in
        a session that reads them, each answer checked by the guard (the analyzer switch
        routes them too)."""
        started = self._now()
        names = _BOOKS
        if isinstance(request, ReadPositions):
            names = _HELD_CANCEL_BOOKS if request.orders else _HELD_CANCEL_BOOKS[:1]
        if self._read_holdings:
            names = (names[0], _HOLDINGS, *names[1:])
        answers = {}
        for name in names:
            if self._guard.locked:
                break
            with self._calling():
                answers[name] = server_answer(getattr(self._client, name)())
            self._guard.check_response(answers[name])
        if self._guard.locked:
            why = "the mode guard is locked: the books answer from the other mode"
            self._post(NotSent(request, NotSentReason.LOCKED, why))
        elif isinstance(request, ReadBooks):
            ended = self._now()
            self._post(Books(request.generation, **answers, started_ns=started, ended_ns=ended))
        else:
            positions, orders = answers["positionbook"], answers.get("orderbook")
            self._post(PositionRows(positions, started, orders, answers.get(_HOLDINGS)))


__all__ = ["HELD_NOTICE_EVERY_S", "REQUEUE_AFTER_S", "Sender"]
