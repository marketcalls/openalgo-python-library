"""What the order worker does with one request it takes (16-session-design 2.4, "Executing
a request"), split from `worker` so each module keeps one job.

- A Place is `OrderPath.send`, posted as `SendResult`; a bound STOP then records its
  leg's journal stop (`_worker_stops`). An entry or add decided more than
  `ENTRY_STALE_AFTER_S` before is `NotSent(stale)` instead.
- A Modify or Cancel of an order the order path knows terminal is `NotSent(order_ended)`;
  else `modify` or `cancel`, and an unanswered or refused answer leaves the order
  unsettled, its further requests held until a read settles it (`_worker_settle`).
- `ModeChangedError` (caught before `LiveError`, its base) is `NotSent(locked)`. Any other
  `LiveError` (the unbound hold, the silent server, a mode that cannot be read before an
  entry) puts a protective request (priority 0) back for `REQUEUE_AFTER_S`, noticed as
  `NotSent(silent)` or `NotSent(held)` at once and then at most every
  `HELD_NOTICE_EVERY_S`, and refuses any other (`NotSent(refused)`: an entry's moment has
  passed; a cancel is then settled by a read, 4.5). A `JournalError` is raised: nothing
  may be sent once an intent cannot be written first.
- A release of a held fill or cancel hands its news over; the book reads go to the guard
  answer by answer and are posted as `Books` or `PositionRows`, or `NotSent(locked)` once
  the guard locks; a probe asks OpenAlgo whether it answers; a stream gap asks for polls.

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
from ._order_answers import Outcome
from ._types import LOGGER_NAME, OrderUpdate, Purpose
from ._venue_requests import CancelRequest, ModifyRequest, PlaceRequest, Priority
from ._worker_queue import (
    Probe,
    ReadBooks,
    ReadPositions,
    ReleaseCancel,
    ReleaseFill,
    Request,
    StreamGap,
    WorkerQueue,
    is_stale,
    order_key,
)
from ._worker_settle import Change, Settlements
from ._worker_stops import StopBook
from .guard import ModeGuard
from .inbox import Books, Message, NotSent, NotSentReason, PositionRows, SendResult
from .orders import OrderPath

logger = logging.getLogger(LOGGER_NAME)

#: Seconds a protective request held back waits before it is tried again (2.4).
REQUEUE_AFTER_S = 0.5
#: Seconds between two notices of one protective request held back, so the loop can alert
#: without being flooded (4.8).
HELD_NOTICE_EVERY_S = 5.0
_REQUEUE_NS = round(REQUEUE_AFTER_S * NANOS_PER_SECOND)
_NOTICE_NS = round(HELD_NOTICE_EVERY_S * NANOS_PER_SECOND)
#: The books a reconciliation reads, as `inbox.Books` names them (5.2).
_BOOKS = ("positionbook", "orderbook", "tradebook", "funds")


class Sender:
    """Executes the requests the worker takes (module docstring), through `path` and, for
    the book reads, `client` (the orders client); `guard`, `queue`, `settle` and `stops`
    are the worker's; `deliver` posts an order's news, `post` any other message, `calling`
    marks the worker busy for a call that may reach the server, `now` gives UTC
    nanoseconds."""

    def __init__(
        self,
        client: Any,
        path: OrderPath,
        *,
        guard: ModeGuard,
        queue: WorkerQueue,
        settle: Settlements,
        stops: StopBook,
        deliver: Callable[[OrderUpdate], None],
        post: Callable[[Message], None],
        calling: Callable[[], AbstractContextManager[None]],
        now: Callable[[], int],
    ) -> None:
        self._client = client
        self._path = path
        self._guard = guard
        self._queue = queue
        self._settle = settle
        self._stops = stops
        self._deliver = deliver
        self._post = post
        self._calling = calling
        self._now = now
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
        else:
            raise LiveError(f"the order worker takes requests, not {request!r}")

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
        """A request the order path would not send: a protective one is put back and
        noticed at a pace; any other is refused (a cancel is then settled by a read)."""
        now = self._now()
        if request.priority is not Priority.PROTECT:
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

    def _release(self, request: ReleaseFill | ReleaseCancel) -> None:
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
            return
        if news is not None:
            self._deliver(news)

    def _read(self, request: ReadBooks | ReadPositions) -> None:
        """The books of a reconciliation, or the position book for a held cancel, each
        answer checked by the guard (the analyzer switch routes them too)."""
        started = self._now()
        names = _BOOKS if isinstance(request, ReadBooks) else ("positionbook",)
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
            self._post(PositionRows(answers["positionbook"], started))


__all__ = ["HELD_NOTICE_EVERY_S", "REQUEUE_AFTER_S", "Sender"]
