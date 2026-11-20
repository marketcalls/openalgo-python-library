"""The order worker's requests and its queue (16-session-design 2.4): one FIFO per kernel
order, ranked at its most urgent priority, so no request of an order overtakes another of
the same order and a protective request never waits behind an entry.

Every request of a kernel order (a `PlaceRequest`, `ModifyRequest` or `CancelRequest`,
`_venue_requests`) joins that order's FIFO, in the order the kernel emitted it, and only
the head of a FIFO is taken. A Place therefore goes before its order's Modify, a Modify
before a later Cancel, and each kernel Modify is a full replacement carrying the terms of
its moment, so none is ever reordered. A FIFO is ranked at the most urgent priority
among its requests, then by the arrival of its head. An order is held while a modify or
cancel of it is unsettled (`hold`), and a request put back (`put_back`) waits until its
`not_before_ns`, at the head of its FIFO.

The worker's own requests (`ReleaseFill`, `ReleaseCancel`, `ReadBooks`, `ReadPositions`,
`Probe`, `StreamGap`, and a restart's `Settle` and the reconciler's `CancelOrphan`) belong
to no order: each is its own FIFO. An entry or add whose
decision is older than `ENTRY_STALE_AFTER_S` when it is taken is not sent (`is_stale`,
open decision 7). Times are UTC nanoseconds. Used by the worker's thread only, but for
`Incoming`: the requests any thread hands the worker, counted until the worker has
executed them (`unfinished`), so the close knows when every answer it waits for has been
posted (16-session-design 3.3 step 5). The close also asks for the requests put back to
be withdrawn (`ask_withdrawal`): its last lines named them as not sent, so they are never
sent after them and the close does not wait for them. The order rate throttle
(`_order_throttle`) takes no request that makes an order call (`is_order_call`) while its
window is full, and withdraws an entry held back too long (`remove`). An entry or add the
kernel cancelled while it waits is withdrawn as it is taken in (`WithdrawPlace`), so a
lock of entries (a daily profit target, the kill switch, the safe stop) is never followed
by a placement the order rate held (18-multi-strategy 2.5a rules 5, 6 and 12).
"""

from __future__ import annotations

import collections
import itertools
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass

from .._layout import NANOS_PER_SECOND
from ._order_state import ENTRY_PURPOSES
from ._venue_requests import CancelRequest, ModifyRequest, OrderRequest, PlaceRequest, Priority
from .inbox import Message, NotSent, NotSentReason

#: An entry or add decided this long before the worker takes it is dropped and reported
#: rejected: decided at a bar close and sent much later, it is a different trade
#: (16-session-design 10.0, decision 7).
ENTRY_STALE_AFTER_S = 10.0
_STALE_NS = round(ENTRY_STALE_AFTER_S * NANOS_PER_SECOND)
#: The words of the `NotSent(cancelled)` of an entry or add withdrawn (`WithdrawPlace`).
CANCELLED_WORDS = "the engine cancelled it before it was sent"


@dataclass(frozen=True)
class ReleaseFill:
    """Hand over the fill the order path holds for broker order `order_id` at
    `average_price` rupees (`OrderPath.release_fill`, 4.11)."""

    order_id: str
    average_price: float

    priority = Priority.UPKEEP


@dataclass(frozen=True)
class ReleaseCancel:
    """Hand over the cancel the order path holds for broker order `order_id` as having
    filled `filled_quantity` units in all at `average_price` rupees
    (`OrderPath.release_cancel`, 4.11)."""

    order_id: str
    filled_quantity: int
    average_price: float

    priority = Priority.UPKEEP


@dataclass(frozen=True)
class ReadBooks:
    """Read the position, order and trade books and the funds for reconciliation number
    `generation` (5.2), answered as `inbox.Books`."""

    generation: int

    priority = Priority.UPKEEP


@dataclass(frozen=True)
class ReadPositions:
    """Read the position book for a held cancel (4.11), answered as `inbox.PositionRows`."""

    priority = Priority.UPKEEP


@dataclass(frozen=True)
class Probe:
    """Ask OpenAlgo whether it answers again while it is silent (`OrderPath.probe`, 4.9)."""

    priority = Priority.UPKEEP


@dataclass(frozen=True)
class StreamGap:
    """The order stream may have missed words since `since_ns` (a feed outage's start):
    poll every followed order once (`OrderPath.stream_gap`). It makes no call, and it
    brings the polls of possibly missed fills forward, so it goes first."""

    since_ns: int

    priority = Priority.PROTECT


@dataclass(frozen=True)
class Settle:
    """Settle `change`, a modify or cancel of a restored order whose answer the restart
    cannot know (16-session-design 3.6: the journal does not record modify or cancel
    calls), by reading the order as for an unanswered one (4.3, 4.5)."""

    change: ModifyRequest | CancelRequest

    priority = Priority.PROTECT


@dataclass(frozen=True)
class CancelOrphan:
    """Cancel broker order `order_id`, an order of the session's journal working at the
    broker that the engine does not hold (5.3 row 6, 3.6): cancelling never opens."""

    order_id: str

    priority = Priority.PROTECT


@dataclass(frozen=True)
class WithdrawPlace:
    """Withdraw `place`, the placement of an entry or add the kernel cancelled before it
    was bound (`venue_map`). Taken in by `Incoming.into`, never queued: while `place` still
    waits in the queue it is taken off and answered `NotSent(cancelled)`; once the worker
    has taken it, nothing (the kernel's cancel waits for the bind, 2.5 E)."""

    place: PlaceRequest

    priority = Priority.PROTECT


#: The worker's own requests.
WorkerRequest = (
    ReleaseFill
    | ReleaseCancel
    | ReadBooks
    | ReadPositions
    | Probe
    | StreamGap
    | Settle
    | CancelOrphan
    | WithdrawPlace
)
#: Everything the worker takes.
Request = OrderRequest | WorkerRequest
#: A kernel order's FIFO key: (strategy, kernel order), since every engine numbers its
#: orders from o1.
OrderKey = tuple[str, int]


def order_key(request: OrderRequest) -> OrderKey:
    """The FIFO key of a request of a kernel order."""
    return request.intent.strategy, request.kernel_order


def is_order_call(request: Request) -> bool:
    """True for a request that makes an order call (place, modify or cancel), the calls
    the order rate counts (18-multi-strategy 2.5a rule 12)."""
    return isinstance(request, PlaceRequest | ModifyRequest | CancelRequest | CancelOrphan)


def is_stale(request: Request, now_ns: int) -> bool:
    """Whether `request` is an entry or add decided more than `ENTRY_STALE_AFTER_S` before
    `now_ns` (its intent's creation, in the iteration of the decision)."""
    return (
        isinstance(request, PlaceRequest)
        and request.intent.purpose in ENTRY_PURPOSES
        and now_ns - request.intent.created_ns > _STALE_NS
    )


@dataclass
class _Waiting:
    request: Request
    arrived: int
    not_before_ns: int = 0
    put_back: bool = False


class WorkerQueue:
    """The worker's queue (module docstring)."""

    def __init__(self) -> None:
        self._fifos: dict[object, collections.deque[_Waiting]] = {}
        self._held: set[OrderKey] = set()
        self._arrivals = itertools.count()
        # The request last taken, so one put back keeps its place among equal ranks.
        self._taken: _Waiting | None = None

    def __len__(self) -> int:
        """The requests waiting."""
        return sum(len(fifo) for fifo in self._fifos.values())

    def put(self, request: Request) -> None:
        """Add `request` at the end of its FIFO."""
        arrived = next(self._arrivals)
        fifo = self._fifos.setdefault(self._key(request, arrived), collections.deque())
        fifo.append(_Waiting(request, arrived))

    def put_back(self, request: Request, not_before_ns: int) -> None:
        """Put the request just taken back at the head of its FIFO, in its place among the
        FIFOs of its rank, to be taken again from `not_before_ns` (UTC nanoseconds)."""
        taken = self._taken
        arrived = taken.arrived if taken and taken.request is request else next(self._arrivals)
        waiting = _Waiting(request, arrived, not_before_ns, put_back=True)
        fifo = self._fifos.setdefault(self._key(request, arrived), collections.deque())
        fifo.appendleft(waiting)

    def withdraw_put_back(self) -> list[Request]:
        """Take every request put back off the queue for good, in order of arrival: the
        session's last lines named them as not sent (16-session-design 3.3 step 5)."""
        withdrawn: list[_Waiting] = []
        for key, fifo in list(self._fifos.items()):
            withdrawn += [waiting for waiting in fifo if waiting.put_back]
            kept = collections.deque(waiting for waiting in fifo if not waiting.put_back)
            if kept:
                self._fifos[key] = kept
            else:
                del self._fifos[key]
        return [waiting.request for waiting in sorted(withdrawn, key=lambda w: w.arrived)]

    def waits(self, request: Request) -> bool:
        """True while `request` waits in the queue: put and not taken, or put back."""
        return any(waiting.request is request for fifo in self._fifos.values() for waiting in fifo)

    def hold(self, key: OrderKey) -> None:
        """Take nothing of order `key` until `release` (an unsettled modify or cancel)."""
        self._held.add(key)

    def release(self, key: OrderKey) -> None:
        """Take order `key`'s requests again."""
        self._held.discard(key)

    def held(self, key: OrderKey) -> bool:
        """True while order `key` is held."""
        return key in self._held

    def ready_orders(self, now_ns: int) -> list[Request]:
        """The heads of the ready FIFOs that make an order call (`is_order_call`), in
        the order of their arrival."""
        heads = [
            fifo[0]
            for key, fifo in self._fifos.items()
            if key not in self._held and fifo[0].not_before_ns <= now_ns
        ]
        heads.sort(key=lambda waiting: waiting.arrived)
        return [waiting.request for waiting in heads if is_order_call(waiting.request)]

    def remove(self, request: Request) -> bool:
        """Take `request`, the head of its FIFO, off the queue for good (an entry the
        order rate withdraws, or one the kernel cancelled); True when it was taken, False
        for a request not at a head."""
        for key, fifo in list(self._fifos.items()):
            if fifo[0].request is request:
                fifo.popleft()
                if not fifo:
                    del self._fifos[key]
                return True
        return False

    def take(
        self, now_ns: int, *, most: Priority = Priority.UPKEEP, orders: bool = True
    ) -> Request | None:
        """The head of the most urgent ready FIFO whose rank is `most` or more urgent,
        taken off the queue; None when none is ready. A FIFO is ready when its order is
        not held and its head's `not_before_ns` has passed at `now_ns`; with `orders`
        False, a head that makes an order call is not ready either (the order rate's
        window is full)."""
        best: tuple[tuple[int, int], object] | None = None
        for key, fifo in self._fifos.items():
            head = fifo[0]
            if key in self._held or head.not_before_ns > now_ns:
                continue
            if not orders and is_order_call(head.request):
                continue
            rank = (min(waiting.request.priority for waiting in fifo), head.arrived)
            if rank[0] <= most and (best is None or rank < best[0]):
                best = (rank, key)
        if best is None:
            return None
        fifo = self._fifos[best[1]]
        self._taken = fifo.popleft()
        if not fifo:
            del self._fifos[best[1]]
        return self._taken.request

    def _key(self, request: Request, arrived: int) -> object:
        if isinstance(request, PlaceRequest | ModifyRequest | CancelRequest):
            return order_key(request)
        return ("", arrived)


class Incoming:
    """The requests handed to the worker until it takes them into its queue, and how many
    it has not finished (module docstring). Any thread puts, and asks for the withdrawal
    of what was put back; the worker's thread takes them into its queue (`into`) and says
    when one is finished. `post` puts a message on the loop's inbox (the answer of an
    entry withdrawn, `WithdrawPlace`)."""

    def __init__(self, post: Callable[[Message], None]) -> None:
        self._requests: queue.SimpleQueue[Request] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._unfinished = 0
        self._withdraw = threading.Event()
        self._post = post

    @property
    def unfinished(self) -> int:
        """The requests handed over and not executed yet, one put back among them. Any
        thread reads it: the worker posts a request's answer before it is finished."""
        return self._unfinished

    def put(self, request: Request) -> None:
        """Hand over `request`, counted unfinished from now; it never waits."""
        with self._lock:
            self._unfinished += 1
        self._requests.put(request)

    def ask_withdrawal(self) -> None:
        """Ask for the withdrawal of every request put back so far, made at the worker's
        next `into` (any thread; it never waits)."""
        self._withdraw.set()

    def into(self, waiting: WorkerQueue) -> list[Request]:
        """Put every request handed over since the last call into `waiting`, the worker's
        queue, oldest first; a `WithdrawPlace` is carried out instead (`_cancelled`). When
        a withdrawal was asked for, first take every request put back off `waiting`
        (`WorkerQueue.withdraw_put_back`), no longer unfinished; those are returned."""
        withdrawn: list[Request] = []
        if self._withdraw.is_set():
            self._withdraw.clear()
            withdrawn = waiting.withdraw_put_back()
            with self._lock:
                self._unfinished -= len(withdrawn)
        while True:
            try:
                request = self._requests.get_nowait()
            except queue.Empty:
                return withdrawn
            if isinstance(request, WithdrawPlace):
                self._cancelled(waiting, request.place)
            else:
                waiting.put(request)

    def _cancelled(self, waiting: WorkerQueue, place: PlaceRequest) -> None:
        """Withdraw `place` while it waits in `waiting`, answered `NotSent(cancelled)`
        before it stops counting (one the worker took already is left to its bind); the
        withdrawal itself is done with either way."""
        if waiting.remove(place):
            self._post(NotSent(place, NotSentReason.CANCELLED, CANCELLED_WORDS))
            self.finished()
        self.finished()

    def finished(self) -> None:
        """The worker executed a request and did not put it back."""
        with self._lock:
            self._unfinished -= 1


__all__ = [
    "CANCELLED_WORDS",
    "ENTRY_STALE_AFTER_S",
    "CancelOrphan",
    "Incoming",
    "OrderKey",
    "Probe",
    "ReadBooks",
    "ReadPositions",
    "ReleaseCancel",
    "ReleaseFill",
    "Request",
    "Settle",
    "StreamGap",
    "WithdrawPlace",
    "WorkerQueue",
    "WorkerRequest",
    "is_order_call",
    "is_stale",
    "order_key",
]
