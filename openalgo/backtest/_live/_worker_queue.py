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
posted (16-session-design 3.3 step 5).
"""

from __future__ import annotations

import collections
import itertools
import queue
import threading
from dataclasses import dataclass

from .._layout import NANOS_PER_SECOND
from ._order_state import ENTRY_PURPOSES
from ._venue_requests import CancelRequest, ModifyRequest, OrderRequest, PlaceRequest, Priority

#: An entry or add decided this long before the worker takes it is dropped and reported
#: rejected: decided at a bar close and sent much later, it is a different trade
#: (16-session-design 10.0, decision 7).
ENTRY_STALE_AFTER_S = 10.0
_STALE_NS = round(ENTRY_STALE_AFTER_S * NANOS_PER_SECOND)


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
)
#: Everything the worker takes.
Request = OrderRequest | WorkerRequest
#: A kernel order's FIFO key: (strategy, kernel order), since every engine numbers its
#: orders from o1.
OrderKey = tuple[str, int]


def order_key(request: OrderRequest) -> OrderKey:
    """The FIFO key of a request of a kernel order."""
    return request.intent.strategy, request.kernel_order


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
        waiting = _Waiting(request, arrived, not_before_ns)
        fifo = self._fifos.setdefault(self._key(request, arrived), collections.deque())
        fifo.appendleft(waiting)

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

    def take(self, now_ns: int, *, most: Priority = Priority.UPKEEP) -> Request | None:
        """The head of the most urgent ready FIFO whose rank is `most` or more urgent,
        taken off the queue; None when none is ready. A FIFO is ready when its order is
        not held and its head's `not_before_ns` has passed at `now_ns`."""
        best: tuple[tuple[int, int], object] | None = None
        for key, fifo in self._fifos.items():
            head = fifo[0]
            if key in self._held or head.not_before_ns > now_ns:
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
    it has not finished (module docstring). Any thread puts; the worker's thread takes
    them and says when one is finished."""

    def __init__(self) -> None:
        self._requests: queue.SimpleQueue[Request] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._unfinished = 0

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

    def take_all(self) -> list[Request]:
        """Every request handed over since the last call, oldest first."""
        taken = []
        while True:
            try:
                taken.append(self._requests.get_nowait())
            except queue.Empty:
                return taken

    def finished(self) -> None:
        """The worker executed a request and did not put it back."""
        with self._lock:
            self._unfinished -= 1


__all__ = [
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
    "WorkerQueue",
    "WorkerRequest",
    "is_stale",
    "order_key",
]
