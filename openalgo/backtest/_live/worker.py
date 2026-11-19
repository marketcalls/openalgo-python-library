"""The order worker of a live session (16-session-design 2.1, 2.4; unit 5): the one thread
that sends the session's orders and reads its order books, so the loop never makes a
network call. It owns the order path (`orders.OrderPath`, with its rate budget, server
watch, follower and resolver), the mode guard once started, and journal handle W. One
worker, serial: the order path is not thread-safe, its holds and send-once rules assume
one sender per account, and the rate budget is the account's.

Its startup phase (`start_up`, on the worker's own thread, since a journal handle is
bound to the thread that opens it) opens W, starts the journal's session for the day,
builds and restores the order path, and posts `inbox.WorkerReady`. Then each `step`:

1. drains the raw order updates (`updates`): each normalized (an unreadable one is logged
   and counted, never raised), taken by the order path, and its news posted as
   `inbox.OrderNews` (`_worker_news`); a STOP intent's terminal news clears its leg's
   journal stop first (`_worker_stops`), and any terminal news settles the order
   (`_worker_settle`);
2. executes every ready request of priority 0 (stops, exits, protective modifies and
   cancels), draining the updates before each (`_worker_send`);
3. polls the mode guard (it reads only every 5 s), after priority 0, so a protective send
   never waits behind a guard read; a lock is posted once as `inbox.GuardLocked`;
4. runs the order path's polling round, posting its news;
5. resolves unbound intents, posting `inbox.Resolutions` (a STOP bound records its stop);
6. posts the held fills (`inbox.HeldFills`) when they changed;
7. settles the orders whose modify or cancel was unanswered or refused, posting the
   check's news and `inbox.Settled` (an expired modify with an ERROR alert; one the order
   path cannot read at all, refused at once: `_worker_send`);
8. executes one ready request of any priority, after draining the updates.

A `JournalError` (any but a refused stop record, which is only alerted) is posted as
`inbox.WorkerFailed`, after which nothing is sent: an intent can no longer be written
first. So is any exception nothing expected, inline (`step`) as on the thread (`run`),
so the fault suite meets what production does. `close` lets the step under way end
without taking another request; `withdraw_put_back` drops what was put back before the
session's last lines (3.3 step 5). The loop reads `busy_since_ns` (set around every call
that may reach the server) and `raw_pending` (the order words not yet posted, in the raw
FIFO or the worker's hands: `_worker_news`) (4.2), the requests not executed yet
(`unfinished`, the close's last wait), and the order path's silence (`silent_since_ns`,
`asked_ns`, 4.9). Times are UTC nanoseconds.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date

from ..errors import JournalError, LiveError
from ._journal_file import DEFAULT_BUSY_TIMEOUT_S
from ._journal_reads import JournalSession
from ._order_state import HeldFill
from ._types import LOGGER_NAME, AlertsLike, Mode, Now, Purpose, Sleep
from ._venue_requests import Priority
from ._worker_news import NewsHandover
from ._worker_queue import Incoming, Request, WorkerQueue
from ._worker_send import Sender
from ._worker_settle import Settlements
from ._worker_stops import StopBook
from .guard import ModeGuard
from .inbox import (
    GuardLocked,
    HeldFills,
    Inbox,
    Message,
    RawUpdates,
    Resolutions,
    WorkerFailed,
    WorkerReady,
)
from .journal import Journal
from .orders import OrderPath, RateBudget

logger = logging.getLogger(LOGGER_NAME)

#: The worker's thread (16-session-design 2.1).
THREAD_NAME = "openalgo-live-orders"
#: Seconds the thread waits for a request or an update when nothing is ready (2.4 step 9).
WORKER_IDLE_WAIT_S = 0.05
#: Seconds `close` waits for the thread by default; a call in flight can take the orders
#: client's timeout, and the thread is a daemon, so the process can end meanwhile.
CLOSE_WAIT_S = 10.0


@dataclass(frozen=True)
class JournalTarget:
    """Where the worker opens journal handle W (`journal.Journal.open`): the file, the
    account, the broker as `ping` names it, the mode the guard adopted, the IST trading
    day of the session, and how long a statement waits for the other handle's lock."""

    path: str | os.PathLike[str]
    account: str
    broker: str
    mode: Mode
    day: date
    busy_timeout_s: float = DEFAULT_BUSY_TIMEOUT_S


@dataclass(frozen=True)
class _Started:
    """What the startup phase built, on the thread that owns it."""

    journal: Journal
    path: OrderPath
    stops: StopBook
    sender: Sender


class OrderWorker:
    """The order worker (module docstring). `client` is the session's orders client,
    `guard` its mode guard (started on the main thread before the worker), `alerts` the
    shared alerts, `inbox` the loop's, `journal` where W is; `now` gives UTC nanoseconds
    and `sleep` waits seconds (the rate budget's waits).

    The loop calls `put` and `withdraw_put_back` and reads `busy_since_ns` and
    `raw_pending` from its thread, and the feed's order callback puts on `updates`;
    `start_up`, `step`, `start` and `close` are called from one thread at a time.
    """

    def __init__(
        self,
        client: object,
        *,
        guard: ModeGuard,
        alerts: AlertsLike,
        inbox: Inbox,
        journal: JournalTarget,
        now: Now,
        sleep: Sleep,
    ) -> None:
        self._client = client
        self._guard = guard
        self._alerts = alerts
        self._inbox = inbox
        self._target = journal
        self._now = now
        self._sleep = sleep
        self._wake = threading.Event()
        #: The raw order updates, put by the feed's order callback (2.3).
        self.updates = RawUpdates(wake=self._wake)
        #: Since when (UTC nanoseconds) the worker is in a call that may reach the server.
        self.busy_since_ns: int | None = None
        self._incoming = Incoming()
        self._queue = WorkerQueue()
        self._settle = Settlements()
        self._news = NewsHandover(
            self.updates, settle=self._settle, queue=self._queue, post=self._post
        )
        self._started: _Started | None = None
        self._held: tuple[HeldFill, ...] = ()
        self._lock_posted = False
        self._failed = False
        self._stopping = False
        self._thread: threading.Thread | None = None

    @property
    def raw_pending(self) -> int:
        """The order words not yet posted to the loop (4.2): the raw updates waiting to be
        drained, and the words the worker took in and has not posted (`_worker_news`)."""
        return self._news.pending

    @property
    def unfinished(self) -> int:
        """The requests handed over and not executed yet, one put back among them until it
        is withdrawn; an answer is posted before its request stops counting (3.3 step 5)."""
        return self._incoming.unfinished

    @property
    def silent_since_ns(self) -> int | None:
        """Since when (UTC nanoseconds) OpenAlgo has not answered the order path after a
        client timeout or a dropped connection (`OrderPath.silent_since_ns`); None while
        it answers, and before the start-up. Read from the loop's thread (4.9)."""
        started = self._started
        return None if started is None else started.path.silent_since_ns

    @property
    def asked_ns(self) -> int | None:
        """When (UTC nanoseconds) the order path's last answer from OpenAlgo was noted,
        whatever it was: when it came, or when its poll or book read began
        (`OrderPath.asked_ns`); None before its first call. Read from the loop (4.9)."""
        started = self._started
        return None if started is None else started.path.asked_ns

    @property
    def unreadable_updates(self) -> int:
        """Raw order updates that could not be read, logged and dropped."""
        return self._news.unreadable

    def put(self, request: Request) -> None:
        """Hand the worker a request (any thread); it never waits."""
        self._incoming.put(request)
        self._wake.set()

    def withdraw_put_back(self) -> None:
        """Withdraw at the next step every request put back so far (OpenAlgo silent, an
        intent of the instrument unbound): the session's last lines named them as not sent,
        so none is sent after them or counted `unfinished`. Any thread; it never waits."""
        self._incoming.ask_withdrawal()
        self._wake.set()

    def start_up(self) -> JournalSession:
        """The startup phase, on the calling thread, which then owns journal W: open it,
        start the journal's session, build and restore the order path, post `WorkerReady`,
        and return the session. Raises JournalError when the journal cannot be opened or
        started, LiveError when the guard has not started or the worker started up already."""
        if self._started is not None:
            raise LiveError("the order worker starts up once")
        target = self._target
        journal = Journal.open(
            target.path,
            account=target.account,
            broker=target.broker,
            mode=target.mode,
            now=self._now,
            busy_timeout_s=target.busy_timeout_s,
        )
        try:
            session = journal.start_session(target.day, target.mode, target.broker)
            path = OrderPath(
                self._client,
                journal=journal,
                guard=self._guard,
                alerts=self._alerts,
                now=self._now,
                sleep=self._sleep,
                rate=RateBudget(self._now),
            )
            path.restore()
        except BaseException:
            journal.close()
            raise
        stops = StopBook(journal, path, alerts=self._alerts, calling=self._calling, news=self._news)
        sender = Sender(
            self._client,
            path,
            journal=journal,
            alerts=self._alerts,
            guard=self._guard,
            queue=self._queue,
            settle=self._settle,
            stops=stops,
            news=self._news,
            post=self._post,
            calling=self._calling,
            now=self._now,
        )
        self._started = _Started(journal, path, stops, sender)
        self._post(WorkerReady(session))
        return session

    def step(self) -> bool:
        """One iteration (module docstring); True when it took a request, so the thread
        steps again at once. A failure is posted as `WorkerFailed`, never raised; nothing
        is done after one or after `close`. Raises LiveError before `start_up`."""
        started = self._started
        if started is None:
            raise LiveError("the order worker steps after start_up()")
        if self._failed or self._stopping:
            # Closed (journal W with it): a step now would only meet the closed journal.
            return False
        try:
            return self._step(started)
        except JournalError as error:
            logger.error("the order worker stops: the journal failed: %s", error)
            self._fail(error)
        # The loop must hear of any failure, or orders would stop unseen: it runs the
        # emergency stop on it (3.5).
        except Exception as error:
            logger.exception("the order worker failed")
            self._fail(error)
        return False

    def _fail(self, error: BaseException) -> None:
        self._failed = True
        self._post(WorkerFailed(error))

    def _step(self, started: _Started) -> bool:
        path = started.path
        self._news.drain(path, started.stops)
        took = False
        while self._run_next(started, Priority.PROTECT):
            took = True
        with self._calling():
            self._guard.poll()
        self._notice_lock()
        with self._calling():
            polled = path.due_polls()
            self._news.taken(len(polled))
        for update in polled:
            self._news.hand_over(update, started.stops)
        self._resolve(started)
        held = tuple(path.held_fills())
        if held != self._held:
            self._held = held
            self._post(HeldFills(held))
        started.sender.settle_due()
        self._notice_lock()
        return self._run_next(started, Priority.UPKEEP) or took

    def _run_next(self, started: _Started, most: Priority) -> bool:
        if self._stopping:
            # Closed: the step under way ends without sending what is still queued.
            return False
        for request in self._incoming.into(self._queue):
            logger.warning("withdrawn, put back before the session's last lines: %r", request)
        request = self._queue.take(self._now(), most=most)
        if request is None:
            return False
        # The updates that came meanwhile first: a modify or cancel is never sent on
        # stale knowledge.
        self._news.drain(started.path, started.stops)
        started.sender.execute(request)
        if not self._queue.waits(request):
            self._incoming.finished()
        self._notice_lock()
        return True

    def _resolve(self, started: _Started) -> None:
        with self._calling():
            resolutions = started.path.resolve_unbound()
            self._news.taken(len(resolutions))
        if not resolutions:
            return
        # Posted first, so the loop knows the orders even if the journal fails next.
        self._post(Resolutions(tuple(resolutions)))
        self._news.handed(len(resolutions))
        for resolution in resolutions:
            update = resolution.update
            ended = update is not None and update.status.terminal
            if resolution.order_id and resolution.intent.purpose is Purpose.STOP and not ended:
                started.stops.bound(resolution.intent, resolution.order_id)

    def _notice_lock(self) -> None:
        breach = self._guard.breach
        if breach is not None and not self._lock_posted:
            self._lock_posted = True
            self._post(GuardLocked(breach))

    @contextlib.contextmanager
    def _calling(self) -> Iterator[None]:
        """Mark the worker busy for a call that may reach the server (4.2)."""
        self.busy_since_ns = self._now()
        try:
            yield
        finally:
            self.busy_since_ns = None

    def _post(self, message: Message) -> None:
        self._inbox.put(message)

    def start(self) -> None:
        """Start the worker's thread (`THREAD_NAME`), whose startup phase opens journal W.
        Raises LiveError when it runs already."""
        if self._thread is not None:
            raise LiveError("the order worker's thread is started once")
        self._thread = threading.Thread(target=self.run, name=THREAD_NAME, daemon=True)
        self._thread.start()

    def run(self) -> None:
        """The worker's thread: the startup phase, then steps until `close`, waiting up to
        `WORKER_IDLE_WAIT_S` when nothing was taken; journal W is closed at the end. A
        failure of the startup, or one nothing expected, is posted as `WorkerFailed`."""
        try:
            self.start_up()
            while not self._stopping and not self._failed:
                self._wake.clear()
                if not self.step():
                    self._wake.wait(WORKER_IDLE_WAIT_S)
        # The startup's failure: `step` posts its own.
        except Exception as error:
            logger.exception("the order worker failed to start")
            self._fail(error)
        finally:
            self._finish()

    def close(self, wait_s: float = CLOSE_WAIT_S) -> None:
        """Stop the worker after its current step, which takes no further request, and
        wait for its thread at most `wait_s` seconds; without a thread, close journal W on
        the calling thread (the one that ran `start_up`). Safe to call twice."""
        self._stopping = True
        self._wake.set()
        if self._thread is None:
            self._finish()
            return
        self._thread.join(wait_s)
        if self._thread.is_alive():
            logger.warning("the order worker's thread did not end within %s s", wait_s)

    def _finish(self) -> None:
        """Close journal W, saying what was left unsent."""
        left = self._incoming.unfinished
        if left and self._stopping:
            logger.warning("the order worker closed with %d requests not sent", left)
        if self._started is not None:
            self._started.journal.close()


__all__ = ["CLOSE_WAIT_S", "THREAD_NAME", "WORKER_IDLE_WAIT_S", "JournalTarget", "OrderWorker"]
