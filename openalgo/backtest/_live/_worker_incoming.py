"""The requests handed to the order worker until it takes them into its queue
(`Incoming`; 16-session-design 2.4), split from `_worker_queue` so each module keeps one
job.

Any thread hands the worker a request (`put`); the worker's thread takes every one handed
over since its last look into its queue (`into`), oldest first. Each request is counted
until the worker has executed it (`unfinished`), so the close knows when every answer it
waits for has been posted (16-session-design 3.3 step 5). The close also asks for the
requests put back to be withdrawn (`ask_withdrawal`): its last lines named them as not
sent, so they are never sent after them and the close does not wait for them. A
`WithdrawPlace` is never queued: it takes its placement off the queue while it still waits
there, answered `NotSent` with the withdrawal's reason (`_venue_withdraw`; 18-multi-strategy
2.5a rules 5, 6 and 12; 2.4.0 item M8-X2).
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable

from ._worker_queue import Request, WithdrawPlace, WorkerQueue
from .inbox import Message, NotSent


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
        queue, oldest first; a `WithdrawPlace` is carried out instead (`_withdrawn`). When
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
                self._withdrawn(waiting, request)
            else:
                waiting.put(request)

    def _withdrawn(self, waiting: WorkerQueue, withdrawal: WithdrawPlace) -> None:
        """Withdraw the placement of `withdrawal` while it waits in `waiting`, answered
        `NotSent` with the withdrawal's reason before it stops counting (one the worker
        took already is left as it is); the withdrawal itself is done with either way."""
        if waiting.remove(withdrawal.place):
            self._post(NotSent(withdrawal.place, withdrawal.reason, withdrawal.words))
            self.finished()
        self.finished()

    def finished(self) -> None:
        """The worker executed a request and did not put it back."""
        with self._lock:
            self._unfinished -= 1


__all__ = ["Incoming"]
