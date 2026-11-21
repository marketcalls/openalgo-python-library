"""The order worker's own thread (16-session-design 2.1, 2.4): started once, it runs the
startup phase and then the worker's steps until the worker closes, and closes journal
handle W at its end. Split from `worker` so each module keeps one job.

`WorkerThread` is a part of `worker.OrderWorker`, which holds the startup phase
(`start_up`), the step (`step`), what is still handed over and not finished
(`_incoming`), what was built at the startup (`_started`), the wake-up event (`_wake`),
whether it failed or closes (`_failed`, `_stopping`), and how a failure is posted
(`_fail`).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from ..errors import LiveError
from ._journal_reads import JournalSession
from ._types import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

#: The worker's thread (16-session-design 2.1).
THREAD_NAME = "openalgo-live-orders"
#: Seconds the thread waits for a request or an update when nothing is ready (2.4 step 9).
WORKER_IDLE_WAIT_S = 0.05
#: Seconds `close` waits for the thread by default (a daemon: a call in flight may take
#: the orders client's timeout, and the process can end meanwhile).
CLOSE_WAIT_S = 10.0


class WorkerThread:
    """The thread of a `worker.OrderWorker` (module docstring)."""

    _incoming: Any
    _started: Any
    _wake: threading.Event
    _failed: bool
    _stopping: bool
    _thread: threading.Thread | None
    start_up: Callable[[], JournalSession]
    step: Callable[[], bool]
    _fail: Callable[[BaseException], None]

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


__all__ = ["CLOSE_WAIT_S", "THREAD_NAME", "WORKER_IDLE_WAIT_S", "WorkerThread"]
