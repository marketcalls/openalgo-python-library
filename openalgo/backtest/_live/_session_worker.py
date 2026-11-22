"""The order worker and the journal at a live session's start (16-session-design 3.1 step 8;
split from `_session_start`): the order worker starts journal handle W, the journal's
session of the day and the order path, on its own thread or inline under the test seam,
and the main thread then opens journal handle L on the same file. Times are UTC
nanoseconds.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Any

from .._layout import NANOS_PER_SECOND
from ..errors import LiveError
from ._journal_reads import JournalSession
from ._shared_alerts import SharedAlerts
from ._types import LOGGER_NAME, Now, Sleep
from .guard import ModeGuard
from .inbox import Inbox, WorkerFailed, WorkerReady
from .journal import Journal
from .worker import JournalTarget, OrderWorker

logger = logging.getLogger(LOGGER_NAME)

#: Seconds the main thread waits for the order worker's startup (3.1 step 8).
WORKER_READY_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class StartedWorker:
    """What step 8 started: the order `worker`, the `journal_session` its startup began, and
    journal handle L (`journal`), the main thread's."""

    worker: OrderWorker
    journal_session: JournalSession
    journal: Journal


def start_worker(
    orders: Any,
    *,
    guard: ModeGuard,
    alerts: SharedAlerts,
    inbox: Inbox,
    target: JournalTarget,
    now: Now,
    sleep: Sleep,
    threads: bool,
    orders_per_second: int,
    stack: contextlib.ExitStack,
) -> StartedWorker:
    """Step 8 (module docstring): the order worker on the `orders` client, posting to the
    loop's `inbox`, its journal handle W opened on `target`, then journal handle L on the
    same file; `threads` False runs the worker's startup inline (the test seam). Both are
    closed by `stack`. Raises the worker's startup error, LiveError when it is not ready in
    `WORKER_READY_TIMEOUT_S`, and JournalError when handle L cannot be opened."""
    worker = OrderWorker(
        orders,
        guard=guard,
        alerts=alerts,
        inbox=inbox,
        journal=target,
        now=now,
        sleep=sleep,
        orders_per_second=orders_per_second,
    )
    stack.callback(worker.close)
    journal_session = _wait_ready(worker, inbox, now=now, threads=threads)
    journal = stack.enter_context(
        Journal.open(
            target.path,
            account=target.account,
            broker=target.broker,
            mode=target.mode,
            now=now,
            busy_timeout_s=target.busy_timeout_s,
        )
    )
    return StartedWorker(worker, journal_session, journal)


def _wait_ready(worker: OrderWorker, inbox: Inbox, *, now: Now, threads: bool) -> JournalSession:
    """Start the order worker and wait for its `WorkerReady` (3.1 step 8). Raises its
    startup's error, or LiveError when it is not ready in `WORKER_READY_TIMEOUT_S`."""
    if not threads:
        return worker.start_up()
    worker.start()
    deadline_ns = now() + round(WORKER_READY_TIMEOUT_S * NANOS_PER_SECOND)
    while now() < deadline_ns:
        envelope = inbox.take(wait_s=0.1)
        if envelope is None:
            continue
        message = envelope.message
        if isinstance(message, WorkerReady):
            return message.session
        if isinstance(message, WorkerFailed):
            raise message.error
        logger.warning("message before the order worker was ready, dropped: %r", message)
    raise LiveError(
        f"the order worker did not start within {WORKER_READY_TIMEOUT_S:g} s: the journal "
        "may be locked by another program"
    )


__all__ = ["WORKER_READY_TIMEOUT_S", "StartedWorker", "start_worker"]
