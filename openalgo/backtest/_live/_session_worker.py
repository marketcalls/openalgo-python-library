"""The order worker and the journal at a live session's start (16-session-design 3.1 step 8;
split from `_session_start`): the order worker starts journal handle W, the journal's
session of the day and the order path, on its own thread or inline under the test seam,
and the main thread then opens journal handle L on the same file. Times are UTC
nanoseconds.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .._layout import NANOS_PER_SECOND
from ..calendar import default_calendar
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
    read_holdings: bool,
    stack: contextlib.ExitStack,
    gtt_legs: tuple[tuple[str, str, str], ...] = (),
) -> StartedWorker:
    """Step 8 (module docstring): the order worker on the `orders` client, posting to the
    loop's `inbox`, its journal handle W opened on `target`, then journal handle L on the
    same file; `threads` False runs the worker's startup inline (the test seam), and
    `read_holdings` has it read the holdings with every book read (a session with a CNC
    strategy). `gtt_legs` are the (symbol, exchange, product) of the CNC and NRML legs:
    the GTT follower reads the GTT book on the clock while their exchanges' sessions are
    open on the trading day of `target` (`market_hours`, the lead's ruling N3b-2), and
    each reconciliation read gives it their units. Both are closed by `stack`. Raises the
    worker's startup error, LiveError when it is not ready in `WORKER_READY_TIMEOUT_S`,
    and JournalError when handle L cannot be opened."""
    worker = OrderWorker(
        orders,
        guard=guard,
        alerts=alerts,
        inbox=inbox,
        journal=target,
        now=now,
        sleep=sleep,
        orders_per_second=orders_per_second,
        read_holdings=read_holdings,
        market_open=market_hours({exchange for _, exchange, _ in gtt_legs}, target.day),
        gtt_legs=gtt_legs,
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


def market_hours(exchanges: Iterable[str], day: dt.date) -> Callable[[int], bool] | None:
    """Whether the market of any of `exchanges` is open at an instant (UTC ns) on `day`, by
    the package calendar's sessions; None for no exchange (no read on the clock)."""
    calendar = default_calendar()
    sessions = [calendar.session(exchange, day) for exchange in sorted(exchanges)]
    spans = [span for span in sessions if span is not None]
    if not sessions:
        return None
    return lambda now_ns: any(opens <= now_ns < closes for opens, closes in spans)


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
