"""The clock of the threads seam (16-session-design 2.11; unit 16): the fake OpenAlgo's clock
moves on only while every thread of the session waits.

A session on its real threads (`threads=True`) runs against the fake's manual clock,
which a driver thread moves on `SPEED` times faster than the wall clock and plays the
test's script on the way. Moved by wall time alone, that clock raced ahead of any thread
the machine was slow to run (another test process, a Rust build), so what the session
saw of fake time, and decided, depended on the load: unit 14's flakes. So the driver
moves the clock only while every thread of the session waits (`Lockstep`): the loop on
its inbox, the order worker, the data worker and the alert sender on their events, the
injected sleep, a client double's held call, a join. While one works, the clock stands
still; the elapsed real time is dropped, not caught up. The session then sees the same
fake time on a loaded machine as on an idle one, only more slowly. The clock moves at
most `STEP_MAX_S` fake seconds a step, and each step wakes whoever waits for it
(`Lockstep.ticked`).

The waits are marked where they happen, for the session's threads only (the test's own
thread, which runs the loop, and the threads named "openalgo-live..."):
`threading.Condition.wait` (every `threading.Event.wait` goes through it),
`threading.Thread.join`, and `Inbox.take` with a wait. A thread that works for
`STALL_LIMIT_S` real seconds lets the clock move one step anyway, so a wait no mark
covers cannot stop a run; each such step is counted (`Lockstep.stalls`).

`lockstep(script)` installs the marks and starts the driver for one run, and takes both
down after it. Times are fake UTC nanoseconds unless said real.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Iterator
from typing import Any

from openalgo.backtest._live.inbox import Inbox

#: Fake seconds per real second while the session's threads wait.
SPEED = 30
#: The most fake seconds the clock moves in one step: a fraction of the session's one
#: second slice grace, so no due time is jumped by much.
STEP_MAX_S = 0.1
#: Real seconds between two looks of the driver.
PACE_S = 0.002
#: Real seconds a thread may work before the clock moves one step anyway.
STALL_LIMIT_S = 5.0
#: How the session names its threads (`worker.THREAD_NAME`, `data_worker.THREAD_NAME`,
#: `alerts.SENDER_THREAD_NAME`, `_feedlink_connect.CONNECT_THREAD_NAME`).
SESSION_THREADS = "openalgo-live"


class Lockstep:
    """Which of the session's threads wait (module docstring). `loop` is the ident of the
    thread that runs the session's loop (the test's)."""

    def __init__(self, loop: int) -> None:
        self._loop = loop
        self._lock = threading.Lock()
        self._depths: dict[int, int] = {}
        #: Notified at each step of the clock.
        self.ticked = threading.Condition()
        #: Steps the clock made while a thread worked past `STALL_LIMIT_S`.
        self.stalls = 0

    def ours(self, thread: threading.Thread) -> bool:
        """Whether `thread` is one of the session's."""
        return thread.ident == self._loop or thread.name.startswith(SESSION_THREADS)

    @contextlib.contextmanager
    def waiting(self) -> Iterator[None]:
        """Mark the calling thread waiting for the block's length, when it is one of the
        session's."""
        if not self.ours(threading.current_thread()):
            yield
            return
        ident = threading.get_ident()
        with self._lock:
            self._depths[ident] = self._depths.get(ident, 0) + 1
        try:
            yield
        finally:
            with self._lock:
                self._depths[ident] -= 1

    def all_waiting(self) -> bool:
        """True when every live thread of the session waits."""
        with self._lock:
            waiting = {ident for ident, depth in self._depths.items() if depth > 0}
        return all(
            thread.ident in waiting
            for thread in threading.enumerate()
            if self.ours(thread) and thread.is_alive()
        )


class Driver(threading.Thread):
    """Moves the fake's clock on while every thread of the session waits, `SPEED` times
    faster than the wall clock, playing the script on the way (module docstring)."""

    def __init__(self, script: Any, step: Lockstep) -> None:
        super().__init__(name="test-clock", daemon=True)
        self._script, self._step = script, step
        self.done = threading.Event()

    def run(self) -> None:
        last = time.perf_counter()
        busy_since: float | None = None
        while not self.done.is_set():
            time.sleep(PACE_S)
            now = time.perf_counter()
            elapsed, last = now - last, now
            if self._step.all_waiting():
                busy_since = None
            else:
                busy_since = now if busy_since is None else busy_since
                if now - busy_since < STALL_LIMIT_S:
                    continue
                self._step.stalls += 1
                busy_since = now
            self._script.sleep(min(elapsed * SPEED, STEP_MAX_S))
            with self._step.ticked:
                self._step.ticked.notify_all()


@contextlib.contextmanager
def lockstep(script: Any) -> Iterator[Lockstep]:
    """For one run on the threads seam: mark the session's waits and move the fake's
    clock by the driver (module docstring), the calling thread running the loop; both
    taken down after it."""
    step = Lockstep(threading.get_ident())
    wait, join, take = threading.Condition.wait, threading.Thread.join, Inbox.take

    def marked_wait(condition: Any, timeout: float | None = None) -> bool:
        with step.waiting():
            return wait(condition, timeout)

    def marked_join(thread: Any, timeout: float | None = None) -> None:
        with step.waiting():
            join(thread, timeout)

    def marked_take(inbox: Any, *, wait_s: float = 0.0, arrived_before_ns: int | None = None):
        if wait_s <= 0:
            return take(inbox, wait_s=wait_s, arrived_before_ns=arrived_before_ns)
        with step.waiting():
            return take(inbox, wait_s=wait_s, arrived_before_ns=arrived_before_ns)

    threading.Condition.wait = marked_wait  # type: ignore[method-assign]
    threading.Thread.join = marked_join  # type: ignore[method-assign]
    Inbox.take = marked_take  # type: ignore[method-assign]
    driver = Driver(script, step)
    driver.start()
    try:
        yield step
    finally:
        driver.done.set()
        driver.join(5)
        threading.Condition.wait = wait  # type: ignore[method-assign]
        threading.Thread.join = join  # type: ignore[method-assign]
        Inbox.take = take  # type: ignore[method-assign]


def wait_for(step: Lockstep, until: Any, *, limit_s: float) -> None:
    """Wait, marked as waiting, until `until()` is true, waking at each step of the clock
    and never longer than `limit_s` real seconds (a broken test cannot hang)."""
    limit = time.perf_counter() + limit_s
    with step.ticked:
        while not until() and time.perf_counter() < limit:
            step.ticked.wait(PACE_S * 10)


__all__ = [
    "PACE_S",
    "SESSION_THREADS",
    "SPEED",
    "STALL_LIMIT_S",
    "STEP_MAX_S",
    "Driver",
    "Lockstep",
    "lockstep",
    "wait_for",
]
