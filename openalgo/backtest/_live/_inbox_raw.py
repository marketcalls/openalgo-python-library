"""The order worker's FIFO of raw order updates (16-session-design 2.3, 2.4 step 1), split
from `inbox`, which re-exports both names: `RawOrderUpdate`, one update as the feed's
order callback received it, and `RawUpdates`, the FIFO between that callback and the
order worker. Times are UTC nanoseconds.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RawOrderUpdate:
    """An order update as the feed's order callback received it, and `arrived_ns`, the
    callback's clock read: the order worker normalizes it (2.4 step 1)."""

    message: Mapping[str, Any]
    arrived_ns: int


class RawUpdates:
    """The FIFO of raw order updates between the feed's order callback and the order
    worker. Any thread puts, and a put never waits; one thread drains (the worker).
    `wake`, when given, is set after each put, so a worker that waits on one event for
    its requests and its updates wakes for an update too."""

    def __init__(self, *, wake: threading.Event | None = None) -> None:
        self._queue: queue.SimpleQueue[RawOrderUpdate] = queue.SimpleQueue()
        self._wake = wake

    def put(self, update: RawOrderUpdate) -> None:
        """Put one raw update at the end. Raises TypeError for anything else."""
        if not isinstance(update, RawOrderUpdate):
            raise TypeError(f"the raw updates are RawOrderUpdate, not {type(update).__name__}")
        self._queue.put(update)
        if self._wake is not None:
            self._wake.set()

    def drain(self) -> list[RawOrderUpdate]:
        """Every update waiting when it is called, oldest first; what arrives meanwhile
        waits for the next drain, so a flood cannot keep the worker here."""
        drained = []
        for _ in range(self._queue.qsize()):
            try:
                drained.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return drained

    def __len__(self) -> int:
        """The updates waiting (the worker's `raw_pending`, 4.2)."""
        return self._queue.qsize()


__all__ = ["RawOrderUpdate", "RawUpdates"]
