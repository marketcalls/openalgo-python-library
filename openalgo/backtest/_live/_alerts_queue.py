"""The bounded queue between a live session's loop and its alert sender (`alerts`).

The loop puts alerts on it and never waits; one sender thread takes them, oldest first,
and posts them to OpenAlgo. A full queue makes room by dropping the oldest alert of the
lowest priority, the new one included: a queue full of urgent alerts keeps them and
drops a routine one. Thread-safe.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

from ._types import AlertKind


@dataclass(frozen=True)
class QueuedAlert:
    """An alert waiting for the sender: its kind, its message (the mode and strategies
    already in front) and its Telegram priority."""

    kind: AlertKind
    message: str
    priority: int


class AlertQueue:
    """At most `size` alerts waiting for the sender, oldest first.

    `dropped` counts the alerts it let go: to make room, put after `close`, or left when
    the sender was given up on (`abandon`).
    """

    def __init__(self, size: int) -> None:
        self._size = size
        self._alerts: deque[QueuedAlert] = deque()
        self._ready = threading.Condition()
        self._closed = False
        self._abandoned = False
        self._dropped = 0

    @property
    def dropped(self) -> int:
        """The alerts dropped so far."""
        with self._ready:
            return self._dropped

    @property
    def closed(self) -> bool:
        """True once `close` was called: nothing more is taken in."""
        with self._ready:
            return self._closed

    def put(self, alert: QueuedAlert) -> QueuedAlert | None:
        """Queue `alert` without waiting. Returns the alert dropped for it (`alert` itself
        when it has the lowest priority, or when the queue is closed), or None."""
        with self._ready:
            if self._closed:
                self._dropped += 1
                return alert
            dropped = None
            if len(self._alerts) >= self._size:
                dropped = self._make_room(alert)
                self._dropped += 1
                if dropped is alert:
                    return alert
            self._alerts.append(alert)
            self._ready.notify()
            return dropped

    def take(self, *, wait: bool) -> QueuedAlert | None:
        """The oldest alert, or None when there is none to send: at once when the queue is
        empty and `wait` is False; with `wait`, once the queue is closed and empty or
        abandoned. A closed queue still gives the alerts it holds."""
        with self._ready:
            while wait and not self._alerts and not self._closed:
                self._ready.wait()
            if self._abandoned or not self._alerts:
                return None
            return self._alerts.popleft()

    def close(self) -> None:
        """Take in nothing more; the sender delivers what is left, then stops."""
        with self._ready:
            self._closed = True
            self._ready.notify_all()

    def abandon(self) -> int:
        """Drop every alert still waiting and stop the sender after its current one;
        returns how many were dropped."""
        with self._ready:
            left = len(self._alerts)
            self._alerts.clear()
            self._dropped += left
            self._closed = self._abandoned = True
            self._ready.notify_all()
            return left

    def _make_room(self, alert: QueuedAlert) -> QueuedAlert:
        """Remove and return the oldest alert of the lowest priority, or return `alert`
        when its priority is lower than every queued one's."""
        lowest = min(queued.priority for queued in self._alerts)
        if alert.priority < lowest:
            return alert
        index = next(i for i, queued in enumerate(self._alerts) if queued.priority == lowest)
        dropped = self._alerts[index]
        del self._alerts[index]
        return dropped


__all__ = ["AlertQueue", "QueuedAlert"]
