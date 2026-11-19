"""The bounded queue between a live session's loop and its alert sender (`alerts`).

The loop puts alerts on it and never waits; the sender takes them highest priority
first, oldest first within a priority. A routine alert (below `urgent_from`) leaves with
every other routine alert waiting that fits beside it in one Telegram message, in the
order they were sent, so a burst costs one post; an urgent alert always leaves alone.
The batch the sender took is held as in flight, under the queue's lock, until the sender
settles it, so a `close` at any moment finds it. A post that failed for a while puts its
alerts back at the head of their priority. A full queue makes room by dropping the
oldest alert of the lowest priority, the new one included: a queue full of urgent
alerts keeps them and drops a routine one. Thread-safe.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace

from ._types import AlertKind

# The text between two alerts posted together.
ALERT_SEPARATOR = "\n"


@dataclass(frozen=True)
class QueuedAlert:
    """An alert waiting for the sender: its kind, its message (the mode and strategies
    already in front), its Telegram priority, its place in the order alerts were sent
    (`seq`), and when it was first posted (UTC nanoseconds, None before)."""

    kind: AlertKind
    message: str
    priority: int
    seq: int
    first_posted_ns: int | None = None


@dataclass(frozen=True)
class Abandoned:
    """What `abandon` found: the alerts posted before and waiting for another try, the
    batch in flight (posted, no answer yet), and how many were never posted."""

    retried: list[QueuedAlert]
    in_flight: list[QueuedAlert]
    never_posted: int


class AlertQueue:
    """At most `size` alerts waiting for the sender. `urgent_from` is the least priority
    that leaves alone and `max_chars` the longest text of one post.

    `dropped` counts the alerts it let go without a post: to make room, put after
    `close`, or never posted when the sender was given up on (`abandon`).
    """

    def __init__(self, size: int, *, urgent_from: int, max_chars: int) -> None:
        self._size = size
        self._urgent_from = urgent_from
        self._max_chars = max_chars
        self._alerts: list[QueuedAlert] = []
        self._in_flight: list[QueuedAlert] = []
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

    def __len__(self) -> int:
        with self._ready:
            return len(self._alerts)

    def head_priority(self) -> int | None:
        """The priority of the alert `take` would hand out first, or None when none
        waits."""
        with self._ready:
            return max((alert.priority for alert in self._alerts), default=None)

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

    def take(self, now_ns: int) -> list[QueuedAlert]:
        """The next post's alerts in the order they were sent, taken off the queue and
        held as in flight until `settle`: the head alone when it is urgent, else every
        routine alert that fits beside it. Each is stamped as first posted at `now_ns`
        (UTC nanoseconds) unless it was posted before. Empty when nothing waits or the
        queue was abandoned."""
        with self._ready:
            if self._abandoned or not self._alerts:
                return []
            head = min(self._alerts, key=_head_order)
            batch = [head]
            if head.priority < self._urgent_from:
                batch = self._routine_batch(head)
            for alert in batch:
                self._alerts.remove(alert)
            self._in_flight = [
                alert
                if alert.first_posted_ns is not None
                else replace(alert, first_posted_ns=now_ns)
                for alert in batch
            ]
            return self._in_flight

    def settle(self, batch: list[QueuedAlert]) -> bool:
        """End the flight of `batch` once its post answered. False when `abandon` took it
        meanwhile (it was counted then, and its answer is to be ignored)."""
        with self._ready:
            if self._in_flight is not batch:
                return False
            self._in_flight = []
            return True

    def end_flight(self) -> list[QueuedAlert]:
        """End whatever flight is open and return its batch (empty when none): after a
        fault of the sender, which never settled it."""
        with self._ready:
            batch, self._in_flight = self._in_flight, []
            return batch

    def put_back(self, alerts: list[QueuedAlert]) -> bool:
        """Return alerts a failed post took: they keep their place (their `seq`), so they
        go first again within their priority, even past the size of a full queue. False,
        and nothing returned, once the queue was abandoned."""
        with self._ready:
            if self._abandoned:
                return False
            self._alerts.extend(alerts)
            self._ready.notify()
            return True

    def wait(self, timeout_seconds: float | None) -> bool:
        """Wait until an alert is put, the queue is closed or `timeout_seconds` pass (None:
        no limit; at once when alerts wait and there is no limit). False once the sender
        should stop: the queue is abandoned, or closed with nothing left."""
        with self._ready:
            if not self._stopping() and (timeout_seconds is not None or not self._alerts):
                self._ready.wait(timeout_seconds)
            return not self._stopping()

    def close(self) -> None:
        """Take in nothing more, and wake the sender: it delivers what is left, then
        stops."""
        with self._ready:
            self._closed = True
            self._ready.notify_all()

    def abandon(self) -> Abandoned:
        """Empty the queue, take the batch in flight and stop the sender after its current
        post. The alerts never posted are counted as dropped; the others are returned, for
        the sender to count as failed."""
        with self._ready:
            retried = [alert for alert in self._alerts if alert.first_posted_ns is not None]
            never_posted = len(self._alerts) - len(retried)
            self._dropped += never_posted
            in_flight, self._in_flight = self._in_flight, []
            self._alerts.clear()
            self._closed = self._abandoned = True
            self._ready.notify_all()
            return Abandoned(retried, in_flight, never_posted)

    def _stopping(self) -> bool:
        return self._abandoned or (self._closed and not self._alerts)

    def _routine_batch(self, head: QueuedAlert) -> list[QueuedAlert]:
        """The routine alerts waiting that fit one post beside the head (which always
        goes), taken in the order they were sent; one too long is skipped for a later
        post, not a stop to the merging."""
        routine = sorted(
            (alert for alert in self._alerts if alert.priority < self._urgent_from),
            key=lambda alert: alert.seq,
        )
        batch, length = [head], len(head.message)
        for alert in routine:
            added = len(ALERT_SEPARATOR) + len(alert.message)
            if alert is head or length + added > self._max_chars:
                continue
            batch.append(alert)
            length += added
        return sorted(batch, key=lambda alert: alert.seq)

    def _make_room(self, alert: QueuedAlert) -> QueuedAlert:
        """Remove and return the oldest alert of the lowest priority, or return `alert`
        when its priority is lower than every queued one's."""
        lowest = min(queued.priority for queued in self._alerts)
        if alert.priority < lowest:
            return alert
        dropped = min(
            (queued for queued in self._alerts if queued.priority == lowest),
            key=lambda queued: queued.seq,
        )
        self._alerts.remove(dropped)
        return dropped


def _head_order(alert: QueuedAlert) -> tuple[int, int]:
    """Highest priority first, oldest first within a priority."""
    return (-alert.priority, alert.seq)


__all__ = ["ALERT_SEPARATOR", "Abandoned", "AlertQueue", "QueuedAlert"]
