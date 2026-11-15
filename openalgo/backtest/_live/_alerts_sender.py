"""The sender of a live session's alerts (`alerts`): it posts the queue through
`client.telegram`, paced, and tries a failed post again while that may help.

Pacing: at most `POSTS_PER_WINDOW` posts in any `POSTING_WINDOW_SECONDS`, on the
session's clock. Retry: a post that failed transiently (`_alerts_telegram.read_post`)
puts its alerts back at the head of their priority and nothing is posted until the wait
a 429 asked for, or a back-off that doubles from `FIRST_BACKOFF_SECONDS` up to
`MAX_BACKOFF_SECONDS`; an alert still failing `GIVE_UP_SECONDS` after its first post
counts as failed. A refusal counts as failed at once. Every alert is counted once:
delivered, failed (`failures`), or dropped by the queue.

`step` posts what may go now and says when the next post may go; the sender thread
(`run`) waits on the queue between steps, so a new alert or `close` wakes it.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from .._layout import NANOS_PER_SECOND
from ._alerts_queue import ALERT_SEPARATOR, AlertQueue, QueuedAlert
from ._alerts_telegram import PostResult, read_post, telegram_text
from ._types import LOGGER_NAME, Now

#: The most posts in any posting window. OpenAlgo's `/notify` allows 30 a minute per
#: client IP (restx_api/telegram_bot.py, TELEGRAM_RATE_LIMIT; limiter.py keys it by the
#: remote address), shared with the test alert and the user's other scripts.
POSTS_PER_WINDOW = 20
#: The posting window, seconds: the minute of the server's limit.
POSTING_WINDOW_SECONDS = 60
#: The first wait after a transient failure with no `retry_after`, seconds; it doubles at
#: each failure in a row, so a server that is restarting is tried a few times a minute.
FIRST_BACKOFF_SECONDS = 1.0
#: The longest back-off, seconds: a server back up is found within half a minute.
MAX_BACKOFF_SECONDS = 30.0
#: How long an alert is tried, seconds since its first post, before it counts as failed:
#: an alert older than that no longer tells the trader what is happening now.
GIVE_UP_SECONDS = 300

_WINDOW_NS = POSTING_WINDOW_SECONDS * NANOS_PER_SECOND
_GIVE_UP_NS = GIVE_UP_SECONDS * NANOS_PER_SECOND
_STILL_FAILING_AT_CLOSE = "still failing when the alerts closed"

logger = logging.getLogger(LOGGER_NAME)

#: Posts one message: (text, priority) to what the client answered or raised.
Post = Callable[[str, int], Any]


class Sender:
    """Posts the alerts of `queue` through `post`, paced and retried by the rules of the
    module docstring; `now` gives the present in UTC nanoseconds. `step` and `run` are
    called from one thread at a time; the counts may be read from any thread."""

    def __init__(self, queue: AlertQueue, post: Post, now: Now) -> None:
        self._queue = queue
        self._post = post
        self._now = now
        self._posted_ns: deque[int] = deque()
        self._paused_until_ns = 0
        self._backoff_seconds = FIRST_BACKOFF_SECONDS
        self._counts = threading.Lock()
        self._failures = 0
        self._last_failure: str | None = None
        self._in_flight: list[QueuedAlert] = []

    @property
    def failures(self) -> int:
        """Alerts posted that OpenAlgo refused, that failed until given up, or whose post
        had not answered when the alerts closed."""
        with self._counts:
            return self._failures

    @property
    def last_failure(self) -> str | None:
        """Why the last alert failed, in words, or None when none failed."""
        with self._counts:
            return self._last_failure

    def note_failure(self, reason: str) -> None:
        """Keep `reason` as the last failure without counting an alert (the test alert)."""
        with self._counts:
            self._last_failure = reason

    def step(self) -> tuple[int, int | None]:
        """Post every batch that may go now. Returns the posts made and when the next may
        go (UTC nanoseconds), or None when nothing waits."""
        posts = 0
        while len(self._queue):
            now_ns = self._now()
            due_ns = self._next_post_ns(now_ns)
            if due_ns > now_ns:
                return posts, due_ns
            batch = self._queue.take()
            if not batch:
                break
            self._send(batch, now_ns)
            posts += 1
        return posts, None

    def run(self) -> None:
        """The sender thread: step, then wait on the queue until the next post may go, an
        alert comes or the alerts close, until the queue is abandoned or closed and
        empty."""
        while True:
            _, due_ns = self.step()
            timeout = None if due_ns is None else max(due_ns - self._now(), 0) / NANOS_PER_SECOND
            if not self._queue.wait(timeout):
                return

    def give_up(self, retried: list[QueuedAlert]) -> None:
        """At close: count as failed the alerts waiting for another try, and the post
        still in flight, which may or may not arrive (its answer is then ignored)."""
        with self._counts:
            in_flight, self._in_flight = self._in_flight, []
        if retried:
            self._fail(retried, _STILL_FAILING_AT_CLOSE)
        if in_flight:
            logger.warning(
                "%s was still being posted when the alerts closed; it may or may not arrive: %s",
                _named(in_flight),
                ALERT_SEPARATOR.join(alert.message for alert in in_flight),
            )
            self._fail(in_flight, "still being posted when the alerts closed", log=False)

    def _next_post_ns(self, now_ns: int) -> int:
        """When the next post may go: after any pause, and within the posting budget."""
        while self._posted_ns and self._posted_ns[0] <= now_ns - _WINDOW_NS:
            self._posted_ns.popleft()
        budget_ns = now_ns
        if len(self._posted_ns) >= POSTS_PER_WINDOW:
            budget_ns = self._posted_ns[0] + _WINDOW_NS
        return max(self._paused_until_ns, budget_ns)

    def _send(self, batch: list[QueuedAlert], now_ns: int) -> None:
        batch = [
            alert if alert.first_posted_ns is not None else replace(alert, first_posted_ns=now_ns)
            for alert in batch
        ]
        self._posted_ns.append(now_ns)
        with self._counts:
            self._in_flight = batch
        text = ALERT_SEPARATOR.join(alert.message for alert in batch)
        result = self._result(telegram_text(text), max(alert.priority for alert in batch))
        with self._counts:
            if self._in_flight is not batch:
                return  # close gave up on this post and counted it
            self._in_flight = []
        if result.delivered:
            self._backoff_seconds = FIRST_BACKOFF_SECONDS
        elif result.transient:
            self._retry(batch, result)
        else:
            self._fail(batch, result.reason or "")

    def _result(self, text: str, priority: int) -> PostResult:
        try:
            return read_post(self._post(text, priority))
        # A client that raises must not stop the sender: the reason is logged, counted
        # and kept in `last_failure`, never dropped.
        except Exception as error:
            return PostResult(f"the client raised {type(error).__name__}: {error}")

    def _retry(self, batch: list[QueuedAlert], result: PostResult) -> None:
        """Put the alerts back and pause, or fail those tried for `GIVE_UP_SECONDS`."""
        now_ns = self._now()
        expired = [alert for alert in batch if now_ns - alert.first_posted_ns >= _GIVE_UP_NS]
        if expired:
            self._fail(expired, f"{result.reason} (tried for {GIVE_UP_SECONDS} s)")
        kept = [alert for alert in batch if alert not in expired]
        wait = result.retry_after_seconds or self._backoff_seconds
        self._backoff_seconds = min(self._backoff_seconds * 2, MAX_BACKOFF_SECONDS)
        self._paused_until_ns = now_ns + round(wait * NANOS_PER_SECOND)
        if not kept:
            return
        if not self._queue.put_back(kept):
            # The alerts closed while this post was answered: no later try will come.
            self._fail(kept, _STILL_FAILING_AT_CLOSE)
            return
        logger.warning(
            "%s not delivered yet (%s): trying again in %g s", _named(kept), result.reason, wait
        )

    def _fail(self, alerts: list[QueuedAlert], reason: str, *, log: bool = True) -> None:
        with self._counts:
            self._failures += len(alerts)
            self._last_failure = reason
        if log:
            logger.warning("%s not delivered: %s", _named(alerts), reason)


def _named(alerts: list[QueuedAlert]) -> str:
    """ "alert entry" or "alerts entry, exit"."""
    kinds = ", ".join(alert.kind.value for alert in alerts)
    return f"alert {kinds}" if len(alerts) == 1 else f"alerts {kinds}"


__all__ = [
    "FIRST_BACKOFF_SECONDS",
    "GIVE_UP_SECONDS",
    "MAX_BACKOFF_SECONDS",
    "POSTING_WINDOW_SECONDS",
    "POSTS_PER_WINDOW",
    "Post",
    "Sender",
]
