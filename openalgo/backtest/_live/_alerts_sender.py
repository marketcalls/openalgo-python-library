"""The sender of a live session's alerts (`alerts`): it posts the queue through
`client.telegram`, paced, and tries a failed post again while that may help.

Pacing, on the session's clock: posts below the highest priority use at most
`POSTS_PER_WINDOW` of any `LIMIT_WINDOW_SECONDS`, and posts of the highest priority up
to `HIGHEST_POSTS_PER_WINDOW`, so a mode breach still goes when routine traffic used its
share. Retry: a post that failed transiently (`_alerts_telegram.read_post`) puts its
alerts back at the head of their priority and nothing is posted until a wait: after the
rate limiter's 429, its `retry_after` up to the server's window, or the whole window
when it gave none; after any other transient failure, a back-off that doubles from
`FIRST_BACKOFF_SECONDS` up to `MAX_BACKOFF_SECONDS`. An alert still failing
`GIVE_UP_SECONDS` after its first post counts as failed. A refusal counts as failed at
once. Every alert is counted once: delivered, failed (`failures`), or dropped by the
queue.

`step` posts what may go now and says when the next post may go; the sender thread
(`run`) waits on the queue between steps, so a new alert or `close` wakes it. An
unexpected fault in the thread is logged, the post it hit counts as failed, and the
thread goes on.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from typing import Any

from .._layout import NANOS_PER_SECOND
from ._alerts_queue import ALERT_SEPARATOR, Abandoned, AlertQueue, QueuedAlert
from ._alerts_telegram import PostResult, read_post, telegram_text
from ._types import LOGGER_NAME, Now

#: The server's rate-limit window, seconds: OpenAlgo's `/notify` allows 30 posts in any
#: minute, a moving window per client IP (restx_api/telegram_bot.py, TELEGRAM_RATE_LIMIT;
#: limiter.py). The posting budget counts posts in it, a 429 without `retry_after` waits
#: it out, and a longer `retry_after` is cut to it.
LIMIT_WINDOW_SECONDS = 60
#: The most posts below the highest priority in any window: of the server's 30, five are
#: left to the test alert and the user's other scripts, and five kept for the highest.
POSTS_PER_WINDOW = 20
#: The most posts of the highest priority in any window, the reserve included.
HIGHEST_POSTS_PER_WINDOW = 25
#: The first wait after a transient failure other than a 429, seconds; it doubles at each
#: failure in a row, so a server that is restarting is tried a few times a minute.
FIRST_BACKOFF_SECONDS = 1.0
#: The longest back-off, seconds: a server back up is found within half a minute.
MAX_BACKOFF_SECONDS = 30.0
#: How long an alert is tried, seconds since its first post, before it counts as failed:
#: an alert older than that no longer tells the trader what is happening now.
GIVE_UP_SECONDS = 300
#: The wait, seconds, after a fault of the sender thread outside a post (its clock, its
#: queue), so a fault that repeats is not retried in a tight loop.
FAULT_PAUSE_SECONDS = 1.0

_WINDOW_NS = LIMIT_WINDOW_SECONDS * NANOS_PER_SECOND
_GIVE_UP_NS = GIVE_UP_SECONDS * NANOS_PER_SECOND
_STILL_FAILING_AT_CLOSE = "still failing when the alerts closed"
_IN_FLIGHT_AT_CLOSE = "still being posted when the alerts closed"
_SENDER_FAULT = "the sender failed while posting it (the error is in the log)"

logger = logging.getLogger(LOGGER_NAME)

#: Posts one message: (text, priority) to what the client answered or raised.
Post = Callable[[str, int], Any]


class Sender:
    """Posts the alerts of `queue` through `post`, paced and retried by the rules of the
    module docstring; `now` gives the present in UTC nanoseconds and `highest_priority`
    is the priority the reserve is kept for. `step` and `run` are called from one thread
    at a time; the counts may be read from any thread."""

    def __init__(self, queue: AlertQueue, post: Post, now: Now, *, highest_priority: int) -> None:
        self._queue = queue
        self._post = post
        self._now = now
        self._highest_priority = highest_priority
        self._posted_ns: deque[int] = deque()
        self._paused_until_ns = 0
        self._backoff_seconds = FIRST_BACKOFF_SECONDS
        self._counts = threading.Lock()
        self._failures = 0
        self._last_failure: str | None = None

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
        while (priority := self._queue.head_priority()) is not None:
            now_ns = self._now()
            due_ns = self._next_post_ns(now_ns, priority)
            if due_ns > now_ns:
                return posts, due_ns
            batch = self._queue.take(now_ns)
            if not batch:
                break
            self._send(batch, now_ns)
            posts += 1
        return posts, None

    def run(self) -> None:
        """The sender thread: step, then wait on the queue until the next post may go, an
        alert comes or the alerts close, until the queue is abandoned or closed and
        empty. A fault is logged and the thread goes on."""
        while True:
            try:
                _, due_ns = self.step()
                timeout = None if due_ns is None else (due_ns - self._now()) / NANOS_PER_SECOND
                if not self._queue.wait(None if timeout is None else max(timeout, 0.0)):
                    return
            # An unexpected fault must not end the sender, or every later alert would wait
            # unseen: it is logged with its traceback and the post it hit counts as failed.
            except Exception:
                logger.exception("the alert sender failed; it goes on")
                if not self._fault_ended_a_post() and not self._queue.wait(FAULT_PAUSE_SECONDS):
                    return

    def give_up(self, abandoned: Abandoned) -> None:
        """At close: count as failed the alerts waiting for another try, and the post
        still in flight, which may or may not arrive (its answer is then ignored)."""
        if abandoned.retried:
            self._fail(abandoned.retried, _STILL_FAILING_AT_CLOSE)
        if abandoned.in_flight:
            logger.warning(
                "%s was %s; it may or may not arrive: %s",
                _named(abandoned.in_flight),
                _IN_FLIGHT_AT_CLOSE,
                ALERT_SEPARATOR.join(alert.message for alert in abandoned.in_flight),
            )
            self._fail(abandoned.in_flight, _IN_FLIGHT_AT_CLOSE, log=False)

    def _next_post_ns(self, now_ns: int, priority: int) -> int:
        """When a post of `priority` may go: after any pause, and within its budget."""
        while self._posted_ns and self._posted_ns[0] <= now_ns - _WINDOW_NS:
            self._posted_ns.popleft()
        limit = POSTS_PER_WINDOW
        if priority >= self._highest_priority:
            limit = HIGHEST_POSTS_PER_WINDOW
        budget_ns = now_ns
        if len(self._posted_ns) >= limit:
            # The post whose leaving the window brings the count below the limit.
            budget_ns = self._posted_ns[len(self._posted_ns) - limit] + _WINDOW_NS
        return max(self._paused_until_ns, budget_ns)

    def _send(self, batch: list[QueuedAlert], now_ns: int) -> None:
        self._posted_ns.append(now_ns)
        text = ALERT_SEPARATOR.join(alert.message for alert in batch)
        result = self._result(telegram_text(text), max(alert.priority for alert in batch))
        if not self._queue.settle(batch):
            return  # close gave up on this post and counted it
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
        wait = self._wait_after(result)
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

    def _wait_after(self, result: PostResult) -> float:
        """Seconds to wait after a transient failure: a 429's `retry_after` cut to the
        server's window, or the window; else the back-off, which then doubles."""
        if result.rate_limited:
            return min(result.retry_after_seconds or LIMIT_WINDOW_SECONDS, LIMIT_WINDOW_SECONDS)
        wait = self._backoff_seconds
        self._backoff_seconds = min(wait * 2, MAX_BACKOFF_SECONDS)
        return wait

    def _fault_ended_a_post(self) -> bool:
        """After a fault: count the batch in flight, if any, as failed. True when there was
        one, so the queue moved on and the sender may step again at once."""
        batch = self._queue.end_flight()
        if batch:
            self._fail(batch, _SENDER_FAULT)
        return bool(batch)

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
    "FAULT_PAUSE_SECONDS",
    "FIRST_BACKOFF_SECONDS",
    "GIVE_UP_SECONDS",
    "HIGHEST_POSTS_PER_WINDOW",
    "LIMIT_WINDOW_SECONDS",
    "MAX_BACKOFF_SECONDS",
    "POSTS_PER_WINDOW",
    "Post",
    "Sender",
]
