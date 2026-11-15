"""Alerts of a live session through OpenAlgo's Telegram bot, the daily summary and the
session logs (08-live-trading section 8).

`Alerts.send` posts alerts to `/api/v1/telegram/notify` through the SDK's
`client.telegram`. Every alert starts with the mode adopted at the session's start, in
words ("SANDBOX" or "LIVE"), not the server's mode at the moment of the post, and with
the session's strategies when known, so a message read on a phone cannot be taken for
the other mode. Each kind of alert has a priority (`ALERT_PRIORITY`); a mode breach, a
dead feed, a reconciliation mismatch and an error are the most urgent.

An alert never raises into the session loop and never makes it wait. `send` logs the
alert and puts it on a bounded queue (`_alerts_queue`); one sender thread, a daemon
started by the first `send` (and by a later one when it could not start or ended),
posts the queue highest priority first, paced below the server's limit with a reserve
for priority 10, and tries again after a transient failure (`_alerts_sender`), so a
server that does not answer holds the sender, not the loop. Routine alerts waiting together
leave as one message. A refused or abandoned alert is logged and counted in `failures`;
a full queue drops the oldest alert of the lowest priority, counted in `dropped`. A
repeated condition (`REPEATED_CONDITIONS`) is queued at most once in
`REPEAT_WINDOW_SECONDS` by the session's clock, counted in `suppressed`, so a condition
the loop reports on every tick reaches the phone once a minute; every other alert is
sent each time. Without an OpenAlgo username alerts go to the log only and no thread is
started. The text is escaped for Telegram and cut to its limit as `_alerts_telegram`
describes.

`SessionSummary` and `summary_text` (in `_alerts_summary`) and `SessionLogs` and
`configure_session_logs` (in `_alerts_logs`) are re-exported here.

The session loop calls `send`, `test` and `close`; the counts may be read from any
thread.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Protocol

from .._layout import NANOS_PER_SECOND
from ..errors import ConfigError
from ._alerts_logs import SessionLogs, configure_session_logs
from ._alerts_queue import AlertQueue, QueuedAlert
from ._alerts_sender import Sender
from ._alerts_summary import SessionSummary, summary_text
from ._alerts_telegram import TELEGRAM_MAX_CHARS, delivery_failure, telegram_text
from ._types import LOGGER_NAME, AlertKind, Mode, Now

#: The priorities OpenAlgo's notify endpoint accepts, low to urgent; it sends any other
#: value as 5 (restx_api/telegram_bot.py, SendNotification).
PRIORITY_LOWEST = 1
PRIORITY_HIGHEST = 10
#: The least priority of an urgent alert: it is logged as a warning and posted alone.
URGENT_PRIORITY = 9

#: The priority of each kind of alert, in the bands of the SDK's `telegram` docstring:
#: 9 to 10 urgent (risk), 7 to 8 high (position updates), 4 to 6 normal (signals and
#: summaries). The four kinds that lock entries or need the operator are the highest.
ALERT_PRIORITY: Mapping[AlertKind, int] = MappingProxyType(
    {
        AlertKind.MODE_BREACH: PRIORITY_HIGHEST,
        AlertKind.FEED_DEAD: PRIORITY_HIGHEST,
        AlertKind.RECONCILE_MISMATCH: PRIORITY_HIGHEST,
        AlertKind.ERROR: PRIORITY_HIGHEST,
        AlertKind.REJECT: URGENT_PRIORITY,
        AlertKind.ENTRY: 7,
        AlertKind.EXIT: 7,
        AlertKind.STOP_MOVED: 6,
        AlertKind.SESSION: 5,
        AlertKind.DAILY_SUMMARY: 5,
    }
)

#: The kinds that report a condition the loop may see again on every tick: the same text
#: of one of them is queued at most once in the repeat window. Every other alert reports
#: an event, and two equal events (two partial exits at one price) are two alerts.
REPEATED_CONDITIONS = frozenset(
    {AlertKind.FEED_DEAD, AlertKind.RECONCILE_MISMATCH, AlertKind.ERROR}
)
#: Seconds within which the same repeated condition is not sent again (the alerts brief,
#: Phase 2 step 8).
REPEAT_WINDOW_SECONDS = 60
#: The most alerts waiting for the sender. Routine alerts leave together in one post, so
#: the queue fills only with urgent alerts, which leave one a post: 60 of them are three
#: minutes of the sender's posting budget (`_alerts_sender.POSTS_PER_WINDOW`).
QUEUE_SIZE = 60
#: Seconds `close` waits for the sender to post what is queued: a server that answers
#: takes a few posts in well under that, and a stopping session is not held longer.
CLOSE_WAIT_SECONDS = 5.0
#: The name of the sender thread.
SENDER_THREAD_NAME = "openalgo-live-alerts"

TEST_ALERT_TEXT = "Test alert from bt.live: the alerts of this session reach you here."

_REPEAT_WINDOW_NS = REPEAT_WINDOW_SECONDS * NANOS_PER_SECOND
_NO_USERNAME = "no OpenAlgo username was given, so alerts go to the log only"

logger = logging.getLogger(LOGGER_NAME)


class _TelegramClient(Protocol):
    """What the alerts need of an `openalgo.api` client: its Telegram call."""

    def telegram(self, *, username: str, message: str, priority: int, **kwargs: Any) -> Any:
        """The SDK's answer to `/api/v1/telegram/notify`."""


class Alerts:
    """The alerts of one live session (implements `AlertsLike`); a context manager that
    closes them.

    `client` is an `openalgo.api` client; only `telegram()` is called, from the sender
    thread and from `test`. `username` is the OpenAlgo login username linked to the
    Telegram bot (not the Telegram handle), or None to send alerts to the log only.
    `mode` is the mode adopted at the session's start, `now` gives the present in UTC
    nanoseconds, and `strategies` names the session's strategies for the start of every
    alert. The sender thread starts on the first `send`; with `start_sender` False it
    never starts and the queue is posted by calling `deliver_queued` (how the tests
    drive it). Call `close` at the session's end.

    Raises ConfigError for a `username` that is neither None nor a non-empty string, a
    `mode` that is not a `Mode`, or `strategies` that are not a sequence of names.
    """

    def __init__(
        self,
        client: _TelegramClient,
        *,
        username: str | None,
        mode: Mode,
        now: Now,
        strategies: Sequence[str] = (),
        start_sender: bool = True,
    ) -> None:
        if username is not None and (not isinstance(username, str) or not username.strip()):
            raise ConfigError(
                f"cannot send alerts to username {username!r}: give your OpenAlgo login "
                f"username (linked to the Telegram bot with /link), or None to send alerts "
                f"to the log only"
            )
        if not isinstance(mode, Mode):
            raise ConfigError(f"mode must be Mode.SANDBOX or Mode.LIVE, got {mode!r}")
        self._client = client
        self._username = username
        self._now = now
        self._prefix = _prefix(mode, _strategy_names(strategies))
        self._last_sent_ns: dict[tuple[AlertKind, str], int] = {}
        self._suppressed = 0
        self._seq = 0
        self._queue = AlertQueue(
            QUEUE_SIZE, urgent_from=URGENT_PRIORITY, max_chars=TELEGRAM_MAX_CHARS
        )
        self._sender = Sender(self._queue, self._post, now, highest_priority=PRIORITY_HIGHEST)
        self._start_sender = start_sender
        self._thread: threading.Thread | None = None

    @property
    def failures(self) -> int:
        """Alerts posted that OpenAlgo refused, that failed until given up, or whose post
        had not answered when the alerts closed (the test alert is not counted)."""
        return self._sender.failures

    @property
    def dropped(self) -> int:
        """Alerts never posted: dropped from a full queue, sent after `close`, or still
        queued when `close` stopped waiting."""
        return self._queue.dropped

    @property
    def undelivered(self) -> int:
        """Alerts of the session that did not reach OpenAlgo: `failures` plus `dropped`
        (the daily summary's figure)."""
        return self.failures + self.dropped

    @property
    def suppressed(self) -> int:
        """Repeated conditions not sent because the same one was sent less than a minute
        before."""
        return self._suppressed

    @property
    def last_failure(self) -> str | None:
        """Why the last alert or test alert failed, in words, or None when none failed."""
        return self._sender.last_failure

    def send(self, kind: AlertKind, text: str) -> None:
        """Queue an alert of `kind`, its text starting with the mode and the strategies.

        Logged at once, as a warning from priority 9 up, and queued for the sender when a
        username was given; it never waits for OpenAlgo. A repeated condition with the
        same text within `REPEAT_WINDOW_SECONDS` of the last time it was queued is counted
        in `suppressed`, logged at debug level only and not queued. Never raises: a `kind`
        that is not an `AlertKind` is sent as an ERROR alert.
        """
        if not isinstance(kind, AlertKind):
            logger.error("alert of unknown kind %r sent as an error alert", kind)
            kind = AlertKind.ERROR
        message = f"{self._prefix}{text}"
        if kind in REPEATED_CONDITIONS and self._repeated(kind, message, self._now()):
            self._suppressed += 1
            logger.debug("alert %s not sent again within a minute: %s", kind.value, message)
            return
        level = logging.WARNING if ALERT_PRIORITY[kind] >= URGENT_PRIORITY else logging.INFO
        logger.log(level, "alert %s: %s", kind.value, message)
        if self._username is not None:
            self._seq += 1
            self._enqueue(QueuedAlert(kind, message, ALERT_PRIORITY[kind], self._seq))
            self._start_thread()

    def test(self) -> bool:
        """Send a test alert now, on the calling thread, and wait for OpenAlgo to say
        Telegram took it.

        True only when Telegram confirmed the delivery; False with no username, for an
        error answer, or when OpenAlgo only queued the message (the bot could not reach
        Telegram), with the reason in `last_failure`. The preflight requires it in LIVE
        mode and warns without it in SANDBOX mode. It waits as long as OpenAlgo waits on
        Telegram (within the client's timeout), so it belongs before the session, not in
        its loop. Bypasses the repeat window, the queue and the pacing; never raises.
        """
        message = f"{self._prefix}{TEST_ALERT_TEXT}"
        failure = _NO_USERNAME
        if self._username is not None:
            logger.info("test alert: %s", message)
            try:
                answer = self._post(
                    telegram_text(message),
                    ALERT_PRIORITY[AlertKind.SESSION],
                    wait_for_delivery=True,
                )
                failure = delivery_failure(answer)
            # A client that raises must not stop the preflight: the reason is kept.
            except Exception as error:
                failure = f"the client raised {type(error).__name__}: {error}"
            if failure is None:
                return True
        self._sender.note_failure(failure)
        logger.warning("test alert not delivered: %s", failure)
        return False

    def deliver_queued(self) -> int:
        """Post on the calling thread every queued alert that may go now (the pacing and
        any pause after a failure hold the rest); returns the posts made. The sender
        thread does this work: call it only on alerts made with `start_sender=False`."""
        posts, _ = self._sender.step()
        return posts

    def close(self, wait_seconds: float = CLOSE_WAIT_SECONDS) -> None:
        """Stop taking alerts and give the sender up to `wait_seconds` to post what is
        queued. Then alerts never posted are dropped, those waiting for another try count
        as failed, and a post still in flight is logged by name and counted as failed (it
        may or may not arrive). Alerts sent afterwards are logged and dropped. Safe to
        call twice."""
        if self._queue.closed:
            return
        self._queue.close()
        if self._thread is not None:
            self._thread.join(wait_seconds)
        abandoned = self._queue.abandon()
        self._sender.give_up(abandoned)
        if abandoned.never_posted:
            logger.warning(
                "%d alert(s) not sent: OpenAlgo had not taken them when the alerts closed",
                abandoned.never_posted,
            )

    def __enter__(self) -> Alerts:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _post(self, text: str, priority: int, **extra: Any) -> Any:
        return self._client.telegram(
            username=self._username, message=text, priority=priority, **extra
        )

    def _start_thread(self) -> None:
        """Start the sender thread unless one is alive. Kept only once it started, so
        `close` joins only a started thread; a thread that cannot start is logged and the
        alerts wait in the queue for the next `send` to try again."""
        if not self._start_sender or self._queue.closed:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        thread = threading.Thread(target=self._sender.run, name=SENDER_THREAD_NAME, daemon=True)
        try:
            thread.start()
        # `send` never raises: a thread refused (at interpreter shutdown, or with no
        # resources left) is reported and tried again on the next send.
        except Exception as error:
            logger.error(
                "the alert sender thread could not start (%s): alerts wait in the queue "
                "until a later send starts it",
                error,
            )
            return
        self._thread = thread

    def _repeated(self, kind: AlertKind, message: str, now_ns: int) -> bool:
        """True when this condition was queued within the repeat window, else records it
        now; forgets the conditions whose window has passed, so the record stays as small
        as a minute's alerts."""
        self._last_sent_ns = {
            key: sent_ns
            for key, sent_ns in self._last_sent_ns.items()
            if now_ns - sent_ns < _REPEAT_WINDOW_NS
        }
        if (kind, message) in self._last_sent_ns:
            return True
        self._last_sent_ns[(kind, message)] = now_ns
        return False

    def _enqueue(self, alert: QueuedAlert) -> None:
        dropped = self._queue.put(alert)
        if dropped is None:
            return
        why = "the alerts are closed" if self._queue.closed else f"{QUEUE_SIZE} alerts wait"
        logger.warning("alert %s dropped (%s): %s", dropped.kind.value, why, dropped.message)


def _strategy_names(strategies: object) -> tuple[str, ...]:
    """The names as a tuple. Raises ConfigError for anything but an iterable of strings;
    a single name is refused rather than read as a sequence of its letters."""
    if not isinstance(strategies, str) and isinstance(strategies, Iterable):
        names = tuple(strategies)
        if all(isinstance(name, str) for name in names):
            return names
    raise ConfigError(
        f"strategies must be a list of strategy names, got {strategies!r}: pass "
        f"['ema_cross'] for one strategy"
    )


def _prefix(mode: Mode, strategies: tuple[str, ...]) -> str:
    """ "SANDBOX: " or "LIVE (ema_cross, orb): "."""
    if not strategies:
        return f"{mode.name}: "
    return f"{mode.name} ({', '.join(strategies)}): "


__all__ = [
    "ALERT_PRIORITY",
    "CLOSE_WAIT_SECONDS",
    "PRIORITY_HIGHEST",
    "PRIORITY_LOWEST",
    "QUEUE_SIZE",
    "REPEATED_CONDITIONS",
    "REPEAT_WINDOW_SECONDS",
    "SENDER_THREAD_NAME",
    "TELEGRAM_MAX_CHARS",
    "TEST_ALERT_TEXT",
    "URGENT_PRIORITY",
    "Alerts",
    "SessionLogs",
    "SessionSummary",
    "configure_session_logs",
    "summary_text",
]
