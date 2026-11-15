"""Alerts of a live session through OpenAlgo's Telegram bot, the daily summary and the
session logs (08-live-trading section 8).

`Alerts.send` posts one alert to `/api/v1/telegram/notify` through the SDK's
`client.telegram`. Every alert starts with the mode OpenAlgo is in, in words ("SANDBOX"
or "LIVE"), and the session's strategies when known, so a message read on a phone cannot
be taken for the other mode. Each kind of alert has a priority (`ALERT_PRIORITY`); a mode
breach, a dead feed, a reconciliation mismatch and an error are the most urgent.

An alert never raises into the session loop and never makes it wait. `send` logs the
alert and puts it on a bounded queue (`_alerts_queue`); one sender thread, a daemon
started with the alerts, posts it, so a server that does not answer holds the sender,
not the loop. A send that fails (an error answer, a transport failure, an exception from
the client) is logged and counted in `failures` by the sender; a full queue drops the
oldest alert of the lowest priority, counted in `dropped`. The same alert is queued at
most once in `REPEAT_WINDOW_SECONDS` by the session's clock, counted in `suppressed`, so
a condition the loop reports on every tick reaches the phone once a minute. Without an
OpenAlgo username alerts go to the log only and no thread is started. The text is
escaped for Telegram and cut to its limit as `_alerts_telegram` describes.

`SessionSummary` and `summary_text` (in `_alerts_summary`) and `SessionLogs` and
`configure_session_logs` (in `_alerts_logs`) are re-exported here.

The session loop calls `send`, `test` and `close`; the counts may be read from any
thread.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Protocol

from .._layout import NANOS_PER_SECOND
from ..errors import ConfigError
from ._alerts_logs import SessionLogs, configure_session_logs
from ._alerts_queue import AlertQueue, QueuedAlert
from ._alerts_summary import SessionSummary, summary_text
from ._alerts_telegram import TELEGRAM_MAX_CHARS, delivery_failure, failure_reason, telegram_text
from ._types import LOGGER_NAME, AlertKind, Mode, Now

#: The priorities OpenAlgo's notify endpoint accepts, low to urgent; it sends any other
#: value as 5 (restx_api/telegram_bot.py, SendNotification).
PRIORITY_LOWEST = 1
PRIORITY_HIGHEST = 10

#: The priority of each kind of alert, in the bands of the SDK's `telegram` docstring:
#: 9 to 10 urgent (risk), 7 to 8 high (position updates), 4 to 6 normal (signals and
#: summaries). The four kinds that lock entries or need the operator are the highest.
ALERT_PRIORITY: Mapping[AlertKind, int] = MappingProxyType(
    {
        AlertKind.MODE_BREACH: PRIORITY_HIGHEST,
        AlertKind.FEED_DEAD: PRIORITY_HIGHEST,
        AlertKind.RECONCILE_MISMATCH: PRIORITY_HIGHEST,
        AlertKind.ERROR: PRIORITY_HIGHEST,
        AlertKind.REJECT: 9,
        AlertKind.ENTRY: 7,
        AlertKind.EXIT: 7,
        AlertKind.STOP_MOVED: 6,
        AlertKind.SESSION: 5,
        AlertKind.DAILY_SUMMARY: 5,
    }
)

#: Seconds within which the same alert is not sent again (the alerts brief, Phase 2 step 8).
REPEAT_WINDOW_SECONDS = 60
#: The most alerts waiting for the sender: two minutes of alerts at the notify endpoint's
#: rate limit of 30 a minute (restx_api/telegram_bot.py, TELEGRAM_RATE_LIMIT).
QUEUE_SIZE = 60
#: Seconds `close` waits for the sender to post what is queued: a server that answers
#: takes a few alerts in well under that, and a stopping session is not held longer.
CLOSE_WAIT_SECONDS = 5.0
#: The name of the sender thread.
SENDER_THREAD_NAME = "openalgo-live-alerts"

TEST_ALERT_TEXT = "Test alert from bt.live: the alerts of this session reach you here."

# Alerts of at least this priority are logged as warnings, the rest as information.
_WARNING_PRIORITY = 9
_REPEAT_WINDOW_NS = REPEAT_WINDOW_SECONDS * NANOS_PER_SECOND
_NO_USERNAME = "no OpenAlgo username was given, so alerts go to the log only"

logger = logging.getLogger(LOGGER_NAME)


class _TelegramClient(Protocol):
    """What the alerts need of an `openalgo.api` client: its Telegram call."""

    def telegram(self, *, username: str, message: str, priority: int, **kwargs: Any) -> Any:
        """The SDK's answer to `/api/v1/telegram/notify`."""


class Alerts:
    """The alerts of one live session (implements `AlertsLike`).

    `client` is an `openalgo.api` client; only `telegram()` is called, from the sender
    thread and from `test`. `username` is the OpenAlgo login username linked to the
    Telegram bot (not the Telegram handle), or None to send alerts to the log only.
    `mode` is the mode adopted at the session's start, `now` gives the present in UTC
    nanoseconds, and `strategies` names the session's strategies for the start of every
    alert. With `start_sender` False no thread is started and the queue is delivered by
    calling `deliver_queued` (how the tests drive it). Call `close` at the session's end.

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
        # A single name is refused rather than read as a sequence of its letters.
        names = () if isinstance(strategies, str) else tuple(strategies)
        if isinstance(strategies, str) or not all(isinstance(name, str) for name in names):
            raise ConfigError(
                f"strategies must be a list of strategy names, got {strategies!r}: pass "
                f"['ema_cross'] for one strategy"
            )
        self._client = client
        self._username = username
        self._now = now
        self._prefix = _prefix(mode, names)
        self._last_sent_ns: dict[tuple[AlertKind, str], int] = {}
        self._suppressed = 0
        self._queue = AlertQueue(QUEUE_SIZE)
        # The sender writes the failure counts while the loop may read them.
        self._counts = threading.Lock()
        self._failures = 0
        self._last_failure: str | None = None
        self._sender: threading.Thread | None = None
        if username is not None and start_sender:
            self._sender = threading.Thread(target=self._run, name=SENDER_THREAD_NAME, daemon=True)
            self._sender.start()

    @property
    def failures(self) -> int:
        """Alerts the sender posted and OpenAlgo did not take (the test alert is not
        counted)."""
        with self._counts:
            return self._failures

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
        """Alerts not sent because the same one was sent less than a minute before."""
        return self._suppressed

    @property
    def last_failure(self) -> str | None:
        """Why the last send or test alert failed, in words, or None when none failed."""
        with self._counts:
            return self._last_failure

    def send(self, kind: AlertKind, text: str) -> None:
        """Queue an alert of `kind`, its text starting with the mode and the strategies.

        Logged at once, as a warning from priority 9 up, and queued for the sender when a
        username was given; it never waits for OpenAlgo. The same kind and text again
        within `REPEAT_WINDOW_SECONDS` of the last time it was queued is counted in
        `suppressed`, logged at debug level only and not queued. Never raises: a `kind`
        that is not an `AlertKind` is sent as an ERROR alert.
        """
        if not isinstance(kind, AlertKind):
            logger.error("alert of unknown kind %r sent as an error alert", kind)
            kind = AlertKind.ERROR
        message = f"{self._prefix}{text}"
        now_ns = self._now()
        if self._repeated(kind, message, now_ns):
            self._suppressed += 1
            logger.debug("alert %s not sent again within a minute: %s", kind.value, message)
            return
        self._last_sent_ns[(kind, message)] = now_ns
        logger.log(_log_level(kind), "alert %s: %s", kind.value, message)
        if self._username is not None:
            self._enqueue(QueuedAlert(kind, message, ALERT_PRIORITY[kind]))

    def test(self) -> bool:
        """Send a test alert now, on the calling thread, and wait for OpenAlgo to say
        Telegram took it.

        True only when Telegram confirmed the delivery; False with no username, for an
        error answer, or when OpenAlgo only queued the message (the bot could not reach
        Telegram), with the reason in `last_failure`. The preflight requires it in LIVE
        mode and warns without it in SANDBOX mode. It waits as long as OpenAlgo waits on
        Telegram (within the client's timeout), so it belongs before the session, not in
        its loop. Bypasses the repeat window and the queue, and never raises.
        """
        message = f"{self._prefix}{TEST_ALERT_TEXT}"
        failure = _NO_USERNAME
        if self._username is not None:
            logger.info("test alert: %s", message)
            failure = self._deliver(AlertKind.SESSION, message, wait=True)
            if failure is None:
                return True
        with self._counts:
            self._last_failure = failure
        logger.warning("test alert not delivered: %s", failure)
        return False

    def deliver_queued(self) -> int:
        """Post every queued alert on the calling thread, oldest first; returns how many.
        The sender thread does this work: call it only on alerts made with
        `start_sender=False`."""
        posted = 0
        while (alert := self._queue.take(wait=False)) is not None:
            self._deliver_queued(alert)
            posted += 1
        return posted

    def close(self, wait_seconds: float = CLOSE_WAIT_SECONDS) -> None:
        """Stop taking alerts and give the sender up to `wait_seconds` to post what is
        queued; what it has not posted by then is dropped, counted and logged. Alerts sent
        afterwards are logged and dropped. Safe to call twice."""
        if self._queue.closed:
            return
        self._queue.close()
        if self._sender is not None:
            self._sender.join(wait_seconds)
        left = self._queue.abandon()
        if left:
            logger.warning(
                "%d alert(s) not sent: OpenAlgo had not taken them when the alerts closed", left
            )

    def _run(self) -> None:
        """The sender thread: post each alert as it is queued, until the queue is closed
        and empty or abandoned."""
        while (alert := self._queue.take(wait=True)) is not None:
            self._deliver_queued(alert)

    def _repeated(self, kind: AlertKind, message: str, now_ns: int) -> bool:
        """True when this alert was queued within the repeat window; forgets the alerts
        whose window has passed, so the record stays as small as a minute's alerts."""
        self._last_sent_ns = {
            key: sent_ns
            for key, sent_ns in self._last_sent_ns.items()
            if now_ns - sent_ns < _REPEAT_WINDOW_NS
        }
        return (kind, message) in self._last_sent_ns

    def _enqueue(self, alert: QueuedAlert) -> None:
        dropped = self._queue.put(alert)
        if dropped is None:
            return
        why = "the alerts are closed" if self._queue.closed else f"{QUEUE_SIZE} alerts wait"
        logger.warning("alert %s dropped (%s): %s", dropped.kind.value, why, dropped.message)

    def _deliver_queued(self, alert: QueuedAlert) -> None:
        failure = self._deliver(alert.kind, alert.message, wait=False)
        if failure is None:
            return
        with self._counts:
            self._failures += 1
            self._last_failure = failure
        logger.warning("alert %s not delivered: %s", alert.kind.value, failure)

    def _deliver(self, kind: AlertKind, message: str, *, wait: bool) -> str | None:
        """Post the alert; the reason it failed, or None. `wait` asks OpenAlgo to answer
        only once Telegram took the message."""
        extra = {"wait_for_delivery": True} if wait else {}
        try:
            answer = self._client.telegram(
                username=self._username,
                message=telegram_text(message),
                priority=ALERT_PRIORITY[kind],
                **extra,
            )
        # A client that raises must not stop the session: the reason is logged, counted
        # and kept in `last_failure`, never dropped.
        except Exception as error:
            return f"the client raised {type(error).__name__}: {error}"
        return delivery_failure(answer) if wait else failure_reason(answer)


def _prefix(mode: Mode, strategies: tuple[str, ...]) -> str:
    """ "SANDBOX: " or "LIVE (ema_cross, orb): "."""
    if not strategies:
        return f"{mode.name}: "
    return f"{mode.name} ({', '.join(strategies)}): "


def _log_level(kind: AlertKind) -> int:
    return logging.WARNING if ALERT_PRIORITY[kind] >= _WARNING_PRIORITY else logging.INFO


__all__ = [
    "ALERT_PRIORITY",
    "CLOSE_WAIT_SECONDS",
    "PRIORITY_HIGHEST",
    "PRIORITY_LOWEST",
    "QUEUE_SIZE",
    "REPEAT_WINDOW_SECONDS",
    "SENDER_THREAD_NAME",
    "TELEGRAM_MAX_CHARS",
    "TEST_ALERT_TEXT",
    "Alerts",
    "SessionLogs",
    "SessionSummary",
    "configure_session_logs",
    "summary_text",
]
