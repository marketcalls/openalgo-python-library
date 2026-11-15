"""Alerts of a live session through OpenAlgo's Telegram bot, the daily summary and the
session logs (08-live-trading section 8).

`Alerts.send` posts one alert to `/api/v1/telegram/notify` through the SDK's
`client.telegram`. Every alert starts with the mode OpenAlgo is in, in words ("SANDBOX"
or "LIVE"), and the session's strategies when known, so a message read on a phone cannot
be taken for the other mode. Each kind of alert has a priority (`ALERT_PRIORITY`); a mode
breach, a dead feed, a reconciliation mismatch and an error are the most urgent.

An alert never raises into the session loop. A send that fails (an error answer, a
transport failure, an exception from the client) is logged and counted in `failures`.
The same alert is sent at most once in `REPEAT_WINDOW_SECONDS` by the session's clock,
counted in `suppressed`, so a condition the loop reports on every tick reaches the phone
once a minute. Without an OpenAlgo username alerts go to the log only.

OpenAlgo sends the text with Telegram's legacy Markdown parse mode
(services/telegram_alert_service.py, send_alert_sync), where the underscore, the
asterisk, the backtick and the opening bracket start formatting: the text between a
strategy named "ema_cross" and a symbol with an underscore would turn italic and lose
both underscores. The text is escaped so it arrives exactly as written, and cut to
Telegram's 4096-character limit (the full alert stays in the log).

`SessionSummary` and `summary_text` (in `_alerts_summary`) and `SessionLogs` and
`configure_session_logs` (in `_alerts_logs`) are re-exported here.

Not thread-safe: the session loop owns its alerts, as it owns the engine.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Protocol

from .._layout import NANOS_PER_SECOND
from ..errors import ConfigError
from ._alerts_logs import SessionLogs, configure_session_logs
from ._alerts_summary import SessionSummary, summary_text
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
#: The longest text Telegram's sendMessage takes, in characters (Telegram Bot API; also
#: in the SDK's `telegram` docstring).
TELEGRAM_MAX_CHARS = 4096

TEST_ALERT_TEXT = "Test alert from bt.live: the alerts of this session reach you here."

# Alerts of at least this priority are logged as warnings, the rest as information.
_WARNING_PRIORITY = 9
_REPEAT_WINDOW_NS = REPEAT_WINDOW_SECONDS * NANOS_PER_SECOND
# No Markdown character in it, so a cut text is exactly the limit long.
_CUT_MARKER = "\n(cut to fit Telegram: the full alert is in the session log)"
# The characters that start formatting in Telegram's legacy Markdown, escaped with a
# backslash to be shown as written (Telegram Bot API, "Markdown style").
_MARKDOWN_SPECIAL = re.compile(r"([_*`\[])")
# The SDK's wrapper of a non-200 answer: "HTTP <code>: <body>" (openalgo/telegram.py,
# _handle_response); the body is the server's JSON answer.
_HTTP_WRAPPER = re.compile(r"HTTP (\d{3}): (.*)", re.DOTALL)
# What the notify endpoint answers when it waited for Telegram and Telegram took the
# message (restx_api/telegram_bot.py, SendNotification, wait_for_delivery). Without the
# wait, or when Telegram refused, it answers "Notification queued for delivery".
_DELIVERED_MESSAGE = "Notification sent successfully"
# The longest server body quoted in a failure reason, in characters: an HTML error page
# of a proxy says nothing more past its first line.
_MAX_REASON_CHARS = 200
_SUCCESS = "success"
_NO_USERNAME = "no OpenAlgo username was given, so alerts go to the log only"

logger = logging.getLogger(LOGGER_NAME)


class _TelegramClient(Protocol):
    """What the alerts need of an `openalgo.api` client: its Telegram call."""

    def telegram(self, *, username: str, message: str, priority: int, **kwargs: Any) -> Any:
        """The SDK's answer to `/api/v1/telegram/notify`."""


class Alerts:
    """The alerts of one live session (implements `AlertsLike`).

    `client` is an `openalgo.api` client; only `telegram()` is called. `username` is the
    OpenAlgo login username linked to the Telegram bot (not the Telegram handle), or None
    to send alerts to the log only. `mode` is the mode adopted at the session's start,
    `now` gives the present in UTC nanoseconds, and `strategies` names the session's
    strategies for the start of every alert.

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
        self._failures = 0
        self._suppressed = 0
        self._last_failure: str | None = None

    @property
    def failures(self) -> int:
        """Sends that failed, the test alert's included."""
        return self._failures

    @property
    def suppressed(self) -> int:
        """Alerts not sent because the same one was sent less than a minute before."""
        return self._suppressed

    @property
    def last_failure(self) -> str | None:
        """Why the last send or test alert failed, in words, or None when none failed."""
        return self._last_failure

    def send(self, kind: AlertKind, text: str) -> None:
        """Send an alert of `kind`, its text starting with the mode and the strategies.

        Logged, as a warning from priority 9 up, and sent through Telegram when a username
        was given. The same kind and text again within `REPEAT_WINDOW_SECONDS` of the last
        time it went out is counted in `suppressed`, logged at debug level only and not
        sent. Never raises: a failed send is logged and counted in `failures`, and a `kind`
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
            self._note(kind, self._deliver(kind, message, wait=False))

    def test(self) -> bool:
        """Send a test alert and wait for OpenAlgo to say Telegram took it.

        True only when Telegram confirmed the delivery; False with no username, for an
        error answer, or when OpenAlgo only queued the message (the bot could not reach
        Telegram), with the reason in `last_failure`. The preflight requires it in LIVE
        mode and warns without it in SANDBOX mode. It waits as long as OpenAlgo waits on
        Telegram (within the client's timeout), so it belongs before the session, not in
        its loop. Bypasses the repeat window and never raises.
        """
        message = f"{self._prefix}{TEST_ALERT_TEXT}"
        if self._username is None:
            self._last_failure = _NO_USERNAME
            logger.warning("test alert not sent: %s", _NO_USERNAME)
            return False
        logger.info("test alert: %s", message)
        failure = self._deliver(AlertKind.SESSION, message, wait=True)
        self._note(AlertKind.SESSION, failure)
        return failure is None

    def _repeated(self, kind: AlertKind, message: str, now_ns: int) -> bool:
        """True when this alert went out within the repeat window; forgets the alerts
        whose window has passed, so the record stays as small as a minute's alerts."""
        self._last_sent_ns = {
            key: sent_ns
            for key, sent_ns in self._last_sent_ns.items()
            if now_ns - sent_ns < _REPEAT_WINDOW_NS
        }
        return (kind, message) in self._last_sent_ns

    def _deliver(self, kind: AlertKind, message: str, *, wait: bool) -> str | None:
        """Post the alert; the reason it failed, or None. `wait` asks OpenAlgo to answer
        only once Telegram took the message."""
        extra = {"wait_for_delivery": True} if wait else {}
        try:
            answer = self._client.telegram(
                username=self._username,
                message=_telegram_text(message),
                priority=ALERT_PRIORITY[kind],
                **extra,
            )
        # A client that raises must not stop the session: the reason is logged, counted
        # and kept in `last_failure`, never dropped.
        except Exception as error:
            return f"the client raised {type(error).__name__}: {error}"
        reason = _failure_reason(answer)
        if reason is None and wait and answer.get("message") != _DELIVERED_MESSAGE:
            return (
                "OpenAlgo accepted the alert but Telegram did not confirm delivery (it is "
                "queued for retry): check the bot token and that OpenAlgo can reach Telegram"
            )
        return reason

    def _note(self, kind: AlertKind, failure: str | None) -> None:
        if failure is None:
            return
        self._failures += 1
        self._last_failure = failure
        logger.warning("alert %s not delivered: %s", kind.value, failure)


def _prefix(mode: Mode, strategies: tuple[str, ...]) -> str:
    """ "SANDBOX: " or "LIVE (ema_cross, orb): "."""
    if not strategies:
        return f"{mode.name}: "
    return f"{mode.name} ({', '.join(strategies)}): "


def _log_level(kind: AlertKind) -> int:
    return logging.WARNING if ALERT_PRIORITY[kind] >= _WARNING_PRIORITY else logging.INFO


def _telegram_text(message: str) -> str:
    """The message cut to Telegram's limit and escaped for its legacy Markdown. The limit
    counts the text after parsing, so the escapes do not count against it."""
    if len(message) > TELEGRAM_MAX_CHARS:
        message = message[: TELEGRAM_MAX_CHARS - len(_CUT_MARKER)] + _CUT_MARKER
    return _MARKDOWN_SPECIAL.sub(r"\\\1", message)


def _failure_reason(answer: Any) -> str | None:
    """Why an answer of `client.telegram` is not a success, or None for a success.

    The SDK passes the server's own error answer on, wraps a non-200 answer as
    `{"status": "error", "message": "HTTP <code>: <body>", "code": <code>}`, and reports
    a transport failure as an error with its own message. The reason is the server's
    message, with the HTTP status when there is one.
    """
    if not isinstance(answer, Mapping):
        return f"OpenAlgo's answer was not a JSON object: {answer!r}"
    if answer.get("status") == _SUCCESS:
        return None
    message = answer.get("message")
    if not isinstance(message, str) or not message.strip():
        message = "an error without a message"
    wrapped = _HTTP_WRAPPER.fullmatch(message)
    if wrapped is None:
        code = answer.get("code")
        return f"{message} (HTTP {code})" if code is not None else message
    return f"{_server_message(wrapped.group(2))} (HTTP {wrapped.group(1)})"


def _server_message(body: str) -> str:
    """The `message` of the server's JSON body, or the body itself, shortened."""
    try:
        parsed = json.loads(body)
    except ValueError:
        parsed = None
    if isinstance(parsed, Mapping) and isinstance(parsed.get("message"), str):
        return parsed["message"]
    text = body.strip() or "an empty answer"
    return text if len(text) <= _MAX_REASON_CHARS else text[:_MAX_REASON_CHARS] + "..."


__all__ = [
    "ALERT_PRIORITY",
    "PRIORITY_HIGHEST",
    "PRIORITY_LOWEST",
    "REPEAT_WINDOW_SECONDS",
    "TELEGRAM_MAX_CHARS",
    "TEST_ALERT_TEXT",
    "Alerts",
    "SessionLogs",
    "SessionSummary",
    "configure_session_logs",
    "summary_text",
]
