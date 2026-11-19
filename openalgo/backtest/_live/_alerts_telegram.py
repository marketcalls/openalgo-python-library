"""What the alerts send to OpenAlgo's Telegram endpoint, and what its answers mean
(restx_api/telegram_bot.py, SendNotification; openalgo/telegram.py). `alerts` and its
sender use them.

OpenAlgo sends the text with Telegram's legacy Markdown parse mode
(services/telegram_alert_service.py, send_alert_sync), where the underscore, the
asterisk, the backtick and the opening bracket start formatting: the text between a
strategy named "ema_cross" and a symbol with an underscore would turn italic and lose
both underscores. The text is escaped so it arrives exactly as written, and cut to
Telegram's 4096-character limit (the full alert stays in the log).

An answer that failed is worth another try when it says nothing of the alert itself:
the rate limiter's HTTP 429, a server or gateway failure from HTTP 500 on, the client's
own timeout, or a connection refused or broken off. Any other failure is a refusal that
another try would repeat: a bad field (400), the key (401, 403), a user not linked
(404), the bot stopped (409).

The 429 of the notify route is the REST layer's own answer, `{"message": "30 per 1
minute"}`, with no `status` and no `retry_after` (the server's application-wide 429
handler, which adds `"retry_after": 60`, does not answer for its REST routes), so the
SDK passes it on as "HTTP 429: <that body>". A `retry_after` is read when an answer has
one; the sender decides how long to wait.

A failure reason is one line: the message is stripped and its runs of whitespace
collapsed, so a body ending in a newline or an HTML page reads on one log line.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._answers import answered_by_server, server_answer

#: The longest text Telegram's sendMessage takes, in characters (Telegram Bot API; also
#: in the SDK's `telegram` docstring).
TELEGRAM_MAX_CHARS = 4096

# No Markdown character in it, so a cut text is exactly the limit long.
_CUT_MARKER = "\n(cut to fit Telegram: the full alert is in the session log)"
# The characters that start formatting in Telegram's legacy Markdown, escaped with a
# backslash to be shown as written (Telegram Bot API, "Markdown style").
_MARKDOWN_SPECIAL = re.compile(r"([_*`\[])")
# What the notify endpoint answers when it waited for Telegram and Telegram took the
# message (SendNotification, wait_for_delivery). Without the wait, or when Telegram
# refused, it answers "Notification queued for delivery".
_DELIVERED_MESSAGE = "Notification sent successfully"
# The longest message quoted in a failure reason, in characters: an HTML error page of a
# proxy, which the SDK quotes whole, says nothing more past its first line.
_MAX_REASON_CHARS = 200
_SUCCESS = "success"
# The rate limiter's status, and the first status of a server or gateway failure.
_RATE_LIMITED = 429
_FIRST_SERVER_ERROR = 500
# The SDK's error types of a post that got no answer: its own timeout and a refused
# connection; its "http_error" with no status is a connection that broke off
# (openalgo/telegram.py, _make_request).
_NO_ANSWER_ERROR_TYPES = frozenset({"timeout_error", "connection_error"})
_HTTP_ERROR_TYPE = "http_error"


@dataclass(frozen=True)
class PostResult:
    """What became of one post. `reason` is None when OpenAlgo took it, else why not in
    words; `transient` is True when another try may succeed, `rate_limited` when the
    answer was the rate limiter's 429, and `retry_after_seconds` the wait that answer
    asked for, when it gave one (the notify route's own 429 gives none)."""

    reason: str | None
    transient: bool = False
    rate_limited: bool = False
    retry_after_seconds: float | None = None

    @property
    def delivered(self) -> bool:
        """True when OpenAlgo took the post."""
        return self.reason is None


def telegram_text(message: str) -> str:
    """The message cut to Telegram's limit and escaped for its legacy Markdown. The limit
    counts the text after parsing, so the escapes do not count against it."""
    if len(message) > TELEGRAM_MAX_CHARS:
        message = message[: TELEGRAM_MAX_CHARS - len(_CUT_MARKER)] + _CUT_MARKER
    return _MARKDOWN_SPECIAL.sub(r"\\\1", message)


def read_post(answer: Any) -> PostResult:
    """What an answer of `client.telegram` says became of the post, by the rules of the
    module docstring."""
    reason = failure_reason(answer)
    if reason is None or not isinstance(answer, Mapping):
        return PostResult(reason)
    answer = server_answer(answer)
    if not _transient(answer):
        return PostResult(reason)
    if answer.get("code") != _RATE_LIMITED:
        return PostResult(reason, transient=True)
    return PostResult(
        reason, transient=True, rate_limited=True, retry_after_seconds=_retry_after(answer)
    )


def failure_reason(answer: Any) -> str | None:
    """Why an answer of `client.telegram` is not a success, or None for a success.

    The reason is the server's own message, with its HTTP status when the SDK wrapped it
    (`_answers.server_answer`); otherwise the SDK's message, which for a reply that is
    not OpenAlgo's JSON already reads "HTTP <code>: <text>". It is one line, shortened.
    """
    if not isinstance(answer, Mapping):
        return f"OpenAlgo's answer was not a JSON object: {answer!r}"
    answer = server_answer(answer)
    if answer.get("status") == _SUCCESS:
        return None
    message = answer.get("message")
    message = " ".join(message.split()) if isinstance(message, str) else ""
    if not message:
        message = "an error without a message"
    elif len(message) > _MAX_REASON_CHARS:
        message = message[:_MAX_REASON_CHARS] + "..."
    code = answer.get("code")
    if answered_by_server(answer) and code is not None:
        return f"{message} (HTTP {code})"
    return message


def delivery_failure(answer: Any) -> str | None:
    """Why an answer to a post that waited for Telegram does not confirm delivery, or
    None when Telegram took the message."""
    reason = failure_reason(answer)
    if reason is None and answer.get("message") != _DELIVERED_MESSAGE:
        return (
            "OpenAlgo accepted the alert but Telegram did not confirm delivery (it is "
            "queued for retry): check the bot token and that OpenAlgo can reach Telegram"
        )
    return reason


def _transient(answer: Mapping[str, Any]) -> bool:
    code = answer.get("code")
    if isinstance(code, int) and not isinstance(code, bool):
        return code == _RATE_LIMITED or code >= _FIRST_SERVER_ERROR
    error_type = answer.get("error_type")
    return error_type in _NO_ANSWER_ERROR_TYPES or error_type == _HTTP_ERROR_TYPE


def _retry_after(answer: Mapping[str, Any]) -> float | None:
    """The seconds a 429 asked to wait, when it gave a positive finite number."""
    value = answer.get("retry_after")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return float(value)


__all__ = [
    "TELEGRAM_MAX_CHARS",
    "PostResult",
    "delivery_failure",
    "failure_reason",
    "read_post",
    "telegram_text",
]
