"""What the alerts send to OpenAlgo's Telegram endpoint, and what its answers mean
(restx_api/telegram_bot.py, SendNotification; openalgo/telegram.py). `alerts` uses them.

OpenAlgo sends the text with Telegram's legacy Markdown parse mode
(services/telegram_alert_service.py, send_alert_sync), where the underscore, the
asterisk, the backtick and the opening bracket start formatting: the text between a
strategy named "ema_cross" and a symbol with an underscore would turn italic and lose
both underscores. The text is escaped so it arrives exactly as written, and cut to
Telegram's 4096-character limit (the full alert stays in the log).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
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


def telegram_text(message: str) -> str:
    """The message cut to Telegram's limit and escaped for its legacy Markdown. The limit
    counts the text after parsing, so the escapes do not count against it."""
    if len(message) > TELEGRAM_MAX_CHARS:
        message = message[: TELEGRAM_MAX_CHARS - len(_CUT_MARKER)] + _CUT_MARKER
    return _MARKDOWN_SPECIAL.sub(r"\\\1", message)


def failure_reason(answer: Any) -> str | None:
    """Why an answer of `client.telegram` is not a success, or None for a success.

    The reason is the server's own message, with its HTTP status when the SDK wrapped it
    (`_answers.server_answer`); otherwise the SDK's message, which for a reply that is
    not OpenAlgo's JSON already reads "HTTP <code>: <text>".
    """
    if not isinstance(answer, Mapping):
        return f"OpenAlgo's answer was not a JSON object: {answer!r}"
    answer = server_answer(answer)
    if answer.get("status") == _SUCCESS:
        return None
    message = answer.get("message")
    if not isinstance(message, str) or not message.strip():
        message = "an error without a message"
    elif len(message) > _MAX_REASON_CHARS:
        message = message[:_MAX_REASON_CHARS] + "..."
    code = answer.get("code")
    if answered_by_server(answer) and code is not None:
        return f"{message} (HTTP {code})"
    return message


def delivery_failure(answer: Any) -> str | None:
    """Why an answer to a send that waited for Telegram does not confirm delivery, or
    None when Telegram took the message."""
    reason = failure_reason(answer)
    if reason is None and answer.get("message") != _DELIVERED_MESSAGE:
        return (
            "OpenAlgo accepted the alert but Telegram did not confirm delivery (it is "
            "queued for retry): check the bot token and that OpenAlgo can reach Telegram"
        )
    return reason


__all__ = ["TELEGRAM_MAX_CHARS", "delivery_failure", "failure_reason", "telegram_text"]
