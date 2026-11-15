"""The fake OpenAlgo, the clock and the answers the alerts tests share.

The fake answers `telegram` as the SDK returns the server's answers
(restx_api/telegram_bot.py, SendNotification; openalgo/telegram.py): a 200 answer as the
server's body, a non-200 answer wrapped as "HTTP <code>: <body>", and a transport failure
as the SDK's own error. It fails the test if the alerts call anything else.
"""

from __future__ import annotations

import datetime as dt
import json
import threading

from openalgo.backtest._live._types import Mode
from openalgo.backtest._live.alerts import SENDER_THREAD_NAME, Alerts

# 2026-11-16 09:15:00 IST is 03:45:00 UTC.
SESSION_OPEN_NS = int(dt.datetime(2026, 11, 16, 3, 45, tzinfo=dt.UTC).timestamp()) * 10**9
SECOND_NS = 10**9
USERNAME = "trader1"
# The longest a test waits for the sender thread before it fails, in seconds; the sender
# answers in microseconds, so this is reached only when something is broken.
WAIT_SECONDS = 10.0

# The server's 200 answers (restx_api/telegram_bot.py, SendNotification).
QUEUED = {"status": "success", "message": "Notification queued for delivery"}
DELIVERED = {"status": "success", "message": "Notification sent successfully"}


def sdk_wrapper(code: int, body: str) -> dict:
    """A non-200 answer as the SDK's `telegram` returns it (openalgo/telegram.py,
    _handle_response)."""
    return {
        "status": "error",
        "message": f"HTTP {code}: {body}",
        "code": code,
        "error_type": "http_error",
    }


def server_error(code: int, message: str) -> dict:
    """A non-200 answer of the notify endpoint, wrapped by the SDK."""
    return sdk_wrapper(code, json.dumps({"status": "error", "message": message}) + "\n")


# Refusals: another try would be refused the same way.
MISSING_FIELD = server_error(400, "Username and message are required")
KEY_REFUSED = server_error(401, "Invalid or missing API key")
FORBIDDEN = server_error(403, "Invalid openalgo apikey")
NOT_LINKED = server_error(404, "User not found or not linked to Telegram")
BOT_STOPPED = server_error(409, "Telegram bot is stopped. Start the bot to send notifications.")
# Transient failures: the server's rate limiter, a server or gateway failure, and the
# SDK's own errors when no answer came back. The notify route's 429 is the REST layer's
# own body, with no status and no retry_after (checked with the server's libraries).
RATE_LIMITED = sdk_wrapper(429, json.dumps({"message": "30 per 1 minute"}) + "\n")


def rate_limited_after(seconds: float) -> dict:
    """A 429 that names its wait, in the shape of the server's application-wide 429
    handler (app.py), which adds `retry_after`."""
    body = {
        "status": "error",
        "message": "Rate limit exceeded. Please slow down your requests.",
        "retry_after": seconds,
    }
    return sdk_wrapper(429, json.dumps(body))


SERVER_DOWN = server_error(503, "Service Unavailable")
PROXY_PAGE = sdk_wrapper(502, "<html><body>Bad Gateway</body></html>")
BROKEN_OFF = {
    "status": "error",
    "message": "HTTP error occurred: Server disconnected without sending a response.",
    "error_type": "http_error",
}
UNREACHABLE = {
    "status": "error",
    "message": "Failed to connect to the server. Please check if the server is running.",
    "error_type": "connection_error",
}
TIMED_OUT = {
    "status": "error",
    "message": "Request timed out. The server took too long to respond.",
    "error_type": "timeout_error",
}
UNWRAPPED_ERROR = {"status": "error", "message": "Username and message are required"}


class ClientRaised(Exception):
    """What the fake client raises when told to."""


class Hold:
    """An answer the fake gives only once the test releases it, as a server that has not
    answered yet: `reached` is set when the call is waiting on it."""

    def __init__(self) -> None:
        self.reached = threading.Event()
        self.release = threading.Event()


class FakeOpenAlgo:
    """An `openalgo.api` client with only `telegram`: it records every call and gives
    the scripted answers in turn, then QUEUED. An exception in the script is raised, and
    a `Hold` is waited on before QUEUED is given."""

    def __init__(self, *answers: object) -> None:
        self.answers = list(answers)
        self.calls: list[dict] = []

    def telegram(self, *, username, message, priority=5, **kwargs):
        self.calls.append({"username": username, "message": message, "priority": priority})
        self.calls[-1].update(kwargs)
        answer = self.answers.pop(0) if self.answers else QUEUED
        if isinstance(answer, Hold):
            answer.reached.set()
            answer.release.wait(WAIT_SECONDS)
            return QUEUED
        if isinstance(answer, Exception):
            raise answer
        return answer

    def __getattr__(self, name):
        raise AssertionError(f"the alerts called client.{name}")

    @property
    def messages(self) -> list[str]:
        return [call["message"] for call in self.calls]


class ManualClock:
    """UTC nanoseconds that move only when a test moves them."""

    def __init__(self, ns: int = SESSION_OPEN_NS) -> None:
        self.ns = ns

    def __call__(self) -> int:
        return self.ns

    def advance(self, seconds: float) -> None:
        self.ns += round(seconds * SECOND_NS)


def make_alerts(client, clock, *, username=USERNAME, mode=Mode.SANDBOX, strategies=()):
    """Alerts with no sender thread: the test posts the queue with `deliver_queued`."""
    return Alerts(
        client, username=username, mode=mode, now=clock, strategies=strategies, start_sender=False
    )


def threaded_alerts(client, clock, *, username=USERNAME):
    """Alerts that start their sender thread on the first send."""
    return Alerts(client, username=username, mode=Mode.SANDBOX, now=clock)


def sender_threads_since(before: set[threading.Thread]) -> list[threading.Thread]:
    """The sender threads started since `before` was taken (`threading.enumerate()`)."""
    return [
        thread
        for thread in threading.enumerate()
        if thread not in before and thread.name == SENDER_THREAD_NAME
    ]
