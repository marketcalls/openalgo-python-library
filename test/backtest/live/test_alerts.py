"""The alerts of a live session through OpenAlgo's Telegram bot: a priority per kind, the
mode and the strategies at the start of every alert, a failed send that never raises, the
one-minute repeat window, the log-only mode without a username, and the test alert
(08-live-trading sections 4 and 8).

The fake OpenAlgo below answers `telegram` as the SDK returns the server's answers
(restx_api/telegram_bot.py, SendNotification; openalgo/telegram.py): a 200 answer as the
server's body, a non-200 answer wrapped as "HTTP <code>: <body>", and a transport failure
as the SDK's own error. It fails the test if the alerts call anything else.
"""

from __future__ import annotations

import datetime as dt
import json
import logging

import pytest

from openalgo.backtest._live._types import LOGGER_NAME, AlertKind, Mode
from openalgo.backtest._live.alerts import (
    ALERT_PRIORITY,
    PRIORITY_HIGHEST,
    PRIORITY_LOWEST,
    TELEGRAM_MAX_CHARS,
    Alerts,
)
from openalgo.backtest.errors import ConfigError

# 2026-11-16 09:15:00 IST is 03:45:00 UTC.
SESSION_OPEN_NS = int(dt.datetime(2026, 11, 16, 3, 45, tzinfo=dt.UTC).timestamp()) * 10**9
SECOND_NS = 10**9
USERNAME = "trader1"

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


NOT_LINKED = server_error(404, "User not found or not linked to Telegram")
BOT_STOPPED = server_error(409, "Telegram bot is stopped. Start the bot to send notifications.")
PROXY_PAGE = sdk_wrapper(502, "<html><body>Bad Gateway</body></html>")
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


class FakeOpenAlgo:
    """An `openalgo.api` client with only `telegram`: it records every call and gives
    the scripted answers in turn, then QUEUED. An exception in the script is raised."""

    def __init__(self, *answers: object) -> None:
        self.answers = list(answers)
        self.calls: list[dict] = []

    def telegram(self, *, username, message, priority=5, **kwargs):
        self.calls.append({"username": username, "message": message, "priority": priority})
        self.calls[-1].update(kwargs)
        answer = self.answers.pop(0) if self.answers else QUEUED
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


@pytest.fixture
def clock():
    return ManualClock()


@pytest.fixture
def client():
    return FakeOpenAlgo()


def make_alerts(client, clock, *, username=USERNAME, mode=Mode.SANDBOX, strategies=()):
    return Alerts(client, username=username, mode=mode, now=clock, strategies=strategies)


@pytest.fixture
def alerts(client, clock):
    return make_alerts(client, clock)


# Priorities


def test_every_kind_of_alert_has_a_priority_openalgo_accepts():
    assert set(ALERT_PRIORITY) == set(AlertKind)
    assert all(
        PRIORITY_LOWEST <= priority <= PRIORITY_HIGHEST for priority in ALERT_PRIORITY.values()
    )


@pytest.mark.parametrize(
    ("kind", "priority"),
    [
        (AlertKind.MODE_BREACH, 10),
        (AlertKind.FEED_DEAD, 10),
        (AlertKind.RECONCILE_MISMATCH, 10),
        (AlertKind.ERROR, 10),
        (AlertKind.REJECT, 9),
        (AlertKind.ENTRY, 7),
        (AlertKind.EXIT, 7),
        (AlertKind.STOP_MOVED, 6),
        (AlertKind.SESSION, 5),
        (AlertKind.DAILY_SUMMARY, 5),
    ],
    ids=lambda value: value.value if isinstance(value, AlertKind) else str(value),
)
def test_an_alert_is_sent_at_the_priority_of_its_kind(alerts, client, kind, priority):
    alerts.send(kind, "something happened")
    assert client.calls[0]["priority"] == priority


def test_the_alert_goes_to_the_openalgo_username_given(alerts, client):
    alerts.send(AlertKind.ENTRY, "Bought 10 SBIN")
    assert client.calls[0]["username"] == USERNAME


def test_an_ordinary_alert_does_not_wait_for_telegram(alerts, client):
    alerts.send(AlertKind.ENTRY, "Bought 10 SBIN")
    assert "wait_for_delivery" not in client.calls[0]


# The start of every alert


def test_a_sandbox_alert_starts_with_the_mode_in_words(alerts, client):
    alerts.send(AlertKind.ENTRY, "Bought 10 SBIN at 812.40")
    assert client.messages == ["SANDBOX: Bought 10 SBIN at 812.40"]


def test_a_live_alert_names_the_session_strategies_after_the_mode(client, clock):
    alerts = make_alerts(client, clock, mode=Mode.LIVE, strategies=["orb", "breakout"])
    alerts.send(AlertKind.EXIT, "Sold 10 SBIN at 815.00")
    assert client.messages == ["LIVE (orb, breakout): Sold 10 SBIN at 815.00"]


def test_markdown_characters_are_escaped_so_the_text_arrives_as_written(client, clock):
    alerts = make_alerts(client, clock, strategies=["ema_cross"])
    alerts.send(AlertKind.ERROR, "NIFTY_INDEX *halted* `code` [see log]")
    assert client.messages == [r"SANDBOX (ema\_cross): NIFTY\_INDEX \*halted\* \`code\` \[see log]"]


def test_an_alert_longer_than_telegram_takes_is_cut_with_a_note(alerts, client):
    alerts.send(AlertKind.ERROR, "x" * 5000)
    sent = client.messages[0]
    assert len(sent) == TELEGRAM_MAX_CHARS
    assert sent.startswith("SANDBOX: xxx")
    assert sent.endswith("x\n(cut to fit Telegram: the full alert is in the session log)")


def test_an_alert_cut_for_telegram_is_logged_whole(alerts, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    alerts.send(AlertKind.ERROR, "x" * 5000)
    assert f"alert error: SANDBOX: {'x' * 5000}" in caplog.messages


# A failed send

FAILED_SENDS = [
    pytest.param(
        NOT_LINKED, "User not found or not linked to Telegram (HTTP 404)", id="not linked"
    ),
    pytest.param(
        BOT_STOPPED,
        "Telegram bot is stopped. Start the bot to send notifications. (HTTP 409)",
        id="bot stopped",
    ),
    pytest.param(PROXY_PAGE, "<html><body>Bad Gateway</body></html> (HTTP 502)", id="proxy page"),
    pytest.param(UNWRAPPED_ERROR, "Username and message are required", id="error answer"),
    pytest.param(UNREACHABLE, UNREACHABLE["message"], id="unreachable"),
    pytest.param(TIMED_OUT, TIMED_OUT["message"], id="timed out"),
    pytest.param(
        ClientRaised("socket closed"),
        "the client raised ClientRaised: socket closed",
        id="client raised",
    ),
    pytest.param(None, "OpenAlgo's answer was not a JSON object: None", id="no JSON object"),
]


@pytest.mark.parametrize(("answer", "reason"), FAILED_SENDS)
def test_a_failed_send_does_not_raise_and_is_counted(clock, answer, reason):
    alerts = make_alerts(FakeOpenAlgo(answer), clock)
    alerts.send(AlertKind.FEED_DEAD, "no tick for 30 s")
    assert alerts.failures == 1


@pytest.mark.parametrize(("answer", "reason"), FAILED_SENDS)
def test_a_failed_send_keeps_the_reason_in_the_servers_words(clock, answer, reason):
    alerts = make_alerts(FakeOpenAlgo(answer), clock)
    alerts.send(AlertKind.FEED_DEAD, "no tick for 30 s")
    assert alerts.last_failure == reason


def test_a_failed_send_is_logged_as_a_warning_with_its_reason(clock, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    alerts = make_alerts(FakeOpenAlgo(NOT_LINKED), clock)
    alerts.send(AlertKind.ENTRY, "Bought 10 SBIN")
    warnings = [
        record.getMessage() for record in caplog.records if record.levelno == logging.WARNING
    ]
    assert warnings == [
        "alert entry not delivered: User not found or not linked to Telegram (HTTP 404)"
    ]


def test_a_long_server_body_is_shortened_in_the_reason(clock):
    alerts = make_alerts(FakeOpenAlgo(sdk_wrapper(502, "y" * 500)), clock)
    alerts.send(AlertKind.ERROR, "the order path failed")
    assert alerts.last_failure == "y" * 200 + "... (HTTP 502)"


def test_a_delivered_alert_counts_no_failure(alerts):
    alerts.send(AlertKind.ENTRY, "Bought 10 SBIN")
    assert (alerts.failures, alerts.last_failure) == (0, None)


def test_an_alert_of_unknown_kind_is_sent_as_an_error_alert(alerts, client):
    alerts.send("entry", "Bought 10 SBIN")
    assert client.calls[0]["priority"] == ALERT_PRIORITY[AlertKind.ERROR]


# The repeat window


def test_the_same_alert_within_a_minute_is_not_sent_again(alerts, client, clock):
    alerts.send(AlertKind.FEED_DEAD, "no tick for 30 s")
    clock.advance(59.999)
    alerts.send(AlertKind.FEED_DEAD, "no tick for 30 s")
    assert len(client.calls) == 1
    assert alerts.suppressed == 1


def test_the_same_alert_is_sent_again_a_minute_after_it_went_out(alerts, client, clock):
    alerts.send(AlertKind.FEED_DEAD, "no tick for 30 s")
    clock.advance(60)
    alerts.send(AlertKind.FEED_DEAD, "no tick for 30 s")
    assert len(client.calls) == 2
    assert alerts.suppressed == 0


def test_the_minute_counts_from_the_last_send_not_the_last_repeat(alerts, client, clock):
    for _ in range(3):
        alerts.send(AlertKind.SESSION, "still not flat")
        clock.advance(30)
    assert len(client.calls) == 2
    assert alerts.suppressed == 1


def test_another_text_within_a_minute_is_sent(alerts, client, clock):
    alerts.send(AlertKind.ENTRY, "Bought 10 SBIN")
    clock.advance(1)
    alerts.send(AlertKind.ENTRY, "Bought 5 INFY")
    assert len(client.calls) == 2


def test_the_same_text_of_another_kind_within_a_minute_is_sent(alerts, client):
    alerts.send(AlertKind.ERROR, "SBIN: order refused")
    alerts.send(AlertKind.REJECT, "SBIN: order refused")
    assert len(client.calls) == 2


def test_a_failed_send_is_not_tried_again_within_a_minute(clock):
    client = FakeOpenAlgo(UNREACHABLE, UNREACHABLE)
    alerts = make_alerts(client, clock)
    alerts.send(AlertKind.FEED_DEAD, "no tick for 30 s")
    clock.advance(10)
    alerts.send(AlertKind.FEED_DEAD, "no tick for 30 s")
    assert (len(client.calls), alerts.failures, alerts.suppressed) == (1, 1, 1)


def test_a_suppressed_repeat_is_not_logged_as_an_alert(alerts, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    alerts.send(AlertKind.FEED_DEAD, "no tick for 30 s")
    alerts.send(AlertKind.FEED_DEAD, "no tick for 30 s")
    assert caplog.messages == ["alert feed_dead: SANDBOX: no tick for 30 s"]


# The log


@pytest.mark.parametrize(
    ("kind", "level"),
    [
        (AlertKind.MODE_BREACH, logging.WARNING),
        (AlertKind.REJECT, logging.WARNING),
        (AlertKind.ENTRY, logging.INFO),
        (AlertKind.DAILY_SUMMARY, logging.INFO),
    ],
    ids=["mode breach", "reject", "entry", "daily summary"],
)
def test_an_urgent_alert_is_logged_as_a_warning_and_the_rest_as_information(
    alerts, caplog, kind, level
):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    alerts.send(kind, "something happened")
    assert [(record.levelno, record.getMessage()) for record in caplog.records] == [
        (level, f"alert {kind.value}: SANDBOX: something happened")
    ]


def test_without_a_username_alerts_go_to_the_log_only(client, clock, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    alerts = make_alerts(client, clock, username=None, mode=Mode.LIVE)
    alerts.send(AlertKind.ENTRY, "Bought 10 SBIN")
    assert client.calls == []
    assert caplog.messages == ["alert entry: LIVE: Bought 10 SBIN"]


def test_without_a_username_an_alert_is_no_failure(client, clock):
    alerts = make_alerts(client, clock, username=None)
    alerts.send(AlertKind.ENTRY, "Bought 10 SBIN")
    assert (alerts.failures, alerts.last_failure) == (0, None)


# The test alert


def test_the_test_alert_is_delivered_when_telegram_confirms_it(clock):
    alerts = make_alerts(FakeOpenAlgo(DELIVERED), clock)
    assert alerts.test() is True


def test_the_test_alert_asks_openalgo_to_wait_for_telegram(clock):
    client = FakeOpenAlgo(DELIVERED)
    make_alerts(client, clock).test()
    assert client.calls[0]["wait_for_delivery"] is True


def test_the_test_alert_starts_with_the_mode_and_the_strategies(clock):
    client = FakeOpenAlgo(DELIVERED)
    make_alerts(client, clock, mode=Mode.LIVE, strategies=["orb"]).test()
    assert client.messages == [
        "LIVE (orb): Test alert from bt.live: the alerts of this session reach you here."
    ]


def test_a_test_alert_openalgo_only_queued_is_not_delivered(clock):
    alerts = make_alerts(FakeOpenAlgo(QUEUED), clock)
    assert alerts.test() is False
    assert alerts.last_failure == (
        "OpenAlgo accepted the alert but Telegram did not confirm delivery (it is queued "
        "for retry): check the bot token and that OpenAlgo can reach Telegram"
    )


@pytest.mark.parametrize(("answer", "reason"), FAILED_SENDS)
def test_a_test_alert_that_failed_is_not_delivered_and_says_why(clock, answer, reason):
    alerts = make_alerts(FakeOpenAlgo(answer), clock)
    assert alerts.test() is False
    assert (alerts.failures, alerts.last_failure) == (1, reason)


def test_without_a_username_the_test_alert_is_not_delivered(client, clock):
    alerts = make_alerts(client, clock, username=None)
    assert alerts.test() is False
    assert client.calls == []
    assert alerts.last_failure == "no OpenAlgo username was given, so alerts go to the log only"


def test_the_test_alert_is_sent_again_within_a_minute(clock):
    client = FakeOpenAlgo(QUEUED, DELIVERED)
    alerts = make_alerts(client, clock)
    alerts.test()
    assert alerts.test() is True
    assert len(client.calls) == 2


# Arguments


@pytest.mark.parametrize("username", ["", "   ", 42], ids=["empty", "blank", "a number"])
def test_a_username_that_is_not_a_name_is_refused(client, clock, username):
    with pytest.raises(ConfigError, match="give your OpenAlgo login username"):
        make_alerts(client, clock, username=username)


def test_a_mode_that_is_not_a_mode_is_refused(client, clock):
    with pytest.raises(ConfigError, match="Mode.SANDBOX or Mode.LIVE"):
        make_alerts(client, clock, mode="sandbox")


def test_a_single_strategy_name_is_refused_rather_than_read_letter_by_letter(client, clock):
    with pytest.raises(ConfigError, match=r"pass \['ema_cross'\] for one strategy"):
        make_alerts(client, clock, strategies="ema_cross")
