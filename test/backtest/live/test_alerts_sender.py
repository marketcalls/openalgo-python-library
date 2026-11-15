"""The queue and the sender thread of the alerts: `send` never waits for OpenAlgo, a full
queue drops the oldest alert of the lowest priority, and `close` posts what is queued,
gives up on a server that does not answer, and is safe to call twice.

The queue tests start no thread and post with `deliver_queued`. The thread tests start
the real sender and wait only on events the fake OpenAlgo sets, never on a timer.
"""

from __future__ import annotations

import logging

import pytest
from alerts_samples import (
    UNREACHABLE,
    WAIT_SECONDS,
    FakeOpenAlgo,
    Hold,
    ManualClock,
    make_alerts,
    started_alerts,
)

from openalgo.backtest._live._types import LOGGER_NAME, AlertKind
from openalgo.backtest._live.alerts import QUEUE_SIZE


@pytest.fixture
def clock():
    return ManualClock()


@pytest.fixture
def client():
    return FakeOpenAlgo()


@pytest.fixture
def alerts(client, clock):
    return make_alerts(client, clock)


def fill(alerts, kind: AlertKind, count: int, start: int = 0) -> None:
    """Queue `count` alerts of `kind` with distinct texts."""
    for number in range(start, start + count):
        alerts.send(kind, f"{kind.value} {number}")


def finish(alerts, threads) -> None:
    """Close the alerts and wait for their sender to end."""
    alerts.close()
    for thread in threads:
        thread.join(WAIT_SECONDS)
        assert not thread.is_alive()


# The queue


def test_an_alert_waits_in_the_queue_until_the_sender_posts_it(alerts, client):
    alerts.send(AlertKind.ENTRY, "Bought 10 SBIN")
    assert client.calls == []
    assert alerts.deliver_queued() == 1
    assert client.messages == ["SANDBOX: Bought 10 SBIN"]


def test_queued_alerts_are_posted_oldest_first(alerts, client):
    alerts.send(AlertKind.SESSION, "session started")
    alerts.send(AlertKind.MODE_BREACH, "the analyzer switch changed")
    alerts.deliver_queued()
    assert client.messages == ["SANDBOX: session started", "SANDBOX: the analyzer switch changed"]


def test_a_full_queue_drops_the_oldest_alert_of_the_lowest_priority(alerts, client):
    alerts.send(AlertKind.ENTRY, "Bought 10 SBIN")
    fill(alerts, AlertKind.SESSION, QUEUE_SIZE - 1)
    alerts.send(AlertKind.EXIT, "Sold 10 SBIN")
    alerts.deliver_queued()
    assert alerts.dropped == 1
    assert client.messages[:2] == ["SANDBOX: Bought 10 SBIN", "SANDBOX: session 1"]
    assert client.messages[-1] == "SANDBOX: Sold 10 SBIN"
    assert "SANDBOX: session 0" not in client.messages


def test_a_new_alert_below_every_queued_priority_is_the_one_dropped(alerts, client):
    fill(alerts, AlertKind.MODE_BREACH, QUEUE_SIZE)
    alerts.send(AlertKind.STOP_MOVED, "stop of SBIN moved to 805.00")
    alerts.deliver_queued()
    assert alerts.dropped == 1
    assert len(client.calls) == QUEUE_SIZE
    assert "SANDBOX: stop of SBIN moved to 805.00" not in client.messages


def test_a_dropped_alert_is_logged_as_a_warning(alerts, caplog):
    fill(alerts, AlertKind.MODE_BREACH, QUEUE_SIZE)
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    caplog.clear()
    alerts.send(AlertKind.STOP_MOVED, "stop of SBIN moved to 805.00")
    assert caplog.messages[-1] == (
        "alert stop_moved dropped (60 alerts wait): SANDBOX: stop of SBIN moved to 805.00"
    )


def test_the_alerts_not_delivered_are_the_failed_ones_and_the_dropped_ones(clock):
    alerts = make_alerts(FakeOpenAlgo(UNREACHABLE), clock)
    alerts.send(AlertKind.FEED_DEAD, "no tick for 30 s")
    alerts.deliver_queued()
    fill(alerts, AlertKind.MODE_BREACH, QUEUE_SIZE)
    alerts.send(AlertKind.SESSION, "still not flat")
    assert (alerts.failures, alerts.dropped, alerts.undelivered) == (1, 1, 2)


# Closing


def test_close_drops_what_no_sender_posted_and_counts_it(alerts, client):
    fill(alerts, AlertKind.ENTRY, 2)
    alerts.close()
    assert client.calls == []
    assert alerts.dropped == 2


def test_an_alert_sent_after_close_is_logged_and_dropped(alerts, client, caplog):
    alerts.close()
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    alerts.send(AlertKind.ENTRY, "Bought 10 SBIN")
    assert caplog.messages == [
        "alert entry: SANDBOX: Bought 10 SBIN",
        "alert entry dropped (the alerts are closed): SANDBOX: Bought 10 SBIN",
    ]
    assert (alerts.deliver_queued(), alerts.dropped) == (0, 1)


def test_close_twice_is_safe(alerts):
    fill(alerts, AlertKind.ENTRY, 2)
    alerts.close()
    alerts.close()
    assert alerts.dropped == 2


# The sender thread


def test_the_sender_is_one_daemon_thread(client, clock):
    alerts, threads = started_alerts(client, clock)
    try:
        assert [thread.daemon for thread in threads] == [True]
    finally:
        finish(alerts, threads)


def test_without_a_username_no_sender_thread_starts(client, clock):
    alerts, threads = started_alerts(client, clock, username=None)
    assert threads == []
    alerts.close()


def test_send_returns_while_openalgo_has_not_answered(clock):
    hold = Hold()
    client = FakeOpenAlgo(hold)
    alerts, threads = started_alerts(client, clock)
    try:
        alerts.send(AlertKind.ENTRY, "Bought 10 SBIN")
        assert hold.reached.wait(WAIT_SECONDS)
        alerts.send(AlertKind.EXIT, "Sold 10 SBIN")
        assert len(client.calls) == 1
    finally:
        hold.release.set()
        finish(alerts, threads)
    assert client.messages == ["SANDBOX: Bought 10 SBIN", "SANDBOX: Sold 10 SBIN"]


def test_close_posts_what_is_queued_before_it_returns(client, clock):
    alerts, threads = started_alerts(client, clock)
    fill(alerts, AlertKind.ENTRY, 3)
    finish(alerts, threads)
    assert len(client.calls) == 3
    assert alerts.dropped == 0


def test_the_sender_counts_a_failed_send(clock):
    alerts, threads = started_alerts(FakeOpenAlgo(UNREACHABLE), clock)
    alerts.send(AlertKind.FEED_DEAD, "no tick for 30 s")
    finish(alerts, threads)
    assert (alerts.failures, alerts.last_failure) == (1, UNREACHABLE["message"])


def test_close_gives_up_on_a_server_that_does_not_answer(clock):
    hold = Hold()
    client = FakeOpenAlgo(hold)
    alerts, threads = started_alerts(client, clock)
    alerts.send(AlertKind.ENTRY, "Bought 10 SBIN")
    assert hold.reached.wait(WAIT_SECONDS)
    alerts.send(AlertKind.EXIT, "Sold 10 SBIN")
    alerts.close(wait_seconds=0)
    assert alerts.dropped == 1
    hold.release.set()
    finish(alerts, threads)
    assert client.messages == ["SANDBOX: Bought 10 SBIN"]
