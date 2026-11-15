"""The queue and the sender of the alerts: highest priority first, routine alerts leaving
together, at most 20 posts a minute, a transient failure tried again after the server's
wait or a back-off, a refusal counted at once, a full queue dropping the oldest alert of
the lowest priority, and a `close` that posts what is queued, gives up on a server that
does not answer and is safe to call twice. The sender thread starts on the first send.

The queue tests start no thread: they post with `deliver_queued` and move the session's
clock by hand, so every wait is on the injected clock. The thread tests start the real
sender and wait only on events the fake OpenAlgo sets, never on a timer.
"""

from __future__ import annotations

import logging
import threading

import pytest
from alerts_samples import (
    BROKEN_OFF,
    NOT_LINKED,
    PROXY_PAGE,
    QUEUED,
    RATE_LIMITED,
    SERVER_DOWN,
    TIMED_OUT,
    UNREACHABLE,
    WAIT_SECONDS,
    FakeOpenAlgo,
    Hold,
    ManualClock,
    make_alerts,
    sender_threads_since,
    threaded_alerts,
)

from openalgo.backtest._live._alerts_queue import AlertQueue, QueuedAlert
from openalgo.backtest._live._alerts_sender import Sender
from openalgo.backtest._live._types import LOGGER_NAME, AlertKind
from openalgo.backtest._live.alerts import QUEUE_SIZE, TELEGRAM_MAX_CHARS, URGENT_PRIORITY


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
    """Send `count` alerts of `kind` with distinct texts."""
    for number in range(start, start + count):
        alerts.send(kind, f"{kind.value} {number}")


def failing_forever(answer) -> FakeOpenAlgo:
    """A fake that gives `answer` to every post."""
    return FakeOpenAlgo(*[answer] * 1000)


def finish(alerts, threads) -> None:
    """Close the alerts and wait for their sender to end."""
    alerts.close()
    for thread in threads:
        thread.join(WAIT_SECONDS)
        assert not thread.is_alive()


# The order of the queue and routine alerts posted together


def test_an_alert_waits_in_the_queue_until_the_sender_posts_it(alerts, client):
    alerts.send(AlertKind.ENTRY, "Bought 10 SBIN")
    assert client.calls == []
    assert alerts.deliver_queued() == 1
    assert client.messages == ["SANDBOX: Bought 10 SBIN"]


def test_the_highest_priority_goes_first_and_the_oldest_first_within_it(alerts, client):
    alerts.send(AlertKind.SESSION, "session started")
    alerts.send(AlertKind.REJECT, "SBIN entry refused")
    alerts.send(AlertKind.MODE_BREACH, "the analyzer switch changed")
    alerts.send(AlertKind.REJECT, "INFY entry refused")
    alerts.deliver_queued()
    assert client.messages == [
        "SANDBOX: the analyzer switch changed",
        "SANDBOX: SBIN entry refused",
        "SANDBOX: INFY entry refused",
        "SANDBOX: session started",
    ]


def test_routine_alerts_waiting_together_leave_as_one_message_in_the_order_sent(alerts, client):
    alerts.send(AlertKind.SESSION, "session started")
    alerts.send(AlertKind.ENTRY, "Bought 10 SBIN")
    alerts.send(AlertKind.STOP_MOVED, "stop of SBIN at 805.00")
    assert alerts.deliver_queued() == 1
    assert client.messages == [
        "SANDBOX: session started\nSANDBOX: Bought 10 SBIN\nSANDBOX: stop of SBIN at 805.00"
    ]


def test_routine_alerts_posted_together_take_the_priority_of_the_highest(alerts, client):
    alerts.send(AlertKind.SESSION, "session started")
    alerts.send(AlertKind.ENTRY, "Bought 10 SBIN")
    alerts.deliver_queued()
    assert client.calls[0]["priority"] == 7


def test_urgent_alerts_always_leave_alone(alerts, client):
    alerts.send(AlertKind.REJECT, "SBIN entry refused")
    alerts.send(AlertKind.REJECT, "INFY entry refused")
    assert alerts.deliver_queued() == 2


def test_routine_alerts_that_would_pass_telegrams_limit_leave_in_the_next_post(alerts, client):
    half = TELEGRAM_MAX_CHARS // 2
    alerts.send(AlertKind.ENTRY, "a" * half)
    alerts.send(AlertKind.ENTRY, "b" * half)
    assert alerts.deliver_queued() == 2
    assert [len(message) for message in client.messages] == [half + 9, half + 9]


def test_a_burst_of_entries_costs_one_post_after_the_mode_breach(alerts, client):
    fill(alerts, AlertKind.ENTRY, 30)
    alerts.send(AlertKind.MODE_BREACH, "the analyzer switch changed: entries locked")
    assert alerts.deliver_queued() == 2
    assert client.messages[0] == "SANDBOX: the analyzer switch changed: entries locked"
    assert client.messages[1].count("\n") == 29


# Pacing: at most 20 posts in any minute, on the session's clock


def test_at_most_20_posts_go_in_any_minute(alerts, clock):
    fill(alerts, AlertKind.MODE_BREACH, 25)
    assert alerts.deliver_queued() == 20
    clock.advance(59.999)
    assert alerts.deliver_queued() == 0
    clock.advance(0.001)
    assert alerts.deliver_queued() == 5


def test_the_posting_budget_frees_as_posts_leave_the_minute(alerts, clock):
    fill(alerts, AlertKind.MODE_BREACH, 10)
    alerts.deliver_queued()
    clock.advance(30)
    fill(alerts, AlertKind.MODE_BREACH, 13, start=10)
    assert alerts.deliver_queued() == 10
    clock.advance(30)
    assert alerts.deliver_queued() == 3


# Retries


@pytest.mark.parametrize(
    "answer",
    [RATE_LIMITED, SERVER_DOWN, PROXY_PAGE, UNREACHABLE, TIMED_OUT, BROKEN_OFF],
    ids=["429", "503", "502 page", "unreachable", "timed out", "broken off"],
)
def test_a_transient_failure_is_tried_again_and_not_counted(clock, answer):
    client = FakeOpenAlgo(answer, QUEUED)
    alerts = make_alerts(client, clock)
    alerts.send(AlertKind.FEED_DEAD, "no tick for 30 s")
    alerts.deliver_queued()
    clock.advance(60)
    alerts.deliver_queued()
    assert client.messages == ["SANDBOX: no tick for 30 s"] * 2
    assert alerts.failures == 0


def test_a_rate_limited_post_waits_as_long_as_the_server_asked(clock):
    client = FakeOpenAlgo(RATE_LIMITED, QUEUED)
    alerts = make_alerts(client, clock)
    alerts.send(AlertKind.MODE_BREACH, "the analyzer switch changed")
    alerts.deliver_queued()
    clock.advance(59.999)
    assert alerts.deliver_queued() == 0
    clock.advance(0.001)
    assert alerts.deliver_queued() == 1


def test_nothing_is_posted_while_the_sender_waits_after_a_failure(clock):
    client = FakeOpenAlgo(RATE_LIMITED)
    alerts = make_alerts(client, clock)
    alerts.send(AlertKind.MODE_BREACH, "the analyzer switch changed")
    alerts.deliver_queued()
    alerts.send(AlertKind.REJECT, "SBIN entry refused")
    clock.advance(30)
    assert alerts.deliver_queued() == 0


def test_the_back_off_doubles_from_one_second_up_to_30_seconds(clock):
    alerts = make_alerts(failing_forever(UNREACHABLE), clock)
    alerts.send(AlertKind.FEED_DEAD, "no tick for 30 s")
    alerts.deliver_queued()
    for wait in [1, 2, 4, 8, 16, 30, 30]:
        clock.advance(wait - 0.001)
        assert alerts.deliver_queued() == 0
        clock.advance(0.001)
        assert alerts.deliver_queued() == 1


def test_the_back_off_starts_again_after_a_post_goes_through(clock):
    alerts = make_alerts(FakeOpenAlgo(UNREACHABLE, UNREACHABLE, QUEUED, UNREACHABLE), clock)
    alerts.send(AlertKind.FEED_DEAD, "no tick for 30 s")
    alerts.deliver_queued()
    clock.advance(1)
    alerts.deliver_queued()
    clock.advance(2)
    alerts.deliver_queued()
    alerts.send(AlertKind.ERROR, "the order path failed")
    alerts.deliver_queued()
    clock.advance(1)
    assert alerts.deliver_queued() == 1


def test_a_retried_alert_goes_before_newer_alerts_of_its_priority(clock):
    client = FakeOpenAlgo(UNREACHABLE)
    alerts = make_alerts(client, clock)
    alerts.send(AlertKind.MODE_BREACH, "first")
    alerts.deliver_queued()
    alerts.send(AlertKind.MODE_BREACH, "second")
    clock.advance(1)
    alerts.deliver_queued()
    assert client.messages == ["SANDBOX: first", "SANDBOX: first", "SANDBOX: second"]


def test_an_alert_failing_for_five_minutes_counts_as_failed(clock):
    alerts = make_alerts(failing_forever(UNREACHABLE), clock)
    alerts.send(AlertKind.FEED_DEAD, "no tick for 30 s")
    alerts.deliver_queued()
    failures_by_second = {}
    for second in range(1, 302):
        clock.advance(1)
        alerts.deliver_queued()
        failures_by_second[second] = alerts.failures
    assert failures_by_second[299] == 0
    assert failures_by_second[301] == 1
    assert alerts.last_failure == f"{UNREACHABLE['message']} (tried for 300 s)"


def test_a_refusal_is_not_tried_again(clock):
    client = FakeOpenAlgo(NOT_LINKED)
    alerts = make_alerts(client, clock)
    alerts.send(AlertKind.FEED_DEAD, "no tick for 30 s")
    alerts.deliver_queued()
    clock.advance(60)
    alerts.deliver_queued()
    assert len(client.calls) == 1
    assert alerts.failures == 1


def test_a_transient_failure_is_logged_with_the_wait(clock, caplog):
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    alerts = make_alerts(FakeOpenAlgo(RATE_LIMITED), clock)
    alerts.send(AlertKind.MODE_BREACH, "the analyzer switch changed")
    alerts.deliver_queued()
    assert caplog.messages[-1] == (
        "alert mode_breach not delivered yet (Rate limit exceeded. Please slow down your "
        "requests. (HTTP 429)): trying again in 60 s"
    )


# A full queue


def deliver_all(alerts, clock) -> None:
    """Post a full queue: three minutes of the posting budget."""
    for _ in range(3):
        alerts.deliver_queued()
        clock.advance(60)


def test_a_full_queue_drops_the_oldest_alert_of_the_lowest_priority(alerts, client, clock):
    alerts.send(AlertKind.REJECT, "SBIN entry refused")
    fill(alerts, AlertKind.MODE_BREACH, QUEUE_SIZE - 2)
    alerts.send(AlertKind.REJECT, "INFY entry refused")
    alerts.send(AlertKind.MODE_BREACH, "one more")
    deliver_all(alerts, clock)
    assert alerts.dropped == 1
    assert "SANDBOX: SBIN entry refused" not in client.messages
    assert "SANDBOX: INFY entry refused" in client.messages


def test_a_new_alert_below_every_queued_priority_is_the_one_dropped(alerts, client, clock):
    fill(alerts, AlertKind.MODE_BREACH, QUEUE_SIZE)
    alerts.send(AlertKind.STOP_MOVED, "stop of SBIN moved to 805.00")
    deliver_all(alerts, clock)
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
    alerts = make_alerts(FakeOpenAlgo(NOT_LINKED), clock)
    alerts.send(AlertKind.FEED_DEAD, "no tick for 30 s")
    alerts.deliver_queued()
    fill(alerts, AlertKind.MODE_BREACH, QUEUE_SIZE)
    alerts.send(AlertKind.SESSION, "still not flat")
    assert (alerts.failures, alerts.dropped, alerts.undelivered) == (1, 1, 2)


# Closing


def test_close_drops_what_no_sender_posted_and_counts_it(alerts, client):
    fill(alerts, AlertKind.REJECT, 2)
    alerts.close()
    assert client.calls == []
    assert alerts.dropped == 2


def test_at_close_an_alert_waiting_for_another_try_counts_as_failed(clock):
    alerts = make_alerts(FakeOpenAlgo(UNREACHABLE), clock)
    alerts.send(AlertKind.FEED_DEAD, "no tick for 30 s")
    alerts.deliver_queued()
    alerts.close()
    assert (alerts.failures, alerts.dropped) == (1, 0)
    assert alerts.last_failure == "still failing when the alerts closed"


def test_a_transient_failure_answered_after_the_queue_was_abandoned_counts_as_failed(clock):
    queue = AlertQueue(QUEUE_SIZE, urgent_from=URGENT_PRIORITY, max_chars=TELEGRAM_MAX_CHARS)

    def post_while_closing(text, priority):
        queue.abandon()
        return UNREACHABLE

    sender = Sender(queue, post_while_closing, clock)
    queue.put(QueuedAlert(AlertKind.FEED_DEAD, "SANDBOX: no tick for 30 s", 10, seq=1))
    sender.step()
    assert (sender.failures, sender.last_failure) == (1, "still failing when the alerts closed")


def test_an_alert_sent_after_close_is_logged_and_dropped(alerts, caplog):
    alerts.close()
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    alerts.send(AlertKind.ENTRY, "Bought 10 SBIN")
    assert caplog.messages == [
        "alert entry: SANDBOX: Bought 10 SBIN",
        "alert entry dropped (the alerts are closed): SANDBOX: Bought 10 SBIN",
    ]
    assert (alerts.deliver_queued(), alerts.dropped) == (0, 1)


def test_close_twice_is_safe(alerts):
    fill(alerts, AlertKind.REJECT, 2)
    alerts.close()
    alerts.close()
    assert alerts.dropped == 2


# The sender thread


def test_alerts_never_sent_through_start_no_thread(client, clock):
    before = set(threading.enumerate())
    alerts = threaded_alerts(client, clock)
    alerts.test()
    started = sender_threads_since(before)
    alerts.close()
    assert started == []


def test_the_first_send_starts_one_daemon_sender_thread(client, clock):
    before = set(threading.enumerate())
    alerts = threaded_alerts(client, clock)
    alerts.send(AlertKind.ENTRY, "Bought 10 SBIN")
    alerts.send(AlertKind.EXIT, "Sold 10 SBIN")
    threads = sender_threads_since(before)
    try:
        assert [thread.daemon for thread in threads] == [True]
    finally:
        finish(alerts, threads)


def test_without_a_username_no_sender_thread_starts(client, clock):
    before = set(threading.enumerate())
    alerts = threaded_alerts(client, clock, username=None)
    alerts.send(AlertKind.ENTRY, "Bought 10 SBIN")
    assert sender_threads_since(before) == []
    alerts.close()


def test_send_returns_while_openalgo_has_not_answered(clock):
    hold = Hold()
    client = FakeOpenAlgo(hold)
    before = set(threading.enumerate())
    alerts = threaded_alerts(client, clock)
    alerts.send(AlertKind.REJECT, "SBIN entry refused")
    threads = sender_threads_since(before)
    try:
        assert hold.reached.wait(WAIT_SECONDS)
        alerts.send(AlertKind.REJECT, "INFY entry refused")
        assert len(client.calls) == 1
    finally:
        hold.release.set()
        finish(alerts, threads)
    assert client.messages == ["SANDBOX: SBIN entry refused", "SANDBOX: INFY entry refused"]


def test_close_posts_what_is_queued_before_it_returns(client, clock):
    before = set(threading.enumerate())
    alerts = threaded_alerts(client, clock)
    fill(alerts, AlertKind.REJECT, 3)
    finish(alerts, sender_threads_since(before))
    assert len(client.calls) == 3
    assert alerts.dropped == 0


def test_the_sender_thread_counts_a_refused_send(clock):
    before = set(threading.enumerate())
    alerts = threaded_alerts(FakeOpenAlgo(NOT_LINKED), clock)
    alerts.send(AlertKind.FEED_DEAD, "no tick for 30 s")
    finish(alerts, sender_threads_since(before))
    assert alerts.failures == 1


def test_leaving_a_with_block_closes_the_alerts(client, clock):
    before = set(threading.enumerate())
    with threaded_alerts(client, clock) as alerts:
        alerts.send(AlertKind.REJECT, "SBIN entry refused")
        threads = sender_threads_since(before)
    for thread in threads:
        thread.join(WAIT_SECONDS)
    assert client.messages == ["SANDBOX: SBIN entry refused"]
    assert [thread.is_alive() for thread in threads] == [False]


def test_close_gives_up_on_a_server_that_does_not_answer(clock):
    hold = Hold()
    client = FakeOpenAlgo(hold)
    before = set(threading.enumerate())
    alerts = threaded_alerts(client, clock)
    alerts.send(AlertKind.REJECT, "SBIN entry refused")
    threads = sender_threads_since(before)
    assert hold.reached.wait(WAIT_SECONDS)
    alerts.send(AlertKind.REJECT, "INFY entry refused")
    alerts.close(wait_seconds=0)
    assert (alerts.failures, alerts.dropped) == (1, 1)
    hold.release.set()
    finish(alerts, threads)
    assert client.messages == ["SANDBOX: SBIN entry refused"]


def test_a_post_in_flight_at_close_is_logged_by_name_and_counted_once(clock, caplog):
    hold = Hold()
    before = set(threading.enumerate())
    alerts = threaded_alerts(FakeOpenAlgo(hold), clock)
    alerts.send(AlertKind.DAILY_SUMMARY, "Session of 2026-11-16 (IST)")
    threads = sender_threads_since(before)
    assert hold.reached.wait(WAIT_SECONDS)
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    alerts.close(wait_seconds=0)
    hold.release.set()
    finish(alerts, threads)
    assert caplog.messages[-1] == (
        "alert daily_summary was still being posted when the alerts closed; it may or may "
        "not arrive: SANDBOX: Session of 2026-11-16 (IST)"
    )
    assert (alerts.failures, alerts.undelivered) == (1, 1)
