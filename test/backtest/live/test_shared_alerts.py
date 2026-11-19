"""The shared alerts of a live session (16-session-design 2.8): one lock around the
session's alerts for the threads that send them, the log until the alerts are attached,
one attachment, a test alert that holds no lock, and every alert echoed on the console
(the sandbox smoke run of 2026-11-17, unit 13).

The thread tests run real threads against alerts doubles that block on an event; every
wait is bounded, and reached only when something is broken.
"""

from __future__ import annotations

import logging
import threading

import pytest
from alerts_samples import FakeOpenAlgo, ManualClock, make_alerts

from openalgo.backtest._live._shared_alerts import SharedAlerts
from openalgo.backtest._live._types import LOGGER_NAME, AlertKind
from openalgo.backtest.errors import ConfigError, LiveError

# The longest a thread test waits before it fails, seconds.
WAIT_SECONDS = 5.0
# How long a test gives a thread that must stay blocked to show it is not, seconds.
QUIET_SECONDS = 0.05


class RecordingAlerts:
    """An `AlertsLike` that keeps what it was sent and answers its test with `tested`."""

    def __init__(self, tested: bool = True) -> None:
        self.sent: list[tuple[AlertKind, str]] = []
        self.tested = tested
        self.tests = 0

    def send(self, kind: AlertKind, text: str) -> None:
        self.sent.append((kind, text))

    def test(self) -> bool:
        self.tests += 1
        return self.tested


class BlockingAlerts(RecordingAlerts):
    """Alerts whose first send, or whose test, waits for `release`; `entered` is set when
    one is waiting. `inside` counts the sends running now, `most_inside` the most at
    once."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.inside = 0
        self.most_inside = 0

    def send(self, kind: AlertKind, text: str) -> None:
        self.inside += 1
        self.most_inside = max(self.most_inside, self.inside)
        if not self.sent:
            self.entered.set()
            assert self.release.wait(WAIT_SECONDS), "the test never released the send"
        super().send(kind, text)
        self.inside -= 1

    def test(self) -> bool:
        self.entered.set()
        assert self.release.wait(WAIT_SECONDS), "the test never released the test alert"
        return super().test()


def started(target, *args) -> threading.Thread:
    thread = threading.Thread(target=target, args=args, daemon=True)
    thread.start()
    return thread


# -- forwarding --------------------------------------------------------------------------------


def test_an_alert_goes_to_the_attached_alerts():
    shared, alerts = SharedAlerts(), RecordingAlerts()
    shared.attach(alerts)

    shared.send(AlertKind.ENTRY, "SBIN bought 10 at 812.00, order 2511160001")

    assert alerts.sent == [(AlertKind.ENTRY, "SBIN bought 10 at 812.00, order 2511160001")]


def test_an_alert_before_the_alerts_are_attached_goes_to_the_log_as_a_warning(caplog):
    shared = SharedAlerts()

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        shared.send(AlertKind.ERROR, "analyzerstatus did not answer")

    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert "error" in record.getMessage()
    assert "analyzerstatus did not answer" in record.getMessage()


def test_alerts_before_the_attachment_are_not_sent_later():
    shared, alerts = SharedAlerts(), RecordingAlerts()
    shared.send(AlertKind.ERROR, "before")

    shared.attach(alerts)

    assert alerts.sent == []


def test_the_shared_alerts_say_whether_alerts_are_attached():
    shared = SharedAlerts()
    before = shared.attached

    shared.attach(RecordingAlerts())

    assert (before, shared.attached) == (False, True)


def test_a_second_attachment_is_refused():
    shared = SharedAlerts()
    shared.attach(RecordingAlerts())

    with pytest.raises(LiveError, match="attached already"):
        shared.attach(RecordingAlerts())


@pytest.mark.parametrize("alerts", [None, "trader1", object()], ids=["None", "a name", "object"])
def test_an_object_without_send_and_test_is_refused(alerts):
    with pytest.raises(ConfigError, match="send and test"):
        SharedAlerts().attach(alerts)


@pytest.mark.parametrize("tested", [True, False], ids=["delivered", "not delivered"])
def test_the_test_alert_is_the_attached_alerts_answer(tested):
    shared, alerts = SharedAlerts(), RecordingAlerts(tested=tested)
    shared.attach(alerts)

    assert (shared.test(), alerts.tests) == (tested, 1)


def test_the_test_alert_before_the_attachment_is_not_delivered(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert SharedAlerts().test() is False

    assert "not set up" in caplog.text


# -- the console's echo ----------------------------------------------------------------------------


def echoing(alerts=None) -> tuple[SharedAlerts, list[str]]:
    """Shared alerts that echo on a console keeping its lines, with `alerts` attached."""
    lines: list[str] = []
    shared = SharedAlerts(echo=lines.append)
    shared.attach(alerts or RecordingAlerts())
    return shared, lines


@pytest.mark.parametrize(
    ("kind", "line"),
    [
        (AlertKind.ENTRY, "ENTRY: BUY 10 SBIN NSE at 812.00"),
        (AlertKind.STOP_MOVED, "STOP MOVED: BUY 10 SBIN NSE at 812.00"),
        (AlertKind.DAILY_SUMMARY, "DAILY SUMMARY: BUY 10 SBIN NSE at 812.00"),
    ],
    ids=["one word", "two words", "the daily summary"],
)
def test_an_alert_is_echoed_on_the_console_after_its_kind(kind, line):
    shared, lines = echoing()
    shared.send(kind, "BUY 10 SBIN NSE at 812.00")
    assert lines == [line]


def test_an_alert_is_echoed_with_no_telegram_user():
    shared, lines = echoing(make_alerts(FakeOpenAlgo(), ManualClock(), username=None))
    shared.send(AlertKind.EXIT, "SELL 10 SBIN NSE at 812.00")
    assert lines == ["EXIT: SELL 10 SBIN NSE at 812.00"]


def test_the_banners_session_alert_is_not_echoed():
    shared, lines = echoing()
    shared.send(AlertKind.SESSION, "OpenAlgo is in LIVE mode: orders go to zerodha")
    assert lines == []


def test_an_alert_whose_line_its_caller_printed_is_not_echoed():
    shared, lines = echoing()
    shared.send(AlertKind.ERROR, "SBIN NSE 10 units: no stop at the broker", echo=False)
    assert lines == []


def test_a_repeated_condition_held_back_is_not_echoed_again():
    clock = ManualClock()
    shared, lines = echoing(make_alerts(FakeOpenAlgo(), clock))
    shared.send(AlertKind.FEED_DEAD, "no tick for 10 s")
    clock.advance(30)
    shared.send(AlertKind.FEED_DEAD, "no tick for 10 s")
    assert lines == ["FEED DEAD: no tick for 10 s"]


def test_an_alert_before_the_attachment_is_echoed_too():
    lines: list[str] = []
    SharedAlerts(echo=lines.append).send(AlertKind.ERROR, "analyzerstatus did not answer")
    assert lines == ["ERROR: analyzerstatus did not answer"]


def test_a_console_that_fails_is_logged_and_the_alert_still_goes(caplog):
    def broken(line: str) -> None:
        raise OSError("the console is closed")

    shared, alerts = SharedAlerts(echo=broken), RecordingAlerts()
    shared.attach(alerts)
    with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
        shared.send(AlertKind.REJECT, "order 2511160001 rejected")
    assert alerts.sent == [(AlertKind.REJECT, "order 2511160001 rejected")]
    assert "could not be printed on the console" in caplog.text


# -- two threads -----------------------------------------------------------------------------------


def test_a_send_from_another_thread_waits_until_the_first_has_left_the_alerts():
    shared, alerts = SharedAlerts(), BlockingAlerts()
    shared.attach(alerts)
    first = started(shared.send, AlertKind.MODE_BREACH, "from the order worker")
    assert alerts.entered.wait(WAIT_SECONDS)

    second = started(shared.send, AlertKind.EXIT, "from the loop")
    second.join(QUIET_SECONDS)
    still_waiting = second.is_alive()
    alerts.release.set()
    first.join(WAIT_SECONDS)
    second.join(WAIT_SECONDS)

    assert still_waiting
    assert alerts.most_inside == 1
    assert [text for _, text in alerts.sent] == ["from the order worker", "from the loop"]


def test_a_send_does_not_wait_for_a_test_alert_in_progress():
    shared, alerts = SharedAlerts(), BlockingAlerts()
    alerts.sent.append((AlertKind.SESSION, "already sent: the next send does not block"))
    shared.attach(alerts)
    testing = started(shared.test)
    assert alerts.entered.wait(WAIT_SECONDS)

    shared.send(AlertKind.ENTRY, "while the test alert waits on OpenAlgo")
    alerts.release.set()
    testing.join(WAIT_SECONDS)

    assert alerts.sent[-1] == (AlertKind.ENTRY, "while the test alert waits on OpenAlgo")


def test_two_threads_sending_at_once_queue_every_alert_once_in_the_real_alerts():
    # 2 x 25 routine alerts fit the queue of 60 together, so none is dropped.
    fake = FakeOpenAlgo()
    alerts = make_alerts(fake, ManualClock())
    shared = SharedAlerts()
    shared.attach(alerts)
    go = threading.Barrier(2)

    def send_from(name: str) -> None:
        go.wait(WAIT_SECONDS)
        for n in range(25):
            shared.send(AlertKind.SESSION, f"{name} {n}")

    threads = [started(send_from, name) for name in ("loop", "worker")]
    for thread in threads:
        thread.join(WAIT_SECONDS)
    alerts.deliver_queued()

    lines = [line for post in fake.messages for line in post.splitlines()]
    expected = {f"SANDBOX: {name} {n}" for name in ("loop", "worker") for n in range(25)}
    assert sorted(lines) == sorted(expected)
    assert (alerts.dropped, alerts.failures) == (0, 0)
