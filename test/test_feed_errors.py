"""Error reporting for the feed (openalgo._feed_errors). No sockets, no real timers."""

import logging
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from openalgo._feed_errors import ErrorReporter, FeedServerError, server_rejection  # noqa: E402
from openalgo._feed_hooks import HookRegistry  # noqa: E402

WINDOW = 10.0


class FakeTimers:
    """Stands in for threading.Timer: a test decides when a window ends."""

    def __init__(self):
        self.started = []

    def __call__(self, interval, function, args=()):
        timer = FakeTimer(interval, function, args)
        self.started.append(timer)
        return timer

    def end_every_window(self):
        for timer in self.started:
            timer.fire()


class FakeTimer:
    def __init__(self, interval, function, args):
        self.interval = interval
        self.daemon = False
        self._function = function
        self._args = args

    def start(self):
        pass

    def fire(self):
        self._function(*self._args)


def raised(exc):
    """Give an exception a traceback, as one caught around a callback has."""
    try:
        raise exc
    except type(exc) as caught:
        return caught


def ignore_failure(event, exc, args):
    """Hook failures are not the subject here."""


@pytest.fixture
def hooks():
    return HookRegistry(("error",), ignore_failure)


@pytest.fixture
def timers():
    return FakeTimers()


@pytest.fixture
def reporter(hooks, timers):
    return ErrorReporter(hooks, lambda: WINDOW, timer=timers)


@pytest.fixture
def logged(caplog):
    """A function that returns the records written to the feed logger so far."""
    caplog.set_level(logging.DEBUG, logger="openalgo.feed")
    return lambda: caplog.records


# -- classifying server frames ------------------------------------------------

KEY_REJECTED = {"status": "error", "code": "AUTHENTICATION_ERROR", "message": "Invalid API key"}
BROKER_ERROR = {"status": "error", "code": "BROKER_ERROR", "message": "Broker adapter not found"}
PARTIAL_SUBSCRIBE = {
    "type": "subscribe",
    "status": "partial",
    "subscriptions": [
        {"symbol": "SBIN", "exchange": "NSE", "status": "success", "mode": "LTP"},
        {"symbol": "XXX", "exchange": "NSE", "status": "error", "message": "Symbol not found"},
    ],
}
FAILED_UNSUBSCRIBE = {
    "type": "unsubscribe",
    "status": "error",
    "successful": [],
    "failed": [{"symbol": "XXX", "exchange": "NSE", "status": "error", "message": "Not owned"}],
}


@pytest.mark.parametrize(
    "frame",
    [
        {"type": "auth", "status": "success", "broker": "b", "user_id": "u"},
        {"type": "subscribe", "status": "success", "subscriptions": [{"status": "success"}]},
        {"type": "unsubscribe", "status": "success", "successful": [{}], "failed": []},
        {"type": "subscribe_orders", "status": "success"},
        {"type": "market_data", "symbol": "SBIN", "exchange": "NSE", "mode": 1, "data": {}},
        {"type": "order_update", "order_status": "rejected"},
        [1, 2, 3],
    ],
)
def test_frame_that_is_not_a_rejection_gives_none(frame):
    assert server_rejection(frame) is None


def test_refused_api_key_is_an_authentication_rejection():
    kind, error, details = server_rejection(KEY_REJECTED)
    assert (kind, error.code, error.message) == (
        "authentication",
        "AUTHENTICATION_ERROR",
        "Invalid API key",
    )


def test_failed_auth_reply_is_an_authentication_rejection():
    kind, error, details = server_rejection({"type": "auth", "status": "error", "message": "no"})
    assert (kind, error.message) == ("authentication", "no")


def test_other_error_frame_is_a_server_rejection_with_the_server_code():
    kind, error, details = server_rejection(BROKER_ERROR)
    assert (kind, error.code, error.message) == (
        "server",
        "BROKER_ERROR",
        "Broker adapter not found",
    )


def test_error_frame_without_a_code_still_gets_one():
    kind, error, details = server_rejection({"status": "error", "message": "odd"})
    assert (kind, error.code) == ("server", "SERVER_ERROR")


def test_partial_subscribe_reply_lists_the_refused_instruments():
    kind, error, details = server_rejection(PARTIAL_SUBSCRIBE)
    assert (kind, error.code, details["rejected"]) == (
        "server",
        "SUBSCRIBE_REJECTED",
        [PARTIAL_SUBSCRIBE["subscriptions"][1]],
    )


def test_failed_unsubscribe_reply_lists_the_refused_instruments():
    kind, error, details = server_rejection(FAILED_UNSUBSCRIBE)
    assert (kind, error.code, details["rejected"]) == (
        "server",
        "UNSUBSCRIBE_REJECTED",
        FAILED_UNSUBSCRIBE["failed"],
    )


def test_server_error_text_names_the_code_and_the_message():
    assert str(FeedServerError("BROKER_ERROR", "not found")) == "BROKER_ERROR: not found"


# -- error hooks ----------------------------------------------------------------


def test_error_goes_to_the_error_hook_with_its_context(hooks, reporter):
    seen = []
    hooks.add("error", lambda exc, context: seen.append((exc, context)))
    error = raised(ValueError("boom"))
    reporter.report(error, kind="callback", callback="ltp", message={"symbol": "SBIN"})
    assert seen == [(error, {"kind": "callback", "callback": "ltp", "message": {"symbol": "SBIN"}})]


def test_nothing_is_logged_when_an_error_hook_is_registered(hooks, reporter, logged):
    hooks.add("error", lambda exc, context: None)
    reporter.report(raised(ValueError("boom")), kind="callback", callback="ltp", message={})
    assert logged() == []


def test_error_hook_receives_every_tick_failure_without_rate_limit(hooks, reporter):
    seen = []
    hooks.add("error", lambda exc, context: seen.append(exc))
    for _ in range(5):
        reporter.report(raised(ValueError("boom")), kind="callback", callback="ltp", message={})
    assert len(seen) == 5


# -- the log, when no error hook is registered ----------------------------------


def test_error_without_a_hook_is_logged_at_error_level(reporter, logged):
    reporter.report(raised(ValueError("boom")), kind="callback", callback="ltp", message={})
    assert [record.levelno for record in logged()] == [logging.ERROR]


def test_log_record_names_the_callback_and_the_exception(reporter, logged):
    reporter.report(raised(ValueError("boom")), kind="callback", callback="ltp", message={})
    message = logged()[0].getMessage()
    assert "ltp" in message and "ValueError: boom" in message


def test_log_record_carries_the_traceback(reporter, logged):
    error = raised(ValueError("boom"))
    reporter.report(error, kind="callback", callback="ltp", message={})
    assert logged()[0].exc_info[1] is error and logged()[0].exc_info[2] is not None


def test_server_rejection_is_logged_with_its_code_and_message(reporter, logged):
    reporter.report(FeedServerError("BROKER_ERROR", "not found"), kind="server", message={})
    assert "BROKER_ERROR: not found" in logged()[0].getMessage()


def test_refused_api_key_is_logged_as_such(reporter, logged):
    error = FeedServerError("AUTHENTICATION_ERROR", "Invalid API key")
    reporter.report(error, kind="authentication", message=KEY_REJECTED)
    assert "rejected the API key" in logged()[0].getMessage()


def test_repeated_tick_callback_failure_is_logged_once_per_window(reporter, logged):
    for _ in range(5):
        reporter.report(raised(ValueError("boom")), kind="callback", callback="ltp", message={})
    assert len(logged()) == 1


def test_different_exception_type_in_the_same_tick_callback_is_logged(reporter, logged):
    reporter.report(raised(ValueError("boom")), kind="callback", callback="ltp", message={})
    reporter.report(raised(KeyError("other")), kind="callback", callback="ltp", message={})
    assert len(logged()) == 2


def test_same_exception_type_in_another_tick_callback_is_logged(reporter, logged):
    reporter.report(raised(ValueError("boom")), kind="callback", callback="ltp", message={})
    reporter.report(raised(ValueError("boom")), kind="callback", callback="quote", message={})
    assert len(logged()) == 2


def test_skipped_count_is_reported_when_the_window_ends(reporter, logged, timers):
    for _ in range(5):
        reporter.report(raised(ValueError("boom")), kind="callback", callback="ltp", message={})
    timers.end_every_window()
    assert "4 more" in logged()[-1].getMessage() and len(logged()) == 2


def test_window_that_skipped_nothing_ends_silently(reporter, logged, timers):
    reporter.report(raised(ValueError("boom")), kind="callback", callback="ltp", message={})
    timers.end_every_window()
    assert len(logged()) == 1


def test_failure_after_the_window_ended_is_logged_again(reporter, logged, timers):
    reporter.report(raised(ValueError("boom")), kind="callback", callback="ltp", message={})
    timers.end_every_window()
    reporter.report(raised(ValueError("boom")), kind="callback", callback="ltp", message={})
    assert len(logged()) == 2


def test_window_lasts_as_long_as_the_configured_interval(reporter, timers):
    reporter.report(raised(ValueError("boom")), kind="callback", callback="ltp", message={})
    assert [timer.interval for timer in timers.started] == [WINDOW]


@pytest.mark.parametrize(
    "context",
    [
        {"kind": "callback", "callback": "order"},
        {"kind": "hook", "callback": "on_disconnect"},
        {"kind": "frame", "callback": None},
        {"kind": "server", "callback": None},
        {"kind": "authentication", "callback": None},
    ],
    ids=lambda context: f"{context['kind']}-{context['callback']}",
)
def test_failure_that_is_not_a_tick_callback_is_never_rate_limited(reporter, logged, context):
    for _ in range(3):
        reporter.report(raised(ValueError("boom")), message={}, **context)
    assert len(logged()) == 3


def test_two_different_order_callback_failures_are_both_logged(reporter, logged):
    reporter.report(raised(KeyError("first fill")), kind="callback", callback="order", message={})
    reporter.report(
        raised(ZeroDivisionError("second fill")), kind="callback", callback="order", message={}
    )
    assert [record.exc_info[0] for record in logged()] == [KeyError, ZeroDivisionError]
