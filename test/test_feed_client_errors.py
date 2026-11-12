"""Errors as an application sees them through the client: exceptions raised
by its callbacks, rejections from the server, and the rule that no part
of the API key is printed or logged.

Fake socket only: see feed_fakes.py. Nothing here reaches the network.
"""

import json
import logging
import time

import pytest
from feed_fakes import (
    API_KEY,
    BROKER_ERROR,
    INFY,
    KEY_REFUSED,
    KINDS,
    SBIN,
    WAIT,
    Recorder,
    feed_log,
    message_for,
    order_update,
    raise_boom,
    shorten,
    subscribe_kind,
    tick,
    wait_for,
)
from feed_fixtures import *  # noqa: F403

# ---------------------------------------------------------------------------
# Exceptions raised by user callbacks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_exception_in_a_data_callback_reaches_the_error_hook(server, connected, recorder, kind):
    assert subscribe_kind(connected, kind, raise_boom) is True
    server.last.deliver(message_for(kind, 1))
    exc, context = recorder.errors[0]
    assert (type(exc), str(exc), context["kind"], context["callback"]) == (
        ValueError,
        "boom",
        "callback",
        kind,
    )


@pytest.mark.parametrize("kind", KINDS)
def test_error_context_carries_the_server_message_being_delivered(
    server, connected, recorder, kind
):
    assert subscribe_kind(connected, kind, raise_boom) is True
    message = message_for(kind, 1)
    server.last.deliver(message)
    assert recorder.errors[0][1]["message"] == message


def test_error_context_carries_the_connection_number(server, connected, recorder):
    connected.subscribe_ltp([SBIN], raise_boom)
    server.last.deliver(tick(SBIN, 100.0))
    assert recorder.errors[0][1]["seq"] == 1


@pytest.mark.parametrize("kind", KINDS)
def test_feed_keeps_delivering_after_a_callback_raised(server, connected, kind):
    received = []

    def fails_on_the_first_message(message):
        received.append(message)
        if len(received) == 1:
            raise ValueError("boom")

    assert subscribe_kind(connected, kind, fails_on_the_first_message) is True
    server.last.deliver(message_for(kind, 1))
    server.last.deliver(message_for(kind, 2))
    assert len(received) == 2


def test_callback_error_does_not_disconnect_the_feed(server, connected, recorder):
    assert connected.subscribe_ltp([SBIN], raise_boom) is True
    server.last.deliver(tick(SBIN, 100.0))
    assert (connected.connected, connected.authenticated, recorder.names()) == (
        True,
        True,
        ["connect"],
    )


def test_error_hook_runs_on_the_websocket_thread(server, connected, recorder):
    assert connected.subscribe_ltp([SBIN], raise_boom) is True
    server.last.deliver(tick(SBIN, 100.0))
    assert recorder.threads[-1] == ("error", server.last.thread)


@pytest.mark.parametrize(
    "raw, exc_type",
    [("this is not json", json.JSONDecodeError), ("[1, 2, 3]", AttributeError)],
)
def test_frame_the_sdk_cannot_process_reaches_the_error_hook(
    server, connected, recorder, raw, exc_type
):
    server.last.deliver_raw(raw)
    exc, context = recorder.errors[0]
    assert (type(exc), context["kind"], context["message"]) == (exc_type, "frame", raw)


@pytest.fixture
def failing_tick_log(server, client, caplog):
    """Log records of a client with a raising LTP callback and no error hook.
    Returns a function that delivers failing ticks and gives the records back."""
    assert client.connect() is True
    assert client.subscribe_ltp([SBIN], raise_boom) is True
    caplog.set_level(logging.DEBUG, logger="openalgo.feed")

    def deliver(count=1):
        for price in range(count):
            server.last.deliver(tick(SBIN, float(price)))
        return feed_log(caplog)

    return deliver


def test_unhandled_callback_error_is_logged_at_error_level(failing_tick_log):
    assert [record.levelno for record in failing_tick_log()] == [logging.ERROR]


def test_log_record_names_the_callback_and_the_exception(failing_tick_log):
    message = failing_tick_log()[0].getMessage()
    assert "ltp" in message and "ValueError: boom" in message


def test_log_record_carries_the_traceback_of_the_exception(failing_tick_log):
    exc_type, exc, traceback = failing_tick_log()[0].exc_info
    assert (exc_type, str(exc), traceback is not None) == (ValueError, "boom", True)


def test_unhandled_callback_error_prints_nothing_at_verbose_zero(failing_tick_log, capsys):
    failing_tick_log()
    assert capsys.readouterr().out == ""


def test_repeated_tick_callback_error_is_logged_once_per_window(failing_tick_log):
    assert len(failing_tick_log(5)) == 1


def test_skipped_tick_callback_errors_are_counted_when_the_window_ends(
    failing_tick_log, client, caplog
):
    shorten(client, error_log=0.05)
    failing_tick_log(5)
    assert wait_for(lambda: len(feed_log(caplog)) == 2)
    assert "4 more" in feed_log(caplog)[1].getMessage()


def test_two_different_order_callback_errors_in_one_window_are_both_logged(server, client, caplog):
    def fails_differently(message):
        if message["orderid"] == "1":
            raise KeyError("first fill: missing key")
        raise ZeroDivisionError("second fill: a different bug")

    caplog.set_level(logging.ERROR, logger="openalgo.feed")
    assert client.connect() is True
    client.subscribe_orders(fails_differently)
    server.last.deliver(order_update("1"))
    server.last.deliver(order_update("2"))
    assert [record.exc_info[0] for record in feed_log(caplog)] == [KeyError, ZeroDivisionError]


def test_registered_error_hook_replaces_the_log_record(server, connected, caplog):
    caplog.set_level(logging.DEBUG, logger="openalgo.feed")
    assert connected.subscribe_ltp([SBIN], raise_boom) is True
    server.last.deliver(tick(SBIN, 100.0))
    assert feed_log(caplog) == []


def test_exception_in_an_error_hook_is_logged(server, client, caplog):
    def failing_error_hook(exc, context):
        raise RuntimeError("error hook boom")

    client.on_error(failing_error_hook)
    assert client.connect() is True
    assert client.subscribe_ltp([SBIN], raise_boom) is True
    caplog.set_level(logging.ERROR, logger="openalgo.feed")
    server.last.deliver(tick(SBIN, 100.0))
    assert [str(record.exc_info[1]) for record in feed_log(caplog)] == ["error hook boom"]


def test_feed_keeps_delivering_after_an_error_hook_raised(server, client):
    def failing_error_hook(exc, context):
        raise RuntimeError("error hook boom")

    client.on_error(failing_error_hook)
    assert client.connect() is True
    assert client.subscribe_ltp([SBIN], raise_boom) is True
    server.last.deliver(tick(SBIN, 100.0))
    server.last.deliver(tick(SBIN, 101.0))
    assert client.get_ltp()["ltp"]["NSE"]["SBIN"]["ltp"] == 101.0


def test_verbose_console_line_for_a_callback_error_is_unchanged(server, make_client, capsys):
    client = make_client(verbose=1)
    client.on_error(lambda exc, context: None)
    assert client.connect() is True
    assert client.subscribe_ltp([SBIN], raise_boom) is True
    capsys.readouterr()
    server.last.deliver(tick(SBIN, 100.0))
    assert capsys.readouterr().out.splitlines()[-1] == "[ERROR]  LTP callback error: boom"


def test_failed_connection_names_the_url_it_tried_and_the_ws_url_override(
    server, make_client, capsys
):
    server.accept = False
    client = make_client(host="https://trade.example.org", verbose=1, auto_reconnect=False)
    shorten(client, connect=0.05)
    client.connect()
    failure = [line for line in capsys.readouterr().out.splitlines() if "Failed to connect" in line]
    assert failure == [
        "[ERROR]  Failed to connect to WebSocket server at wss://trade.example.org/ws. "
        "If the feed is served at another URL, pass ws_url= to the client."
    ]


def test_refused_connection_returns_without_waiting_for_the_connect_timeout(server, make_client):
    server.accept = False
    client = make_client(auto_reconnect=False)
    shorten(client, connect=2 * WAIT)
    started = time.time()
    assert client.connect() is False
    assert time.time() - started < WAIT


# ---------------------------------------------------------------------------
# Rejections from the server are surfaced, and not acted on
# ---------------------------------------------------------------------------


@pytest.fixture
def key_refused(server, make_client):
    """A client (no automatic reconnect) whose API key the server refuses,
    with a recorder attached. connect() has returned False."""
    server.auth_reply = KEY_REFUSED
    client = make_client(auto_reconnect=False)
    shorten(client, authenticate=0.05)
    recorder = Recorder(client)
    assert client.connect() is False
    return client, recorder


def test_refused_api_key_reaches_the_error_hook_with_the_server_code_and_text(key_refused):
    _, recorder = key_refused
    exc, context = recorder.errors[0]
    assert (exc.code, exc.message, context["code"]) == (
        "AUTHENTICATION_ERROR",
        "Invalid API key",
        "AUTHENTICATION_ERROR",
    )


def test_refused_api_key_is_reported_as_an_authentication_error(key_refused):
    _, recorder = key_refused
    assert recorder.errors[0][1]["kind"] == "authentication"


def test_refused_api_key_fires_no_connect_hook(key_refused):
    client, recorder = key_refused
    assert (client.authenticated, recorder.names()) == (False, [])


def test_refused_api_key_is_logged_at_verbose_zero_when_no_hook_is_registered(
    server, make_client, caplog
):
    caplog.set_level(logging.ERROR, logger="openalgo.feed")
    server.auth_reply = KEY_REFUSED
    client = make_client(auto_reconnect=False)
    shorten(client, authenticate=0.05)
    client.connect()
    message = feed_log(caplog)[0].getMessage()
    assert "AUTHENTICATION_ERROR" in message and "Invalid API key" in message


def test_feed_keeps_retrying_after_the_key_was_refused(server, make_client):
    server.auth_reply = KEY_REFUSED
    client = make_client()  # auto_reconnect on
    shorten(client, authenticate=0.02)
    recorder = Recorder(client)
    client.connect()
    server.last.server_close(4401, "auth timeout")
    assert wait_for(lambda: len(server.sockets) >= 3)
    assert wait_for(lambda: len(recorder.errors) >= 3)


def test_failed_auth_reply_is_reported_as_an_authentication_error(server, make_client):
    server.auth_reply = {"type": "auth", "status": "error", "message": "Key expired"}
    client = make_client(auto_reconnect=False)
    shorten(client, authenticate=0.05)
    recorder = Recorder(client)
    client.connect()
    exc, context = recorder.errors[0]
    assert (context["kind"], exc.message) == ("authentication", "Key expired")


def test_server_error_frame_reaches_the_error_hook(server, connected, recorder):
    server.last.deliver(BROKER_ERROR)
    exc, context = recorder.errors[0]
    assert (context["kind"], exc.code, exc.message, context["message"]) == (
        "server",
        "BROKER_ERROR",
        "Broker adapter not found",
        BROKER_ERROR,
    )


def test_server_error_frame_is_logged_at_verbose_zero_when_no_hook_is_registered(
    server, client, caplog
):
    caplog.set_level(logging.ERROR, logger="openalgo.feed")
    assert client.connect() is True
    server.last.deliver(BROKER_ERROR)
    assert "BROKER_ERROR: Broker adapter not found" in feed_log(caplog)[0].getMessage()


def test_instrument_the_server_refuses_reaches_the_error_hook(server, connected, recorder):
    server.refuse = [INFY]
    assert connected.subscribe_ltp([SBIN, INFY]) is True  # the frame was sent
    assert wait_for(lambda: len(recorder.errors) == 1)
    exc, context = recorder.errors[0]
    assert (exc.code, [entry["symbol"] for entry in context["rejected"]]) == (
        "SUBSCRIBE_REJECTED",
        ["INFY"],
    )


def test_server_rejection_does_not_disconnect_the_feed(server, connected, recorder):
    server.last.deliver(BROKER_ERROR)
    assert (connected.connected, connected.authenticated, recorder.names()) == (
        True,
        True,
        ["connect"],
    )


# ---------------------------------------------------------------------------
# No part of the API key is ever printed or logged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", [API_KEY, "PLACEHOLDER-KEY1"], ids=["long key", "16 characters"])
def test_no_part_of_the_api_key_is_printed_or_logged(server, make_client, caplog, capsys, key):
    caplog.set_level(logging.DEBUG)
    client = make_client(key=key, verbose=2)
    reconnected = []
    client.on_reconnect(reconnected.append)
    assert client.connect() is True
    client.subscribe_ltp([SBIN], raise_boom)
    server.last.deliver(tick(SBIN, 100.0))  # a logged callback error
    server.last.deliver(BROKER_ERROR)  # a logged server error
    server.last.drop()
    assert wait_for(lambda: len(reconnected) == 1)
    client.disconnect()

    captured = capsys.readouterr()
    logged = "\n".join(record.getMessage() + str(record.exc_info) for record in caplog.records)
    output = captured.out + captured.err + logged
    pieces = {key[start : start + 4] for start in range(len(key) - 3)}
    assert "Authenticating" in output  # the authentication step was printed
    assert [piece for piece in sorted(pieces) if piece in output] == []
