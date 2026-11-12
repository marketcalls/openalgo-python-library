"""The requested subscriptions, the order stream included: every call updates
them, a frame is sent when it can go, and the set is replayed whenever a
connection authenticates.

Fake socket only: see feed_fakes.py. Nothing here reaches the network.
"""

import threading
import time

import pytest
from feed_fakes import (
    AUTHENTICATE,
    INFY,
    KINDS,
    NIFTY,
    NOTHING_REQUESTED,
    QUIET,
    SBIN,
    TCS,
    WAIT,
    Recorder,
    come_back,
    drop_and_recover,
    go_down,
    shorten,
    subscribe_frame,
    tick,
    wait_for,
)
from feed_fixtures import *  # noqa: F403

# ---------------------------------------------------------------------------
# Requests while the socket is up, down, or not yet connected
# ---------------------------------------------------------------------------


REQUESTS = [
    ("subscribe_ltp", ([SBIN],)),
    ("subscribe_quote", ([SBIN],)),
    ("subscribe_depth", ([SBIN],)),
    ("subscribe_orders", ()),
    ("unsubscribe_ltp", ([SBIN],)),
    ("unsubscribe_quote", ([SBIN],)),
    ("unsubscribe_depth", ([SBIN],)),
    ("unsubscribe_orders", ()),
]


REQUEST_IDS = [name for name, _ in REQUESTS]


def test_reconnect_replays_every_requested_subscription_once(server, connected, recorder):
    connected.subscribe_ltp([SBIN, INFY])
    connected.subscribe_quote([TCS])
    connected.subscribe_depth([NIFTY])
    connected.subscribe_orders()
    drop_and_recover(server, recorder)
    assert server.last.sent == [
        AUTHENTICATE,
        subscribe_frame(1, SBIN, INFY),
        subscribe_frame(2, TCS),
        subscribe_frame(3, NIFTY),
        {"action": "subscribe_orders"},
    ]


@pytest.mark.parametrize("method, args", REQUESTS, ids=REQUEST_IDS)
def test_request_returns_false_while_the_socket_is_down_as_in_2_0_5(
    server, connected, method, args
):
    go_down(server, connected)
    assert getattr(connected, method)(*args) is False


@pytest.mark.parametrize("method, args", REQUESTS, ids=REQUEST_IDS)
def test_request_returns_false_before_the_first_connect_as_in_2_0_5(client, method, args):
    assert getattr(client, method)(*args) is False


@pytest.mark.parametrize("method, args", REQUESTS, ids=REQUEST_IDS)
def test_request_returns_false_after_disconnect_as_in_2_0_5(connected, method, args):
    connected.disconnect()
    assert getattr(connected, method)(*args) is False


@pytest.mark.parametrize("kind", ["ltp", "quote", "depth"])
def test_instrument_unsubscribed_while_reconnecting_is_not_replayed(
    server, connected, recorder, kind
):
    getattr(connected, f"subscribe_{kind}")([SBIN, INFY])
    go_down(server, connected)
    getattr(connected, f"unsubscribe_{kind}")([SBIN])
    come_back(server, recorder)
    assert server.last.sent == [AUTHENTICATE, subscribe_frame(KINDS.index(kind) + 1, INFY)]


def test_unsubscribe_while_reconnecting_drops_the_cached_snapshot(server, connected):
    connected.subscribe_ltp([SBIN])
    server.last.deliver(tick(SBIN, 7.0))
    go_down(server, connected)
    connected.unsubscribe_ltp([SBIN])
    assert connected.get_ltp() == {"ltp": {}}


def test_instrument_subscribed_while_reconnecting_is_replayed(server, connected, recorder):
    connected.subscribe_ltp([SBIN])
    go_down(server, connected)
    connected.subscribe_ltp([INFY])
    come_back(server, recorder)
    assert server.last.sent == [AUTHENTICATE, subscribe_frame(1, SBIN, INFY)]


def test_callback_given_while_reconnecting_receives_ticks_afterwards(server, connected, recorder):
    received = []
    go_down(server, connected)
    connected.subscribe_ltp([INFY], received.append)
    come_back(server, recorder)
    server.last.deliver(tick(INFY, 7.0))
    assert [message["symbol"] for message in received] == ["INFY"]


def test_order_stream_unsubscribed_while_reconnecting_is_not_replayed(server, connected, recorder):
    connected.subscribe_orders()
    go_down(server, connected)
    connected.unsubscribe_orders()
    come_back(server, recorder)
    assert server.last.sent == [AUTHENTICATE]


def test_order_stream_subscribed_while_reconnecting_is_replayed(server, connected, recorder):
    go_down(server, connected)
    connected.subscribe_orders()
    come_back(server, recorder)
    assert server.last.sent == [AUTHENTICATE, {"action": "subscribe_orders"}]


def test_order_stream_requested_when_the_send_fails_is_replayed_like_market_data(
    server, connected, recorder
):
    server.last.break_transport()  # send() fails, the close has not been delivered yet
    requested = (connected.subscribe_ltp([SBIN]), connected.subscribe_orders())
    drop_and_recover(server, recorder)
    assert (requested, server.last.sent) == (
        (False, False),
        [AUTHENTICATE, subscribe_frame(1, SBIN), {"action": "subscribe_orders"}],
    )


def test_replayed_set_is_what_the_user_last_asked_for(server, connected, recorder):
    shorten(connected, connect=0.02)
    connected.subscribe_ltp([SBIN, INFY, TCS])
    server.accept = False  # stay down across several refused attempts
    server.last.drop()
    connected.unsubscribe_ltp([SBIN])
    connected.subscribe_ltp([NIFTY])
    connected.unsubscribe_ltp([NIFTY])
    connected.subscribe_ltp([SBIN])
    connected.unsubscribe_ltp([INFY])
    assert wait_for(lambda: len(server.sockets) >= 4)
    server.accept = True
    assert wait_for(lambda: recorder.count("reconnect") == 1)
    assert server.last.sent == [AUTHENTICATE, subscribe_frame(1, TCS, SBIN)]


def test_unsubscribe_issued_during_the_replay_is_sent_after_it(server, connected, recorder):
    connected.subscribe_ltp([SBIN, INFY])
    replay_started = threading.Event()
    release_replay = threading.Event()
    send_subscribe = connected._send_subscribe_msg

    def held_send(mode, instruments, depth=5):
        replay_started.set()
        release_replay.wait(WAIT)
        return send_subscribe(mode, instruments, depth)

    go_down(server, connected)
    connected._send_subscribe_msg = held_send
    server.gate.set()
    assert replay_started.wait(WAIT)
    unsubscribe = threading.Thread(target=connected.unsubscribe_ltp, args=([SBIN],))
    unsubscribe.start()
    time.sleep(QUIET)  # the unsubscribe has to wait for the replay
    release_replay.set()
    unsubscribe.join(WAIT)

    assert server.last.sent[1:] == [
        subscribe_frame(1, SBIN, INFY),
        {"action": "unsubscribe", "symbols": [{"symbol": "SBIN", "exchange": "NSE", "mode": 1}]},
    ]


def test_subscription_requested_before_the_first_connect_is_sent_when_it_authenticates(
    server, client
):
    client.subscribe_ltp([SBIN])
    client.subscribe_orders()
    assert client.connect() is True
    assert server.last.sent == [
        AUTHENTICATE,
        subscribe_frame(1, SBIN),
        {"action": "subscribe_orders"},
    ]


def test_subscription_requested_after_a_refused_first_connect_is_sent_when_it_authenticates(
    server, make_client
):
    server.accept = False
    client = make_client()
    shorten(client, connect=0.05)
    client.connect()
    client.subscribe_ltp([SBIN])
    server.accept = True
    assert wait_for(lambda: client.authenticated)
    assert wait_for(lambda: server.last.sent == [AUTHENTICATE, subscribe_frame(1, SBIN)])


def test_unsubscribe_before_the_first_connect_cancels_the_earlier_request(server, client):
    client.subscribe_ltp([SBIN])
    client.unsubscribe_ltp([SBIN])
    assert client.connect() is True
    assert server.last.sent == [AUTHENTICATE]


def test_disconnect_forgets_the_requested_subscriptions(connected):
    connected.subscribe_ltp([SBIN])
    connected.subscribe_orders()
    connected.disconnect()
    assert connected.get_subscriptions() == NOTHING_REQUESTED


def test_connect_after_disconnect_sends_no_old_subscription(server, connected):
    connected.subscribe_ltp([SBIN])
    connected.disconnect()
    assert connected.connect() is True
    assert server.last.sent == [AUTHENTICATE]


def test_subscription_requested_after_disconnect_is_sent_by_the_next_connect(server, connected):
    connected.disconnect()
    connected.subscribe_ltp([INFY])
    assert connected.connect() is True
    assert server.last.sent == [AUTHENTICATE, subscribe_frame(1, INFY)]


def test_subscription_from_before_a_disconnect_is_not_resurrected_by_a_later_drop(
    server, connected, recorder
):
    connected.subscribe_ltp([SBIN])
    connected.disconnect()
    assert connected.connect() is True
    connected.subscribe_ltp([INFY])
    drop_and_recover(server, recorder)
    assert server.last.sent == [AUTHENTICATE, subscribe_frame(1, INFY)]


# ---------------------------------------------------------------------------
# Without auto_reconnect
# ---------------------------------------------------------------------------


@pytest.fixture
def dropped_without_auto_reconnect(server, make_client):
    """A client with auto_reconnect=False whose socket has just dropped."""
    client = make_client(auto_reconnect=False)
    recorder = Recorder(client)
    assert client.connect() is True
    assert client.subscribe_ltp([SBIN]) is True
    server.last.drop()
    return client, recorder


def test_without_auto_reconnect_a_drop_opens_no_new_socket(server, dropped_without_auto_reconnect):
    time.sleep(QUIET)
    assert len(server.sockets) == 1


def test_without_auto_reconnect_the_disconnect_hook_says_no_reconnect(
    dropped_without_auto_reconnect,
):
    _, recorder = dropped_without_auto_reconnect
    assert recorder.infos("disconnect")[0]["will_reconnect"] is False


def test_without_auto_reconnect_a_manual_connect_replays_nothing_as_in_2_0_5(
    server, dropped_without_auto_reconnect
):
    client, _ = dropped_without_auto_reconnect
    assert client.connect() is True
    assert server.last.sent == [AUTHENTICATE]


def test_without_auto_reconnect_a_request_made_while_down_is_sent_by_the_manual_connect(
    server, dropped_without_auto_reconnect
):
    client, _ = dropped_without_auto_reconnect
    client.subscribe_quote([TCS])
    assert client.connect() is True
    assert server.last.sent == [AUTHENTICATE, subscribe_frame(2, TCS)]


def test_without_auto_reconnect_a_manual_connect_fires_no_reconnect_hook(
    dropped_without_auto_reconnect,
):
    client, recorder = dropped_without_auto_reconnect
    assert client.connect() is True
    assert wait_for(lambda: recorder.names() == ["connect", "disconnect", "connect"])


# ---------------------------------------------------------------------------
# get_subscriptions
# ---------------------------------------------------------------------------


def test_get_subscriptions_lists_what_was_requested(connected):
    connected.subscribe_ltp([SBIN])
    connected.subscribe_depth([NIFTY])
    connected.subscribe_orders()
    assert connected.get_subscriptions() == {
        "ltp": [SBIN],
        "quote": [],
        "depth": [NIFTY],
        "orders": True,
    }


def test_get_subscriptions_reflects_requests_made_while_reconnecting(server, connected):
    connected.subscribe_ltp([SBIN, INFY])
    go_down(server, connected)
    connected.unsubscribe_ltp([SBIN])
    connected.subscribe_quote([TCS])
    assert connected.get_subscriptions() == {
        "ltp": [INFY],
        "quote": [TCS],
        "depth": [],
        "orders": False,
    }


def test_get_subscriptions_returns_a_copy(connected):
    connected.subscribe_ltp([SBIN])
    snapshot = connected.get_subscriptions()
    snapshot["ltp"][0]["symbol"] = "CHANGED"
    snapshot["quote"].append(TCS)
    assert connected.get_subscriptions()["ltp"] == [SBIN]
