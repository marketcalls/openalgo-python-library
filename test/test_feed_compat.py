"""What must stay as it was in 2.0.5: the WebSocket URL, the constructor and
its attributes, every method signature, and the frames, payloads and
snapshots of a client that registers no hook. A quote payload and snapshot keep
every 2.0.5 key and have gained the fields the server sends that 2.0.5 dropped.

Fake socket only: see feed_fakes.py. Nothing here reaches the network.
"""

import inspect
import logging

import pytest
from feed_fakes import (
    AUTHENTICATE,
    CLIENT_CLASSES,
    INFY,
    KINDS,
    SBIN,
    Recorder,
    drop_and_recover,
    feed_log,
    message_for,
    order_update,
    subscribe_frame,
    subscribe_kind,
    tick,
)
from feed_fixtures import *  # noqa: F403

import openalgo
from openalgo.feed import FeedAPI

# ---------------------------------------------------------------------------
# The WebSocket URL with ws_port passed is the one 2.0.5 derived (see test_feed_url.py)
# ---------------------------------------------------------------------------


URL_CASES = [
    # host, ws_port, WebSocket URL
    ("http://127.0.0.1:5000", 8765, "ws://127.0.0.1:8765"),
    ("http://example.org:5000/openalgo/v2", 8765, "ws://example.org:8765"),
    ("example.org:5000/path", 9001, "ws://example.org:9001"),
    ("https://example.org", 8765, "ws://example.org:8765"),
    ("https://example.org:8443/openalgo", 9443, "ws://example.org:9443"),
    ("http://[::1]:5000", 8765, "ws://[::1]:8765"),
    ("HTTPS://Example.org", 8765, "ws://Example.org:8765"),
]


@CLIENT_CLASSES
@pytest.mark.parametrize("host, ws_port, expected", URL_CASES)
def test_ws_url_takes_the_hostname_from_host_and_the_port_from_ws_port(
    make_client, cls, host, ws_port, expected
):
    assert make_client(cls, host=host, ws_port=ws_port).ws_url == expected


@CLIENT_CLASSES
def test_default_host_gives_the_local_plain_ws_url(make_client, cls):
    assert make_client(cls).ws_url == "ws://127.0.0.1:8765"


@CLIENT_CLASSES
@pytest.mark.parametrize(
    "host, ws_url",
    [
        ("https://example.org", "wss://example.org/ws"),
        ("https://example.org", "ws://10.0.0.5:9999/feed"),
        ("http://example.org:5000", "wss://example.org/ws"),
    ],
)
def test_explicit_ws_url_is_used_verbatim(make_client, cls, host, ws_url):
    assert make_client(cls, host=host, ws_url=ws_url).ws_url == ws_url


@CLIENT_CLASSES
def test_ws_host_is_not_set_when_ws_url_is_given(make_client, cls):
    client = make_client(cls, host="https://example.org", ws_url="wss://example.org/ws")
    assert not hasattr(client, "ws_host")


@CLIENT_CLASSES
def test_ws_host_is_the_bare_hostname(make_client, cls):
    assert make_client(cls, host="https://example.org:8443/openalgo").ws_host == "example.org"


def test_connect_opens_the_derived_url(server, make_client):
    client = make_client(host="https://example.org")
    assert client.connect() is True
    assert server.last.url == "wss://example.org/ws"


# ---------------------------------------------------------------------------
# FeedAPI and the unified api class share one feed initialiser
# ---------------------------------------------------------------------------


FEED_DEFAULTS = {
    "verbose": 0,
    "ws_port": 8765,
    "ws_host": "127.0.0.1",
    "ws_url": "ws://127.0.0.1:8765",
    "ws": None,
    "connected": False,
    "authenticated": False,
    "ws_thread": None,
    "message_queue": [],
    "ltp_data": {},
    "quotes_data": {},
    "depth_data": {},
    "ltp_callback": None,
    "quote_callback": None,
    "quotes_callback": None,
    "depth_callback": None,
    "order_callback": None,
    "order_data": [],
    "_orders_subscribed": False,
    "auto_reconnect": True,
    "_shutting_down": False,
    "_reconnect_thread": None,
    "_active_subs": {1: {}, 2: {}, 3: {}},
}


@CLIENT_CLASSES
@pytest.mark.parametrize("name, expected", sorted(FEED_DEFAULTS.items()))
def test_existing_feed_attribute_keeps_its_default(make_client, cls, name, expected):
    assert getattr(make_client(cls), name) == expected


@CLIENT_CLASSES
@pytest.mark.parametrize("verbose, level", [(False, 0), (0, 0), (True, 1), (1, 1), (2, 2)])
def test_verbose_argument_maps_to_the_same_level_as_before(make_client, cls, verbose, level):
    assert make_client(cls, verbose=verbose).verbose == level


@CLIENT_CLASSES
def test_auto_reconnect_argument_is_stored(make_client, cls):
    assert make_client(cls, auto_reconnect=False).auto_reconnect is False


def test_unified_client_has_exactly_the_feed_attributes_of_feed_api(make_client):
    assert set(vars(make_client(openalgo.api))) == set(vars(make_client(FeedAPI)))


def test_unified_client_fires_the_connection_hooks(server, make_client):
    client = make_client(openalgo.api)
    recorder = Recorder(client)
    assert client.connect() is True
    drop_and_recover(server, recorder)
    assert recorder.names() == ["connect", "disconnect", "connect", "reconnect"]


def test_unified_client_replays_subscriptions_after_a_drop(server, make_client):
    client = make_client(openalgo.api)
    recorder = Recorder(client)
    assert client.connect() is True
    assert client.subscribe_ltp([SBIN]) is True
    drop_and_recover(server, recorder)
    assert server.last.frames("subscribe") == [subscribe_frame(1, SBIN)]


# ---------------------------------------------------------------------------
# Every method that existed in 2.0.5 keeps its signature
# ---------------------------------------------------------------------------


SIGNATURES_2_0_5 = {
    "_log": "(self, level: int, category: str, message: str) -> None",
    "connect": "(self) -> bool",
    "_do_connect": "(self) -> bool",
    "disconnect": "(self) -> None",
    "_schedule_reconnect": "(self) -> None",
    "_reconnect_loop": "(self) -> None",
    "_replay_subscriptions": "(self) -> None",
    "_send_subscribe_msg": (
        "(self, mode: int, instruments: List[Dict[str, Any]], depth: int = 5) -> bool"
    ),
    "_send_unsubscribe_msg": "(self, mode: int, instruments: List[Dict[str, Any]]) -> bool",
    "_authenticate": "(self) -> None",
    "_process_message": "(self, message_str: str) -> None",
    "subscribe_ltp": (
        "(self, instruments: List[Dict[str, Any]], "
        "on_data_received: Optional[Callable] = None) -> bool"
    ),
    "unsubscribe_ltp": "(self, instruments: List[Dict[str, Any]]) -> bool",
    "subscribe_quote": (
        "(self, instruments: List[Dict[str, Any]], "
        "on_data_received: Optional[Callable] = None) -> bool"
    ),
    "unsubscribe_quote": "(self, instruments: List[Dict[str, Any]]) -> bool",
    "subscribe_depth": (
        "(self, instruments: List[Dict[str, Any]], "
        "on_data_received: Optional[Callable] = None) -> bool"
    ),
    "unsubscribe_depth": "(self, instruments: List[Dict[str, Any]]) -> bool",
    "subscribe_orders": "(self, on_order_update: Optional[Callable] = None) -> bool",
    "unsubscribe_orders": "(self) -> bool",
    "get_orders": "(self) -> Dict[str, Any]",
    "get_ltp": "(self, exchange: str = None, symbol: str = None) -> Dict[str, Any]",
    "get_quotes": "(self, exchange: str = None, symbol: str = None) -> Dict[str, Any]",
    "get_depth": "(self, exchange: str = None, symbol: str = None) -> Dict[str, Any]",
}


@pytest.mark.parametrize("name, signature", sorted(SIGNATURES_2_0_5.items()))
def test_method_keeps_its_2_0_5_signature(name, signature):
    assert str(inspect.signature(getattr(FeedAPI, name))) == signature.replace("typing.", "")


def test_constructor_keeps_its_2_0_5_parameters():
    assert list(inspect.signature(FeedAPI.__init__).parameters) == [
        "self",
        "api_key",
        "host",
        "version",
        "ws_port",
        "ws_url",
        "verbose",
        "auto_reconnect",
    ]


def test_subclass_overriding_process_message_with_the_old_signature_still_connects(make_client):
    class Mine(FeedAPI):
        def _process_message(self, message_str):
            return super()._process_message(message_str)

    assert make_client(Mine).connect() is True


def test_subclass_overriding_authenticate_with_the_old_signature_still_connects(make_client):
    class Mine(FeedAPI):
        def _authenticate(self):
            return super()._authenticate()

    assert make_client(Mine).connect() is True


# ---------------------------------------------------------------------------
# Behaviour that must not change when no hook is registered
# ---------------------------------------------------------------------------


def test_connect_sends_the_same_authenticate_frame(server, client):
    assert client.connect() is True
    assert server.last.sent == [AUTHENTICATE]


def test_connect_runs_the_socket_with_the_same_keepalive_settings(server, client):
    assert client.connect() is True
    assert server.last.run_kwargs == {"ping_interval": 20, "ping_timeout": 10}


def test_subscribe_calls_send_the_same_frames(server, client):
    assert client.connect() is True
    client.subscribe_ltp([SBIN, {"exchange": "NSE", "exchange_token": "INFY"}])
    client.subscribe_quote([SBIN])
    client.subscribe_depth([SBIN])
    client.subscribe_orders()
    assert server.last.sent[1:] == [
        subscribe_frame(1, SBIN, INFY),
        subscribe_frame(2, SBIN),
        subscribe_frame(3, SBIN),
        {"action": "subscribe_orders"},
    ]


def test_unsubscribe_calls_send_the_same_frames(server, client):
    assert client.connect() is True
    client.subscribe_ltp([SBIN])
    client.subscribe_orders()
    client.unsubscribe_ltp([SBIN])
    client.unsubscribe_orders()
    assert server.last.sent[3:] == [
        {"action": "unsubscribe", "symbols": [{"symbol": "SBIN", "exchange": "NSE", "mode": 1}]},
        {"action": "unsubscribe_orders"},
    ]


def test_instrument_without_a_symbol_is_refused(server, client):
    assert client.connect() is True
    assert client.subscribe_ltp([{"exchange": "NSE"}]) is False
    assert server.last.frames("subscribe") == []


CALLBACK_PAYLOADS = {
    "ltp": {
        "type": "market_data",
        "symbol": "SBIN",
        "exchange": "NSE",
        "mode": 1,
        "data": {"ltp": 7.0, "timestamp": 1700000000000},
    },
    "quote": {
        "type": "market_data",
        "symbol": "SBIN",
        "exchange": "NSE",
        "mode": 2,
        "data": {
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": 1.5,
            "ltp": 7.0,
            "volume": 10,
            "last_trade_quantity": 0,
            "avg_trade_price": 0,
            "change": 0,
            "change_percent": 0,
            # Added after 2.0.5, 0 when the server sends none (test_feed_quotes.py).
            "ltt": 0,
            "total_buy_quantity": 0,
            "total_sell_quantity": 0,
            "oi": 0,
            "timestamp": 1700000000000,
        },
    },
    "depth": {
        "type": "market_data",
        "symbol": "SBIN",
        "exchange": "NSE",
        "mode": 3,
        "data": {
            "ltp": 7.0,
            "timestamp": 1700000000000,
            "depth": {
                "buy": [{"price": 6.0, "quantity": 5, "orders": 1}],
                "sell": [{"price": 8.0, "quantity": 7, "orders": 2}],
            },
        },
    },
    "order": order_update("7"),
}


@pytest.mark.parametrize("kind", KINDS)
def test_data_callback_receives_the_same_payload(server, client, kind):
    received = []
    assert client.connect() is True
    assert subscribe_kind(client, kind, received.append) is True
    server.last.deliver(message_for(kind, 7))
    assert received == [CALLBACK_PAYLOADS[kind]]


def test_polled_ltp_snapshot_is_unchanged(server, client):
    assert client.connect() is True
    server.last.deliver(tick(SBIN, 7.0))
    assert client.get_ltp() == {"ltp": {"NSE": {"SBIN": {"timestamp": 1700000000000, "ltp": 7.0}}}}


def test_polled_quote_snapshot_is_unchanged(server, client):
    assert client.connect() is True
    server.last.deliver(tick(SBIN, 7.0, 2))
    assert client.get_quotes()["quote"]["NSE"]["SBIN"] == {
        "timestamp": 1700000000000,
        "open": 1.0,
        "high": 2.0,
        "low": 0.5,
        "close": 1.5,
        "ltp": 7.0,
        "volume": 10,
        "last_trade_quantity": 0,
        "avg_trade_price": 0,
        "change": 0,
        "change_percent": 0,
        # Added after 2.0.5, 0 when the server sends none (test_feed_quotes.py).
        "ltt": 0,
        "total_buy_quantity": 0,
        "total_sell_quantity": 0,
        "oi": 0,
    }


def test_polled_depth_snapshot_is_unchanged(server, client):
    assert client.connect() is True
    server.last.deliver(tick(SBIN, 7.0, 3))
    assert client.get_depth()["depth"]["NSE"]["SBIN"]["buyBook"]["1"] == {
        "price": 6.0,
        "qty": 5,
        "orders": 1,
    }


def test_polled_order_updates_are_unchanged(server, client):
    assert client.connect() is True
    server.last.deliver(order_update("7"))
    assert client.get_orders() == {"orders": [order_update("7")]}


def test_unsubscribe_drops_the_cached_snapshot(server, client):
    assert client.connect() is True
    client.subscribe_ltp([SBIN])
    server.last.deliver(tick(SBIN, 7.0))
    client.unsubscribe_ltp([SBIN])
    assert client.get_ltp() == {"ltp": {}}


def test_nothing_is_logged_while_callbacks_succeed(server, client, caplog):
    received = []
    caplog.set_level(logging.DEBUG, logger="openalgo.feed")
    assert client.connect() is True
    assert client.subscribe_ltp([SBIN], received.append) is True
    server.last.deliver(tick(SBIN, 7.0))
    assert feed_log(caplog) == []


def test_disconnect_leaves_the_same_public_state(connected):
    connected.disconnect()
    assert (connected.ws, connected.connected, connected.authenticated) == (None, False, False)


def test_shutting_down_flag_is_set_by_disconnect_and_cleared_by_connect(connected):
    connected.disconnect()
    after_disconnect = connected._shutting_down
    connected.connect()
    assert (after_disconnect, connected._shutting_down) == (True, False)
