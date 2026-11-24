"""Depth ticks as the server's broker adapters send them (websocket_proxy): a depth
frame carries the quote fields as well as the book, and OHLC, volume, OI and the
rest reach a depth callback under the SDK's quote keys.

Fake socket only: see feed_fakes.py. Nothing here reaches the network.
"""

import pytest
from feed_fakes import SBIN
from feed_fixtures import *  # noqa: F403

SERVER_DEPTH = {
    "type": "market_data",
    "exchange": "NSE",
    "symbol": "SBIN",
    "mode": 3,
    "data": {
        "ltp": 812.4,
        "ltt": 1762941600000,
        "timestamp": 1762941600500,
        "volume": 4210,
        "open": 805.0,
        "high": 815.0,
        "low": 801.2,
        "close": 806.0,
        "last_quantity": 3,
        "average_price": 809.85,
        "total_buy_quantity": 1200,
        "total_sell_quantity": 950,
        "oi": 56000,
        "depth": {
            "buy": [{"price": 812.3, "quantity": 40, "orders": 2}],
            "sell": [{"price": 812.5, "quantity": 25, "orders": 1}],
        },
    },
}
# Each SDK key and the value the server sent for it.
EXPECTED = [
    ("open", 805.0),
    ("high", 815.0),
    ("low", 801.2),
    ("close", 806.0),
    ("volume", 4210),
    ("oi", 56000),
    ("avg_trade_price", 809.85),
    ("last_trade_quantity", 3),
    ("ltt", 1762941600000),
    ("total_buy_quantity", 1200),
    ("total_sell_quantity", 950),
]


@pytest.mark.parametrize("key,value", EXPECTED)
def test_a_depth_callback_carries_what_the_server_sent(server, client, key, value):
    received = []
    assert client.connect() is True
    assert client.subscribe_depth([SBIN], on_data_received=received.append) is True
    server.last.deliver(SERVER_DEPTH)
    assert received[0]["data"][key] == value


def test_a_depth_callback_still_carries_the_price_and_the_book(server, client):
    received = []
    assert client.connect() is True
    assert client.subscribe_depth([SBIN], on_data_received=received.append) is True
    server.last.deliver(SERVER_DEPTH)
    data = received[0]["data"]
    assert (data["ltp"], data["timestamp"]) == (812.4, 1762941600500)
    assert data["depth"] == SERVER_DEPTH["data"]["depth"]


def test_the_polled_depth_snapshot_keeps_its_shape(server, client):
    assert client.connect() is True
    server.last.deliver(SERVER_DEPTH)
    snapshot = client.get_depth()["depth"]["NSE"]["SBIN"]
    assert list(snapshot) == ["timestamp", "ltp", "buyBook", "sellBook"]
    assert snapshot["buyBook"]["1"] == {"price": 812.3, "qty": 40, "orders": 2}
