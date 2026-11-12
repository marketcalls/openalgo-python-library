"""Quote ticks as the server's broker adapters send them (websocket_proxy): the
average traded price, the last traded quantity, the last trade time, the book
totals and the open interest reach a quote callback and the get_quotes() snapshot
under the SDK's keys. A data callback that edits its update leaves the polled
snapshot as the server sent it.

Fake socket only: see feed_fakes.py. Nothing here reaches the network.
"""

import pytest
from feed_fakes import SBIN, tick
from feed_fixtures import *  # noqa: F403

SERVER_QUOTE = {
    "type": "market_data",
    "exchange": "NSE",
    "symbol": "SBIN",
    "mode": 2,
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
    },
}
# Each SDK key and the value the server sent for it.
EXPECTED = [
    ("avg_trade_price", 809.85),
    ("last_trade_quantity", 3),
    ("ltt", 1762941600000),
    ("total_buy_quantity", 1200),
    ("total_sell_quantity", 950),
    ("oi", 0),
]


@pytest.mark.parametrize("key,value", EXPECTED)
def test_a_quote_callback_carries_what_the_server_sent(server, client, key, value):
    received = []
    assert client.connect() is True
    assert client.subscribe_quote([SBIN], received.append) is True
    server.last.deliver(SERVER_QUOTE)
    assert received[0]["data"][key] == value


@pytest.mark.parametrize("key,value", EXPECTED)
def test_the_polled_quote_carries_what_the_server_sent(server, client, key, value):
    assert client.connect() is True
    server.last.deliver(SERVER_QUOTE)
    assert client.get_quotes()["quote"]["NSE"]["SBIN"][key] == value


def test_depth_callback_that_edits_a_level_leaves_the_polled_depth_as_received(server, client):
    def edits_the_best_bid(message):
        message["data"]["depth"]["buy"][0]["price"] = 0.0

    assert client.connect() is True
    assert client.subscribe_depth([SBIN], edits_the_best_bid) is True
    server.last.deliver(tick(SBIN, 7.0, 3))  # best bid 6.0
    assert client.get_depth()["depth"]["NSE"]["SBIN"]["buyBook"]["1"]["price"] == 6.0
