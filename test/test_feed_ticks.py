"""The shape of stored and delivered market data (openalgo._feed_ticks). Pure functions."""

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from openalgo import _feed_ticks as ticks  # noqa: E402

STAMP = 1700000000000


@pytest.fixture
def frozen_clock(monkeypatch):
    """The local clock reads 1234.5 seconds, that is 1234500 milliseconds."""
    monkeypatch.setattr(ticks.time, "time", lambda: 1234.5)
    return 1234500


def test_ltp_update_holds_price_and_timestamp():
    assert ticks.ltp_update({"ltp": 7.0, "timestamp": STAMP}) == {"ltp": 7.0, "timestamp": STAMP}


def test_ltp_update_takes_the_last_trade_time_from_the_top_level():
    assert ticks.ltp_update({"ltp": 7.0, "timestamp": STAMP, "ltt": 5})["ltt"] == 5


def test_ltp_update_takes_the_last_trade_time_from_the_nested_data():
    assert ticks.ltp_update({"ltp": 7.0, "timestamp": STAMP, "data": {"ltt": 6}})["ltt"] == 6


def test_ltp_update_has_no_last_trade_time_when_the_server_sent_none():
    assert "ltt" not in ticks.ltp_update({"ltp": 7.0, "timestamp": STAMP})


@pytest.mark.parametrize("build", [ticks.ltp_update, ticks.quote_update, ticks.depth_update])
def test_missing_timestamp_is_filled_with_the_local_time_in_milliseconds(build, frozen_clock):
    assert build({"ltp": 7.0})["timestamp"] == frozen_clock


def test_quote_update_holds_the_quote_fields_in_their_order_then_the_timestamp():
    assert list(ticks.quote_update({"timestamp": STAMP})) == [
        "open",
        "high",
        "low",
        "close",
        "ltp",
        "volume",
        "last_trade_quantity",
        "avg_trade_price",
        "change",
        "change_percent",
        "ltt",
        "total_buy_quantity",
        "total_sell_quantity",
        "oi",
        "timestamp",
    ]


def test_quote_field_the_server_left_out_is_zero():
    assert ticks.quote_update({"ltp": 7.0, "timestamp": STAMP})["volume"] == 0


# A quote frame's data as the server's broker adapters send it (websocket_proxy).
SERVER_QUOTE = {
    "ltp": 6512.0,
    "ltt": 1762941600000,
    "timestamp": STAMP,
    "volume": 4210,
    "last_quantity": 3,
    "average_price": 6498.35,
    "total_buy_quantity": 1200,
    "total_sell_quantity": 950,
    "oi": 15800,
}


def test_quote_update_reads_the_average_price_the_server_sends():
    assert ticks.quote_update(SERVER_QUOTE)["avg_trade_price"] == 6498.35


def test_quote_update_reads_the_last_quantity_the_server_sends():
    assert ticks.quote_update(SERVER_QUOTE)["last_trade_quantity"] == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("ltt", 1762941600000),
        ("total_buy_quantity", 1200),
        ("total_sell_quantity", 950),
        ("oi", 15800),
    ],
)
def test_quote_update_keeps_the_trade_time_the_book_totals_and_open_interest(field, value):
    assert ticks.quote_update(SERVER_QUOTE)[field] == value


def test_quote_update_still_reads_a_server_that_sends_the_sdk_names():
    update = ticks.quote_update({"avg_trade_price": 9.5, "last_trade_quantity": 4})
    assert (update["avg_trade_price"], update["last_trade_quantity"]) == (9.5, 4)


def test_quote_update_ignores_a_field_that_is_not_a_quote_field():
    assert "bid" not in ticks.quote_update({"bid": 6.9, "timestamp": STAMP})


def test_depth_update_holds_price_timestamp_and_both_sides():
    book = {"buy": [{"price": 6.0}], "sell": [{"price": 8.0}]}
    assert ticks.depth_update({"ltp": 7.0, "timestamp": STAMP, "depth": book}) == {
        "ltp": 7.0,
        "timestamp": STAMP,
        "depth": book,
    }


def test_callback_message_names_the_instrument_and_the_mode():
    message = {"type": "market_data", "symbol": "SBIN", "exchange": "NSE", "mode": 2, "data": {}}
    assert ticks.callback_message(message, {"ltp": 7.0}) == {
        "type": "market_data",
        "symbol": "SBIN",
        "exchange": "NSE",
        "mode": 2,
        "data": {"ltp": 7.0},
    }


def side(price, quantity, orders):
    """One side of a ladder row, in the column widths 2.0.5 printed."""
    return f"{price:<10} {quantity:<10} {orders:<8}"


def test_depth_ladder_prints_the_lines_2_0_5_printed():
    depth_data = {
        "depth": {
            "buy": [{"price": 6.0, "quantity": 5, "orders": 1}, {"price": 5.5, "quantity": 9}],
            "sell": [{"price": 8.0, "quantity": 7, "orders": 2}],
        }
    }
    indent = " " * 9
    assert ticks.depth_ladder(depth_data) == [
        f"{indent}{'BUY':<30} | {'SELL':<30}",
        f"{indent}{side('Price', 'Qty', 'Orders')} | {side('Price', 'Qty', 'Orders')}",
        f"{indent}{'-' * 62}",
        f"{indent}{side('6.0', '5', '1')} | {side('8.0', '7', '2')}",
        f"{indent}{side('5.5', '9', '-')} | {side('-', '-', '-')}",
    ]


def test_depth_ladder_of_an_empty_book_has_one_row_of_dashes():
    rows = ticks.depth_ladder({"depth": {"buy": [], "sell": []}})[3:]
    assert rows == [f"{' ' * 9}{side('-', '-', '-')} | {side('-', '-', '-')}"]
