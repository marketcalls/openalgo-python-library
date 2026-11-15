"""The tick adapter: one market-data message of the SDK's feed as the shared `Tick`.

The messages are shaped as the SDK's data callbacks receive them (openalgo/feed.py and
openalgo/_feed_ticks.py: LTP data holds ltp, timestamp and the ltt the server sent; quote
data every quote field, 0 for one the server left out; depth data ltp, timestamp and the
book), and the last trade times as the OpenAlgo server's broker adapters send them
(broker/*/streaming/: epoch milliseconds, epoch seconds, seconds as a string of digits,
seconds counted from 1980, clock times and dates as text).
"""

from __future__ import annotations

import math

import pytest

from openalgo.backtest._live._types import Tick
from openalgo.backtest._live.ticks import TickTally, tick_from_feed

NS = 1_000_000_000
DAY_S = 86_400
#: Monday 2026-11-16, 10:00:00 IST, in UTC seconds (the NSE open, 09:15 IST, is
#: 1794800700).
RECV_S = 1_794_803_400
RECV_NS = RECV_S * NS
#: 09:59:59 IST of the same day: the last trade, one second before receipt.
TRADE_S = RECV_S - 1
TRADE_NS = TRADE_S * NS


def ltp_message(data: dict, symbol: str = "SBIN", exchange: str = "NSE") -> dict:
    return {"type": "market_data", "symbol": symbol, "exchange": exchange, "mode": 1, "data": data}


def quote_message(**fields) -> dict:
    """A quote callback message: every quote field, 0 where the server sent none."""
    data = {
        "open": 806.0,
        "high": 815.5,
        "low": 804.1,
        "close": 803.9,
        "ltp": 812.4,
        "volume": 1_904_211,
        "last_trade_quantity": 25,
        "avg_trade_price": 810.77,
        "change": 0,
        "change_percent": 0,
        "ltt": TRADE_S * 1000,
        "total_buy_quantity": 120_400,
        "total_sell_quantity": 98_100,
        "oi": 0,
        "timestamp": RECV_S * 1000,
    }
    data.update(fields)
    return {"type": "market_data", "symbol": "SBIN", "exchange": "NSE", "mode": 2, "data": data}


def depth_message(buy: list, sell: list, **fields) -> dict:
    data = {"ltp": 812.4, "timestamp": RECV_S * 1000, "depth": {"buy": buy, "sell": sell}}
    data.update(fields)
    return {"type": "market_data", "symbol": "SBIN", "exchange": "NSE", "mode": 3, "data": data}


def level(price: float, quantity: int = 100, orders: int = 3) -> dict:
    return {"price": price, "quantity": quantity, "orders": orders}


def adapt(message: object) -> tuple[Tick | None, TickTally]:
    tally = TickTally()
    return tick_from_feed(message, recv_ns=RECV_NS, tally=tally), tally


# ----- whole messages ---------------------------------------------------------------


def test_a_quote_message_becomes_a_tick_with_its_price_volume_and_trade_time():
    tick, _ = adapt(quote_message())

    assert tick == Tick("SBIN", "NSE", 812.4, RECV_NS, TRADE_NS, 1_904_211.0, None, None)


def test_an_ltp_message_becomes_a_tick_without_a_volume():
    tick, _ = adapt(ltp_message({"ltp": 812.4, "timestamp": RECV_S * 1000, "ltt": TRADE_S}))

    assert tick == Tick("SBIN", "NSE", 812.4, RECV_NS, TRADE_NS, None, None, None)


def test_a_depth_message_becomes_a_tick_with_the_best_bid_and_ask():
    tick, _ = adapt(depth_message([level(812.35), level(812.3)], [level(812.45), level(812.5)]))

    assert tick == Tick("SBIN", "NSE", 812.4, RECV_NS, None, None, 812.35, 812.45)


def test_the_receive_time_is_the_one_given_not_the_servers_timestamp():
    tick, _ = adapt(quote_message(timestamp=(RECV_S - 60) * 1000))

    assert tick.ts_recv_ns == RECV_NS


def test_a_quote_message_has_no_bid_or_ask_even_when_its_data_holds_a_book():
    # The server hands depth data to a quote subscriber with the mode set to quote.
    tick, _ = adapt(quote_message(depth={"buy": [level(812.35)], "sell": [level(812.45)]}))

    assert (tick.bid, tick.ask) == (None, None)


def test_the_symbol_and_exchange_come_from_the_message():
    tick, _ = adapt(ltp_message({"ltp": 24_310.5}, symbol="NIFTY", exchange="NSE_INDEX"))

    assert (tick.symbol, tick.exchange) == ("NIFTY", "NSE_INDEX")


# ----- the last trade time ----------------------------------------------------------


@pytest.mark.parametrize(
    "ltt, event_ns",
    [
        pytest.param(TRADE_S * 1000, TRADE_NS, id="epoch milliseconds"),
        pytest.param(TRADE_S, TRADE_NS, id="epoch seconds"),
        pytest.param(str(TRADE_S), TRADE_NS, id="epoch seconds as a string of digits"),
        pytest.param(str(TRADE_S * 1000), TRADE_NS, id="epoch milliseconds as a string"),
        pytest.param(TRADE_S * 1_000_000, TRADE_NS, id="epoch microseconds"),
        pytest.param(TRADE_NS, TRADE_NS, id="epoch nanoseconds"),
        pytest.param(TRADE_S + 0.25, TRADE_NS + 250_000_000, id="seconds with a fraction"),
        pytest.param(float(TRADE_S * 1000), TRADE_NS, id="milliseconds as a float"),
        pytest.param(RECV_S - 365 * DAY_S, (RECV_S - 365 * DAY_S) * NS, id="a year ago"),
        pytest.param(RECV_S + 3_600, (RECV_S + 3_600) * NS, id="an hour ahead"),
    ],
)
def test_a_last_trade_time_is_read_in_the_unit_its_magnitude_names(ltt, event_ns):
    tick, tally = adapt(ltp_message({"ltp": 812.4, "ltt": ltt}))

    assert (tick.ts_event_ns, tally.event_times_rejected) == (event_ns, 0)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param({"ltp": 812.4}, id="no ltt"),
        pytest.param({"ltp": 812.4, "ltt": None}, id="None"),
        pytest.param({"ltp": 812.4, "ltt": 0}, id="0, the quote data's fill"),
        pytest.param({"ltp": 812.4, "ltt": "0"}, id="0 as text"),
        pytest.param({"ltp": 812.4, "ltt": ""}, id="empty text"),
    ],
)
def test_a_missing_last_trade_time_gives_no_event_time_and_is_not_counted(data):
    tick, tally = adapt(ltp_message(data))

    assert (tick.ts_event_ns, tally.event_times_rejected) == (None, 0)


@pytest.mark.parametrize(
    "ltt",
    [
        pytest.param(TRADE_S - 315_532_800, id="seconds counted from 1980"),
        pytest.param(TRADE_S + 19_800, id="an IST clock time counted as UTC"),
        pytest.param("09:59:59", id="a clock time as text"),
        pytest.param("2026-11-16 09:59:59", id="a date and time as text"),
        pytest.param(-62_135_596_800_000, id="a negative count"),
        pytest.param(50_000_000_000, id="between seconds and milliseconds"),
        pytest.param(35_999, id="seconds since midnight"),
        pytest.param(RECV_S - 400 * DAY_S, id="more than a year before receipt"),
        pytest.param(RECV_S + 3_601, id="more than an hour after receipt"),
        pytest.param(math.nan, id="not a number"),
        pytest.param(True, id="a bool"),
        pytest.param([TRADE_S], id="a list"),
    ],
)
def test_an_implausible_last_trade_time_is_dropped_from_the_tick_and_counted(ltt):
    tick, tally = adapt(ltp_message({"ltp": 812.4, "ltt": ltt}))

    assert (tick.ltp, tick.ts_event_ns, tally.event_times_rejected) == (812.4, None, 1)


# ----- volume, bid and ask ------------------------------------------------------------


def test_the_volume_the_quote_data_fills_with_0_is_kept_as_0():
    tick, _ = adapt(quote_message(volume=0))

    assert tick.volume == 0.0


@pytest.mark.parametrize(
    "buy, sell, best",
    [
        pytest.param([], [level(812.45)], (None, 812.45), id="no bid"),
        pytest.param([level(812.35)], [], (812.35, None), id="no ask"),
        pytest.param([level(0), level(0)], [level(812.45)], (None, 812.45), id="a zero bid"),
        pytest.param([level(812.35)], [level(0.0)], (812.35, None), id="a zero ask"),
    ],
)
def test_an_empty_side_of_the_book_gives_no_price_for_it(buy, sell, best):
    tick, tally = adapt(depth_message(buy, sell))

    assert ((tick.bid, tick.ask), tally.unusable) == (best, 0)


def test_a_depth_message_without_a_book_gives_no_bid_or_ask():
    message = depth_message([], [])
    del message["data"]["depth"]

    tick, _ = adapt(message)

    assert (tick.bid, tick.ask) == (None, None)


# ----- messages that make no tick -----------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(quote_message(ltp=0), id="an LTP of 0"),
        pytest.param(ltp_message({"ltp": None}), id="an LTP of None"),
        pytest.param(ltp_message({"ltt": TRADE_S}), id="no LTP"),
        pytest.param(ltp_message({"ltp": "812.4"}), id="an LTP as text"),
        pytest.param(ltp_message({"ltp": math.nan}), id="an LTP that is not a number"),
        pytest.param(ltp_message({"ltp": -812.4}), id="a negative LTP"),
        pytest.param(ltp_message({"ltp": True}), id="an LTP that is a bool"),
        pytest.param(quote_message(volume=-1), id="a negative volume"),
        pytest.param(quote_message(volume="1904211"), id="a volume as text"),
        pytest.param(quote_message(volume=math.inf), id="an infinite volume"),
        pytest.param(depth_message([level("abc")], []), id="a bid as text"),
        pytest.param(depth_message([level(812.35)], [level(-1.0)]), id="a negative ask"),
        pytest.param(depth_message(["812.35"], []), id="a level that is no mapping"),
    ],
)
def test_market_data_without_a_usable_price_or_value_makes_no_tick_and_is_counted(message):
    tick, tally = adapt(message)

    assert (tick, tally.unusable, tally.not_ticks) == (None, 1, 0)


def test_a_book_that_is_not_a_mapping_makes_no_tick():
    message = depth_message([], [])
    message["data"]["depth"] = []

    tick, tally = adapt(message)

    assert (tick, tally.unusable) == (None, 1)


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(
            {
                "type": "subscribe",
                "status": "success",
                "subscriptions": [{"symbol": "SBIN", "exchange": "NSE", "status": "success"}],
                "message": "Subscription processing complete",
                "broker": "zerodha",
            },
            id="a subscription acknowledgement",
        ),
        pytest.param(
            {"status": "error", "code": "INVALID_PARAMETERS", "message": "Invalid mode"},
            id="a server error",
        ),
        pytest.param(
            {
                "type": "order_update",
                "orderid": "25111600000001",
                "symbol": "SBIN",
                "exchange": "NSE",
                "order_status": "complete",
            },
            id="an order update",
        ),
        pytest.param({**ltp_message({"ltp": 812.4}), "symbol": ""}, id="no symbol"),
        pytest.param({**ltp_message({"ltp": 812.4}), "exchange": None}, id="no exchange"),
        pytest.param({**ltp_message({"ltp": 812.4}), "mode": 4}, id="an unknown mode"),
        pytest.param({**ltp_message({"ltp": 812.4}), "mode": "LTP"}, id="a mode as text"),
        pytest.param({**ltp_message({"ltp": 812.4}), "mode": True}, id="a mode that is a bool"),
        pytest.param({**ltp_message({"ltp": 812.4}), "data": [812.4]}, id="data not a mapping"),
        pytest.param("market_data", id="not a mapping"),
    ],
)
def test_a_message_that_is_no_market_data_of_an_instrument_makes_no_tick_and_is_counted(message):
    tick, tally = adapt(message)

    assert (tick, tally.not_ticks, tally.unusable) == (None, 1, 0)


def test_the_tally_counts_the_messages_that_made_no_tick_as_dropped():
    tally = TickTally()
    for message in (quote_message(ltp=0), {"type": "subscribe"}, quote_message()):
        tick_from_feed(message, recv_ns=RECV_NS, tally=tally)

    assert tally.dropped == 2


def test_a_message_without_a_tally_is_still_adapted():
    assert tick_from_feed(quote_message(), recv_ns=RECV_NS).ts_event_ns == TRADE_NS
