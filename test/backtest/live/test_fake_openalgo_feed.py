"""The fake OpenAlgo's WebSocket feed, received by the SDK's real `FeedAPI` over the feed
tests' in-process socket: ticks in the proxy's market-data shape for LTP, Quote and Depth
(websocket_proxy/server.py; the broker adapter's fields), order updates in the shape of
subscribers/wsproxy_subscriber.py in live and sandbox mode, and the stream's and the
feed's faults. A message is processed by the SDK before the call that caused it returns.
"""

from __future__ import annotations

import pytest
from fake_openalgo import BROKER, USERNAME, FakeOpenAlgo
from fake_openalgo.fixtures import *  # noqa: F403
from fake_openalgo.fixtures import SESSION_START_NS
from fake_openalgo.market import Instrument, MarketState
from fake_openalgo.messages import DEPTH, LTP, QUOTE, market_data
from feed_fakes import shorten, wait_for

from openalgo import feed as feed_module

STRATEGY = "ema_cross"
SBIN = {"exchange": "NSE", "symbol": "SBIN"}
START_MS = SESSION_START_NS // 1_000_000


def buy(client, **changes):
    fields = {
        "strategy": STRATEGY, "symbol": "SBIN", "action": "BUY", "exchange": "NSE",
        "price_type": "MARKET", "product": "MIS", "quantity": 10,
    }  # fmt: skip
    return client.placeorder(**(fields | changes))


def subscribed(fake, key=("SBIN", "NSE")):
    """Whether the open socket has subscribed `key` in some mode."""
    sock = fake.feed_server.live_socket()
    if sock is None:
        return False
    entries, _ = fake.feed_server.subscriptions(sock)
    return any((symbol, exchange) == key for symbol, exchange, _ in entries)


@pytest.fixture
def market(fake_openalgo):
    """The fake in live mode with SBIN and INFY known."""
    fake_openalgo.add_instrument("SBIN", "NSE")
    fake_openalgo.add_instrument("INFY", "NSE")
    return fake_openalgo


@pytest.fixture
def connected(market):
    """A client of the fake, connected, its order stream subscribed into `updates`."""
    client = market.client()
    assert client.connect() is True
    client.updates = []
    client.subscribe_orders(client.updates.append)
    return client


def test_the_sdk_authenticates_to_the_proxy_with_the_fakes_key(market):
    client = market.client()

    assert client.connect() is True

    assert client.authenticated is True
    assert market.feed_server.live_socket().frames("authenticate")[0]["api_key"] == client.api_key


def test_the_proxy_refuses_a_key_it_does_not_know(market):
    client = market.client(api_key="not-the-fakes-key", auto_reconnect=False)
    shorten(client, authenticate=0.05)

    assert client.connect() is False


# -- ticks ---------------------------------------------------------------------------


def test_an_ltp_tick_reaches_the_callback_stamped_by_the_fakes_clock(connected, market):
    ticks = []
    connected.subscribe_ltp([SBIN], ticks.append)

    market.trade("SBIN", "NSE", 812.5)

    assert ticks == [
        {
            "type": "market_data",
            "symbol": "SBIN",
            "exchange": "NSE",
            "mode": 1,
            "data": {"ltp": 812.5, "timestamp": START_MS, "ltt": START_MS},
        }
    ]


def test_a_quote_tick_carries_the_days_open_high_low_and_volume(connected, market):
    ticks = []
    connected.subscribe_quote([SBIN], ticks.append)
    market.trade("SBIN", "NSE", 810.0, volume=100)
    market.trade("SBIN", "NSE", 815.0, volume=40)

    market.trade("SBIN", "NSE", 812.0, volume=10)

    data = ticks[-1]["data"]
    assert (ticks[-1]["mode"], data["ltp"], data["volume"]) == (2, 812.0, 150.0)
    assert (data["open"], data["high"], data["low"]) == (810.0, 815.0, 810.0)


def test_a_depth_tick_carries_the_best_bid_and_ask(connected, market):
    ticks = []
    connected.subscribe_depth([SBIN], ticks.append)

    market.trade("SBIN", "NSE", 812.0, bid=811.95, ask=812.05)

    depth = ticks[-1]["data"]["depth"]
    assert (depth["buy"][0]["price"], depth["sell"][0]["price"]) == (811.95, 812.05)


@pytest.mark.parametrize(
    ("mode", "ltt"),
    [(LTP, START_MS), (QUOTE, START_MS), (DEPTH, SESSION_START_NS // 1_000_000_000)],
    ids=["ltp", "quote", "depth"],
)
def test_a_ticks_last_trade_time_is_milliseconds_but_seconds_in_depth(mode, ltt):
    # The adapter's own stamp in milliseconds (zerodha_websocket.py:740), the exchange's
    # in seconds in a depth packet (zerodha_websocket.py:826). The SDK keeps `ltt` from
    # LTP ticks only, so the proxy's message is read here.
    state = MarketState(Instrument("SBIN", "NSE"))
    state.trade(SESSION_START_NS, 812.0, 3)

    message = market_data(state, mode, broker=BROKER, ts_ns=SESSION_START_NS)

    assert (message["data"]["ltt"], message["data"]["timestamp"]) == (ltt, START_MS)


@pytest.mark.parametrize("mode", [LTP, QUOTE, DEPTH], ids=["ltp", "quote", "depth"])
def test_another_brokers_tick_carries_the_trades_time_in_milliseconds(mode):
    # The flavour of 16-session-design 7.3 (F61): most adapters give the exchange's last
    # trade time in milliseconds. The message is sent 2 s after the trade.
    state = MarketState(Instrument("SBIN", "NSE"))
    state.trade(SESSION_START_NS, 812.0, 3)

    message = market_data(state, mode, broker="angel", ts_ns=SESSION_START_NS + 2 * 10**9)

    assert (message["data"]["ltt"], message["data"]["timestamp"]) == (START_MS, START_MS + 2000)


def test_a_quote_ticks_volume_is_a_whole_number(connected, market):
    # The packet's quantities are integers (zerodha_websocket.py:779-780).
    ticks = []
    connected.subscribe_quote([SBIN], ticks.append)

    market.trade("SBIN", "NSE", 812.0, volume=3)

    volume = ticks[-1]["data"]["volume"]
    assert (volume, type(volume)) == (3, int)


def test_a_subscription_to_an_unknown_instrument_is_refused_and_never_ticks(connected, market):
    # broker/zerodha/streaming/zerodha_adapter.py:282-284 finds no token.
    nope = {"exchange": "NSE", "symbol": "NOPE"}
    ticks = []

    connected.subscribe_ltp([nope, SBIN], ticks.append)

    sock = market.feed_server.live_socket()
    entries, _ = market.feed_server.subscriptions(sock)
    assert entries == {("SBIN", "NSE", 1)}
    market.add_instrument("NOPE", "NSE")
    market.trade("NOPE", "NSE", 10.0)
    assert ticks == []


def test_the_proxy_answers_a_refused_subscription_in_the_adapters_words(market):
    nope = {"exchange": "NSE", "symbol": "NOPE"}
    frame = {"action": "subscribe", "symbols": [nope, SBIN], "mode": 1, "depth": 5}

    reply = market.feed_server.reply_to(frame)

    assert reply["status"] == "partial"
    assert reply["subscriptions"][0] == {
        "symbol": "NOPE",
        "exchange": "NSE",
        "status": "error",
        "message": "Token not found for NOPE on NSE",
        "broker": BROKER,
    }


def test_ticks_and_updates_reach_every_subscribed_socket(market):
    # websocket_proxy/server.py:2102-2147 fans a tick out to every subscriber.
    first, second = market.client(), market.client()
    ticks, updates = [], []
    for client in (first, second):
        assert client.connect() is True
        client.subscribe_ltp([SBIN], ticks.append)
        client.subscribe_orders(updates.append)

    market.trade("SBIN", "NSE", 812.0)
    buy(first)

    assert [tick["data"]["ltp"] for tick in ticks] == [812.0, 812.0]
    assert [u["order_status"] for u in updates] == ["open", "open", "complete", "complete"]


def test_a_trade_sends_no_tick_for_an_instrument_not_subscribed(connected, market):
    ticks = []
    connected.subscribe_ltp([SBIN], ticks.append)

    market.trade("INFY", "NSE", 1500.0)

    assert ticks == []
    assert market.log.of("tick")[-1].what == "not sent"


# -- order updates -------------------------------------------------------------------


def test_a_live_fill_is_announced_open_then_complete(connected, market):
    market.trade("SBIN", "NSE", 812.0)

    order_id = buy(connected)["orderid"]

    assert [u["order_status"] for u in connected.updates] == ["open", "complete"]
    assert connected.updates[-1] == {
        "type": "order_update",
        "user_id": USERNAME,
        "mode": "live",
        "broker": BROKER,
        "orderid": order_id,
        "symbol": "SBIN",
        "exchange": "NSE",
        "action": "BUY",
        "quantity": 10,
        "price": 0.0,
        "trigger_price": 0.0,
        "pricetype": "MARKET",
        "product": "MIS",
        "order_status": "complete",
        "filled_quantity": 10,
        "pending_quantity": 0,
        "average_price": 812.0,
        "rejection_reason": "",
    }


def test_a_live_stop_is_announced_trigger_pending_with_a_space(connected, market):
    # broker/zerodha/streaming/zerodha_order_adapter.py:72 maps "TRIGGER PENDING" so.
    market.trade("SBIN", "NSE", 812.0)

    buy(connected, action="SELL", price_type="SL-M", trigger_price=800.0)

    assert [u["order_status"] for u in connected.updates] == ["trigger pending"]


def test_a_live_modify_is_announced_open_even_for_a_resting_stop(connected, market):
    # The broker reports a modification as UPDATE, read as open (order_data.py:39).
    market.trade("SBIN", "NSE", 812.0)
    order_id = buy(connected, action="SELL", price_type="SL-M", trigger_price=800.0)["orderid"]

    connected.modifyorder(
        order_id=order_id, strategy=STRATEGY, symbol="SBIN", action="SELL", exchange="NSE",
        price_type="SL-M", product="MIS", quantity=10, price=0, trigger_price=799.0,
    )  # fmt: skip

    assert [u["order_status"] for u in connected.updates] == ["trigger pending", "open"]
    assert market.orders()[0].status == "trigger pending"


def test_a_sandbox_refusal_at_placement_is_announced_rejected_only(make_fake_openalgo):
    # sandbox/order_manager.py:666-668 publishes the rejection, nothing before it.
    fake = make_fake_openalgo(analyzer=True)
    fake.add_instrument("SBIN", "NSE")
    fake.trade("SBIN", "NSE", 812.0)
    client = fake.client()
    client.connect()
    updates = []
    client.subscribe_orders(updates.append)
    fake.reject_next("Insufficient holdings", at_placement=True)

    buy(client, action="SELL")

    assert [(u["order_status"], u["rejection_reason"]) for u in updates] == [
        ("rejected", "Insufficient holdings")
    ]


def test_a_live_part_fill_is_announced_with_its_filled_and_pending_units(connected, market):
    market.trade("SBIN", "NSE", 812.0)
    market.split_next_fill(4)

    buy(connected)

    last = connected.updates[-1]
    assert (last["order_status"], last["filled_quantity"], last["pending_quantity"]) == (
        "open",
        4,
        6,
    )


def test_a_rejected_order_is_announced_with_its_reason(connected, market):
    market.trade("SBIN", "NSE", 812.0)
    market.reject_next("RMS: margin exceeds")

    buy(connected)

    last = connected.updates[-1]
    assert (last["order_status"], last["rejection_reason"]) == ("rejected", "RMS: margin exceeds")


def test_a_sandbox_fill_is_announced_in_the_sandboxs_shape(make_fake_openalgo):
    fake = make_fake_openalgo(analyzer=True)
    fake.add_instrument("SBIN", "NSE")
    fake.trade("SBIN", "NSE", 812.0)
    client = fake.client()
    client.connect()
    updates = []
    client.subscribe_orders(updates.append)

    buy(client)

    figures = [
        (u["mode"], u["broker"], u["order_status"], u["filled_quantity"], u["pending_quantity"],
         u["average_price"])
        for u in updates
    ]  # fmt: skip
    assert figures == [
        ("analyze", "sandbox", "open", 0, 0, 0.0),
        ("analyze", "sandbox", "complete", 10, 0, 812.0),
    ]


def test_a_sandbox_stop_is_announced_trigger_pending_with_a_space(make_fake_openalgo):
    fake = make_fake_openalgo(analyzer=True)
    fake.add_instrument("SBIN", "NSE")
    fake.trade("SBIN", "NSE", 812.0)
    client = fake.client()
    client.connect()
    updates = []
    client.subscribe_orders(updates.append)

    buy(client, action="SELL", price_type="SL-M", trigger_price=800.0)

    assert [u["order_status"] for u in updates] == ["trigger pending"]


def test_each_update_carries_the_mode_its_order_was_handled_in(connected, market):
    market.trade("SBIN", "NSE", 812.0)
    buy(connected)

    market.set_analyzer(True)
    buy(connected)

    assert [u["mode"] for u in connected.updates] == ["live", "live", "analyze", "analyze"]


# -- the order stream's faults ------------------------------------------------------------


def test_a_silent_order_stream_loses_every_update_until_it_resumes(connected, market):
    market.trade("SBIN", "NSE", 812.0)
    market.faults.silence_order_updates()
    buy(connected)
    assert connected.updates == []

    market.faults.resume_order_updates()
    buy(connected)

    assert [u["order_status"] for u in connected.updates] == ["open", "complete"]
    lost = [e for e in market.log.of("order_update") if e.what == "lost"]
    assert [e.detail["reason"] for e in lost] == ["order stream silent"] * 2


def test_one_dropped_update_is_lost_and_the_next_ones_arrive(connected, market):
    market.trade("SBIN", "NSE", 812.0)
    market.faults.drop_next_order_update()

    buy(connected)

    assert [u["order_status"] for u in connected.updates] == ["complete"]
    assert "drop_order_update" in market.log.faults()


def test_a_duplicated_update_arrives_twice(connected, market):
    market.trade("SBIN", "NSE", 812.0)
    market.faults.duplicate_next_order_update()

    buy(connected)

    assert [u["order_status"] for u in connected.updates] == ["open", "open", "complete"]


def test_held_updates_arrive_after_the_answer_and_can_come_out_of_order(connected, market):
    market.trade("SBIN", "NSE", 812.0)
    market.faults.hold_order_updates()
    answer = buy(connected)
    assert (answer["status"], connected.updates) == ("success", [])

    released = market.faults.release_order_updates(newest_first=True)

    assert released == 2
    assert [u["order_status"] for u in connected.updates] == ["complete", "open"]


def test_a_fill_reported_at_price_zero_keeps_its_price_in_the_books(connected, market):
    market.trade("SBIN", "NSE", 812.0)
    market.faults.zero_price_next_fill()

    order_id = buy(connected)["orderid"]

    assert connected.updates[-1]["average_price"] == 0.0
    status = connected.orderstatus(order_id=order_id, strategy=STRATEGY)
    assert status["data"]["average_price"] == 812.0


def test_the_sandbox_reports_every_fill_at_its_price(make_fake_openalgo):
    fake = make_fake_openalgo(analyzer=True)

    with pytest.raises(ValueError, match="sandbox"):
        fake.faults.zero_price_next_fill()


# -- the market-data feed's faults --------------------------------------------------------


@pytest.mark.parametrize(
    ("lose", "restore"),
    [("expire_session", "restore_session"), ("revoke_login", "restore_login")],
    ids=["expired", "revoked"],
)
def test_a_lost_broker_login_stops_the_ticks_until_restored(connected, market, lose, restore):
    ticks = []
    connected.subscribe_ltp([SBIN], ticks.append)
    getattr(market.faults, lose)()
    market.trade("SBIN", "NSE", 812.0)
    assert ticks == []

    getattr(market.faults, restore)()
    market.trade("SBIN", "NSE", 813.0)

    assert [tick["data"]["ltp"] for tick in ticks] == [813.0]
    assert market.log.of("tick")[0].detail["reason"] == "broker login gone"


def test_a_dead_feed_sends_no_ticks_while_the_connection_stays_up(connected, market):
    ticks = []
    connected.subscribe_ltp([SBIN], ticks.append)
    market.faults.feed_dead()
    market.trade("SBIN", "NSE", 812.0)
    assert (ticks, connected.connected) == ([], True)

    market.faults.feed_alive()
    market.trade("SBIN", "NSE", 813.0)

    assert [tick["data"]["ltp"] for tick in ticks] == [813.0]


def test_a_dropped_feed_reconnects_and_its_subscriptions_are_replayed(connected, market):
    ticks = []
    connected.subscribe_ltp([SBIN], ticks.append)
    first = market.feed_server.live_socket()

    market.faults.drop_feed()

    assert wait_for(lambda: market.feed_server.live_socket() not in (None, first))
    assert wait_for(lambda: subscribed(market))
    market.trade("SBIN", "NSE", 812.0)
    assert [tick["data"]["ltp"] for tick in ticks] == [812.0]


def test_ticks_and_updates_while_a_reconnect_is_held_are_lost(connected, market):
    ticks = []
    connected.subscribe_ltp([SBIN], ticks.append)
    market.trade("SBIN", "NSE", 812.0)
    market.faults.drop_feed(hold_reconnect=True)

    market.trade("SBIN", "NSE", 813.0)
    buy(connected)
    market.faults.let_feed_reconnect()
    assert wait_for(lambda: subscribed(market))
    market.trade("SBIN", "NSE", 814.0)

    assert [tick["data"]["ltp"] for tick in ticks] == [812.0, 814.0]
    assert connected.updates == []


def test_a_tick_gap_skips_ticks_while_the_candles_keep_the_trades(connected, market):
    ticks = []
    connected.subscribe_ltp([SBIN], ticks.append)
    market.faults.tick_gap("SBIN", "NSE", trades=2)

    for price in (812.0, 813.0, 814.0):
        market.trade("SBIN", "NSE", price, volume=1)

    assert [tick["data"]["ltp"] for tick in ticks] == [814.0]
    candles = connected.history(
        symbol="SBIN", exchange="NSE", interval="1m", start_date="2026-11-16", end_date="2026-11-16"
    )
    assert candles["volume"].tolist() == [3]


def test_a_rest_call_from_the_feeds_thread_is_refused_as_a_problem(monkeypatch):
    fake = FakeOpenAlgo(start_ns=SESSION_START_NS)
    monkeypatch.setattr(feed_module, "websocket", fake.feed_server)
    answers = []
    try:
        fake.add_instrument("SBIN", "NSE")
        client = fake.client()
        client.connect()
        client.subscribe_ltp([SBIN], lambda tick: answers.append(client.ping()))

        fake.trade("SBIN", "NSE", 812.0)

        assert [answer["error_type"] for answer in answers] == ["unknown_error"]
        [problem] = fake.log.problems()
        assert "feed's thread" in problem.detail["error"]
    finally:
        fake.close()
