"""The fake OpenAlgo's direct feed (16-session-design 2.11 and 7.3): the part of the SDK's
`FeedAPI` a live session uses, delivered synchronously on the thread of the fake's call
that caused it, in a fixed order, with no socket; what it hands over equals what the
SDK's callbacks receive over the socket; the stream's faults and `refuse_feed` act on it.
"""

from __future__ import annotations

import threading

import pytest
from fake_openalgo import BROKER, USERNAME, FakeOpenAlgo
from fake_openalgo.direct_feed import DIRECT_URL
from fake_openalgo.fixtures import *  # noqa: F403
from fake_openalgo.fixtures import SESSION_START_NS

from openalgo import feed as feed_module
from openalgo._feed_errors import FeedServerError

STRATEGY = "ema_cross"
SBIN = {"exchange": "NSE", "symbol": "SBIN"}
NOPE = {"exchange": "NSE", "symbol": "NOPE"}


def buy(client, **changes):
    fields = {
        "strategy": STRATEGY, "symbol": "SBIN", "action": "BUY", "exchange": "NSE",
        "price_type": "MARKET", "product": "MIS", "quantity": 10,
    }  # fmt: skip
    return client.placeorder(**(fields | changes))


class Received:
    """Every hook and callback of one feed, in the order they fired, with the thread."""

    def __init__(self, feed):
        self.items: list[tuple[str, object]] = []
        self.threads: list[threading.Thread] = []
        for name in ("connect", "disconnect", "reconnect"):
            getattr(feed, f"on_{name}")(self._keeping(name))
        feed.on_error(lambda error, context: self._keep("error", (error, context)))

    def quote(self, message):
        self._keep("quote", message)

    def order(self, message):
        self._keep("order", message)

    def names(self):
        return [name for name, _ in self.items]

    def of(self, name):
        return [item for kind, item in self.items if kind == name]

    def _keeping(self, name):
        return lambda payload: self._keep(name, payload)

    def _keep(self, name, item):
        self.items.append((name, item))
        self.threads.append(threading.current_thread())


@pytest.fixture
def market(fake_openalgo):
    """The fake in live mode with SBIN known and traded at 812."""
    fake_openalgo.add_instrument("SBIN", "NSE")
    fake_openalgo.trade("SBIN", "NSE", 812.0)
    return fake_openalgo


@pytest.fixture
def direct(market):
    """A direct feed, its hooks and callbacks kept in `direct.received`, connected and
    subscribed to SBIN quotes and the order stream."""
    feed = market.direct_feed()
    feed.received = Received(feed)
    assert feed.connect() is True
    feed.subscribe_quote([SBIN], feed.received.quote)
    feed.subscribe_orders(feed.received.order)
    return feed


# -- connecting ------------------------------------------------------------------------------


def test_connect_fires_the_connect_hook_with_the_first_connection_number(market):
    feed = market.direct_feed()
    received = Received(feed)

    assert feed.connect() is True

    assert received.of("connect") == [
        {
            "event": "connect",
            "seq": 1,
            "url": DIRECT_URL,
            "reconnect": False,
            "broker": BROKER,
            "user_id": USERNAME,
        }
    ]
    assert (feed.connected, feed.authenticated) == (True, True)


def test_a_subscription_before_connect_is_not_sent_until_the_feed_connects(market):
    feed = market.direct_feed()

    sent_early = feed.subscribe_quote([SBIN])
    frames_early = list(feed.frames)
    feed.connect()

    assert (sent_early, frames_early) == (False, [])
    assert feed.frames == [
        {"action": "subscribe", "symbols": [{"symbol": "SBIN", "exchange": "NSE"}], "mode": 2,
         "depth": 5}
    ]  # fmt: skip


def test_disconnect_fires_a_requested_disconnect_and_forgets_the_subscriptions(direct, market):
    direct.disconnect()
    market.trade("SBIN", "NSE", 813.0)

    [payload] = direct.received.of("disconnect")
    assert (payload["seq"], payload["requested"], payload["will_reconnect"]) == (1, True, False)
    assert direct.received.of("quote") == []
    assert direct.get_subscriptions() == {"ltp": [], "quote": [], "depth": [], "orders": False}


def test_a_refused_feed_does_not_connect_and_fires_nothing_until_accepted(market):
    feed = market.direct_feed()
    received = Received(feed)
    market.refuse_feed()

    refused = feed.connect()
    fired_while_refused = list(received.items)
    market.accept_feed()

    assert (refused, fired_while_refused, feed.connect()) == (False, [], True)
    assert market.log.faults()[-2:] == ["refuse_feed", "accept_feed"]


def test_a_refused_feed_refuses_the_sdks_socket_too(market):
    client = market.client(auto_reconnect=False)
    market.refuse_feed()

    assert client.connect() is False


def test_a_removed_hook_fires_no_more(market):
    feed = market.direct_feed()
    seen = []
    feed.on_connect(seen.append)

    assert feed.remove_hook(seen.append) is True
    feed.connect()

    assert (seen, feed.remove_hook(seen.append)) == ([], False)


# -- what it delivers ---------------------------------------------------------------------------


def test_a_quote_tick_is_what_the_sdks_quote_callback_receives(direct, market):
    client = market.client()
    client.connect()
    through_socket = []
    client.subscribe_quote([SBIN], through_socket.append)

    market.trade("SBIN", "NSE", 813.5, volume=40)

    assert direct.received.of("quote") == through_socket
    assert through_socket[0]["data"]["ltp"] == 813.5


def test_an_order_update_is_what_the_sdks_order_callback_receives(direct, market):
    client = market.client()
    client.connect()
    through_socket = []
    client.subscribe_orders(through_socket.append)

    buy(client)

    assert direct.received.of("order") == through_socket
    assert [update["order_status"] for update in through_socket] == ["open", "complete"]


def test_a_trade_delivers_its_tick_then_the_updates_it_caused_in_that_order(direct, market):
    client = market.client()
    buy(client, price_type="LIMIT", price=810.0)

    market.trade("SBIN", "NSE", 809.0)

    names = direct.received.names()
    assert names[-3:] == ["order", "quote", "order"]
    assert direct.received.of("quote")[-1]["data"]["ltp"] == 809.0
    assert direct.received.of("order")[-1]["order_status"] == "complete"


def test_delivery_is_done_on_the_thread_of_the_fakes_call_before_it_returns(direct, market):
    market.trade("SBIN", "NSE", 813.0)

    assert len(direct.received.of("quote")) == 1
    assert set(direct.received.threads) == {threading.current_thread()}


def test_an_instrument_the_broker_does_not_know_is_refused_and_never_ticks(direct, market):
    direct.subscribe_quote([NOPE])
    market.add_instrument("NOPE", "NSE")
    market.trade("NOPE", "NSE", 10.0)

    [(error, context)] = direct.received.of("error")
    assert isinstance(error, FeedServerError)
    assert (error.code, context["kind"], context["seq"]) == ("SUBSCRIBE_REJECTED", "server", 1)
    assert context["rejected"][0]["message"] == "Token not found for NOPE on NSE"
    assert direct.received.of("quote") == []


def test_a_callback_that_raises_reaches_the_error_hooks(market):
    feed = market.direct_feed()
    received = Received(feed)
    feed.connect()

    def fail(message):
        raise ValueError("boom")

    feed.subscribe_quote([SBIN], fail)
    market.trade("SBIN", "NSE", 813.0)

    [(error, context)] = received.of("error")
    assert (str(error), context["kind"], context["callback"]) == ("boom", "callback", "quote")


def test_nothing_is_delivered_before_the_feed_connects(market):
    feed = market.direct_feed()
    received = Received(feed)
    feed.subscribe_quote([SBIN], received.quote)
    feed.subscribe_orders(received.order)

    market.trade("SBIN", "NSE", 813.0)
    buy(market.client())

    assert received.items == []


# -- the faults act on it -------------------------------------------------------------------------


def test_a_dead_feed_sends_it_no_tick(direct, market):
    market.faults.feed_dead()

    market.trade("SBIN", "NSE", 813.0)

    assert direct.received.of("quote") == []


def test_a_silent_order_stream_sends_it_no_update(direct, market):
    market.faults.silence_order_updates()

    buy(market.client())

    assert direct.received.of("order") == []


def test_held_updates_reach_it_when_released_newest_first(direct, market):
    market.faults.hold_order_updates()
    buy(market.client())

    market.faults.release_order_updates(newest_first=True)

    assert [u["order_status"] for u in direct.received.of("order")] == ["complete", "open"]


def test_a_call_into_the_fake_from_a_direct_callback_is_refused_as_a_problem(monkeypatch):
    fake = FakeOpenAlgo(start_ns=SESSION_START_NS)
    monkeypatch.setattr(feed_module, "websocket", fake.feed_server)
    answers = []
    try:
        fake.add_instrument("SBIN", "NSE")
        client = fake.client()
        feed = fake.direct_feed()
        feed.connect()
        feed.subscribe_quote([SBIN], lambda tick: answers.append(client.ping()))

        fake.trade("SBIN", "NSE", 812.0)

        assert [answer["error_type"] for answer in answers] == ["unknown_error"]
        [problem] = fake.log.problems()
        assert "feed's thread" in problem.detail["error"]
    finally:
        fake.close()
