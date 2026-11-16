"""The session's feed link over the SDK's real `FeedAPI` on the fake OpenAlgo's in-process
socket (16-session-design 2.11: the feed link's own tests, and the mechanics of F41, F85
and F86): the hooks become events with their connection numbers across a drop, a drop
and its reconnect make one outage by the loop's clock, the SDK replays the subscriptions
so nothing is subscribed again, ticks keep reaching the loop's inbox from the feed's
thread, and a feed that refuses to connect fails the start and stops trying.

The feed's threads are real, so a test waits (`wait_for`, bounded) for what they do; the
link's own clock is the fake's manual clock.
"""

from __future__ import annotations

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from feed_fakes import QUIET, wait_for

from openalgo.backtest._layout import NANOS_PER_SECOND
from openalgo.backtest._live._tick_spans import Outage
from openalgo.backtest._live.feedlink import FeedLink
from openalgo.backtest._live.inbox import FeedEvent, FeedEventKind, FeedTick, Inbox, RawUpdates
from openalgo.backtest.errors import LiveError

SBIN = {"exchange": "NSE", "symbol": "SBIN"}


class Loop:
    """Plays the session's loop: takes what reaches the inbox, and hands each feed event
    to the link in order, keeping what `note` returned."""

    def __init__(self, link: FeedLink, inbox: Inbox) -> None:
        self.link, self.inbox = link, inbox
        self.messages: list = []
        self.outages: list[Outage] = []

    def take(self) -> list:
        while (envelope := self.inbox.take()) is not None:
            message = envelope.message
            self.messages.append(message)
            if isinstance(message, FeedEvent):
                outage = self.link.note(message)
                if outage is not None:
                    self.outages.append(outage)
        return self.messages

    def until(self, predicate) -> bool:
        """Take until `predicate(messages)` holds (bounded)."""
        return wait_for(lambda: predicate(self.take()))

    def events(self) -> list[tuple[str, int]]:
        return [(m.kind.value, m.seq) for m in self.messages if isinstance(m, FeedEvent)]

    def ticks(self) -> list[float]:
        return [m.message["data"]["ltp"] for m in self.messages if isinstance(m, FeedTick)]


def has(hook: str):
    """Whether the messages hold an event of `hook`."""
    kind = FeedEventKind(hook)
    return lambda messages: any(isinstance(m, FeedEvent) and m.kind is kind for m in messages)


@pytest.fixture
def market(fake_openalgo):
    """The fake in live mode with SBIN known and traded."""
    fake_openalgo.add_instrument("SBIN", "NSE")
    fake_openalgo.trade("SBIN", "NSE", 812.0)
    return fake_openalgo


@pytest.fixture
def client(market):
    """The session's feed client: a real `openalgo.api` of the fake."""
    return market.client()


def make_link(market, client, **options):
    inbox = Inbox(market.clock.now)
    link = FeedLink(
        client,
        instruments=[SBIN],
        inbox=inbox,
        updates=RawUpdates(),
        now=market.clock.now,
        sleep=options.pop("sleep", market.clock.sleep),
        **options,
    )
    return link, Loop(link, inbox)


@pytest.fixture
def connected(market, client):
    """A link connected through the real SDK, its loop past the connect event."""
    link, loop = make_link(market, client)
    link.connect()
    assert loop.until(has("connect"))
    return loop


def subscribed(market) -> bool:
    """Whether the newest socket has SBIN in quote mode and the order stream."""
    sock = market.feed_server.live_socket()
    if sock is None:
        return False
    entries, orders = market.feed_server.subscriptions(sock)
    return ("SBIN", "NSE", 2) in entries and orders


# -- over the SDK's socket ---------------------------------------------------------------------


def test_connect_subscribes_quotes_and_the_order_stream_on_the_sdks_socket(connected, market):
    assert subscribed(market)


def test_a_tick_reaches_the_loops_inbox_from_the_feeds_thread(connected, market):
    market.trade("SBIN", "NSE", 812.5)

    assert connected.until(lambda messages: connected.ticks() == [812.5])


def test_the_hooks_become_events_with_their_connection_numbers_across_a_drop(connected, market):
    market.faults.drop_feed()

    assert connected.until(has("reconnect"))
    assert connected.events() == [
        ("connect", 1),
        ("disconnect", 1),
        ("connect", 2),
        ("reconnect", 2),
    ]


def test_a_drop_and_its_reconnect_make_one_outage_by_the_loops_clock(connected, market):
    market.faults.drop_feed(hold_reconnect=True)
    assert connected.until(has("disconnect"))
    down_at = market.clock.now()
    market.clock.advance(7)

    market.faults.let_feed_reconnect()

    assert connected.until(has("reconnect"))
    assert connected.outages == [Outage(down_at, down_at + 7 * NANOS_PER_SECOND)]
    assert connected.link.down_since_ns is None


def test_the_sdk_replays_the_subscriptions_so_nothing_is_subscribed_again(connected, market):
    market.faults.drop_feed()
    assert connected.until(has("reconnect"))

    market.trade("SBIN", "NSE", 813.0)

    assert subscribed(market)
    assert connected.until(lambda messages: connected.ticks() == [813.0])
    assert connected.link.resubscriptions == 0


def test_ticks_while_the_reconnect_is_held_never_reach_the_loop(connected, market):
    market.faults.drop_feed(hold_reconnect=True)
    assert connected.until(has("disconnect"))

    market.trade("SBIN", "NSE", 813.0)
    market.faults.let_feed_reconnect()
    assert connected.until(has("reconnect"))
    market.trade("SBIN", "NSE", 814.0)

    assert connected.until(lambda messages: connected.ticks() == [814.0])


def test_nothing_waits_on_the_feeds_thread_while_the_loop_takes_nothing(connected, market):
    for n in range(500):
        market.trade("SBIN", "NSE", 812.0 + n % 10 / 20)

    # Each trade returns only once the SDK's thread has run the callback.
    assert len(connected.inbox) == 500


# -- a feed that does not connect ---------------------------------------------------------------


def test_a_feed_that_refuses_to_connect_fails_the_start_and_stops_the_sdk_trying(market, client):
    link, loop = make_link(market, client)
    market.refuse_feed()

    with pytest.raises(LiveError, match="within 30 s"):
        link.connect()

    assert wait_for(
        lambda: client._reconnect_thread is None or not client._reconnect_thread.is_alive()
    )
    attempts = len(market.feed_server.sockets)
    market.accept_feed()
    assert not wait_for(lambda: len(market.feed_server.sockets) > attempts, timeout=QUIET)
    assert client.get_subscriptions() == {"ltp": [], "quote": [], "depth": [], "orders": False}


def test_a_feed_the_sdk_connects_on_its_own_retry_is_waited_for(market, client):
    accepted = []

    def sleep(seconds):
        # At the link's first look the server takes connections again, and the SDK's own
        # retry gets through.
        market.clock.sleep(seconds)
        if not accepted:
            accepted.append(market.clock.now())
            market.accept_feed()
            assert wait_for(lambda: client.authenticated)

    link, loop = make_link(market, client, sleep=sleep)
    market.refuse_feed()

    link.connect()

    assert wait_for(lambda: subscribed(market))
    assert loop.until(has("connect"))
