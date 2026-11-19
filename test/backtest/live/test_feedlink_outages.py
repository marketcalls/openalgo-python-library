"""The session's feed link over the SDK's real `FeedAPI` on the fake OpenAlgo's in-process
socket (16-session-design 2.11: the feed link's own tests, and the mechanics of F41, F85
and F86): the hooks become events with their connection numbers across a drop, a drop
and its reconnect make one outage by the loop's clock that ends when the new connection
is up, the stream gap waits for the order stream, the SDK replays the subscriptions so
nothing is subscribed again, a drop before the link subscribed changes nothing, ticks
keep reaching the loop's inbox from the feed's thread, the loop keeps turning while the
SDK's own connect waits (and a stop then ends it), and a feed that refuses to connect
fails the start and stops trying.

The feed's threads are real, so a test waits (`wait_for`, bounded) for what they do; the
link's own clock is the fake's manual clock.
"""

from __future__ import annotations

import threading

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from feed_fakes import QUIET, WAIT, shorten, wait_for

from openalgo.backtest._layout import NANOS_PER_SECOND
from openalgo.backtest._live._tick_spans import Outage
from openalgo.backtest._live.feedlink import FeedLink
from openalgo.backtest._live.inbox import FeedEvent, FeedEventKind, FeedTick, Inbox, RawUpdates
from openalgo.backtest.errors import LiveError

SBIN = {"exchange": "NSE", "symbol": "SBIN"}


class Stop(Exception):
    """The session's own stop signal, raised from its sleep (3.1 step 13)."""


def raise_stop(seconds: float) -> None:
    """A sleep in which the session's stop was asked for."""
    raise Stop


class Loop:
    """Plays the session's loop: takes what reaches the inbox, and hands each feed event
    to the link in order, keeping what `note` returned."""

    def __init__(self, link: FeedLink, inbox: Inbox) -> None:
        self.link, self.inbox = link, inbox
        self.messages: list = []
        self.recoveries: list = []

    def take(self) -> list:
        while (envelope := self.inbox.take()) is not None:
            message = envelope.message
            self.messages.append(message)
            if isinstance(message, FeedEvent):
                recovery = self.link.note(message)
                if recovery is not None:
                    self.recoveries.append(recovery)
        return self.messages

    def outages(self) -> list[Outage]:
        """The spans `note` handed over for the bar builders to resume over."""
        return [r.outage for r in self.recoveries if r.outage is not None]

    def stream_gaps(self) -> list[int]:
        """The times `note` asked the order worker's stream gap from."""
        return [r.stream_gap_since_ns for r in self.recoveries if r.stream_gap_since_ns]

    def until(self, predicate) -> bool:
        """Take until `predicate(messages)` holds (bounded)."""
        return wait_for(lambda: predicate(self.take()))

    def events(self) -> list[tuple[str, int]]:
        return [(m.kind.value, m.seq) for m in self.messages if isinstance(m, FeedEvent)]

    def ticks(self) -> list[float]:
        return [m.message["data"]["ltp"] for m in self.messages if isinstance(m, FeedTick)]


def has(hook: str, seq: int | None = None):
    """Whether the messages hold an event of `hook` (of connection `seq`, when given)."""
    kind = FeedEventKind(hook)
    return lambda messages: any(
        isinstance(m, FeedEvent) and m.kind is kind and seq in (None, m.seq) for m in messages
    )


class WatchedConnect:
    """Stands in for the client's `connect`: calls the SDK's own once `go` is set (it is
    at first), and sets `returned` once that call has returned."""

    def __init__(self, client) -> None:
        self._sdk_connect = client.connect
        self.go = threading.Event()
        self.go.set()
        self.returned = threading.Event()
        client.connect = self

    def __call__(self) -> bool:
        assert self.go.wait(WAIT), "the test never let connect() begin"
        try:
            return self._sdk_connect()
        finally:
            self.returned.set()


def answer_no_order_stream(market, monkeypatch) -> None:
    """From now the proxy answers no `subscribe_orders` frame (the other frames as
    before), as a proxy that is slow to answer one."""
    answer = market.feed_server.reply_to
    monkeypatch.setattr(
        market.feed_server,
        "reply_to",
        lambda frame: None if frame.get("action") == "subscribe_orders" else answer(frame),
    )


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
    assert connected.outages() == [Outage(down_at, down_at + 7 * NANOS_PER_SECOND)]
    assert (connected.stream_gaps(), connected.link.down_since_ns) == ([down_at], None)


def test_the_outage_ends_when_the_new_connection_is_up_not_at_the_order_streams_answer(
    connected, market, client, monkeypatch
):
    # The SDK fires its reconnect hook once the proxy has answered every replayed frame;
    # the new connection's ticks flow as soon as its quote subscription is taken.
    shorten(client, replay_ack=1.0)
    answer_no_order_stream(market, monkeypatch)
    market.faults.drop_feed(hold_reconnect=True)
    assert connected.until(has("disconnect"))
    down_at = market.clock.now()
    market.clock.advance(7)

    market.faults.let_feed_reconnect()

    assert connected.until(has("connect", seq=2))
    assert connected.outages() == [Outage(down_at, down_at + 7 * NANOS_PER_SECOND)]
    assert connected.link.down_since_ns is None


def test_a_reconnect_with_the_order_stream_unanswered_asks_for_the_gap_and_subscribes_it(
    connected, market, client, monkeypatch
):
    shorten(client, replay_ack=0.2)
    answer_no_order_stream(market, monkeypatch)
    market.faults.drop_feed(hold_reconnect=True)
    assert connected.until(has("disconnect"))
    down_at = market.clock.now()

    market.faults.let_feed_reconnect()

    assert connected.until(has("reconnect"))
    assert (connected.stream_gaps(), connected.link.resubscriptions) == ([down_at], 1)


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


@pytest.mark.parametrize(
    "noted_while_waiting", [True, False], ids=["noted while connect waits", "noted later"]
)
def test_a_drop_and_reconnect_before_the_link_subscribed_change_nothing(
    market, client, noted_while_waiting
):
    early: list = []
    noted: list = []

    def take_early() -> bool:
        while (envelope := inbox.take()) is not None:
            early.append(envelope.message)
            if noted_while_waiting and isinstance(envelope.message, FeedEvent):
                noted.append(link.note(envelope.message))
        return has("reconnect")(early)

    def sleep(seconds):
        # At the link's first look the server takes connections again, and the
        # connection that comes up drops and is restored, all before the link has
        # subscribed. The session's sleep runs loop iterations meanwhile, which note
        # the events; or the loop takes them only once connect has returned.
        market.clock.sleep(seconds)
        if early:
            return
        market.accept_feed()
        assert wait_for(lambda: client.authenticated)
        market.faults.drop_feed()
        assert wait_for(take_early)

    link, loop = make_link(market, client, sleep=sleep)
    inbox = loop.inbox
    market.refuse_feed()

    link.connect()
    if not noted_while_waiting:
        noted.extend(link.note(m) for m in early if isinstance(m, FeedEvent))

    sock = market.feed_server.live_socket()
    assert (len(sock.frames("subscribe")), len(sock.frames("subscribe_orders"))) == (1, 1)
    assert all(recovery is None for recovery in noted)
    assert (link.outages, link.down_since_ns, link.resubscriptions) == ((), None, 0)


# -- the SDK's own connect, off the loop's thread ------------------------------------------------


def test_the_loop_turns_while_the_sdks_own_connect_waits_for_its_socket(market, client):
    watched = WatchedConnect(client)
    looks = []

    def sleep(seconds):
        looks.append(watched.returned.is_set())
        raise Stop

    link, loop = make_link(market, client, sleep=sleep)
    market.feed_server.gate.clear()  # the socket hangs before it opens, for 5 s

    try:
        with pytest.raises(Stop):
            link.connect()
    finally:
        market.feed_server.gate.set()

    assert looks == [False]


def test_a_stop_while_the_sdks_own_connect_waits_ends_that_wait_and_nothing_connects(
    market, client
):
    watched = WatchedConnect(client)
    link, loop = make_link(market, client, sleep=raise_stop)
    market.feed_server.gate.clear()
    with pytest.raises(Stop):
        link.connect()

    # The stop's disconnect ends the SDK's wait for its socket, which would last 5 s.
    assert watched.returned.wait(1.0)
    market.feed_server.gate.set()
    assert not wait_for(lambda: client.connected, timeout=QUIET)
    assert loop.take() == []


def test_a_connect_that_begins_after_the_stop_is_disconnected_once_it_returns(market, client):
    # The SDK's connect() re-arms a feed that disconnect() stopped: one that only begins
    # after the stop's disconnect connects anew, and the link's connect thread ends it.
    watched = WatchedConnect(client)
    watched.go.clear()
    link, loop = make_link(market, client, sleep=raise_stop)
    with pytest.raises(Stop):
        link.connect()

    watched.go.set()

    assert watched.returned.wait(WAIT)
    assert [sock.frames("authenticate") != [] for sock in market.feed_server.sockets] == [True]
    assert wait_for(lambda: not client.connected)
    assert loop.take() == []


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
