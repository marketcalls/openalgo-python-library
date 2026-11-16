"""The session's feed link (16-session-design 2.3, 2.5 A.12, 3.1 step 14) on the fake
OpenAlgo's direct feed, which delivers on the test's thread with no socket (the link runs
the feed's connect there too: `threads=False`): the callbacks only queue and stamp, the
hooks become events with their connection number, the subscriptions, the start that
fails when the feed does not connect, the outage spans and stream gaps `note` makes of
the events, the subscriptions sent again after a reconnect, what `note` leaves alone (the
events before the link subscribed, and after close), and close.

The feed's own threads, a real drop and the SDK's replay are in
`test_feedlink_outages.py`, over the SDK's `FeedAPI` on the fake's in-process socket.
"""

from __future__ import annotations

import threading

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from feed_fakes import WAIT

from openalgo.backtest._layout import NANOS_PER_SECOND
from openalgo.backtest._live._feedlink_connect import CONNECT_THREAD_NAME
from openalgo.backtest._live._tick_spans import Outage
from openalgo.backtest._live.feedlink import (
    CONNECT_POLL_S,
    FEED_CONNECT_TIMEOUT_S,
    FeedLink,
    Recovery,
)
from openalgo.backtest._live.inbox import (
    FeedEvent,
    FeedEventKind,
    FeedTick,
    Inbox,
    RawOrderUpdate,
    RawUpdates,
)
from openalgo.backtest.errors import ConfigError, LiveError

SBIN = {"exchange": "NSE", "symbol": "SBIN"}
INFY = {"exchange": "NSE", "symbol": "INFY"}
SUBSCRIBE_BOTH = {
    "action": "subscribe",
    "symbols": [{"exchange": "NSE", "symbol": "SBIN"}, {"exchange": "NSE", "symbol": "INFY"}],
    "mode": 2,
    "depth": 5,
}


def buy(client, **changes):
    fields = {
        "strategy": "ema_cross", "symbol": "SBIN", "action": "BUY", "exchange": "NSE",
        "price_type": "MARKET", "product": "MIS", "quantity": 10,
    }  # fmt: skip
    return client.placeorder(**(fields | changes))


def taken(inbox: Inbox) -> list:
    """Every message waiting on the inbox, oldest first, taken off it."""
    messages = []
    while (envelope := inbox.take()) is not None:
        messages.append(envelope.message)
    return messages


@pytest.fixture
def market(fake_openalgo):
    """The fake in live mode with SBIN and INFY known and traded."""
    for symbol, price in (("SBIN", 812.0), ("INFY", 1500.0)):
        fake_openalgo.add_instrument(symbol, "NSE")
        fake_openalgo.trade(symbol, "NSE", price)
    return fake_openalgo


class Wiring:
    """A feed link on a direct feed of the fake, with its inbox and raw updates, on the
    fake's manual clock, the feed's connect run on the test's thread."""

    def __init__(self, fake, *, instruments=(SBIN, INFY), sleep=None, **options):
        self.fake = fake
        self.source = fake.direct_feed()
        self.inbox = Inbox(fake.clock.now)
        self.updates = RawUpdates()
        self.link = FeedLink(
            self.source,
            instruments=list(instruments),
            inbox=self.inbox,
            updates=self.updates,
            now=fake.clock.now,
            sleep=fake.clock.sleep if sleep is None else sleep,
            threads=False,
            **options,
        )

    def take(self) -> list:
        """Take what waits on the inbox, oldest first, as the loop does: each feed event
        goes to `note`. Returns what `note` returned for them."""
        return [self.link.note(m) for m in taken(self.inbox) if isinstance(m, FeedEvent)]


@pytest.fixture
def wired(market):
    """The link, connected, its loop past the connect event."""
    wiring = Wiring(market)
    wiring.link.connect()
    wiring.take()
    return wiring


def event(hook, seq=1, **payload):
    """A feed event of `hook` as the link's hook makes it (its arrival time is not used
    by `note`)."""
    return FeedEvent(FeedEventKind(hook), {"event": hook, "seq": seq, **payload}, seq, 0)


# -- connecting and subscribing ---------------------------------------------------------------


def test_connect_subscribes_the_instruments_in_quote_mode_and_the_order_stream(wired):
    assert wired.source.frames == [SUBSCRIBE_BOTH, {"action": "subscribe_orders"}]


def test_the_instruments_are_named_in_upper_case_as_openalgo_writes_them(market):
    wiring = Wiring(market, instruments=[{"exchange": "nse", "symbol": "sbin"}])

    wiring.link.connect()

    assert wiring.link.instruments == (("NSE", "SBIN"),)
    assert wiring.source.get_subscriptions()["quote"] == [{"exchange": "NSE", "symbol": "SBIN"}]


def test_the_connect_hook_becomes_a_connect_event_with_its_connection_number(market):
    wiring = Wiring(market)
    connected_at = market.clock.now()

    wiring.link.connect()

    [message] = taken(wiring.inbox)
    assert (message.kind, message.seq, message.arrived_ns) == (
        FeedEventKind.CONNECT,
        1,
        connected_at,
    )
    assert message.payload["url"] == wiring.source.ws_url


def test_a_feed_that_refuses_to_connect_fails_the_start_after_30_seconds(market):
    wiring = Wiring(market)
    market.refuse_feed()
    started_at = market.clock.now()

    with pytest.raises(LiveError, match="did not connect to direct://fake-openalgo within 30 s"):
        wiring.link.connect()

    assert market.clock.now() - started_at == FEED_CONNECT_TIMEOUT_S * NANOS_PER_SECOND
    assert set(market.clock.slept) == {CONNECT_POLL_S}
    assert wiring.source.frames == []


def test_a_failed_start_removes_the_hooks_so_a_later_connection_reaches_nothing(market):
    wiring = Wiring(market)
    market.refuse_feed()
    with pytest.raises(LiveError):
        wiring.link.connect()
    market.accept_feed()

    wiring.source.connect()
    market.trade("SBIN", "NSE", 813.0)

    assert taken(wiring.inbox) == []


def test_a_feed_that_connects_on_its_own_retry_while_the_link_waits_is_taken(market):
    sleeps = []

    def sleep(seconds):
        # The feed goes on trying by itself after a refused connect: here, on the third
        # look, the server takes connections again and the feed's retry gets through.
        sleeps.append(seconds)
        market.clock.sleep(seconds)
        if len(sleeps) == 3:
            market.accept_feed()

    wiring = Wiring(market, sleep=sleep)
    market.refuse_feed()

    wiring.link.connect()

    assert len(sleeps) == 3
    assert wiring.source.frames == [SUBSCRIBE_BOTH, {"action": "subscribe_orders"}]


def test_the_sessions_stop_while_waiting_stops_the_feed_and_goes_on(market):
    def stop(seconds):
        raise KeyboardInterrupt

    wiring = Wiring(market, sleep=stop)
    market.refuse_feed()

    with pytest.raises(KeyboardInterrupt):
        wiring.link.connect()

    market.accept_feed()
    wiring.source.connect()
    assert taken(wiring.inbox) == []


class ReturnsBetweenLooks:
    """A feed whose connect() returns True while the link reads `authenticated` once
    `armed`, and which reads as not connected until then."""

    ws_url = "test://between-looks"

    def __init__(self):
        self._go = threading.Event()
        self.armed = False
        self.subscribed = False

    @property
    def authenticated(self):
        if self.armed and not self._go.is_set():
            self._go.set()
            for thread in threading.enumerate():
                if thread.name == CONNECT_THREAD_NAME:
                    thread.join(WAIT)
        return False

    def connect(self):
        return self._go.wait(WAIT)

    def disconnect(self):
        pass

    def on_connect(self, callback):
        return callback

    on_disconnect = on_reconnect = on_error = on_connect

    def remove_hook(self, callback):
        return True

    def subscribe_quote(self, instruments, on_data_received=None):
        self.subscribed = True
        return True

    def subscribe_orders(self, on_order_update=None):
        return True


def test_a_connect_that_returns_while_the_link_looks_is_taken_not_timed_out(market):
    # The feed's own connect() runs on the link's connect thread (threads=True) and
    # returns between two of the link's reads, once the deadline has passed.
    source = ReturnsBetweenLooks()

    def sleep(seconds):
        market.clock.advance(FEED_CONNECT_TIMEOUT_S + 1)
        source.armed = True

    link = FeedLink(
        source,
        instruments=[SBIN],
        inbox=Inbox(market.clock.now),
        updates=RawUpdates(),
        now=market.clock.now,
        sleep=sleep,
    )

    link.connect()

    assert source.subscribed


def test_a_link_connects_once(wired):
    with pytest.raises(LiveError, match="connects once"):
        wired.link.connect()


def test_a_shorter_connect_timeout_is_kept(market):
    wiring = Wiring(market, connect_timeout_s=2.5)
    market.refuse_feed()
    started_at = market.clock.now()

    with pytest.raises(LiveError, match="within 2.5 s"):
        wiring.link.connect()

    assert market.clock.now() - started_at == 2.5 * NANOS_PER_SECOND


# -- callbacks: queue and stamp ------------------------------------------------------------------


def test_a_quote_tick_is_queued_for_the_loop_with_the_callbacks_clock_read(wired, market):
    taken(wired.inbox)
    market.clock.advance(1.5)

    market.trade("SBIN", "NSE", 812.5, volume=10)

    [tick] = taken(wired.inbox)
    assert isinstance(tick, FeedTick)
    assert tick.arrived_ns == market.clock.now()
    assert (tick.message["symbol"], tick.message["mode"], tick.message["data"]["ltp"]) == (
        "SBIN",
        2,
        812.5,
    )


def test_an_order_update_goes_to_the_workers_updates_and_not_to_the_loop(wired, market):
    taken(wired.inbox)

    buy(market.client())

    updates = wired.updates.drain()
    assert [type(update) for update in updates] == [RawOrderUpdate, RawOrderUpdate]
    assert [update.message["order_status"] for update in updates] == ["open", "complete"]
    assert {update.arrived_ns for update in updates} == {market.clock.now()}
    assert taken(wired.inbox) == []


def test_ticks_stay_in_the_order_they_came(wired, market):
    taken(wired.inbox)

    for price in (812.1, 1500.5, 812.2):
        symbol = "INFY" if price > 1000 else "SBIN"
        market.trade(symbol, "NSE", price)

    ticks = taken(wired.inbox)
    assert [tick.message["data"]["ltp"] for tick in ticks] == [812.1, 1500.5, 812.2]


def test_the_callbacks_only_put_on_unbounded_queues_so_they_never_wait(wired, market):
    taken(wired.inbox)

    for n in range(2000):
        market.trade("SBIN", "NSE", 812.0 + n % 10 / 20)

    assert len(wired.inbox) == 2000


def test_a_subscription_the_broker_refuses_arrives_as_an_error_event(market):
    wiring = Wiring(market, instruments=[SBIN, {"exchange": "NSE", "symbol": "NOPE"}])

    wiring.link.connect()

    errors = [m for m in taken(wiring.inbox) if m.kind is FeedEventKind.ERROR]
    [error] = errors
    assert (error.payload["kind"], error.seq) == ("server", 1)
    assert error.payload["rejected"][0]["symbol"] == "NOPE"
    assert "SUBSCRIBE_REJECTED" in str(error.error)


# -- outages and reconnects: what note makes of the events -------------------------------------


def test_a_drop_opens_an_outage_and_the_restoring_connections_connect_ends_it(wired, market):
    wired.link.note(event("disconnect", requested=False, will_reconnect=True))
    down_at = market.clock.now()
    market.clock.advance(7)

    recovery = wired.link.note(event("connect", seq=2, reconnect=True))

    outage = Outage(down_at, down_at + 7 * NANOS_PER_SECOND)
    assert recovery == Recovery(outage=outage, stream_gap_since_ns=None)
    assert (wired.link.outages, wired.link.down_since_ns) == ((outage,), None)


def test_the_reconnect_asks_for_the_stream_gap_from_the_drop(wired, market):
    wired.link.note(event("disconnect", requested=False))
    down_at = market.clock.now()
    market.clock.advance(7)
    wired.link.note(event("connect", seq=2, reconnect=True))
    market.clock.advance(3)

    recovery = wired.link.note(
        event("reconnect", seq=2, subscriptions=wired.source.get_subscriptions())
    )

    assert recovery == Recovery(outage=None, stream_gap_since_ns=down_at)


def test_the_feed_is_down_from_the_drop_until_the_restoring_connection_is_up(wired, market):
    before = wired.link.down_since_ns

    wired.link.note(event("disconnect", requested=False))
    down = wired.link.down_since_ns
    wired.link.note(event("connect", seq=2, reconnect=True))

    assert (before, down, wired.link.down_since_ns) == (None, market.clock.now(), None)


def test_a_second_drop_noted_while_down_keeps_the_outages_start(wired, market):
    wired.link.note(event("disconnect", requested=False))
    down_at = market.clock.now()
    market.clock.advance(3)
    wired.link.note(event("disconnect", requested=False))
    market.clock.advance(4)

    recovery = wired.link.note(event("connect", seq=2, reconnect=True))

    assert recovery.outage == Outage(down_at, down_at + 7 * NANOS_PER_SECOND)


def test_a_drop_before_the_reconnect_keeps_the_stream_gap_from_the_first_drop(wired, market):
    # The order stream is not known to be back until a reconnect: a connection that came
    # up and dropped again before its reconnect leaves the gap open from the first drop.
    wired.link.note(event("disconnect", requested=False))
    first_drop = market.clock.now()
    market.clock.advance(3)
    wired.link.note(event("connect", seq=2, reconnect=True))
    wired.link.note(event("disconnect", seq=2, requested=False))
    market.clock.advance(4)
    wired.link.note(event("connect", seq=3, reconnect=True))

    recovery = wired.link.note(
        event("reconnect", seq=3, subscriptions=wired.source.get_subscriptions())
    )

    assert recovery == Recovery(outage=None, stream_gap_since_ns=first_drop)
    assert wired.link.outages == (
        Outage(first_drop, first_drop + 3 * NANOS_PER_SECOND),
        Outage(first_drop + 3 * NANOS_PER_SECOND, first_drop + 7 * NANOS_PER_SECOND),
    )


def test_a_reconnect_while_still_down_ends_the_outage_and_asks_for_the_stream_gap(wired, market):
    wired.link.note(event("disconnect", requested=False))
    down_at = market.clock.now()
    market.clock.advance(5)

    recovery = wired.link.note(
        event("reconnect", seq=2, subscriptions=wired.source.get_subscriptions())
    )

    outage = Outage(down_at, down_at + 5 * NANOS_PER_SECOND)
    assert recovery == Recovery(outage=outage, stream_gap_since_ns=down_at)


@pytest.mark.parametrize(
    "events",
    [
        [event("disconnect", requested=True)],
        [event("connect")],
        [event("connect", seq=2, reconnect=True)],
        [event("error", kind="frame")],
    ],
    ids=["the session's own disconnect", "a connect", "a reconnecting connect", "an error"],
)
def test_events_that_are_no_drop_open_no_outage(wired, events):
    for one in events:
        assert wired.link.note(one) is None

    assert wired.link.down_since_ns is None


def test_a_reconnect_with_no_drop_noted_asks_for_nothing(wired):
    recovery = wired.link.note(
        event("reconnect", seq=2, subscriptions=wired.source.get_subscriptions())
    )

    assert (recovery, wired.link.outages) == (None, ())


def test_a_reconnect_subscribes_again_what_the_server_did_not_take_back(wired):
    took_back = {"quote": [{"symbol": "SBIN", "exchange": "NSE"}], "orders": True}
    sent_before = len(wired.source.frames)

    wired.link.note(event("reconnect", seq=2, subscriptions=took_back))

    assert wired.source.frames[sent_before:] == [
        {"action": "subscribe", "symbols": [{"exchange": "NSE", "symbol": "INFY"}], "mode": 2,
         "depth": 5}
    ]  # fmt: skip
    assert wired.link.resubscriptions == 1


def test_a_reconnect_without_the_order_stream_subscribes_it_again(wired):
    took_back = dict(wired.source.get_subscriptions(), orders=False)
    sent_before = len(wired.source.frames)

    wired.link.note(event("reconnect", seq=2, subscriptions=took_back))

    assert wired.source.frames[sent_before:] == [{"action": "subscribe_orders"}]


def test_a_reconnect_that_took_everything_back_subscribes_nothing(wired):
    sent_before = len(wired.source.frames)

    wired.link.note(event("reconnect", seq=2, subscriptions=wired.source.get_subscriptions()))

    assert (len(wired.source.frames), wired.link.resubscriptions) == (sent_before, 0)


def test_an_instrument_subscribed_again_ticks_to_the_loop(wired, market):
    wired.link.note(event("reconnect", seq=2, subscriptions={"quote": [], "orders": True}))
    taken(wired.inbox)

    market.trade("INFY", "NSE", 1501.0)

    [tick] = taken(wired.inbox)
    assert tick.message["symbol"] == "INFY"


def test_each_error_is_counted(wired):
    for _ in range(3):
        wired.link.note(event("error", kind="callback"))

    assert wired.link.errors == 3


def test_note_takes_feed_events_only(wired):
    with pytest.raises(TypeError, match="FeedEvent"):
        wired.link.note({"event": "connect"})


def test_events_noted_after_close_change_nothing(wired):
    wired.link.close()

    noted = [
        wired.link.note(event("disconnect", requested=False, will_reconnect=True)),
        wired.link.note(event("reconnect", seq=2, subscriptions={"quote": [], "orders": False})),
    ]

    assert noted == [None, None]
    assert (wired.link.down_since_ns, wired.link.outages, wired.link.resubscriptions) == (
        None,
        (),
        0,
    )
    assert wired.source.get_subscriptions() == {
        "ltp": [], "quote": [], "depth": [], "orders": False,
    }  # fmt: skip


# -- closing -----------------------------------------------------------------------------------


def test_after_close_nothing_more_is_queued(wired, market):
    taken(wired.inbox)

    wired.link.close()
    wired.source.connect()
    wired.source.subscribe_quote([SBIN])
    wired.source.subscribe_orders()
    market.trade("SBIN", "NSE", 813.0)
    buy(market.client())

    assert (taken(wired.inbox), wired.updates.drain()) == ([], [])


def test_close_disconnects_the_feed_and_forgets_its_subscriptions(wired):
    wired.link.close()

    assert wired.source.connected is False
    assert wired.source.get_subscriptions()["quote"] == []


def test_close_ends_an_outage_under_way(wired, market):
    wired.link.note(event("disconnect", requested=False))
    down_at = market.clock.now()
    market.clock.advance(2)

    wired.link.close()

    assert wired.link.outages == (Outage(down_at, down_at + 2 * NANOS_PER_SECOND),)


def test_closing_twice_is_harmless(wired):
    wired.link.close()

    wired.link.close()

    assert wired.source.connected is False


def test_a_closed_link_does_not_connect(market):
    wiring = Wiring(market)
    wiring.link.close()

    with pytest.raises(LiveError, match="connects once"):
        wiring.link.connect()


# -- what is refused ---------------------------------------------------------------------------


class NotAFeed:
    """Has a feed's connect and nothing else."""

    def connect(self):
        return True


class NoFlags:
    """Has every call a link makes of a feed, but not the feed's `authenticated` and
    `ws_url`, which the link reads too."""

    def connect(self):
        return False

    def disconnect(self):
        pass

    def on_connect(self, callback):
        return callback

    on_disconnect = on_reconnect = on_error = on_connect

    def remove_hook(self, callback):
        return True

    def subscribe_quote(self, instruments, on_data_received=None):
        return True

    def subscribe_orders(self, on_order_update=None):
        return True


@pytest.mark.parametrize(
    ("changes", "says"),
    [
        ({"source": NotAFeed()}, "the feed has no disconnect, on_connect"),
        ({"source": NoFlags()}, "the feed has no authenticated, ws_url"),
        ({"instruments": []}, "instruments is empty"),
        ({"instruments": [{"exchange": "NSE"}]}, "both named"),
        ({"connect_timeout_s": 0}, "positive number of seconds"),
        ({"connect_timeout_s": float("nan")}, "positive number of seconds"),
        ({"threads": "no"}, "threads is True or False"),
    ],
    ids=[
        "not a feed",
        "no connection flags",
        "no instrument",
        "no symbol",
        "a zero timeout",
        "a timeout of nan",
        "threads not a bool",
    ],
)
def test_unusable_arguments_are_refused_saying_what_to_give(market, changes, says):
    arguments = {
        "source": market.direct_feed(),
        "instruments": [SBIN],
        "inbox": Inbox(market.clock.now),
        "updates": RawUpdates(),
        "now": market.clock.now,
        "sleep": market.clock.sleep,
        "threads": False,
    }
    arguments.update(changes)
    source = arguments.pop("source")

    with pytest.raises(ConfigError, match=says):
        FeedLink(source, **arguments)
