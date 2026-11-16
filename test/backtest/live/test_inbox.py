"""The messages between a live session's threads and the inboxes that carry them
(16-session-design 2.3): the loop's inbox takes them in the order they were put, stamped
when put, never makes a put wait, and can leave a later message first for the next take;
the order worker's raw updates drain in order and wake the worker; every message is
frozen.
"""

from __future__ import annotations

import dataclasses
import threading
from datetime import date

import pytest

from openalgo.backtest._live._guard_breach import ModeBreach, SeenIn
from openalgo.backtest._live._journal_reads import JournalSession
from openalgo.backtest._live._order_answers import Outcome, SendOutcome
from openalgo.backtest._live._order_check import OrderCheck
from openalgo.backtest._live._types import Mode, OrderStatus, OrderUpdate
from openalgo.backtest._live.inbox import (
    Books,
    Candles,
    DataFailed,
    Envelope,
    FeedEvent,
    FeedEventKind,
    FeedTick,
    GuardLocked,
    HeldFills,
    Inbox,
    NotSent,
    NotSentReason,
    OrderNews,
    PositionRows,
    Quote,
    RawOrderUpdate,
    RawUpdates,
    Resolutions,
    SendResult,
    Settled,
    WorkerFailed,
    WorkerReady,
)

# 2026-11-16 10:15:00 IST.
START_NS = 1_794_804_300 * 10**9
SECOND_NS = 10**9
# The longest a thread test waits before it fails, seconds.
WAIT_SECONDS = 5.0


class Clock:
    """UTC nanoseconds that move only when a test moves them."""

    def __init__(self) -> None:
        self.ns = START_NS

    def __call__(self) -> int:
        return self.ns


def tick(ltp: float, arrived_ns: int = START_NS) -> FeedTick:
    data = {"ltp": ltp, "timestamp": arrived_ns // 10**6}
    message = {"type": "market_data", "symbol": "SBIN", "exchange": "NSE", "mode": 2, "data": data}
    return FeedTick(message, arrived_ns)


def raw_update(order_id: str) -> RawOrderUpdate:
    return RawOrderUpdate({"type": "order_update", "orderid": order_id}, START_NS)


UPDATE = OrderUpdate(
    "2511160001", "SBIN", "NSE", "BUY", 10, 10, 0, 812.0, OrderStatus.COMPLETE, "", Mode.LIVE,
    START_NS, "stream",
)  # fmt: skip
BREACH = ModeBreach(Mode.LIVE, Mode.SANDBOX, START_NS, SeenIn.POLL, ())
ONE_OF_EACH = [
    tick(812.0),
    FeedEvent(FeedEventKind.CONNECT, {"event": "connect", "seq": 1}, 1, START_NS),
    WorkerReady(JournalSession(date(2026, 11, 16), Mode.LIVE, "zerodha", START_NS, START_NS, 1)),
    OrderNews(UPDATE, "20261116-live-1-ema_cross-o1"),
    SendResult(1, SendOutcome(Outcome.ACCEPTED, order_id="2511160001"), intent_id="i1"),
    NotSent(object(), NotSentReason.STALE),
    Settled(object(), OrderCheck("2511160001", status=OrderStatus.CANCELLED)),
    Resolutions(()),
    HeldFills(()),
    GuardLocked(BREACH),
    Books(1, {}, {}, {}, {}, START_NS, START_NS),
    PositionRows({}, START_NS),
    WorkerFailed(RuntimeError("journal")),
    Candles("SBIN", "NSE", (), START_NS),
    Quote("SBIN", "NSE", 812.0, START_NS),
    DataFailed(object(), "timed out"),
]


# -- the loop's inbox ----------------------------------------------------------------------------


def test_messages_come_out_in_the_order_they_were_put():
    inbox = Inbox(Clock())
    for message in ONE_OF_EACH:
        inbox.put(message)

    taken = [inbox.take().message for _ in ONE_OF_EACH]

    assert taken == ONE_OF_EACH


def test_each_message_is_stamped_when_it_is_put():
    clock = Clock()
    inbox = Inbox(clock)
    inbox.put(tick(812.0))
    clock.ns += 3 * SECOND_NS
    inbox.put(tick(812.5))

    stamps = [inbox.take().put_ns, inbox.take().put_ns]

    assert stamps == [START_NS, START_NS + 3 * SECOND_NS]


def test_an_empty_inbox_answers_none_at_once():
    assert Inbox(Clock()).take() is None


def test_a_message_put_at_or_after_the_cut_stays_first_for_the_next_take():
    clock = Clock()
    inbox = Inbox(clock)
    inbox.put(tick(812.0))
    clock.ns += SECOND_NS
    inbox.put(tick(812.5))
    cut_ns = START_NS + SECOND_NS

    before_cut = inbox.take(arrived_before_ns=cut_ns)
    at_cut = inbox.take(arrived_before_ns=cut_ns)
    next_take = inbox.take()

    assert before_cut.message.message["data"]["ltp"] == 812.0
    assert at_cut is None
    assert next_take.message.message["data"]["ltp"] == 812.5


def test_the_depth_counts_a_message_left_first():
    clock = Clock()
    inbox = Inbox(clock)
    inbox.put(tick(812.0))
    inbox.put(tick(812.5))

    inbox.take(arrived_before_ns=START_NS)

    assert len(inbox) == 2


def test_a_waiting_take_wakes_when_another_thread_puts():
    inbox = Inbox(Clock())
    putter = threading.Timer(0.01, inbox.put, args=(tick(812.0),))
    putter.start()

    envelope = inbox.take(wait_s=WAIT_SECONDS)

    putter.join()
    assert isinstance(envelope, Envelope)


def test_puts_from_many_threads_lose_nothing():
    inbox = Inbox(Clock())
    go = threading.Barrier(4)

    def put_from(thread: int) -> None:
        go.wait(WAIT_SECONDS)
        for n in range(500):
            inbox.put(tick(800.0 + thread, arrived_ns=n))

    threads = [threading.Thread(target=put_from, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(WAIT_SECONDS)

    taken = []
    while (envelope := inbox.take()) is not None:
        taken.append(envelope.message)
    assert len(taken) == 2000
    for thread in range(4):
        mine = [m.arrived_ns for m in taken if m.message["data"]["ltp"] == 800.0 + thread]
        assert mine == list(range(500))


@pytest.mark.parametrize(
    "thing",
    [{"type": "market_data"}, raw_update("1"), None],
    ids=["a dict", "a raw update", "None"],
)
def test_the_inbox_takes_only_its_messages(thing):
    with pytest.raises(TypeError, match="messages of inbox.py"):
        Inbox(Clock()).put(thing)


# -- the order worker's raw updates ----------------------------------------------------------------


def test_raw_updates_drain_in_the_order_they_came():
    updates = RawUpdates()
    for order_id in ("1", "2", "3"):
        updates.put(raw_update(order_id))

    drained = updates.drain()

    assert [update.message["orderid"] for update in drained] == ["1", "2", "3"]
    assert updates.drain() == []


def test_the_raw_updates_waiting_are_counted():
    updates = RawUpdates()
    updates.put(raw_update("1"))
    updates.put(raw_update("2"))

    assert len(updates) == 2


def test_a_raw_update_wakes_a_worker_that_waits_on_the_event():
    wake = threading.Event()
    updates = RawUpdates(wake=wake)

    updates.put(raw_update("1"))

    assert wake.is_set()


def test_the_raw_updates_take_only_raw_updates():
    with pytest.raises(TypeError, match="RawOrderUpdate"):
        RawUpdates().put(tick(812.0))


# -- the messages ----------------------------------------------------------------------------------


@pytest.mark.parametrize("message", ONE_OF_EACH, ids=lambda m: type(m).__name__)
def test_every_message_is_frozen(message):
    first = dataclasses.fields(message)[0].name

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(message, first, None)


@pytest.mark.parametrize(
    ("payload", "requested"),
    [({"requested": True}, True), ({"requested": False}, False), ({}, False)],
    ids=["asked for", "a drop", "not said"],
)
def test_a_disconnect_says_whether_the_session_asked_for_it(payload, requested):
    event = FeedEvent(FeedEventKind.DISCONNECT, {"event": "disconnect", **payload}, 1, 0)

    assert event.requested is requested


def test_only_a_disconnect_is_ever_requested():
    event = FeedEvent(FeedEventKind.CONNECT, {"event": "connect", "requested": True}, 1, 0)

    assert event.requested is False


@pytest.mark.parametrize(
    ("check", "expired"),
    [(OrderCheck("2511160001", status=OrderStatus.OPEN), False), (None, True)],
    ids=["settled by a read", "past its deadline"],
)
def test_a_settled_order_says_whether_its_deadline_passed(check, expired):
    assert Settled(object(), check).expired is expired
