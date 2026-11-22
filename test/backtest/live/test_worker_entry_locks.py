"""The order worker refuses an entry or add of a strategy whose entries the loop has locked,
just before it would send it (`_locked_entries`, `_worker_send`; 2.4.0 item P3-1).

The loop withdraws an entry still waiting in the worker's queue when its strategy's entries
lock, but only when it hands over its next commands (2.4.0 item M8-X2). On its own thread
the worker can take the entry off its queue in between and send it after the lock. So the
call that changes a strategy's locks in the kernel (`Trader.lock`, `Trader.unlock`) copies
them to a record the worker reads before each entry or add: such an entry is answered
`NotSent(entries_locked)`, which the loop rejects with `live_lock` as a withdrawal is.
Exits and stops never read it.

The worker runs on the fake OpenAlgo with its real clients, guard and journal
(`worker_samples`); its record is written here as the loop's `Trader.lock` writes it, and
the copy itself is shown on a real kernel session (`venue_samples`). The thread test holds
the worker inside another order's call with events, so the order of what happens is fixed.
A session started on the fake shows its traders write to the worker's own record (2.4.0
item P4-2): a trader given a record of its own would leave the worker's check blind.
"""

from __future__ import annotations

import dataclasses
import threading
from types import SimpleNamespace

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import Hold, ist
from session_samples import rig as session_rig
from venue_samples import new_session
from worker_samples import PRICE, STRATEGY, WorkerRig

from openalgo.backtest._live._locked_entries import LockedEntries
from openalgo.backtest._live._session_bookkeeping import Bookkeeping
from openalgo.backtest._live._session_records import StrategyRecords
from openalgo.backtest._live._types import Purpose
from openalgo.backtest._live._venue_requests import PlaceRequest
from openalgo.backtest._live.inbox import NotSent, NotSentReason, SendResult, WorkerReady

#: Seconds a thread test waits for the worker's thread at most.
THREAD_WAIT_S = 5.0
FEED_DEAD = "feed_dead"


# ----- the loop copies each change of a strategy's locks -----------------------------------


class Host(Bookkeeping):
    """The bookkeeping part of a `Trader` of strategy `name` over a real kernel session."""

    def __init__(self, record: LockedEntries, name: str = STRATEGY) -> None:
        self.name = name
        self.part = SimpleNamespace(engine=new_session())
        self.finished = False
        self.records = StrategyRecords(name)
        self.locked_entries = record
        self._now = lambda: 0


def test_a_lock_reaches_the_order_workers_record_in_the_same_call():
    record = LockedEntries()
    Host(record).lock(FEED_DEAD)

    assert record.reasons(STRATEGY) == (FEED_DEAD,)


def test_a_strategy_stays_locked_while_any_reason_is_set():
    record = LockedEntries()
    host = Host(record)
    host.lock(FEED_DEAD)
    host.lock("reconciliation")

    host.unlock(FEED_DEAD)
    left = record.reasons(STRATEGY)
    host.unlock("reconciliation")

    assert (left, record.reasons(STRATEGY)) == (("reconciliation",), ())


def test_a_lock_of_one_strategy_leaves_another_unlocked():
    record = LockedEntries()
    Host(record, "ema_cross").lock(FEED_DEAD)
    Host(record, "breakout")

    assert (record.reasons("ema_cross"), record.reasons("breakout")) == ((FEED_DEAD,), ())


# ----- the worker reads it before an entry or add ------------------------------------------


@pytest.fixture
def broker(fake_openalgo):
    fake_openalgo.add_instrument("SBIN", "NSE")
    fake_openalgo.add_instrument("INFY", "NSE")
    fake_openalgo.trade("SBIN", "NSE", PRICE)
    fake_openalgo.trade("INFY", "NSE", 1500.0)
    return fake_openalgo


@pytest.fixture
def rig(broker, tmp_path):
    made = WorkerRig(broker, tmp_path)
    made.messages()
    yield made
    made.worker.close()


def placed(rig) -> list[tuple[str, str]]:
    """(symbol, price type) of each placement OpenAlgo was asked for."""
    calls = rig.fake.log.calls("placeorder")
    return [(c.detail["request"]["symbol"], c.detail["request"]["pricetype"]) for c in calls]


@pytest.mark.parametrize("purpose", [Purpose.ENTRY, Purpose.ADD])
def test_an_entry_or_add_of_a_locked_strategy_is_not_sent(rig, purpose):
    rig.worker.locked_entries.note(STRATEGY, [FEED_DEAD])
    place = rig.place(purpose)

    posted = rig.run(place)

    expected = "the strategy's entries were locked (feed_dead) before it was sent"
    assert rig.of(NotSent, posted) == [NotSent(place, NotSentReason.ENTRIES_LOCKED, expected)]
    assert placed(rig) == []


@pytest.mark.parametrize("purpose", [Purpose.STOP, Purpose.EXIT, Purpose.SQUARE_OFF])
def test_a_stop_or_exit_of_a_locked_strategy_is_sent(rig, purpose):
    rig.worker.locked_entries.note(STRATEGY, [FEED_DEAD])
    shape = {"price_type": "SL-M", "trigger_price": 800.0} if purpose is Purpose.STOP else {}

    posted = rig.run(rig.place(purpose, action="SELL", **shape))

    assert len(rig.of(SendResult, posted)) == 1
    assert placed(rig) == [("SBIN", shape.get("price_type", "MARKET"))]


def test_an_entry_is_sent_once_its_strategys_entries_are_unlocked(rig):
    rig.worker.locked_entries.note(STRATEGY, [FEED_DEAD])
    rig.worker.locked_entries.note(STRATEGY, [])

    posted = rig.run(rig.place())

    assert len(rig.of(SendResult, posted)) == 1
    assert placed(rig) == [("SBIN", "MARKET")]


def test_another_strategys_entry_is_sent(rig):
    rig.worker.locked_entries.note(STRATEGY, [FEED_DEAD])
    place = rig.place()
    others = PlaceRequest(
        place.request_id,
        place.kernel_order,
        dataclasses.replace(place.intent, strategy="breakout", intent_id="breakout-o1"),
        place.priority,
    )

    posted = rig.run(others)

    assert len(rig.of(SendResult, posted)) == 1
    assert placed(rig) == [("SBIN", "MARKET")]


# ----- the window of M8-X2, the worker on its own thread -----------------------------------


def test_an_entry_taken_after_its_strategys_entries_locked_is_not_sent(broker, tmp_path):
    # The worker, on its thread, is inside the placement of INFY's stop when the loop
    # hands it an SBIN entry and then locks the strategy's entries (the feed died). No
    # withdrawal follows: the loop step that would hand it over has not come. Once the
    # stop's call returns the worker takes the entry: it must not send it.
    rig = WorkerRig(broker, tmp_path, start=False)
    inside, carry_on = threading.Event(), threading.Event()
    client = broker.client()

    def placeorder(**fields):
        if fields["symbol"] == "INFY":
            inside.set()
            carry_on.wait(THREAD_WAIT_S)
        return client.placeorder(**fields)

    rig.orders.placeorder = placeorder
    infy_stop = rig.place(
        Purpose.STOP, action="SELL", price_type="SL-M", trigger_price=1490.0, symbol="INFY"
    )
    entry = rig.place()
    rig.worker.start()
    try:
        assert isinstance(rig.inbox.take(wait_s=THREAD_WAIT_S).message, WorkerReady)
        rig.worker.put(infy_stop)
        assert inside.wait(THREAD_WAIT_S)
        rig.worker.put(entry)
        rig.worker.locked_entries.note(STRATEGY, [FEED_DEAD])
        carry_on.set()
        answer = answer_to(rig, entry)
    finally:
        carry_on.set()
        rig.worker.close(wait_s=THREAD_WAIT_S)

    assert isinstance(answer, NotSent) and answer.reason is NotSentReason.ENTRIES_LOCKED
    assert placed(rig) == [("INFY", "SL-M")]


# ----- a session's traders write to the worker's own record (2.4.0 item P4-2) -------------


@pytest.fixture
def session_world(make_fake_openalgo, monkeypatch, tmp_path):
    """A session of `session_samples.Hold` on the fake (SBIN trading every 10 s at 500.00
    from 14:40:05 IST; it buys 10 at 14:42:01 and holds them), Ctrl+C at 14:44:30."""
    world = session_rig(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(ist(14, 44, 30))
    return world


def read_the_workers_record(world, seen: list, *, lock: bool = False):
    """An action of the world's script: note the order worker's record of the running
    session's one trader in `seen`, after the trader locks its entries for the feed when
    `lock` (lifted again at once, so the session runs on as before)."""

    def act():
        session = world.trace.runs[-1].session
        (trader,) = session.traders
        if lock:
            trader.lock(FEED_DEAD)
        seen.append(session.parts.worker.locked_entries.reasons(trader.name))
        if lock:
            trader.unlock(FEED_DEAD)

    return act


def test_a_session_traders_lock_reaches_the_order_workers_own_record(session_world):
    seen: list = []
    session_world.script.at(
        ist(14, 43, 30), read_the_workers_record(session_world, seen, lock=True)
    )

    session_world.run(Hold)

    assert seen == [(FEED_DEAD,)]


def test_a_dead_feeds_lock_in_a_session_reaches_the_order_workers_own_record(session_world):
    # The feed dies at 14:43:30 after its tick of 14:43:25: the risk watch locks the
    # strategy's entries from 14:43:35, 10 s without a tick, and they stay locked.
    seen: list = []
    session_world.script.at(ist(14, 43, 30), session_world.fake.faults.feed_dead)
    session_world.script.at(ist(14, 43, 50), read_the_workers_record(session_world, seen))

    session_world.run(Hold)

    assert seen == [(FEED_DEAD,)]


def answer_to(rig: WorkerRig, request: PlaceRequest):
    """The worker's first answer to `request` (a `SendResult` or a `NotSent`), waiting for
    it on the inbox; None when none came in time."""
    while (envelope := rig.inbox.take(wait_s=THREAD_WAIT_S)) is not None:
        message = envelope.message
        if isinstance(message, NotSent) and message.request is request:
            return message
        if isinstance(message, SendResult) and message.request_id == request.request_id:
            return message
    return None
