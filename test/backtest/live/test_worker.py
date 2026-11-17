"""The order worker (`worker`, 16-session-design 2.4) on the fake OpenAlgo: its startup
phase, the order of its step (priority 0 before the guard poll), what each request
becomes (`NotSent` kinds included), the book reads, the releases, and what the loop reads
of it (`busy_since_ns`, `raw_pending`, `unfinished`).

The worker runs with the fake's real SDK clients, a real mode guard and a real journal;
the test drives `step()` on the fake's manual clock (`worker_samples`).
"""

from __future__ import annotations

import threading

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from worker_samples import DAY, PRICE, WorkerRig, looks_idle, watch_handover

from openalgo.backtest._live._order_answers import Outcome
from openalgo.backtest._live._types import AlertKind, Mode, OrderStatus, Purpose
from openalgo.backtest._live._venue_requests import OrderTerms, Priority
from openalgo.backtest._live._worker_queue import (
    Probe,
    ReadBooks,
    ReadPositions,
    ReleaseFill,
    StreamGap,
)
from openalgo.backtest._live.inbox import (
    Books,
    GuardLocked,
    HeldFills,
    NotSent,
    NotSentReason,
    OrderNews,
    PositionRows,
    RawOrderUpdate,
    SendResult,
    WorkerFailed,
    WorkerReady,
)
from openalgo.backtest._live.orders import OrderPath
from openalgo.backtest.errors import JournalError, LiveError

NS = 1_000_000_000
#: Seconds a thread test waits for the worker's thread at most.
THREAD_WAIT_S = 5.0


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


def endpoints(rig: WorkerRig) -> list[str]:
    return [entry.what for entry in rig.fake.log.calls()]


def test_the_startup_posts_the_journal_session(broker, tmp_path):
    rig = WorkerRig(broker, tmp_path)

    (ready,) = rig.messages()

    assert isinstance(ready, WorkerReady)
    assert (ready.session.day, ready.session.mode, ready.session.starts) == (DAY, Mode.LIVE, 1)
    rig.worker.close()


def test_a_restart_counts_one_more_start(broker, tmp_path):
    rig = WorkerRig(broker, tmp_path)
    rig.restart()

    ready = rig.of(WorkerReady, rig.messages())

    assert [message.session.starts for message in ready] == [1, 2]
    rig.worker.close()


def test_the_worker_starts_up_once(rig):
    with pytest.raises(LiveError, match="once"):
        rig.worker.start_up()


def test_the_startup_phase_on_the_workers_thread_opens_its_journal_there(broker, tmp_path):
    rig = WorkerRig(broker, tmp_path, start=False)
    rig.worker.start()
    ready = rig.inbox.take(wait_s=5.0)

    rig.worker.put(rig.place())
    answer = rig.inbox.take(wait_s=5.0)
    rig.worker.close()

    assert isinstance(ready.message, WorkerReady)
    assert isinstance(answer.message, SendResult), answer.message
    assert answer.message.outcome.kind is Outcome.ACCEPTED


def test_an_entry_is_sent_and_its_news_names_its_intent(rig):
    place = rig.place()

    posted = rig.run(place)
    rig.worker.step()
    posted += rig.messages()

    (result,) = rig.of(SendResult, posted)
    assert (result.request_id, result.intent_id, result.outcome.kind) == (
        place.request_id,
        place.intent.intent_id,
        Outcome.ACCEPTED,
    )
    news = rig.of(OrderNews, posted)
    assert {item.intent_id for item in news} == {place.intent.intent_id}
    assert news[-1].update.status is OrderStatus.COMPLETE


def test_a_protective_request_goes_before_the_guards_poll(rig):
    rig.clock.advance(6)
    rig.run(rig.stop())

    # The guard's start, the stop, then the guard's poll.
    assert endpoints(rig) == ["analyzer", "placeorder", "analyzer"]


def test_a_stop_goes_before_an_entry_put_ahead_of_it(rig):
    rig.worker.put(rig.place())
    rig.worker.put(rig.stop())

    rig.worker.step()

    sent = [entry.detail["request"]["pricetype"] for entry in rig.fake.log.calls("placeorder")]
    assert sent == ["SL-M", "MARKET"]


def test_a_lock_of_the_guard_is_posted_once_and_nothing_is_sent_after_it(rig):
    rig.fake.set_analyzer(True)
    rig.clock.advance(6)

    rig.worker.step()
    first = rig.messages()
    posted = rig.run(rig.stop())

    assert len(rig.of(GuardLocked, first)) == 1
    (not_sent,) = rig.of(NotSent, posted)
    assert (not_sent.reason, rig.of(GuardLocked, posted)) == (NotSentReason.LOCKED, [])


def test_an_entry_decided_too_long_ago_is_not_sent(rig):
    place = rig.place(created_ns=rig.clock.now())
    rig.clock.advance(10.5)

    posted = rig.run(place)

    (not_sent,) = rig.of(NotSent, posted)
    assert (not_sent.request, not_sent.reason) == (place, NotSentReason.STALE)
    assert "placeorder" not in endpoints(rig)


def unbound_entry(rig: WorkerRig, *, reached_broker: bool = False) -> None:
    """An SBIN entry answered by a gateway's 502 page: unbound, so SBIN is held until the
    order book settles it."""
    rig.fake.faults.bad_gateway_page("placeorder", reached_broker=reached_broker)
    rig.run(rig.place())


def test_an_entry_the_order_path_holds_back_is_refused_and_not_tried_again(rig):
    unbound_entry(rig)
    entry = rig.place()

    posted = rig.run(entry)
    rig.clock.advance(1)
    rig.worker.step()

    (not_sent,) = rig.of(NotSent, posted)
    assert (not_sent.request, not_sent.reason) == (entry, NotSentReason.REFUSED)
    assert len(rig.fake.log.calls("placeorder")) == 1


def steps(rig: WorkerRig, seconds: float) -> list:
    """Step every half second for `seconds`; what was posted."""
    posted = []
    for _ in range(round(seconds / 0.5)):
        rig.clock.advance(0.5)
        rig.worker.step()
        posted += rig.messages()
    return posted


def test_a_stop_held_back_is_put_back_noticed_at_a_pace_and_sent_when_the_hold_lifts(rig):
    # The order book cannot be read for 7 seconds, then shows the entry: SBIN is held
    # until then.
    for _ in range(7):
        rig.fake.faults.gateway_error("orderbook")
    unbound_entry(rig, reached_broker=True)
    stop = rig.stop()

    notices = rig.of(NotSent, rig.run(stop)) + rig.of(NotSent, steps(rig, 5.0))
    held_for = len(rig.fake.log.calls("placeorder"))
    steps(rig, 3.0)

    assert [(n.request, n.reason) for n in notices] == [(stop, NotSentReason.HELD)] * 2
    sent = [entry.detail["request"]["pricetype"] for entry in rig.fake.log.calls("placeorder")]
    assert (held_for, sent) == (1, ["MARKET", "SL-M"])


def test_a_stop_held_while_openalgo_is_silent_is_noticed_so_and_a_probe_lifts_it(rig):
    rig.fake.faults.timeout("placeorder")
    rig.run(rig.place(symbol="INFY"))
    stop = rig.stop()

    (not_sent,) = rig.of(NotSent, rig.run(stop))
    rig.run(Probe())
    rig.clock.advance(1)
    rig.worker.step()

    assert not_sent.reason is NotSentReason.SILENT
    assert [e.detail["request"]["pricetype"] for e in rig.fake.log.calls("placeorder")][-1] == (
        "SL-M"
    )


def test_a_protective_modify_held_back_is_put_back_and_a_trailing_one_is_refused(rig):
    sbin = rig.stop()
    infy = rig.place(
        Purpose.STOP, action="SELL", price_type="SL-M", trigger_price=1490.0, symbol="INFY"
    )
    sbin_id = rig.of(SendResult, rig.run(sbin))[0].outcome.order_id
    infy_id = rig.of(SendResult, rig.run(infy))[0].outcome.order_id
    rig.fake.faults.drop_connection("orderbook")
    rig.run(Probe())
    units = rig.modify(sbin, sbin_id, OrderTerms(20, "SL-M", 0.0, 800.0), Priority.PROTECT)
    trailing = rig.modify(infy, infy_id, OrderTerms(10, "SL-M", 0.0, 1495.0), Priority.ADJUST)

    rig.worker.put(units)
    rig.worker.put(trailing)
    rig.worker.step()

    posted = [(n.request, n.reason) for n in rig.of(NotSent, rig.messages())]
    assert posted == [(units, NotSentReason.SILENT), (trailing, NotSentReason.REFUSED)]
    assert rig.fake.log.calls("modifyorder") == []


def test_a_protective_request_the_order_path_refuses_for_good_is_refused_with_an_alert(rig):
    # A cancel of an order this session never sent: no hold explains it, so it is not
    # put back to be refused again every half second.
    cancel = rig.cancel(rig.stop(), "260101000000001", Priority.PROTECT)

    posted = rig.run(cancel) + steps(rig, 6.0)

    assert [(n.request, n.reason) for n in rig.of(NotSent, posted)] == [
        (cancel, NotSentReason.REFUSED)
    ]
    assert rig.alerts.kinds() == [AlertKind.ERROR]


def filled_entry(rig: WorkerRig):
    place = rig.place()
    posted = rig.run(place)
    rig.worker.step()
    rig.messages()
    (result,) = rig.of(SendResult, posted)
    return place, result.outcome.order_id


def test_a_change_of_an_order_known_ended_is_not_sent(rig):
    place, order_id = filled_entry(rig)
    change = rig.modify(place, order_id, OrderTerms(20, "MARKET"), Priority.ADJUST)

    (not_sent,) = rig.of(NotSent, rig.run(change))

    assert (not_sent.reason, "modifyorder" in endpoints(rig)) == (NotSentReason.ORDER_ENDED, False)


def test_a_journal_that_cannot_take_an_intent_stops_the_worker(rig):
    tomorrow = rig.clock.now() + 24 * 3600 * NS
    posted = rig.run(rig.place(created_ns=tomorrow))
    after = rig.run(rig.stop())

    (failed,) = rig.of(WorkerFailed, posted)
    assert isinstance(failed.error, JournalError)
    assert (after, len(rig.fake.log.calls("placeorder"))) == ([], 0)


def test_an_unexpected_error_in_a_step_stops_the_worker_as_on_its_thread(rig, monkeypatch):
    def broken(path: OrderPath) -> list:
        raise RuntimeError("a defect")

    monkeypatch.setattr(OrderPath, "held_fills", broken)
    took = rig.worker.step()
    (failed,) = rig.of(WorkerFailed, rig.messages())
    monkeypatch.undo()
    after = rig.run(rig.stop())

    assert (took, str(failed.error)) == (False, "a defect")
    assert (after, rig.fake.log.calls("placeorder")) == ([], [])


def test_a_step_under_way_when_the_worker_closes_sends_nothing_more(broker, tmp_path):
    rig = WorkerRig(broker, tmp_path, start=False)
    inside, carry_on = threading.Event(), threading.Event()
    client = broker.client()

    def placeorder(**fields):
        inside.set()
        carry_on.wait(THREAD_WAIT_S)
        return client.placeorder(**fields)

    rig.orders.placeorder = placeorder
    infy_stop = rig.place(
        Purpose.STOP, action="SELL", price_type="SL-M", trigger_price=1490.0, symbol="INFY"
    )
    rig.worker.put(rig.stop())
    rig.worker.put(infy_stop)
    rig.worker.start()
    assert inside.wait(THREAD_WAIT_S)

    rig.worker.close(wait_s=0.0)
    carry_on.set()
    rig.worker.close(wait_s=THREAD_WAIT_S)

    assert len(broker.log.calls("placeorder")) == 1


def test_the_books_are_read_and_posted_with_their_generation(rig):
    (books,) = rig.of(Books, rig.run(ReadBooks(7)))

    assert books.generation == 7
    answers = (books.positionbook, books.orderbook, books.tradebook, books.funds)
    assert [answer["status"] for answer in answers] == ["success"] * 4
    assert books.started_ns <= books.ended_ns


def test_the_position_book_is_read_for_a_held_cancel(rig):
    (rows,) = rig.of(PositionRows, rig.run(ReadPositions()))

    assert (rows.positionbook["status"], rows.read_ns) == ("success", rig.clock.now())


def test_no_book_is_read_once_the_guard_locked(rig):
    rig.fake.set_analyzer(True)
    rig.clock.advance(6)
    rig.worker.step()
    rig.messages()

    (not_sent,) = rig.of(NotSent, rig.run(ReadBooks(1)))

    assert (not_sent.reason, "positionbook" in endpoints(rig)) == (NotSentReason.LOCKED, False)


def test_a_held_fill_is_posted_and_released_at_a_price(rig):
    rig.fake.faults.zero_price_next_fill()
    rig.fake.faults.gateway_error("tradebook")
    place = rig.place()

    posted = rig.run(place)
    rig.worker.step()
    posted += rig.messages()
    (result,) = rig.of(SendResult, posted)
    (held,) = rig.of(HeldFills, posted)
    released = rig.run(ReleaseFill(result.outcome.order_id, 811.5))

    assert [fill.filled_quantity for fill in held.held] == [10]
    (news,) = rig.of(OrderNews, released)
    assert (news.update.filled_quantity, news.update.average_price) == (10, 811.5)


def test_a_stream_gap_asks_for_a_poll_of_every_working_order(rig):
    rig.run(rig.place(price_type="LIMIT", price=700.0))
    rig.clock.advance(1)

    rig.run(StreamGap(rig.clock.now()))

    assert len(rig.fake.log.calls("orderstatus")) == 1


def test_the_worker_is_busy_during_each_call_and_idle_between_them(rig):
    rig.run(rig.place())

    assert rig.orders.busy_during[0] == ("placeorder", rig.clock.now())
    assert rig.worker.busy_since_ns is None


def test_a_request_is_unfinished_until_the_worker_has_executed_it(rig):
    rig.worker.put(rig.place())

    waiting = rig.worker.unfinished
    rig.worker.step()

    assert (waiting, rig.worker.unfinished) == (1, 0)


def test_a_request_put_back_stays_unfinished_until_it_is_sent(rig):
    # As above: SBIN is held for 7 seconds while the order book cannot be read.
    for _ in range(7):
        rig.fake.faults.gateway_error("orderbook")
    unbound_entry(rig, reached_broker=True)
    rig.run(rig.stop())

    put_back = rig.worker.unfinished
    steps(rig, 8.0)

    assert (put_back, rig.worker.unfinished) == (1, 0)
    assert [e.detail["request"]["pricetype"] for e in rig.fake.log.calls("placeorder")] == [
        "MARKET",
        "SL-M",
    ]


def test_the_raw_updates_waiting_are_counted_until_drained(rig):
    rig.worker.updates.put(RawOrderUpdate({"type": "order_update"}, rig.clock.now()))
    rig.worker.updates.put(RawOrderUpdate({"type": "order_update"}, rig.clock.now()))

    waiting = rig.worker.raw_pending
    rig.worker.step()

    assert (waiting, rig.worker.raw_pending) == (2, 0)


def test_a_drained_update_counts_as_waiting_until_its_news_is_posted(rig, monkeypatch):
    rig.run(rig.stop())  # its confirming word waits in the raw FIFO
    seen = watch_handover(rig.worker, monkeypatch)

    rig.worker.step()

    assert rig.of(OrderNews, rig.messages()) != []
    assert (seen != [], looks_idle(seen)) == (True, [])


def test_polled_news_counts_as_waiting_until_it_is_posted(rig, monkeypatch):
    rig.fake.faults.silence_order_updates()
    rig.run(rig.stop())
    rig.clock.advance(2.5)
    seen = watch_handover(rig.worker, monkeypatch)

    rig.worker.step()

    assert [news.update.source for news in rig.of(OrderNews, rig.messages())] == ["poll"]
    assert (seen != [], looks_idle(seen)) == (True, [])


def test_a_released_held_word_counts_as_waiting_until_it_is_posted(rig, monkeypatch):
    rig.fake.faults.zero_price_next_fill()
    rig.fake.faults.gateway_error("tradebook")
    (result,) = rig.of(SendResult, rig.run(rig.place()))
    rig.worker.step()
    rig.messages()
    seen = watch_handover(rig.worker, monkeypatch)

    released = rig.run(ReleaseFill(result.outcome.order_id, 811.5))

    assert len(rig.of(OrderNews, released)) == 1
    assert (seen != [], looks_idle(seen)) == (True, [])


def test_an_update_that_cannot_be_read_is_counted_and_dropped(rig):
    rig.worker.updates.put(RawOrderUpdate({"type": "order_update"}, rig.clock.now()))

    rig.worker.step()

    assert (rig.worker.unreadable_updates, rig.messages()) == (1, [])


def test_a_step_before_the_startup_is_refused(broker, tmp_path):
    rig = WorkerRig(broker, tmp_path, start=False)
    with pytest.raises(LiveError, match="start_up"):
        rig.worker.step()
