"""The order worker's queue (`_worker_queue`, 16-session-design 2.4): one FIFO per kernel
order ranked at its most urgent priority, only a FIFO's head taken, an order held while
its modify or cancel is unsettled, a request put back waiting at the head of its FIFO,
and an entry decided too long ago recognised as stale.
"""

from __future__ import annotations

import pytest

from openalgo.backtest._live._types import OrderIntent, Purpose
from openalgo.backtest._live._venue_requests import (
    CancelRequest,
    ModifyRequest,
    OrderTerms,
    PlaceRequest,
    Priority,
)
from openalgo.backtest._live._worker_queue import (
    ENTRY_STALE_AFTER_S,
    Probe,
    ReadBooks,
    ReleaseFill,
    StreamGap,
    WorkerQueue,
    is_stale,
    order_key,
)

NS = 1_000_000_000
NOW_NS = 1_794_804_300 * NS
TERMS = OrderTerms(10, "SL-M", 0.0, 800.0)


def intent(order: int, purpose: Purpose = Purpose.ENTRY, created_ns: int = NOW_NS) -> OrderIntent:
    return OrderIntent(
        intent_id=f"20261116-live-1-ema_cross-o{order}",
        strategy="ema_cross",
        symbol="SBIN",
        exchange="NSE",
        action="BUY",
        quantity=10,
        price_type="MARKET",
        product="MIS",
        purpose=purpose,
        created_ns=created_ns,
    )


def place(number: int, order: int, priority: Priority) -> PlaceRequest:
    return PlaceRequest(number, order, intent(order), priority)


def modify(number: int, order: int, priority: Priority) -> ModifyRequest:
    return ModifyRequest(number, order, intent(order), f"B{order}", TERMS, TERMS, priority)


def cancel(number: int, order: int, priority: Priority) -> CancelRequest:
    return CancelRequest(number, order, intent(order), f"B{order}", priority)


def taken(queue: WorkerQueue, now_ns: int = NOW_NS, most: Priority = Priority.UPKEEP) -> list:
    out = []
    while (request := queue.take(now_ns, most=most)) is not None:
        out.append(request)
    return out


def queue_of(*requests) -> WorkerQueue:
    queue = WorkerQueue()
    for request in requests:
        queue.put(request)
    return queue


def test_the_most_urgent_request_goes_first():
    entry, stop = place(1, 1, Priority.ENTER), place(2, 2, Priority.PROTECT)
    assert taken(queue_of(entry, stop)) == [stop, entry]


def test_requests_of_one_priority_go_in_their_order_of_arrival():
    first, second = place(1, 1, Priority.PROTECT), place(2, 2, Priority.PROTECT)
    assert taken(queue_of(first, second)) == [first, second]


def test_an_orders_fifo_is_ranked_at_its_most_urgent_request_and_its_head_goes_first():
    other = place(1, 9, Priority.ADJUST)
    entry, protective = place(2, 1, Priority.ENTER), modify(3, 1, Priority.PROTECT)
    assert taken(queue_of(other, entry, protective)) == [entry, protective, other]


def test_a_trailing_modify_goes_before_a_later_exit_when_its_order_waits_to_be_cancelled():
    # The stop's FIFO ranks at its cancel's priority and its head is older than the exit;
    # once the modify is gone, the exit is older than the cancel.
    trailing, exit_cancel = modify(1, 1, Priority.ADJUST), cancel(3, 1, Priority.PROTECT)
    exit_place = place(2, 2, Priority.PROTECT)
    assert taken(queue_of(trailing, exit_place, exit_cancel)) == [
        trailing,
        exit_place,
        exit_cancel,
    ]


def test_only_the_protective_requests_are_taken_when_asked_for():
    trailing, stop = modify(1, 1, Priority.ADJUST), place(2, 2, Priority.PROTECT)
    queue = queue_of(trailing, stop)
    assert (taken(queue, most=Priority.PROTECT), len(queue)) == ([stop], 1)


def test_nothing_of_a_held_order_is_taken_until_it_is_released():
    change, other = modify(1, 1, Priority.PROTECT), place(2, 2, Priority.ENTER)
    queue = queue_of(change, other)
    queue.hold(order_key(change))

    first = taken(queue)
    queue.release(order_key(change))

    assert (first, taken(queue)) == ([other], [change])


def test_a_request_put_back_waits_at_the_head_of_its_fifo_and_keeps_its_place():
    stop, later = place(1, 1, Priority.PROTECT), modify(2, 1, Priority.PROTECT)
    other = place(3, 2, Priority.PROTECT)
    queue = queue_of(stop, later)
    assert queue.take(NOW_NS) is stop
    queue.put(other)
    queue.put_back(stop, NOW_NS + NS // 2)

    waiting = taken(queue, NOW_NS + NS // 4)
    after = taken(queue, NOW_NS + NS // 2)

    assert (waiting, after) == ([other], [stop, later])


def test_the_workers_own_requests_each_wait_on_their_own():
    release, books, probe = ReleaseFill("B1", 812.0), ReadBooks(1), Probe()
    gap = StreamGap(NOW_NS)
    assert taken(queue_of(release, books, probe, gap)) == [gap, release, books, probe]


@pytest.mark.parametrize(
    ("purpose", "age_s", "stale"),
    [
        (Purpose.ENTRY, ENTRY_STALE_AFTER_S, False),
        (Purpose.ENTRY, ENTRY_STALE_AFTER_S + 0.001, True),
        (Purpose.ADD, ENTRY_STALE_AFTER_S + 1, True),
        (Purpose.STOP, 60.0, False),
        (Purpose.SQUARE_OFF, 60.0, False),
    ],
    ids=["entry at the limit", "entry past it", "add past it", "stop", "square-off"],
)
def test_an_entry_or_add_decided_more_than_ten_seconds_before_is_stale(purpose, age_s, stale):
    request = PlaceRequest(1, 1, intent(1, purpose), Priority.ENTER)
    assert is_stale(request, NOW_NS + round(age_s * NS)) is stale


def test_the_queue_counts_what_waits():
    assert len(queue_of(place(1, 1, Priority.ENTER), modify(2, 1, Priority.ADJUST), Probe())) == 3
