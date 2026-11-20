"""The live order rate throttle (`_live/_order_throttle.py`; 18-multi-strategy 2.5a rule 12):
at most `max_orders_per_second` order calls in any one-second window, a call beyond it
waiting in the worker's queue with stops and exits first, a note after 2 s, an entry or
add withdrawn after 5 s, an exit never dropped. On a manual clock: nothing waits for real
and nothing contacts a server.

A request is taken as the worker takes it (`OrderThrottle.take`), and each order request
taken is counted as the order path counts the call it makes (`RateBudget.take`).
"""

from __future__ import annotations

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from worker_samples import PRICE, WorkerRig

from openalgo.backtest._live._order_throttle import (
    NOTICE_AFTER_S,
    WITHDRAW_AFTER_S,
    OrderThrottle,
)
from openalgo.backtest._live._types import OrderIntent, Purpose
from openalgo.backtest._live._venue_requests import CancelRequest, PlaceRequest, Priority
from openalgo.backtest._live._worker_queue import ReadBooks, WorkerQueue, is_order_call
from openalgo.backtest._live.inbox import NotSent, NotSentReason, Throttled
from openalgo.backtest.errors import LiveError

NS = 1_000_000_000
START_NS = 1_794_804_300 * NS


class Clock:
    def __init__(self) -> None:
        self.now_ns = START_NS

    def now(self) -> int:
        return self.now_ns

    def advance(self, seconds: float) -> None:
        self.now_ns += round(seconds * NS)


def intent(order: int, purpose: Purpose) -> OrderIntent:
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
        created_ns=START_NS,
    )


def entry(order: int) -> PlaceRequest:
    return PlaceRequest(order, order, intent(order, Purpose.ENTRY), Priority.ENTER)


def exit_(order: int) -> PlaceRequest:
    return PlaceRequest(order, order, intent(order, Purpose.EXIT), Priority.PROTECT)


def stop_cancel(order: int) -> CancelRequest:
    stop = intent(order, Purpose.STOP)
    return CancelRequest(order, order, stop, f"B{order}", Priority.PROTECT)


class World:
    """A throttle of `per_second`, its queue, the messages it posts and the requests it
    finished with."""

    def __init__(self, per_second: int) -> None:
        self.clock = Clock()
        self.posted: list = []
        self.finished = 0
        self.queue = WorkerQueue()
        self.throttle = OrderThrottle(
            self.clock.now, per_second, post=self.posted.append, finished=self._finish
        )

    def _finish(self) -> None:
        self.finished += 1

    def put(self, *requests) -> None:
        for request in requests:
            self.queue.put(request)

    def step(self) -> list:
        """Every request the worker takes now, each order call counted."""
        taken = []
        while (request := self.throttle.take(self.queue, Priority.UPKEEP)) is not None:
            if is_order_call(request):
                self.throttle.budget.take()
            taken.append(request)
        return taken


def test_no_more_order_calls_than_the_rate_in_any_second():
    world = World(per_second=2)
    first, second, third = entry(1), entry(2), entry(3)
    world.put(first, second, third)
    assert world.step() == [first, second]
    world.clock.advance(0.999)
    assert world.step() == []
    world.clock.advance(0.001)
    assert world.step() == [third]


def test_when_the_window_frees_stops_and_exits_go_before_entries_queued_earlier():
    world = World(per_second=1)
    world.put(entry(1))
    world.step()
    later_entry, cancel, exit_order = entry(2), stop_cancel(3), exit_(4)
    world.put(later_entry, cancel, exit_order)
    taken = []
    for _ in range(3):
        world.clock.advance(1.0)
        taken += world.step()
    assert taken == [cancel, exit_order, later_entry]


def test_a_request_that_makes_no_order_call_is_never_held():
    world = World(per_second=1)
    world.put(entry(1), entry(2))
    world.step()
    read = ReadBooks(generation=1)
    world.put(read)
    assert world.step() == [read]


def full_window(world: World) -> None:
    """Keep the throttle's window full whatever the clock says."""
    world.throttle.budget.wait_seconds = lambda: 1.0


def test_a_request_held_two_seconds_is_noted_once():
    world = World(per_second=1)
    held = exit_(1)
    world.put(held)
    full_window(world)
    world.throttle.take(world.queue, Priority.UPKEEP)
    world.clock.advance(NOTICE_AFTER_S - 0.1)
    world.throttle.take(world.queue, Priority.UPKEEP)
    assert world.posted == []
    world.clock.advance(0.1)
    world.throttle.take(world.queue, Priority.UPKEEP)
    world.clock.advance(1.0)
    world.throttle.take(world.queue, Priority.UPKEEP)
    assert world.posted == [Throttled(held, NOTICE_AFTER_S, withdrawn=False)]


def test_an_entry_held_five_seconds_is_withdrawn_as_not_sent_for_the_order_rate():
    world = World(per_second=1)
    held = entry(1)
    world.put(held)
    full_window(world)
    world.throttle.take(world.queue, Priority.UPKEEP)
    world.clock.advance(WITHDRAW_AFTER_S)
    assert world.throttle.take(world.queue, Priority.UPKEEP) is None
    assert world.posted == [
        Throttled(held, WITHDRAW_AFTER_S, withdrawn=True),
        NotSent(held, NotSentReason.THROTTLED, "waited 5.0 s for the order rate"),
    ]
    assert len(world.queue) == 0
    assert world.finished == 1
    assert NotSentReason.THROTTLED.value == "order_rate"


def test_an_exit_is_never_dropped_however_long_it_waits():
    world = World(per_second=1)
    held = exit_(1)
    world.put(held)
    full_window(world)
    for _ in range(30):
        world.throttle.take(world.queue, Priority.UPKEEP)
        world.clock.advance(1.0)
    assert len(world.queue) == 1
    assert world.posted == [Throttled(held, NOTICE_AFTER_S, withdrawn=False)]
    assert world.finished == 0


@pytest.mark.parametrize("per_second", [0, -3, 1.5])
def test_a_rate_that_is_not_a_whole_number_of_at_least_one_is_refused(per_second):
    with pytest.raises(LiveError, match="at least 1"):
        OrderThrottle(Clock().now, per_second, post=print, finished=lambda: None)


# ----- the order worker on the fake OpenAlgo ------------------------------------------


@pytest.fixture
def broker(fake_openalgo):
    fake_openalgo.add_instrument("SBIN", "NSE")
    fake_openalgo.trade("SBIN", "NSE", PRICE)
    return fake_openalgo


def test_the_worker_sends_at_most_eight_order_calls_in_a_second_by_default(broker, tmp_path):
    # Ten exits handed over at once (an exit reads no mode before it is sent, so only the
    # order calls count): eight placeorder calls now, the other two once the first of them
    # is a second old.
    rig = WorkerRig(broker, tmp_path)
    rig.messages()
    for _ in range(10):
        rig.worker.put(rig.place(Purpose.EXIT, action="SELL"))
    for _ in range(12):
        rig.worker.step()
    assert len(broker.log.calls("placeorder")) == 8
    broker.clock.advance(1.0)
    for _ in range(4):
        rig.worker.step()
    assert len(broker.log.calls("placeorder")) == 10
    rig.worker.close()
