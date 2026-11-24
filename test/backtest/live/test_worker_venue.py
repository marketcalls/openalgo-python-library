"""The kernel, the venue map and the order worker together on the fake OpenAlgo
(16-session-design 2.5 A and E): the kernel's commands become requests, the worker sends
them, and its answers and the order's news come back to the kernel through the venue map,
as the session loop will route them.

The kernel is the real binding's session of the fake's trading day; the worker runs on
the fake's real SDK clients, a real mode guard and a real journal (`worker_samples`).
"""

from __future__ import annotations

import itertools

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from venue_samples import SBIN, new_session, order, release_minute
from worker_samples import DAY, PRICE, STRATEGY, WorkerRig

from openalgo.backtest._live.inbox import NotSent, OrderNews, SendResult, Settled, WorkerReady
from openalgo.backtest._live.venue_map import VenueMap


class Session:
    """A kernel session of the fake's day, its venue map and a worker on the fake, wired
    as the loop wires them (the answers routed by kind)."""

    def __init__(self, rig: WorkerRig) -> None:
        (ready,) = rig.of(WorkerReady, rig.messages())
        self.rig = rig
        self.engine = new_session(DAY)
        self.map = VenueMap(
            self.engine,
            strategy=STRATEGY,
            instruments=[SBIN],
            product="MIS",
            session=ready.session,
            request_ids=itertools.count(1),
            alerts=rig.alerts,
            now=rig.clock.now,
        )
        self.minute = 0
        assert release_minute(self.engine, 0, day=DAY, price=PRICE) == "bar_close"

    def hand_over(self) -> None:
        """Leave the decision point and hand the commands to the worker."""
        assert self.engine.advance()[0] == "waiting"
        for request in self.map.requests(self.engine.take_commands()):
            self.rig.worker.put(request)

    def work(self, steps: int = 2) -> None:
        """Let the worker step and route what it posts to the venue map."""
        for _ in range(steps):
            self.rig.worker.step()
            for message in self.rig.messages():
                self.route(message)

    def route(self, message) -> None:
        if isinstance(message, SendResult):
            handled = self.map.sent(message)
        elif isinstance(message, NotSent):
            handled = self.map.not_sent(message)
        elif isinstance(message, Settled):
            handled = self.map.settled(message)
        elif isinstance(message, OrderNews):
            self.map.news(message.update)
            return
        else:
            return
        for request in handled.requests:
            self.rig.worker.put(request)

    def next_close(self) -> None:
        self.minute += 1
        assert release_minute(self.engine, self.minute, day=DAY, price=PRICE) == "bar_close"


@pytest.fixture
def session(fake_openalgo, tmp_path):
    fake_openalgo.add_instrument("SBIN", "NSE")
    fake_openalgo.trade("SBIN", "NSE", PRICE)
    rig = WorkerRig(fake_openalgo, tmp_path)
    yield Session(rig)
    rig.worker.close()


def test_a_market_entry_reaches_the_broker_and_its_fill_the_kernel(session):
    session.engine.submit(order("BUY", 10.0))
    session.hand_over()

    session.work()

    position = session.engine.position(0, "MIS")
    assert (position["quantity_units"], position["average_price_rupees"]) == (10.0, PRICE)
    assert session.rig.fake.position("SBIN", "NSE") == 10


def test_a_modify_of_a_resting_entry_takes_effect_in_both_once_the_broker_confirms_it(session):
    _, kernel_order, _ = session.engine.submit(order("BUY", 10.0, order_type="LIMIT", price=700.0))
    session.hand_over()
    session.work()
    session.next_close()
    session.engine.modify(kernel_order, order("BUY", 20.0, order_type="LIMIT", price=705.0))
    session.hand_over()

    session.work()

    (at_broker,) = session.rig.fake.orders()
    assert (at_broker.quantity, session.engine.order(kernel_order)["quantity_units"]) == (20, 20.0)


@pytest.mark.parametrize(
    ("silent_stream", "units_before_the_read"),
    [(False, 20.0), (True, 10.0)],
    ids=["the stream shows it", "the read shows it"],
)
def test_a_modify_whose_answer_was_lost_is_confirmed_once_the_broker_shows_it(
    session, silent_stream, units_before_the_read
):
    _, kernel_order, _ = session.engine.submit(order("BUY", 10.0, order_type="LIMIT", price=700.0))
    session.hand_over()
    session.work()
    session.next_close()
    session.engine.modify(kernel_order, order("BUY", 20.0, order_type="LIMIT", price=705.0))
    if silent_stream:
        session.rig.fake.faults.silence_order_updates()
    session.rig.fake.faults.lose_answer("modifyorder")
    session.hand_over()
    session.work()
    before = session.engine.order(kernel_order)["quantity_units"]

    session.rig.clock.advance(1)
    session.work()

    assert (before, session.engine.order(kernel_order)["quantity_units"]) == (
        units_before_the_read,
        20.0,
    )


def test_a_cancel_ends_the_order_in_the_kernel_once_the_broker_has_cancelled_it(session):
    _, kernel_order, _ = session.engine.submit(order("BUY", 10.0, order_type="LIMIT", price=700.0))
    session.hand_over()
    session.work()
    session.next_close()
    session.engine.cancel(kernel_order)
    session.hand_over()

    session.work()

    assert session.engine.order(kernel_order)["status"] == "Canceled"
    assert [placed.status for placed in session.rig.fake.orders()] == ["cancelled"]
