"""The order worker settling a modify or cancel whose answer was lost or refused
(`_worker_settle`, 16-session-design 2.4 step 7, 4.3, 4.5) on the fake OpenAlgo: the
order's further requests wait, a read of its row on the back-off settles it, a modify no
row settles in 10 s expires with an ERROR alert, and the order's terminal news settles it
before any read.
"""

from __future__ import annotations

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from worker_samples import PRICE, WorkerRig

from openalgo.backtest._live._order_answers import Outcome
from openalgo.backtest._live._types import AlertKind, OrderStatus
from openalgo.backtest._live._venue_requests import OrderTerms, Priority
from openalgo.backtest._live._worker_settle import (
    MODIFY_SETTLE_DEADLINE_S,
    ORDER_CHECK_DELAYS_S,
)
from openalgo.backtest._live.inbox import NotSent, NotSentReason, OrderNews, SendResult, Settled

NEW_TERMS = OrderTerms(20, "LIMIT", 705.0, 0.0)


@pytest.fixture
def rig(fake_openalgo, tmp_path):
    fake_openalgo.add_instrument("SBIN", "NSE")
    fake_openalgo.trade("SBIN", "NSE", PRICE)
    made = WorkerRig(fake_openalgo, tmp_path)
    made.messages()
    yield made
    made.worker.close()


def resting(rig: WorkerRig):
    """A LIMIT buy of 10 at 700.00 resting at the broker: its request and order ID."""
    place = rig.place(price_type="LIMIT", price=700.0)
    (result,) = rig.of(SendResult, rig.run(place))
    return place, result.outcome.order_id


def calls(rig: WorkerRig) -> list[str]:
    return [entry.what for entry in rig.fake.log.calls() if entry.what != "analyzer"]


def test_the_reads_back_off_as_the_order_paths_polls_do():
    assert (ORDER_CHECK_DELAYS_S, MODIFY_SETTLE_DEADLINE_S) == ((1.0, 2.0, 5.0), 10.0)


def test_an_unanswered_modify_holds_its_order_until_its_row_shows_the_new_terms(rig):
    place, order_id = resting(rig)
    rig.fake.faults.lose_answer("modifyorder")
    change = rig.modify(place, order_id, NEW_TERMS, Priority.ADJUST)
    (answer,) = rig.of(SendResult, rig.run(change))
    rig.run(rig.cancel(place, order_id, Priority.ADJUST))
    held = calls(rig)

    rig.clock.advance(1)
    rig.worker.step()

    (settled,) = rig.of(Settled, rig.messages())
    assert answer.outcome.kind is Outcome.UNANSWERED
    assert (settled.request, settled.check.quantity, settled.check.price) == (change, 20, 705.0)
    assert held == ["placeorder", "modifyorder"]
    assert calls(rig) == ["placeorder", "modifyorder", "orderstatus", "cancelorder"]


def test_a_refused_modify_is_settled_by_a_row_showing_the_old_terms(rig):
    place, order_id = resting(rig)
    rig.fake.faults.rate_limit("modifyorder")
    change = rig.modify(place, order_id, NEW_TERMS, Priority.ADJUST)
    (answer,) = rig.of(SendResult, rig.run(change))

    rig.clock.advance(1)
    rig.worker.step()

    (settled,) = rig.of(Settled, rig.messages())
    assert answer.outcome.kind is Outcome.REJECTED
    assert (settled.check.quantity, settled.check.price) == (10, 700.0)


def test_a_modify_no_row_settles_within_its_deadline_expires_with_an_alert(rig):
    place, order_id = resting(rig)
    rig.fake.faults.lose_answer("modifyorder")
    for _ in range(3):
        rig.fake.faults.gateway_error("orderstatus")
    change = rig.modify(place, order_id, NEW_TERMS, Priority.ADJUST)
    rig.run(change)

    posted = []
    for wait in (1, 2, 5, 2):
        rig.clock.advance(wait)
        rig.worker.step()
        posted += rig.messages()

    (settled,) = rig.of(Settled, posted)
    assert (settled.request, settled.expired) == (change, True)
    assert len(rig.fake.log.calls("orderstatus")) == 3
    assert rig.alerts.kinds() == [AlertKind.ERROR]


def test_a_refused_cancel_of_an_order_still_working_is_settled_by_its_row(rig):
    place, order_id = resting(rig)
    rig.fake.faults.rate_limit("cancelorder")
    rig.run(rig.cancel(place, order_id, Priority.ADJUST))

    rig.clock.advance(1)
    rig.worker.step()

    (settled,) = rig.of(Settled, rig.messages())
    assert settled.check.status is OrderStatus.OPEN


def test_a_cancel_the_order_path_holds_back_is_refused_and_settled_by_a_read(rig):
    place, order_id = resting(rig)
    rig.fake.faults.bad_gateway_page("placeorder")
    rig.run(rig.place())
    cancel = rig.cancel(place, order_id, Priority.ADJUST)

    (not_sent,) = rig.of(NotSent, rig.run(cancel))
    rig.clock.advance(1)
    rig.worker.step()

    (settled,) = rig.of(Settled, rig.messages())
    assert (not_sent.reason, settled.request, settled.check.status) == (
        NotSentReason.REFUSED,
        cancel,
        OrderStatus.OPEN,
    )


def test_the_orders_terminal_news_settles_it_before_any_read(rig):
    place, order_id = resting(rig)
    rig.fake.faults.lose_answer("cancelorder")
    (answer,) = rig.of(SendResult, rig.run(rig.cancel(place, order_id, Priority.ADJUST)))

    rig.worker.step()
    news = rig.of(OrderNews, rig.messages())
    rig.clock.advance(5)
    rig.worker.step()

    assert answer.outcome.kind is Outcome.UNANSWERED
    assert [item.update.status for item in news] == [OrderStatus.CANCELLED]
    assert (rig.of(Settled, rig.messages()), rig.fake.log.calls("orderstatus")) == ([], [])
