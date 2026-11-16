"""The journal's stop of each leg, kept by the order worker (`_worker_stops`,
16-session-design 4.1) on the fake OpenAlgo: a STOP that binds is recorded, its terminal
news clears it before the news is posted, and a stop the journal still names is settled
before the new one is recorded: cleared when known ended or read ended, cancelled as an
orphan when it still works, and never fatal when it cannot be read.
"""

from __future__ import annotations

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from worker_samples import DAY, PRICE, STRATEGY, WorkerRig

from openalgo.backtest._live._types import AlertKind, Mode, OrderStatus, OrderUpdate
from openalgo.backtest._live.inbox import OrderNews, Resolutions, SendResult

LEG = (STRATEGY, "SBIN", "NSE")


@pytest.fixture
def rig(fake_openalgo, tmp_path):
    fake_openalgo.add_instrument("SBIN", "NSE")
    fake_openalgo.trade("SBIN", "NSE", PRICE)
    made = WorkerRig(fake_openalgo, tmp_path)
    made.messages()
    yield made
    made.worker.close()


def journal_stop(rig: WorkerRig) -> str | None:
    with rig.journal() as journal:
        return journal.stop_of(*LEG, day=DAY)


def placed_stop(rig: WorkerRig) -> str:
    """A stop of the leg placed and accepted: its order ID."""
    (result,) = rig.of(SendResult, rig.run(rig.stop()))
    return result.outcome.order_id


def test_a_stop_that_binds_is_recorded_as_its_legs_stop(rig):
    order_id = placed_stop(rig)
    assert journal_stop(rig) == order_id


def test_a_stops_terminal_news_clears_its_leg_before_the_news_is_posted(rig):
    order_id = placed_stop(rig)

    rig.fake.trade("SBIN", "NSE", 799.0)
    rig.worker.step()

    (news,) = [item for item in rig.of(OrderNews, rig.messages()) if item.update.status.terminal]
    assert (news.update.order_id, news.update.status, news.stop_cleared) == (
        order_id,
        OrderStatus.COMPLETE,
        True,
    )
    assert journal_stop(rig) is None


def test_an_entrys_terminal_news_clears_no_stop(rig):
    order_id = placed_stop(rig)

    posted = rig.run(rig.place())
    rig.worker.step()
    posted += rig.messages()

    news = rig.of(OrderNews, posted)
    assert news[-1].update.status is OrderStatus.COMPLETE
    assert {item.stop_cleared for item in news} == {False}
    assert journal_stop(rig) == order_id


def test_a_stop_bound_by_the_order_book_is_recorded(rig):
    rig.fake.faults.bad_gateway_page("placeorder", reached_broker=True)

    posted = rig.run(rig.stop())

    (resolutions,) = rig.of(Resolutions, posted)
    (bound,) = resolutions.resolutions
    assert journal_stop(rig) == bound.order_id


def ended_update(order_id: str, ts_ns: int) -> OrderUpdate:
    return OrderUpdate(
        order_id=order_id,
        symbol="SBIN",
        exchange="NSE",
        action="SELL",
        quantity=10,
        filled_quantity=10,
        pending_quantity=0,
        average_price=799.0,
        status=OrderStatus.COMPLETE,
        rejection_reason="",
        mode=Mode.LIVE,
        ts_recv_ns=ts_ns,
        source="stream",
    )


def test_a_new_stop_clears_a_stop_the_journal_knows_ended(rig):
    old = placed_stop(rig)
    rig.worker.close()
    with rig.journal() as journal:
        journal.record_update(ended_update(old, rig.clock.now()))
    rig.restart()
    rig.messages()

    new = placed_stop(rig)

    assert (journal_stop(rig), rig.fake.log.calls("orderstatus"), rig.alerts.sent) == (new, [], [])


def test_a_new_stop_clears_a_stop_its_row_shows_ended(rig):
    old = placed_stop(rig)
    rig.fake.faults.silence_order_updates()
    rig.fake.trade("SBIN", "NSE", 799.0)
    rig.restart()
    rig.messages()

    new = placed_stop(rig)

    assert [entry.detail["request"]["orderid"] for entry in rig.fake.log.calls("orderstatus")] == [
        old
    ]
    assert (journal_stop(rig), rig.alerts.sent) == (new, [])


def test_a_stop_still_working_that_the_journal_names_is_cancelled_as_an_orphan(rig):
    old = placed_stop(rig)
    rig.restart()
    rig.messages()

    new = placed_stop(rig)
    named_meanwhile = journal_stop(rig)
    rig.worker.step()

    assert rig.alerts.kinds() == [AlertKind.RECONCILE_MISMATCH]
    assert [entry.detail["request"]["orderid"] for entry in rig.fake.log.calls("cancelorder")] == [
        old
    ]
    assert (named_meanwhile, journal_stop(rig)) == (old, new)


def test_a_stop_the_journal_names_whose_row_cannot_be_read_keeps_it_and_alerts(rig):
    old = placed_stop(rig)
    rig.restart()
    rig.messages()
    rig.fake.faults.gateway_error("orderstatus")

    posted = rig.run(rig.stop())

    assert rig.of(SendResult, posted) != []
    assert (journal_stop(rig), rig.alerts.kinds()) == (old, [AlertKind.ERROR])
