"""Following orders once sent: news taken once and in order with its status as the broker
reported it, words that repeat or lose what is known, updates of orders not bound kept
aside, and the orders taken back after a restart."""

import pytest
from orders_fakes import (
    NS,
    STRATEGY,
    TIMEOUT,
    http_error,
    make_intent,
    make_rig,
    placed,
    server_error,
    status_answer,
    trades,
    update,
)

from openalgo.backtest._live._types import OrderStatus, Purpose

NOT_FOUND = http_error(404, server_error("Order 1 not found"))
STOP_ROW = {"pricetype": "SL", "action": "SELL"}
#: A trailing modify of the stop: its trigger moved up.
TRAIL = {
    "symbol": "SBIN",
    "action": "SELL",
    "exchange": "NSE",
    "price_type": "SL",
    "product": "MIS",
    "quantity": 10,
    "price": 768.0,
    "trigger_price": 808.0,
}


def sent_rig(**answers):
    """A rig whose intent i-1, a market buy, was placed as order 1 at the clock's start."""
    rig = make_rig(placeorder=[placed("1")], **answers)
    rig.path.send(make_intent("i-1"))
    return rig


def stop_rig(**answers):
    """A rig whose stop i-stop, an SL sell, was placed as order 1."""
    rig = make_rig(placeorder=[placed("1")], **answers)
    stop = make_intent(
        "i-stop",
        purpose=Purpose.STOP,
        action="SELL",
        price_type="SL",
        price=760.0,
        trigger_price=800.0,
    )
    rig.path.send(stop)
    return rig


def test_a_new_update_is_journaled_checked_by_the_guard_and_returned():
    rig = sent_rig()
    new = update("1", OrderStatus.OPEN, 0)

    assert rig.path.on_update(new) is new
    assert rig.journal.updates == [new]
    assert rig.guard.updates == [new]


def test_a_duplicate_update_is_dropped():
    rig = sent_rig()
    rig.path.on_update(update("1", OrderStatus.OPEN, 4))

    assert rig.path.on_update(update("1", OrderStatus.OPEN, 4)) is None


@pytest.mark.parametrize(
    "changed",
    [
        update("1", OrderStatus.TRIGGER_PENDING, 0, quantity=20),
        update("1", OrderStatus.TRIGGER_PENDING, 0, price=1.0),
        update("1", OrderStatus.OPEN, 0),
    ],
    ids=["order quantity", "average price", "status"],
)
def test_a_change_of_the_reported_status_quantity_or_price_is_news(changed):
    rig = stop_rig()
    rig.path.on_update(update("1", OrderStatus.TRIGGER_PENDING, 0))

    assert rig.path.on_update(changed) is changed


def test_a_status_is_passed_on_as_reported_whichever_way_it_changes():
    rig = stop_rig()
    words = [
        update("1", OrderStatus.TRIGGER_PENDING, 0),
        update("1", OrderStatus.OPEN, 0, source="poll"),
        update("1", OrderStatus.TRIGGER_PENDING, 0),
    ]

    taken = [rig.path.on_update(word) for word in words]

    assert taken == words


def test_an_open_after_the_sessions_own_modify_of_a_stop_is_passed_on_as_reported():
    # A broker reports a modify of a waiting stop (zerodha's MODIFY PENDING) as open: the
    # order path does not read a trigger into it, nor out of it.
    rig = stop_rig(modifyorder=[placed("1")])
    rig.path.on_update(update("1", OrderStatus.TRIGGER_PENDING, 0))
    rig.path.modify("1", **TRAIL)

    told = rig.path.on_update(update("1", OrderStatus.OPEN, 0))

    assert told.status is OrderStatus.OPEN


def test_a_poll_saying_open_for_a_stop_is_passed_on_as_open():
    rig = stop_rig(orderstatus=[status_answer("1", "open", **STOP_ROW)], tradebook=[trades()])
    rig.clock.advance(2)

    (first,) = rig.path.due_polls()

    assert first.status is OrderStatus.OPEN


def test_an_update_with_a_lower_filled_quantity_than_seen_is_stale():
    rig = sent_rig()
    rig.path.on_update(update("1", OrderStatus.OPEN, 6))

    assert rig.path.on_update(update("1", OrderStatus.OPEN, 4)) is None


@pytest.mark.parametrize(
    "after", [update("1", OrderStatus.OPEN, 10), update("1", OrderStatus.CANCELLED, 10)]
)
def test_any_change_after_a_terminal_status_is_stale(after):
    rig = sent_rig()
    rig.path.on_update(update("1", OrderStatus.COMPLETE, 10))

    assert rig.path.on_update(after) is None


def test_a_further_fill_with_the_same_status_is_new():
    rig = sent_rig()
    rig.path.on_update(update("1", OrderStatus.OPEN, 4))

    assert rig.path.on_update(update("1", OrderStatus.OPEN, 7)) is not None


def test_a_stale_update_is_neither_journaled_nor_checked():
    rig = sent_rig()
    first = update("1", OrderStatus.OPEN, 6)
    rig.path.on_update(first)

    rig.path.on_update(update("1", OrderStatus.OPEN, 4))

    assert rig.journal.updates == [first]
    assert rig.guard.updates == [first]


def test_a_stream_word_that_repeats_what_is_known_is_journaled_but_not_returned():
    rig = sent_rig()
    first = update("1", OrderStatus.OPEN, 4)
    rig.path.on_update(first)
    again = update("1", OrderStatus.OPEN, 4)

    assert rig.path.on_update(again) is None
    assert rig.journal.updates == [first, again]


def test_a_word_without_the_price_of_fills_known_at_a_price_is_neither_news_nor_journaled():
    rig = sent_rig()
    priced = update("1", OrderStatus.OPEN, 4, price=812.0)
    rig.path.on_update(priced)

    assert rig.path.on_update(update("1", OrderStatus.OPEN, 4, price=0.0)) is None
    assert rig.journal.updates == [priced]


def test_a_word_after_a_terminal_status_is_not_journaled():
    rig = sent_rig()
    done = update("1", OrderStatus.COMPLETE, 10)
    rig.path.on_update(done)

    rig.path.on_update(update("1", OrderStatus.COMPLETE, 10))

    assert rig.journal.updates == [done]


def test_an_update_of_an_order_the_session_never_sent_is_ignored():
    rig = sent_rig()
    manual = update("77", OrderStatus.COMPLETE, 10)

    assert rig.path.on_update(manual) is None
    assert rig.journal.updates == []
    assert rig.guard.updates == []


def test_restore_takes_back_the_journals_bound_orders_to_cancel_them():
    first = sent_rig()
    rig = make_rig(journal=first.journal, cancelorder=[placed("1")])

    rig.path.restore()

    assert rig.path.cancel("1").order_id == "1"
    assert rig.client.calls_to("cancelorder") == [{"order_id": "1", "strategy": STRATEGY}]
    assert rig.guard.sent == ["1"]


def test_restored_working_orders_are_polled_and_terminal_ones_are_not():
    first = make_rig(placeorder=[placed("1"), placed("2")])
    first.path.send(make_intent("i-1"))
    first.path.send(make_intent("i-2"))
    first.path.on_update(update("2", OrderStatus.COMPLETE, 10))
    rig = make_rig(journal=first.journal, orderstatus=[NOT_FOUND])
    rig.path.restore()
    rig.clock.advance(2)

    rig.path.due_polls()

    assert [kwargs["order_id"] for kwargs in rig.client.calls_to("orderstatus")] == ["1"]


def test_after_a_restart_an_order_the_stream_spoke_for_keeps_its_safety_net_poll():
    # A poll said the stop waits before the stream did; the stream's repeat is journaled,
    # so the restored order is known to the stream and is not polled every second.
    first = stop_rig(orderstatus=[status_answer("1", "trigger pending", **STOP_ROW)])
    first.clock.advance(2)
    first.path.due_polls()
    first.path.on_update(update("1", OrderStatus.TRIGGER_PENDING, 0, ts_recv_ns=first.clock.now_ns))
    rig = make_rig(journal=first.journal, clock=first.clock, orderstatus=[NOT_FOUND])
    rig.path.restore()

    rig.clock.advance(29)
    assert rig.path.due_polls() == []
    rig.clock.advance(1)
    rig.path.due_polls()

    assert len(rig.client.calls_to("orderstatus")) == 1
    assert rig.clock.now_ns - first.journal.updates[-1].ts_recv_ns == 30 * NS


def test_a_restored_order_takes_its_fills_from_the_tradebook_like_one_sent():
    first = sent_rig()
    rig = make_rig(journal=first.journal, tradebook=[trades(("1", 4, 812.0))])
    rig.path.restore()
    rig.path.on_update(update("1", OrderStatus.CANCELLED, 0))
    rig.clock.advance(1)

    (cancelled,) = rig.path.due_polls()

    assert (cancelled.filled_quantity, cancelled.average_price) == (4, 812.0)


def test_an_unanswered_cancel_changes_nothing_of_how_the_order_is_read():
    rig = stop_rig(cancelorder=[TIMEOUT])
    rig.path.on_update(update("1", OrderStatus.TRIGGER_PENDING, 0))
    rig.path.cancel("1")

    told = rig.path.on_update(update("1", OrderStatus.OPEN, 0))

    assert told.status is OrderStatus.OPEN
