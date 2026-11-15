"""Following orders once sent: news taken once and in order, a stop read as waiting for
its trigger until the stream says otherwise, updates of orders not bound kept aside, and
the orders taken back after a restart."""

import pytest
from orders_fakes import (
    STRATEGY,
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


def sent_rig(**answers):
    """A rig whose intent i-1, a market buy, was placed as order 1 at the clock's start."""
    rig = make_rig(placeorder=[placed("1")], **answers)
    rig.path.send(make_intent("i-1"))
    return rig


def stop_rig(**answers):
    """A rig whose stop i-stop, an SL-M sell, was placed as order 1."""
    rig = make_rig(placeorder=[placed("1")], **answers)
    stop = make_intent(
        "i-stop", purpose=Purpose.STOP, action="SELL", price_type="SL-M", trigger_price=806.5
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


def test_a_change_of_quantity_with_the_same_status_and_fills_is_news():
    rig = stop_rig()
    rig.path.on_update(update("1", OrderStatus.TRIGGER_PENDING, 0))

    modified = update("1", OrderStatus.TRIGGER_PENDING, 0, quantity=20)

    assert rig.path.on_update(modified) is modified


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


def test_a_dropped_update_is_neither_journaled_nor_checked():
    rig = sent_rig()
    first = update("1", OrderStatus.OPEN, 4)
    rig.path.on_update(first)

    rig.path.on_update(update("1", OrderStatus.OPEN, 4))

    assert rig.journal.updates == [first]
    assert rig.guard.updates == [first]


def test_a_stop_the_stream_reports_waiting_then_open_is_triggered():
    rig = stop_rig()
    rig.path.on_update(update("1", OrderStatus.TRIGGER_PENDING, 0))

    triggered = rig.path.on_update(update("1", OrderStatus.OPEN, 0))

    assert triggered.status is OrderStatus.OPEN


def test_a_stop_polled_open_before_the_stream_spoke_is_waiting_for_its_trigger():
    # Some brokers' books write a stop waiting for its trigger as open.
    rig = stop_rig(
        orderstatus=[status_answer("1", "open", pricetype="SL-M", action="SELL")],
        tradebook=[trades()],
    )
    rig.clock.advance(2)

    (first,) = rig.path.due_polls()

    assert first.status is OrderStatus.TRIGGER_PENDING


def test_the_streams_word_that_a_stop_waits_is_not_taken_as_going_back():
    rig = stop_rig(
        orderstatus=[status_answer("1", "open", pricetype="SL-M", action="SELL")],
        tradebook=[trades()],
    )
    rig.clock.advance(2)
    rig.path.due_polls()

    rig.path.on_update(update("1", OrderStatus.TRIGGER_PENDING, 0))
    triggered = rig.path.on_update(update("1", OrderStatus.OPEN, 0))

    assert rig.journal.updates[-1].status is OrderStatus.OPEN
    assert triggered is not None


def test_a_stop_the_stream_reports_open_with_no_word_of_waiting_still_waits():
    rig = stop_rig()

    first = rig.path.on_update(update("1", OrderStatus.OPEN, 0))

    assert first.status is OrderStatus.TRIGGER_PENDING


def test_a_poll_saying_open_for_a_stop_the_stream_said_waits_is_not_news():
    rig = stop_rig()
    rig.path.on_update(update("1", OrderStatus.TRIGGER_PENDING, 0))

    assert rig.path.on_update(update("1", OrderStatus.OPEN, 0, source="poll")) is None


def test_a_fill_of_a_stop_is_news_whatever_word_its_book_uses():
    rig = stop_rig()
    rig.path.on_update(update("1", OrderStatus.TRIGGER_PENDING, 0))

    filled = rig.path.on_update(update("1", OrderStatus.OPEN, 4))

    assert (filled.status, filled.filled_quantity) == (OrderStatus.OPEN, 4)


def test_a_status_never_moves_back_from_open_to_trigger_pending():
    rig = stop_rig()
    rig.path.on_update(update("1", OrderStatus.TRIGGER_PENDING, 0))
    rig.path.on_update(update("1", OrderStatus.OPEN, 0))

    assert rig.path.on_update(update("1", OrderStatus.TRIGGER_PENDING, 0)) is None


def test_a_modify_that_changes_the_price_type_changes_how_the_order_is_read():
    rig = stop_rig(modifyorder=[placed("1")])
    fields = {"symbol": "SBIN", "action": "SELL", "exchange": "NSE", "product": "MIS"}
    rig.path.modify("1", **fields, price_type="LIMIT", quantity=10, price=806.0, trigger_price=0)

    now_open = rig.path.on_update(update("1", OrderStatus.OPEN, 0))

    assert now_open.status is OrderStatus.OPEN


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


def test_a_restored_stop_the_stream_said_waits_is_triggered_by_the_streams_open():
    first = stop_rig()
    first.path.on_update(update("1", OrderStatus.TRIGGER_PENDING, 0))
    rig = make_rig(journal=first.journal)
    rig.path.restore()

    triggered = rig.path.on_update(update("1", OrderStatus.OPEN, 0))

    assert triggered.status is OrderStatus.OPEN


def test_a_restored_order_takes_its_fills_from_the_tradebook_like_one_sent():
    first = sent_rig()
    rig = make_rig(journal=first.journal, tradebook=[trades(("1", 4, 812.0))])
    rig.path.restore()
    rig.path.on_update(update("1", OrderStatus.CANCELLED, 0))
    rig.clock.advance(1)

    (cancelled,) = rig.path.due_polls()

    assert (cancelled.filled_quantity, cancelled.average_price) == (4, 812.0)
