"""Following orders once sent: news from the stream taken once and in order, and the
polling fallback (on a manual clock)."""

import pytest
from orders_fakes import (
    NS,
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

from openalgo.backtest._live._types import OrderStatus

NOT_FOUND = http_error(404, server_error("Order 1 not found"))


def sent_rig(**answers):
    """A rig whose intent i-1 was placed as order 1 at the start of the clock."""
    rig = make_rig(placeorder=[placed("1")], **answers)
    rig.path.send(make_intent("i-1"))
    return rig


def poll_times(rig, until_seconds, step_seconds=0.5):
    """Call `due_polls` every `step_seconds` until `until_seconds` after now, and return
    the seconds at which `orderstatus` was called."""
    start = rig.clock.now_ns
    called_at = []

    def note(name, kwargs):
        if name == "orderstatus":
            called_at.append((rig.clock.now_ns - start) / NS)

    rig.client.before_call = note
    while rig.clock.now_ns - start < until_seconds * NS:
        rig.clock.advance(step_seconds)
        rig.path.due_polls()
    return called_at


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
    rig = sent_rig()
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


def test_a_stop_the_stream_reports_triggered_is_news():
    rig = sent_rig()
    rig.path.on_update(update("1", OrderStatus.TRIGGER_PENDING, 0))

    assert rig.path.on_update(update("1", OrderStatus.OPEN, 0)) is not None


def test_a_status_never_moves_back_from_open_to_trigger_pending():
    rig = sent_rig()
    rig.path.on_update(update("1", OrderStatus.OPEN, 0))

    assert rig.path.on_update(update("1", OrderStatus.TRIGGER_PENDING, 0)) is None


def test_a_poll_saying_open_for_a_stop_the_stream_said_waits_for_its_trigger_is_not_news():
    # Some brokers' books write a trigger-pending stop as open.
    rig = sent_rig()
    rig.path.on_update(update("1", OrderStatus.TRIGGER_PENDING, 0))

    assert rig.path.on_update(update("1", OrderStatus.OPEN, 0, source="poll")) is None


def test_a_dropped_update_is_neither_journaled_nor_checked():
    rig = sent_rig()
    first = update("1", OrderStatus.OPEN, 4)
    rig.path.on_update(first)

    rig.path.on_update(update("1", OrderStatus.OPEN, 4))

    assert rig.journal.updates == [first]
    assert rig.guard.updates == [first]


def test_a_fill_with_no_price_is_held_not_returned_nor_journaled():
    rig = sent_rig()

    assert rig.path.on_update(update("1", OrderStatus.COMPLETE, 10, price=0.0)) is None
    assert rig.journal.updates == []


def test_a_priced_update_replaces_a_held_priceless_one():
    rig = sent_rig()
    rig.path.on_update(update("1", OrderStatus.COMPLETE, 10, price=0.0))

    priced = update("1", OrderStatus.COMPLETE, 10, price=812.35)

    assert rig.path.on_update(priced) is priced
    assert rig.journal.updates == [priced]


def test_a_held_fill_is_priced_from_the_tradebook_in_the_next_round():
    rig = sent_rig(tradebook=[trades(("1", 6, 812.0), ("1", 4, 813.0), ("9", 50, 90.0))])
    rig.path.on_update(update("1", OrderStatus.COMPLETE, 10, price=0.0))
    rig.clock.advance(1)

    (priced,) = rig.path.due_polls()

    assert (priced.status, priced.filled_quantity) == (OrderStatus.COMPLETE, 10)
    assert priced.average_price == pytest.approx((6 * 812.0 + 4 * 813.0) / 10)


def test_a_held_fill_stays_held_while_the_tradebook_does_not_cover_it():
    rig = sent_rig(tradebook=[trades(("1", 6, 812.0))])
    rig.path.on_update(update("1", OrderStatus.COMPLETE, 10, price=0.0))
    rig.clock.advance(1)

    assert rig.path.due_polls() == []
    assert rig.path.on_update(update("1", OrderStatus.COMPLETE, 10)) is not None


def test_a_priceless_update_with_fills_already_priced_takes_their_price():
    rig = sent_rig()
    rig.path.on_update(update("1", OrderStatus.OPEN, 4, price=812.0))

    cancelled = rig.path.on_update(update("1", OrderStatus.CANCELLED, 4, price=0.0))

    assert (cancelled.status, cancelled.average_price) == (OrderStatus.CANCELLED, 812.0)


def test_a_complete_poll_without_a_price_is_priced_from_the_tradebook():
    rig = sent_rig(
        orderstatus=[status_answer("1", "complete")], tradebook=[trades(("1", 10, 812.35))]
    )
    rig.clock.advance(2)

    (polled,) = rig.path.due_polls()

    assert (polled.status, polled.average_price, polled.source) == (
        OrderStatus.COMPLETE,
        812.35,
        "poll",
    )


def test_an_order_with_no_word_is_polled_from_two_seconds_then_every_second():
    rig = sent_rig(orderstatus=[NOT_FOUND] * 4)

    assert poll_times(rig, until_seconds=5) == [2.0, 3.0, 4.0, 5.0]


def test_an_order_is_not_polled_before_two_seconds():
    rig = sent_rig()

    assert poll_times(rig, until_seconds=1.5) == []


def test_a_stream_update_stops_the_fast_polls_and_leaves_a_thirty_second_safety_net():
    rig = sent_rig(orderstatus=[status_answer("1", "trigger pending")])
    rig.clock.advance(0.5)
    rig.path.on_update(update("1", OrderStatus.TRIGGER_PENDING, 0, ts_recv_ns=rig.clock.now_ns))

    assert poll_times(rig, until_seconds=59) == [30.0]


def test_a_resting_order_acknowledged_by_a_poll_is_polled_every_thirty_seconds():
    rig = sent_rig(orderstatus=[status_answer("1", "trigger pending")] * 3)

    assert poll_times(rig, until_seconds=63) == [2.0, 32.0, 62.0]


def test_a_terminal_stream_update_stops_the_polls():
    rig = sent_rig()
    rig.path.on_update(update("1", OrderStatus.CANCELLED, 0))

    assert poll_times(rig, until_seconds=40) == []


def test_a_terminal_poll_stops_the_polls():
    rig = sent_rig(orderstatus=[status_answer("1", "complete", average_price=812.35)])

    assert poll_times(rig, until_seconds=40) == [2.0]


def test_a_safety_poll_that_repeats_what_the_stream_said_returns_nothing():
    rig = sent_rig(orderstatus=[status_answer("1", "trigger pending")])
    rig.path.on_update(update("1", OrderStatus.TRIGGER_PENDING, 0))
    rig.clock.advance(30)

    assert rig.path.due_polls() == []
    assert len(rig.client.calls_to("orderstatus")) == 1


def test_a_poll_asks_for_the_order_with_its_strategy():
    rig = sent_rig(orderstatus=[status_answer("1", "trigger pending")])
    rig.clock.advance(2)

    rig.path.due_polls()

    assert rig.client.calls_to("orderstatus") == [{"order_id": "1", "strategy": STRATEGY}]


def test_a_failed_poll_is_made_again_a_second_later():
    rig = sent_rig(orderstatus=[NOT_FOUND, status_answer("1", "complete", average_price=812.35)])

    assert poll_times(rig, until_seconds=4) == [2.0, 3.0]


def test_a_cancelled_order_whose_status_omits_its_fills_reads_them_from_the_tradebook():
    rig = sent_rig(
        orderstatus=[status_answer("1", "cancelled")],
        tradebook=[trades(("1", 3, 100.0), ("7", 50, 90.0), ("1", "2", "101.0"))],
    )
    rig.clock.advance(2)

    (polled,) = rig.path.due_polls()

    assert (polled.status, polled.filled_quantity) == (OrderStatus.CANCELLED, 5)
    assert polled.average_price == pytest.approx((3 * 100.0 + 2 * 101.0) / 5)


def test_rounds_are_at_most_one_a_second_however_often_they_are_asked_for():
    rig = sent_rig(orderstatus=[NOT_FOUND] * 100)

    times = poll_times(rig, until_seconds=10, step_seconds=0.1)

    assert len(times) == 9


def test_a_round_reads_the_tradebook_once_for_all_its_orders():
    rig = make_rig(
        placeorder=[placed("1"), placed("2")],
        orderstatus=[status_answer("1", "cancelled"), status_answer("2", "cancelled")],
        tradebook=[trades(("1", 4, 812.0), ("2", 5, 813.0))],
    )
    rig.path.send(make_intent("i-1"))
    rig.path.send(make_intent("i-2"))
    rig.clock.advance(2)

    polled = rig.path.due_polls()

    assert [(u.order_id, u.filled_quantity) for u in polled] == [("1", 4), ("2", 5)]
    assert len(rig.client.calls_to("tradebook")) == 1


def test_a_round_polls_at_most_three_orders_the_most_overdue_first():
    ids = [str(n) for n in range(1, 6)]
    rig = make_rig(placeorder=[placed(order_id) for order_id in ids], orderstatus=[NOT_FOUND] * 10)
    for order_id in ids:
        rig.path.send(make_intent(f"i-{order_id}"))
        rig.clock.advance(0.1)
    rig.clock.advance(2)

    rig.path.due_polls()
    rig.clock.advance(1)
    rig.path.due_polls()

    polled = [kwargs["order_id"] for kwargs in rig.client.calls_to("orderstatus")]
    assert polled == ["1", "2", "3", "4", "5", "1"]


def test_a_poll_with_an_unknown_status_is_logged_and_the_round_goes_on(caplog):
    rig = make_rig(
        placeorder=[placed("1"), placed("2")],
        orderstatus=[
            status_answer("1", "after market order req received"),
            status_answer("2", "complete", average_price=812.35),
        ],
    )
    rig.path.send(make_intent("i-1"))
    rig.path.send(make_intent("i-2"))
    rig.clock.advance(2)

    polled = rig.path.due_polls()

    assert [u.order_id for u in polled] == ["2"]
    assert "after market order req received" in caplog.text


def test_polls_stop_while_the_guard_is_locked():
    rig = sent_rig(orderstatus=[NOT_FOUND])
    rig.guard.locked = True

    assert poll_times(rig, until_seconds=5) == []


def test_a_poll_answered_in_the_other_mode_is_not_taken():
    sandbox = {**status_answer("1", "complete", average_price=812.35), "mode": "analyze"}
    rig = sent_rig(orderstatus=[sandbox])
    rig.clock.advance(2)

    assert rig.path.due_polls() == []
    assert rig.guard.locked
    assert rig.journal.updates == []


def test_a_tradebook_answered_in_the_other_mode_prices_nothing():
    sandbox = {**trades(("1", 10, 812.35)), "mode": "analyze"}
    rig = sent_rig(orderstatus=[status_answer("1", "complete")], tradebook=[sandbox])
    rig.clock.advance(2)

    assert rig.path.due_polls() == []
    assert rig.guard.responses[-1] is sandbox


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
