"""Following orders once sent: updates from the stream deduplicated and kept in order, and
the polling fallback when no terminal update arrives (on a manual clock)."""

import pytest
from orders_fakes import (
    NS,
    STRATEGY,
    make_intent,
    make_rig,
    placed,
    status_answer,
    update,
)

from openalgo.backtest._live._types import OrderStatus
from openalgo.backtest.errors import LiveError


def sent_rig(**answers):
    """A rig whose intent i-1 was placed as order 1 at the start of the clock."""
    rig = make_rig(placeorder=[placed("1")], **answers)
    rig.path.send(make_intent("i-1"))
    return rig


def poll_times(rig, until_seconds, step_seconds=0.5):
    """Call `due_polls` every `step_seconds` until `until_seconds` after the start, and
    return the seconds at which `orderstatus` was called."""
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


def test_an_order_with_no_terminal_update_is_polled_from_two_seconds_then_every_second():
    rig = sent_rig(orderstatus=[status_answer("1", "trigger pending")] * 4)

    assert poll_times(rig, until_seconds=5) == [2.0, 3.0, 4.0, 5.0]


def test_an_order_is_not_polled_before_two_seconds():
    rig = sent_rig()

    assert poll_times(rig, until_seconds=1.5) == []


def test_a_terminal_stream_update_stops_the_polls():
    rig = sent_rig(orderstatus=[status_answer("1", "trigger pending")])
    rig.clock.advance(2)
    rig.path.due_polls()

    rig.path.on_update(update("1", OrderStatus.CANCELLED, 0))

    assert poll_times(rig, until_seconds=3) == []


def test_a_terminal_poll_stops_the_polls():
    rig = sent_rig(orderstatus=[status_answer("1", "complete", average_price=812.35)])

    assert poll_times(rig, until_seconds=5) == [2.0]


def test_a_poll_returns_the_new_update_with_source_poll():
    rig = sent_rig(orderstatus=[status_answer("1", "complete", average_price=812.35)])
    rig.clock.advance(2)

    (polled,) = rig.path.due_polls()

    assert (polled.order_id, polled.status, polled.filled_quantity, polled.source) == (
        "1",
        OrderStatus.COMPLETE,
        10,
        "poll",
    )
    assert rig.journal.updates == [polled]


def test_a_poll_that_repeats_what_the_stream_said_returns_nothing():
    rig = sent_rig(orderstatus=[status_answer("1", "trigger pending")])
    rig.path.on_update(update("1", OrderStatus.TRIGGER_PENDING, 0))
    rig.clock.advance(2)

    assert rig.path.due_polls() == []


def test_a_poll_asks_for_the_order_with_its_strategy():
    rig = sent_rig(orderstatus=[status_answer("1", "trigger pending")])
    rig.clock.advance(2)

    rig.path.due_polls()

    assert rig.client.calls_to("orderstatus") == [{"order_id": "1", "strategy": STRATEGY}]


def test_a_failed_poll_is_made_again_a_second_later():
    not_found = {"status": "error", "message": "HTTP 404: Order 1 not found", "code": 404}
    rig = sent_rig(orderstatus=[not_found, status_answer("1", "complete", average_price=812.35)])

    assert poll_times(rig, until_seconds=4) == [2.0, 3.0]


def test_a_cancelled_order_whose_status_omits_its_fills_reads_them_from_the_tradebook():
    trades = [
        {"orderid": "1", "quantity": 3, "average_price": 100.0},
        {"orderid": "7", "quantity": 50, "average_price": 90.0},
        {"orderid": "1", "quantity": "2", "average_price": "101.0"},
    ]
    rig = sent_rig(
        orderstatus=[status_answer("1", "cancelled")],
        tradebook=[{"status": "success", "data": trades}],
    )
    rig.clock.advance(2)

    (polled,) = rig.path.due_polls()

    assert (polled.status, polled.filled_quantity) == (OrderStatus.CANCELLED, 5)
    assert polled.average_price == pytest.approx((3 * 100.0 + 2 * 101.0) / 5)


def test_a_poll_with_an_unknown_status_raises_naming_it():
    rig = sent_rig(orderstatus=[status_answer("1", "after market order req received")])
    rig.clock.advance(2)

    with pytest.raises(LiveError, match="after market order req received"):
        rig.path.due_polls()
