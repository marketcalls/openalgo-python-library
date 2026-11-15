"""Fills handed to the engine with their price and full count: a polled fill priced from
the tradebook, a cancel's fills confirmed by it (the brokers the server polls itself say
0 filled), a fill with no price held, alerted after 10 seconds, and released at a price."""

import logging

import pytest
from orders_fakes import (
    NS,
    make_intent,
    make_rig,
    placed,
    status_answer,
    trades,
    update,
)

from openalgo.backtest._live._types import AlertKind, OrderStatus
from openalgo.backtest._live.orders import UNSUPPORTED_BROKERS, HeldFill
from openalgo.backtest.errors import LiveError

BROKER_BUSY = {"status": "error", "message": "broker busy"}


def sent_rig(**answers):
    """A rig whose intent i-1 was placed as order 1, a LIMIT buy of 10, at the start."""
    rig = make_rig(placeorder=[placed("1")], **answers)
    rig.path.send(make_intent("i-1", price_type="LIMIT", price=812.0))
    return rig


def next_round(rig, seconds=1):
    rig.clock.advance(seconds)
    return rig.path.due_polls()


def test_a_stream_fill_with_its_own_price_is_taken_as_given():
    rig = sent_rig()
    done = update("1", OrderStatus.COMPLETE, 10, price=812.35)

    assert rig.path.on_update(done) is done
    assert rig.client.calls_to("tradebook") == []


def test_a_cancel_reported_with_no_fill_waits_for_the_tradebook():
    rig = sent_rig()
    rig.path.on_update(update("1", OrderStatus.OPEN, 0))

    assert rig.path.on_update(update("1", OrderStatus.CANCELLED, 0)) is None
    assert rig.journal.updates[-1].status is OrderStatus.OPEN


def test_a_cancel_reported_with_no_fill_takes_the_fills_the_tradebook_shows():
    rig = sent_rig(tradebook=[trades(("1", 3, 812.0), ("1", 1, 811.0))])
    rig.path.on_update(update("1", OrderStatus.OPEN, 0))
    rig.path.on_update(update("1", OrderStatus.CANCELLED, 0))

    (cancelled,) = next_round(rig)

    assert (cancelled.status, cancelled.filled_quantity) == (OrderStatus.CANCELLED, 4)
    assert cancelled.average_price == pytest.approx((3 * 812.0 + 811.0) / 4)
    assert rig.journal.updates[-1] == cancelled


def test_a_cancel_the_tradebook_shows_no_trade_for_is_handed_over_unfilled():
    rig = sent_rig(tradebook=[trades(("9", 50, 90.0))])
    rig.path.on_update(update("1", OrderStatus.CANCELLED, 0))

    (cancelled,) = next_round(rig)

    assert (cancelled.status, cancelled.filled_quantity) == (OrderStatus.CANCELLED, 0)


def test_a_cancel_waits_while_the_tradebook_cannot_be_read():
    rig = sent_rig(tradebook=[BROKER_BUSY, trades(("1", 4, 812.0))])
    rig.path.on_update(update("1", OrderStatus.CANCELLED, 0))

    assert next_round(rig) == []
    (cancelled,) = next_round(rig)
    assert cancelled.filled_quantity == 4


def test_a_terminal_update_with_fewer_fills_than_known_keeps_them_for_the_tradebook():
    rig = sent_rig(tradebook=[trades(("1", 5, 812.0))])
    rig.path.on_update(update("1", OrderStatus.OPEN, 5, price=0.0))
    rig.path.on_update(update("1", OrderStatus.CANCELLED, 0))

    (cancelled,) = next_round(rig)

    assert (cancelled.status, cancelled.filled_quantity, cancelled.average_price) == (
        OrderStatus.CANCELLED,
        5,
        812.0,
    )


def test_a_cancel_waits_while_the_tradebook_has_fewer_fills_than_the_stream_reported():
    rig = sent_rig(tradebook=[trades(("1", 3, 812.0)), trades(("1", 5, 812.0))])
    rig.path.on_update(update("1", OrderStatus.OPEN, 5, price=812.0))
    rig.path.on_update(update("1", OrderStatus.CANCELLED, 0))

    assert next_round(rig) == []
    (cancelled,) = next_round(rig)
    assert cancelled.filled_quantity == 5


def test_a_polled_complete_fill_is_priced_at_the_tradebooks_average_not_its_first_trade():
    # orderstatus prices a complete order at its first trade
    # (services/orderstatus_service.py).
    rig = sent_rig(
        orderstatus=[status_answer("1", "complete", average_price=812.0)],
        tradebook=[trades(("1", 6, 812.0), ("1", 4, 815.0))],
    )

    (polled,) = next_round(rig, 2)

    assert polled.average_price == pytest.approx((6 * 812.0 + 4 * 815.0) / 10)


def test_a_polled_partial_fill_is_priced_from_the_tradebook():
    rig = sent_rig(
        orderstatus=[status_answer("1", "open", filled_quantity=4, average_price=812.0)],
        tradebook=[trades(("1", 2, 812.0), ("1", 2, 813.0))],
    )

    (polled,) = next_round(rig, 2)

    assert (polled.filled_quantity, polled.average_price) == (4, 812.5)


def test_a_cancelled_order_whose_status_omits_its_fills_reads_them_from_the_tradebook():
    rig = sent_rig(
        orderstatus=[status_answer("1", "cancelled")],
        tradebook=[trades(("1", 3, 100.0), ("7", 50, 90.0), ("1", "2", "101.0"))],
    )

    (polled,) = next_round(rig, 2)

    assert (polled.status, polled.filled_quantity) == (OrderStatus.CANCELLED, 5)
    assert polled.average_price == pytest.approx((3 * 100.0 + 2 * 101.0) / 5)


def test_a_fill_with_no_price_is_held_not_returned_nor_journaled():
    rig = sent_rig()

    assert rig.path.on_update(update("1", OrderStatus.COMPLETE, 10, price=0.0)) is None
    assert rig.journal.updates == []


def test_a_priced_update_replaces_a_held_priceless_one():
    rig = sent_rig()
    rig.path.on_update(update("1", OrderStatus.COMPLETE, 10, price=0.0))

    priced = update("1", OrderStatus.COMPLETE, 10, price=812.35)

    assert rig.path.on_update(priced) is priced
    assert rig.path.held_fills() == []


def test_a_held_fill_is_priced_from_the_tradebook_in_the_next_round():
    rig = sent_rig(tradebook=[trades(("1", 6, 812.0), ("1", 4, 813.0), ("9", 50, 90.0))])
    rig.path.on_update(update("1", OrderStatus.COMPLETE, 10, price=0.0))

    (priced,) = next_round(rig)

    assert (priced.status, priced.filled_quantity) == (OrderStatus.COMPLETE, 10)
    assert priced.average_price == pytest.approx((6 * 812.0 + 4 * 813.0) / 10)


def test_a_held_fill_stays_held_while_the_tradebook_does_not_cover_it():
    rig = sent_rig(tradebook=[trades(("1", 6, 812.0))])
    rig.path.on_update(update("1", OrderStatus.COMPLETE, 10, price=0.0))

    assert next_round(rig) == []
    assert [held.filled_quantity for held in rig.path.held_fills()] == [10]


def test_a_held_fills_tradebook_retries_back_off_to_five_seconds():
    rig = sent_rig(tradebook=[trades(("1", 6, 812.0))] * 10)
    rig.path.on_update(update("1", OrderStatus.COMPLETE, 10, price=0.0))
    read_at = []
    rig.client.before_call = lambda name, kwargs: read_at.append(rig.clock.now_ns // NS)
    start_s = rig.clock.now_ns // NS

    for _ in range(20):
        next_round(rig)

    assert [second - start_s for second in read_at] == [1, 2, 4, 9, 14, 19]


def test_fivepaisa_is_refused_since_its_tradebook_names_trades_by_another_order_id():
    assert "ExchOrderID" in UNSUPPORTED_BROKERS["fivepaisa"]
    with pytest.raises(TypeError):
        UNSUPPORTED_BROKERS["groww"] = "writable"  # type: ignore[index]


def test_held_fills_lists_each_held_fill_with_its_units_and_since_when():
    rig = sent_rig()
    rig.clock.advance(0.5)
    rig.path.on_update(update("1", OrderStatus.OPEN, 4, price=0.0))

    assert rig.path.held_fills() == [HeldFill("1", 4, rig.clock.now_ns)]


def test_a_fill_held_past_ten_seconds_sends_one_error_alert_naming_the_order_and_units():
    rig = sent_rig(tradebook=[trades()] * 20)
    rig.path.on_update(update("1", OrderStatus.COMPLETE, 10, price=0.0))
    for _ in range(15):
        next_round(rig)

    ((kind, text),) = rig.alerts.sent
    assert kind is AlertKind.ERROR
    assert "order 1" in text
    assert "10 units" in text


def test_a_fill_held_under_ten_seconds_is_not_alerted():
    rig = sent_rig(tradebook=[trades()] * 9)
    rig.path.on_update(update("1", OrderStatus.COMPLETE, 10, price=0.0))
    for _ in range(9):
        next_round(rig)

    assert rig.alerts.sent == []


def test_a_held_fill_is_logged_at_debug_not_info(caplog):
    rig = sent_rig()

    with caplog.at_level(logging.DEBUG, logger="openalgo.backtest.live"):
        rig.path.on_update(update("1", OrderStatus.COMPLETE, 10, price=0.0))

    assert [record.levelno for record in caplog.records] == [logging.DEBUG]


def test_release_fill_hands_the_held_fill_over_at_the_price_given():
    rig = sent_rig()
    rig.path.on_update(update("1", OrderStatus.COMPLETE, 10, price=0.0))

    released = rig.path.release_fill("1", 812.5)

    assert (released.status, released.filled_quantity, released.average_price) == (
        OrderStatus.COMPLETE,
        10,
        812.5,
    )
    assert rig.journal.updates == [released]
    assert rig.path.held_fills() == []


@pytest.mark.parametrize(
    ("held", "price"),
    [(False, 812.5), (True, 0.0), (True, float("nan"))],
    ids=["nothing held", "price 0", "price nan"],
)
def test_release_fill_raises_without_a_held_fill_or_a_positive_price(held, price):
    rig = sent_rig()
    if held:
        rig.path.on_update(update("1", OrderStatus.COMPLETE, 10, price=0.0))

    with pytest.raises(LiveError):
        rig.path.release_fill("1", price)


def test_a_held_fill_is_dated_from_when_it_was_first_held():
    rig = sent_rig()
    rig.path.on_update(update("1", OrderStatus.OPEN, 4, price=0.0))
    first_held_ns = rig.clock.now_ns
    rig.clock.advance(3)

    rig.path.on_update(update("1", OrderStatus.OPEN, 6, price=0.0))

    assert rig.path.held_fills() == [HeldFill("1", 6, first_held_ns)]
    assert rig.clock.now_ns - first_held_ns == 3 * NS
