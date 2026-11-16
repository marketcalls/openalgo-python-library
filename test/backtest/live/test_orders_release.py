"""Held words the session releases (16-session-design 4.11): a held fill carries its own
average price, and a held cancel is handed over with the units the session found it
filled before it ended, never more than the order's units."""

import math

import pytest
from orders_fakes import make_intent, make_rig, placed, status_answer, update

from openalgo.backtest._live._types import OrderStatus, Purpose
from openalgo.backtest.errors import LiveError

STOP = make_intent("i-1", purpose=Purpose.STOP, action="SELL", price_type="SL-M", trigger_price=805)


def stop_rig(**answers):
    """A rig whose SL-M stop of 10 SBIN was placed as order 1 and reported waiting."""
    rig = make_rig(placeorder=[placed("1")], **answers)
    rig.path.send(STOP)
    rig.path.on_update(update("1", OrderStatus.TRIGGER_PENDING, 0))
    return rig


def held_cancel(rig):
    """The stream reports the stop cancelled with no fill: held for the tradebook."""
    rig.path.on_update(update("1", OrderStatus.CANCELLED, 0))


# ----- a held fill's own price -------------------------------------------------------


def test_a_polled_fill_held_for_the_tradebook_carries_the_price_orderstatus_gave():
    row = {"action": "SELL", "pricetype": "SL-M", "trigger_price": 805.0, "average_price": 804.5}
    rig = stop_rig(orderstatus=[status_answer("1", "complete", **row)])

    rig.path.check_order("1")

    (held,) = rig.path.held_fills()
    assert (held.filled_quantity, held.average_price) == (10, 804.5)


def test_a_fill_held_for_want_of_a_price_carries_none():
    rig = stop_rig()
    rig.path.on_update(update("1", OrderStatus.COMPLETE, 10, price=0.0))

    (held,) = rig.path.held_fills()
    assert held.average_price == 0.0


# ----- a held cancel released with its fills -----------------------------------------


def test_release_cancel_hands_over_the_cancel_with_the_units_found_at_the_price_given():
    rig = stop_rig()
    held_cancel(rig)

    released = rig.path.release_cancel("1", 4, 806.0)

    assert (released.status, released.filled_quantity, released.pending_quantity) == (
        OrderStatus.CANCELLED,
        4,
        0,
    )
    assert released.average_price == 806.0
    assert rig.journal.updates[-1] == released
    assert rig.guard.updates[-1] == released
    assert rig.path.held_fills() == []


def test_release_cancel_with_no_units_hands_the_cancel_over_unfilled():
    rig = stop_rig()
    held_cancel(rig)

    released = rig.path.release_cancel("1", 0, 0.0)

    assert (released.status, released.filled_quantity, released.average_price) == (
        OrderStatus.CANCELLED,
        0,
        0.0,
    )


def test_release_cancel_refuses_more_than_the_orders_units():
    rig = stop_rig()
    held_cancel(rig)

    with pytest.raises(LiveError, match="up to its 10, not 11"):
        rig.path.release_cancel("1", 11, 806.0)
    assert [held.order_id for held in rig.path.held_fills()] == ["1"]
    assert rig.journal.updates[-1].status is OrderStatus.TRIGGER_PENDING


def test_release_cancel_refuses_fewer_units_than_already_reported_filled():
    rig = stop_rig()
    rig.path.on_update(update("1", OrderStatus.OPEN, 5))
    held_cancel(rig)

    with pytest.raises(LiveError, match="from the 5 reported filled"):
        rig.path.release_cancel("1", 3, 806.0)


@pytest.mark.parametrize(
    "units", [-1, 2.5, True, "4"], ids=["negative", "fraction", "bool", "text"]
)
def test_release_cancel_refuses_units_that_are_not_a_whole_number_in_range(units):
    rig = stop_rig()
    held_cancel(rig)

    with pytest.raises(LiveError, match="released with a whole number of units"):
        rig.path.release_cancel("1", units, 806.0)


@pytest.mark.parametrize("price", [0.0, -806.0, math.nan], ids=["zero", "negative", "nan"])
def test_release_cancel_refuses_a_fill_at_a_price_that_is_not_positive(price):
    rig = stop_rig()
    held_cancel(rig)

    with pytest.raises(LiveError, match="released at a positive price"):
        rig.path.release_cancel("1", 4, price)


def test_a_price_already_known_for_the_units_stands():
    rig = stop_rig()
    rig.path.on_update(update("1", OrderStatus.OPEN, 5, price=804.75))
    held_cancel(rig)

    released = rig.path.release_cancel("1", 5, 806.0)

    assert (released.filled_quantity, released.average_price) == (5, 804.75)


# ----- a fill handed over after the cancel was held ----------------------------------


def late_fill_rig():
    """The stop's cancel is held, then a late stream word hands over 3 units at 804."""
    rig = stop_rig()
    held_cancel(rig)
    rig.path.on_update(update("1", OrderStatus.OPEN, 3, price=804.0))
    return rig


def test_a_held_cancel_keeps_the_fills_handed_over_after_it_was_held():
    rig = late_fill_rig()

    assert [held.filled_quantity for held in rig.path.held_fills()] == [3]
    known = rig.path.last_known("1")
    assert (known.status, known.filled_quantity) == (OrderStatus.CANCELLED, 3)


@pytest.mark.parametrize("units", [0, 2], ids=["none", "fewer"])
def test_release_cancel_refuses_fewer_units_than_handed_over_after_the_cancel_was_held(units):
    rig = late_fill_rig()

    with pytest.raises(LiveError, match="from the 3 reported filled"):
        rig.path.release_cancel("1", units, 805.0)
    assert [held.order_id for held in rig.path.held_fills()] == ["1"]


def test_release_cancel_hands_over_a_cancel_held_before_a_late_fill_at_the_known_price():
    rig = late_fill_rig()

    released = rig.path.release_cancel("1", 3, 805.0)

    assert (released.status, released.filled_quantity, released.average_price) == (
        OrderStatus.CANCELLED,
        3,
        804.0,
    )
    assert rig.journal.updates[-1] == released


def test_release_fill_hands_over_a_cancel_held_before_a_late_fill_at_the_known_price():
    rig = late_fill_rig()

    released = rig.path.release_fill("1", 805.0)

    assert (released.status, released.filled_quantity, released.average_price) == (
        OrderStatus.CANCELLED,
        3,
        804.0,
    )
    assert rig.path.last_known("1") == released
    assert rig.path.held_fills() == []


def test_release_cancel_refuses_an_order_holding_nothing():
    rig = stop_rig()

    with pytest.raises(LiveError, match="order 1 holds no cancel to release"):
        rig.path.release_cancel("1", 0, 0.0)


def test_release_cancel_refuses_a_held_fill_and_points_at_release_fill():
    rig = stop_rig()
    rig.path.on_update(update("1", OrderStatus.COMPLETE, 10, price=0.0))

    with pytest.raises(LiveError, match="not a cancel: release it with release_fill"):
        rig.path.release_cancel("1", 10, 806.0)
