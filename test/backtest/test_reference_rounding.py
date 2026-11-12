"""Tick and quantity helpers the kernel uses (FR-EXEC-04)."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from openalgo.backtest.errors import ConfigError
from openalgo.backtest.reference.rounding import (
    check_freeze,
    is_tick_valid,
    max_order_quantity,
    round_to_lot,
    round_to_tick,
    split_by_freeze,
    units_to_lots,
)

TICKS = [0.01, 0.05, 0.10, 0.20, 0.25, 0.50, 1.0, 5.0, 0.0025]
LOTS = [1, 15, 20, 25, 30, 35, 40, 50, 65, 75, 100, 120, 140, 1250, 2500]

# ---- prices -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "price,tick,mode,expected",
    [
        (100.12, 0.05, "nearest", 100.10),
        (100.13, 0.05, "nearest", 100.15),
        (100.125, 0.05, "nearest", 100.15),  # exactly half way goes up
        (100.12, 0.05, "down", 100.10),
        (100.12, 0.05, "up", 100.15),
        (100.15, 0.05, "down", 100.15),  # already on a tick
        (100.15, 0.05, "up", 100.15),
        (24010.37, 0.10, "nearest", 24010.4),
        (51234.55, 0.20, "nearest", 51234.6),
        (51234.55, 0.20, "down", 51234.4),
        (99.994, 0.01, "nearest", 99.99),
        (99.995, 0.01, "nearest", 100.0),
        (75012.4, 1.0, "nearest", 75012.0),
        (22503.0, 5.0, "up", 22505.0),
        (83.12625, 0.0025, "nearest", 83.1275),
        (0.03, 0.05, "nearest", 0.05),
        (0.02, 0.05, "nearest", 0.0),
        (-37.63, 0.05, "down", -37.65),  # down is towards minus infinity
        (-37.63, 0.05, "up", -37.60),
    ],
)
def test_price_rounds_to_a_multiple_of_the_tick(price, tick, mode, expected):
    assert round_to_tick(price, tick, mode) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "noisy,mode",
    [
        (0.1 + 0.2 + 99.85, "up"),  # 100.15 plus binary noise
        (0.1 + 0.2 + 99.85, "down"),
        (100.15000000000002, "up"),
        (100.14999999999998, "down"),
    ],
)
def test_floating_point_noise_does_not_move_a_price_by_a_tick(noisy, mode):
    assert round_to_tick(noisy, 0.05, mode) == 100.15


@pytest.mark.parametrize(
    "price,tick,valid",
    [
        (100.05, 0.05, True),
        (100.06, 0.05, False),
        (100.06, 0.01, True),
        (24010.4, 0.10, True),
        (24010.45, 0.10, False),
        (0.1 + 0.2 + 99.85, 0.05, True),  # noise is not an off-tick price
    ],
)
def test_tick_validity(price, tick, valid):
    assert is_tick_valid(price, tick) is valid


@pytest.mark.parametrize(
    "price,tick,mode",
    [
        (100.0, 0, "nearest"),
        (100.0, -0.05, "nearest"),
        (float("nan"), 0.05, "nearest"),
        (float("inf"), 0.05, "nearest"),
        ("abc", 0.05, "nearest"),
        (True, 0.05, "nearest"),
        (100.0, 0.05, "sideways"),
    ],
)
def test_round_to_tick_rejects_bad_input(price, tick, mode):
    with pytest.raises(ConfigError):
        round_to_tick(price, tick, mode)


@settings(max_examples=400, deadline=None)
@given(
    price=st.floats(min_value=0.01, max_value=500000.0, allow_nan=False, allow_infinity=False),
    tick=st.sampled_from(TICKS),
)
def test_rounded_prices_bracket_the_price_within_one_tick(price, tick):
    slack = 1e-9
    down = round_to_tick(price, tick, "down")
    up = round_to_tick(price, tick, "up")
    nearest = round_to_tick(price, tick, "nearest")
    assert down <= price + slack and price - slack <= up
    assert up - down <= tick + slack
    assert abs(nearest - price) <= tick / 2 + slack
    assert nearest in (down, up)


@settings(max_examples=200, deadline=None)
@given(steps=st.integers(min_value=0, max_value=10_000_000), tick=st.sampled_from(TICKS))
def test_a_price_on_a_tick_is_left_where_it_is(steps, tick):
    price = steps * tick  # a whole number of ticks, up to binary noise
    for mode in ("nearest", "down", "up"):
        assert round_to_tick(price, tick, mode) == pytest.approx(price, abs=tick * 1e-6)


# ---- quantities ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "quantity,lot,mode,expected",
    [
        (160, 75, "down", 150),
        (160, 75, "up", 225),
        (160, 75, "nearest", 150),
        (190, 75, "nearest", 225),
        (187.5, 75, "nearest", 225),  # exactly half a lot goes away from zero
        (74, 75, "down", 0),
        (75, 75, "down", 75),
        (1, 75, "up", 75),
        (0, 75, "down", 0),
        (-160, 75, "down", -150),  # down is towards zero
        (-160, 75, "up", -225),
        (12.7, 1, "down", 12),
        (149.99999999999997, 75, "down", 150),  # noise, not a missing lot
    ],
)
def test_quantity_rounds_to_a_whole_number_of_lots_in_units(quantity, lot, mode, expected):
    result = round_to_lot(quantity, lot, mode)
    assert result == expected
    assert isinstance(result, int)


@settings(max_examples=300, deadline=None)
@given(
    quantity=st.floats(min_value=0, max_value=5_000_000, allow_nan=False), lot=st.sampled_from(LOTS)
)
def test_rounded_quantities_bracket_the_quantity_within_one_lot(quantity, lot):
    down = round_to_lot(quantity, lot, "down")
    up = round_to_lot(quantity, lot, "up")
    assert down % lot == 0 and up % lot == 0
    slack = lot * 1e-6  # a quantity within a millionth of a lot of a multiple is that multiple
    assert down <= quantity + slack and quantity - slack <= up
    assert up - down in (0, lot)


@pytest.mark.parametrize("lot", [0, -1, 1.5, True, "75"])
def test_lot_size_must_be_a_positive_whole_number(lot):
    with pytest.raises(ConfigError):
        round_to_lot(100, lot)


def test_units_convert_to_lots():
    assert units_to_lots(225, 75) == 3


def test_units_that_are_not_whole_lots_are_refused_with_a_reminder_of_the_unit():
    with pytest.raises(ConfigError, match="units, not lots"):
        units_to_lots(2, 75)


# ---- freeze quantity ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "quantity,freeze_qty,allowed",
    [
        (1800, 1800, True),  # the limit itself passes
        (1801, 1800, False),
        (-1800, 1800, True),  # a sell order is checked on its size
        (-1875, 1800, False),
        (10**9, None, True),  # no limit applies
    ],
)
def test_order_is_within_the_freeze_limit_up_to_and_including_it(quantity, freeze_qty, allowed):
    assert check_freeze(quantity, freeze_qty) is allowed


@pytest.mark.parametrize("freeze_qty", [0, -5, 1800.5, True])
def test_freeze_quantity_must_be_a_positive_whole_number(freeze_qty):
    with pytest.raises(ConfigError):
        check_freeze(100, freeze_qty)


@pytest.mark.parametrize(
    "freeze_qty,lot,largest",
    [
        (1800, 75, 1800),  # 24 lots
        (1800, 65, 1755),  # 27 lots
        (2800, 120, 2760),  # 23 lots
        (600, 25, 600),
        (None, 75, None),
    ],
)
def test_largest_single_order_is_the_biggest_lot_multiple_within_the_limit(
    freeze_qty, lot, largest
):
    assert max_order_quantity(freeze_qty, lot) == largest


def test_freeze_limit_smaller_than_one_lot_is_inconsistent_data():
    with pytest.raises(ConfigError, match="inconsistent"):
        max_order_quantity(50, 75)


@pytest.mark.parametrize(
    "quantity,freeze_qty,lot,slices",
    [
        (5250, 1800, 75, [1800, 1800, 1650]),
        (1800, 1800, 75, [1800]),
        (75, 1800, 75, [75]),
        (3900, 1800, 65, [1755, 1755, 390]),
        (750_000, None, 75, [750_000]),
    ],
)
def test_order_above_the_freeze_limit_is_sliced_into_orders_within_it(
    quantity, freeze_qty, lot, slices
):
    assert split_by_freeze(quantity, freeze_qty, lot) == slices


@pytest.mark.parametrize("quantity", [5000, 0, -75, 150.0])
def test_only_a_positive_whole_number_of_lots_can_be_sliced(quantity):
    with pytest.raises(ConfigError):
        split_by_freeze(quantity, 1800, 75)


@settings(max_examples=200, deadline=None)
@given(
    lots=st.integers(min_value=1, max_value=5000),
    lot=st.sampled_from([15, 25, 30, 65, 75, 120]),
    freeze=st.sampled_from([600, 900, 1200, 1800, 2800]),
)
def test_slices_add_up_and_each_is_a_lot_multiple_within_the_limit(lots, lot, freeze):
    quantity = lots * lot
    slices = split_by_freeze(quantity, freeze, lot)
    lots_per_slice = freeze // lot
    assert sum(slices) == quantity
    assert all(piece % lot == 0 and 0 < piece <= freeze for piece in slices)
    assert len(slices) == -(-lots // lots_per_slice)  # as few orders as the limit allows


# ---- numpy integers and the half-way rule -------------------------------------------------


def test_numpy_integer_lot_and_quantity_are_accepted():
    import numpy as np

    assert round_to_lot(np.int64(160), np.int64(75)) == 150
    assert split_by_freeze(np.int64(3600), 1800, np.int64(75)) == [1800, 1800]
    assert check_freeze(np.int64(1800), np.int64(1800))


@pytest.mark.parametrize(
    "price,tick,expected",
    [
        (100.025, 0.05, 100.05),
        (-100.025, 0.05, -100.05),  # half way rounds away from zero on both sides
        (0.025, 0.05, 0.05),
        (-0.025, 0.05, -0.05),
    ],
)
def test_price_exactly_half_way_between_ticks_rounds_away_from_zero(price, tick, expected):
    assert round_to_tick(price, tick) == expected
