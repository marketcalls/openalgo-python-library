"""The session invariants' checks at every change of an order (`session_invariants`, 16-session-
design 7.1, unit 14), on order changes written by hand: each check finds what it must and
nothing more. The session's client is "s", another client of the account "o"; SBIN's open
side is BUY.
"""

from __future__ import annotations

import pytest
from session_invariants import _one_stop_per_leg, _within_the_position
from session_trace import OrderChange

from openalgo.backtest._live._types import Mode

MINE = frozenset({"s"})
SBIN = (Mode.LIVE, "SBIN", "NSE")
SIDES = {SBIN: "BUY"}


def change(
    seq: int,
    order_id: str,
    action: str,
    price_type: str,
    status: str,
    quantity: int,
    *,
    filled: int = 0,
    fill: int = 0,
    position: int,
    client: str = "s",
) -> OrderChange:
    """An order of SBIN as it stood after a change, the position after it."""
    return OrderChange(
        seq, seq, Mode.LIVE, order_id, client, "StopHold", "SBIN", "NSE", action, price_type,
        status, quantity, filled, fill, 500.0 if fill else 0.0, position,
    )  # fmt: skip


#: The entry of 10 filled, then its stop working.
ENTERED = [
    change(0, "e", "BUY", "MARKET", "complete", 10, filled=10, fill=10, position=10),
    change(1, "s1", "SELL", "SL-M", "trigger pending", 10, position=10),
]
#: Another client sells 4 of the 10 by hand while the stop of 10 works.
SOLD_4 = change(2, "h", "SELL", "MARKET", "complete", 4, filled=4, fill=4, position=6, client="o")


def test_a_second_stop_working_beside_the_first_is_found():
    second = change(2, "s2", "SELL", "SL-M", "trigger pending", 10, position=10)
    with pytest.raises(AssertionError, match="two stops"):
        _one_stop_per_leg([*ENTERED, second], MINE)


def test_a_stop_placed_again_once_the_first_ended_is_one_stop():
    ended = change(2, "s1", "SELL", "SL-M", "cancelled", 10, position=10)
    again = change(3, "s2", "SELL", "SL-M", "trigger pending", 10, position=10)
    _one_stop_per_leg([*ENTERED, ended, again], MINE)


def test_an_exit_placed_while_the_stop_still_works_is_found():
    exit_ = change(2, "x", "SELL", "MARKET", "open", 10, position=10)
    with pytest.raises(AssertionError, match="working exits"):
        _within_the_position([*ENTERED, exit_], MINE, SIDES)


def test_an_exit_placed_once_the_stop_is_cancelled_is_within_the_position():
    cancelled = change(2, "s1", "SELL", "SL-M", "cancelled", 10, position=10)
    exit_ = change(3, "x", "SELL", "MARKET", "open", 10, position=10)
    assert _within_the_position([*ENTERED, cancelled, exit_], MINE, SIDES) == {}


def test_a_fill_of_the_sessions_order_through_zero_is_found():
    cut = change(2, "s1", "SELL", "SL-M", "complete", 10, filled=10, fill=10, position=0)
    oversold = change(3, "x", "SELL", "MARKET", "complete", 4, filled=4, fill=4, position=-4)
    with pytest.raises(AssertionError, match="other side"):
        _within_the_position([*ENTERED, cut, oversold], MINE, SIDES)


def test_a_sale_by_hand_against_the_stop_leaves_its_units_as_the_operators_excess():
    assert _within_the_position([*ENTERED, SOLD_4], MINE, SIDES) == {SBIN: 4}


def test_the_stop_filling_through_zero_after_a_sale_by_hand_is_found():
    # Finding 1 of unit 14 round 1: the sale by hand excuses none of the session's fills.
    filled = change(3, "s1", "SELL", "SL-M", "complete", 10, filled=10, fill=10, position=-4)
    with pytest.raises(AssertionError, match="other side"):
        _within_the_position([*ENTERED, SOLD_4, filled], MINE, SIDES)


def test_an_exit_the_session_adds_beside_a_stop_a_sale_by_hand_left_too_large_is_found():
    trailed = change(3, "s1", "SELL", "SL-M", "trigger pending", 10, position=6)
    exit_ = change(4, "x", "SELL", "MARKET", "open", 6, position=6)
    _within_the_position([*ENTERED, SOLD_4, trailed], MINE, SIDES)
    with pytest.raises(AssertionError, match="working exits"):
        _within_the_position([*ENTERED, SOLD_4, trailed, exit_], MINE, SIDES)


def test_once_the_session_fits_the_stop_to_a_sale_by_hand_the_excess_is_gone():
    fitted = change(3, "s1", "SELL", "SL-M", "trigger pending", 6, position=6)
    exit_ = change(4, "x", "SELL", "MARKET", "open", 6, position=6)
    assert _within_the_position([*ENTERED, SOLD_4, fitted], MINE, SIDES) == {}
    with pytest.raises(AssertionError, match="working exits"):
        _within_the_position([*ENTERED, SOLD_4, fitted, exit_], MINE, SIDES)


def test_a_session_buy_back_while_another_client_took_the_account_short_is_no_break():
    sold_15 = change(
        2, "h", "SELL", "MARKET", "complete", 15, filled=15, fill=15, position=-5, client="o"
    )
    cancelled = change(3, "s1", "SELL", "SL-M", "cancelled", 10, position=-5)
    bought = change(4, "b", "BUY", "MARKET", "complete", 2, filled=2, fill=2, position=-3)
    _within_the_position([*ENTERED, sold_15, cancelled, bought], MINE, SIDES)


def test_a_session_sale_deeper_on_the_side_another_client_took_the_account_to_is_found():
    sold_15 = change(
        2, "h", "SELL", "MARKET", "complete", 15, filled=15, fill=15, position=-5, client="o"
    )
    deeper = change(3, "s1", "SELL", "SL-M", "complete", 10, filled=10, fill=10, position=-15)
    with pytest.raises(AssertionError, match="other side"):
        _within_the_position([*ENTERED, sold_15, deeper], MINE, SIDES)


def test_a_waived_check_is_not_asserted():
    exit_ = change(2, "x", "SELL", "MARKET", "open", 10, position=10)
    _within_the_position([*ENTERED, exit_], MINE, SIDES, within=False)
