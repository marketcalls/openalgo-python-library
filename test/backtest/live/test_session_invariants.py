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
    assert _within_the_position([*ENTERED, cancelled, exit_], MINE, SIDES) == set()


def test_a_fill_of_the_sessions_order_through_zero_is_found():
    cut = change(2, "s1", "SELL", "SL-M", "complete", 10, filled=10, fill=10, position=0)
    oversold = change(3, "x", "SELL", "MARKET", "complete", 4, filled=4, fill=4, position=-4)
    with pytest.raises(AssertionError, match="other side"):
        _within_the_position([*ENTERED, cut, oversold], MINE, SIDES)


def test_a_sale_by_hand_against_the_stop_is_excused_until_the_session_fits_it():
    sold = change(2, "h", "SELL", "MARKET", "complete", 4, filled=4, fill=4, position=6, client="o")
    assert _within_the_position([*ENTERED, sold], MINE, SIDES) == {SBIN}


def test_once_the_session_fits_the_stop_to_a_sale_by_hand_the_checks_apply_again():
    sold = change(2, "h", "SELL", "MARKET", "complete", 4, filled=4, fill=4, position=6, client="o")
    fitted = change(3, "s1", "SELL", "SL-M", "trigger pending", 6, position=6)
    exit_ = change(4, "x", "SELL", "MARKET", "open", 6, position=6)
    with pytest.raises(AssertionError, match="working exits"):
        _within_the_position([*ENTERED, sold, fitted, exit_], MINE, SIDES)
