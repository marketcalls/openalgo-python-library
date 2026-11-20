"""The session invariants' checks at every change of an order (`session_invariants_changes`,
16-session-design 7.1, unit 14), on order changes written by hand: each check finds what it
must and nothing more. The session's client is "s", another client of the account "o";
SBIN's open side is BUY, and a reversal (an entry of the session's against the position,
twice its units) moves it to SELL.
"""

from __future__ import annotations

import pytest
from session_invariants_changes import SaleRoom, one_stop_per_leg, sides_of, within_the_position
from session_trace import OrderChange

from openalgo.backtest._live._types import Mode

MINE = frozenset({"s"})
SBIN = (Mode.LIVE, "SBIN", "NSE")
SIDES = {SBIN: "BUY"}
#: A test's decided end that lets `StopHold`'s overtaken stop fill through zero.
OVERSOLD = {"StopHold": "a decided end"}


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
FILLED = change(0, "e", "BUY", "MARKET", "complete", 10, filled=10, fill=10, position=10)
ENTERED = [FILLED, change(1, "s1", "SELL", "SL-M", "trigger pending", 10, position=10)]


def sold(seq: int, units: int, *, position: int) -> OrderChange:
    """Another client sells `units` SBIN by hand, the position after it."""
    return change(
        seq, "h", "SELL", "MARKET", "complete", units, filled=units, fill=units,
        position=position, client="o",
    )  # fmt: skip


#: Another client sells 4 of the 10 by hand while the stop of 10 works.
SOLD_4 = sold(2, 4, position=6)
#: The stop of 10 then fills against the 6 left, taking the account short 4.
STOP_FILLED = change(3, "s1", "SELL", "SL-M", "complete", 10, filled=10, fill=10, position=-4)


def test_a_second_stop_working_beside_the_first_is_found():
    second = change(2, "s2", "SELL", "SL-M", "trigger pending", 10, position=10)
    with pytest.raises(AssertionError, match="two stops"):
        one_stop_per_leg([*ENTERED, second], MINE)


def test_a_stop_placed_again_once_the_first_ended_is_one_stop():
    ended = change(2, "s1", "SELL", "SL-M", "cancelled", 10, position=10)
    again = change(3, "s2", "SELL", "SL-M", "trigger pending", 10, position=10)
    one_stop_per_leg([*ENTERED, ended, again], MINE)


def test_an_exit_placed_while_the_stop_still_works_is_found():
    exit_ = change(2, "x", "SELL", "MARKET", "open", 10, position=10)
    with pytest.raises(AssertionError, match="working exits"):
        within_the_position([*ENTERED, exit_], MINE, SIDES)


def test_an_exit_placed_once_the_stop_is_cancelled_is_within_the_position():
    cancelled = change(2, "s1", "SELL", "SL-M", "cancelled", 10, position=10)
    exit_ = change(3, "x", "SELL", "MARKET", "open", 10, position=10)
    assert within_the_position([*ENTERED, cancelled, exit_], MINE, SIDES) == {}


def test_a_fill_of_the_sessions_order_through_zero_is_found():
    cut = change(2, "s1", "SELL", "SL-M", "complete", 10, filled=10, fill=10, position=0)
    oversold = change(3, "x", "SELL", "MARKET", "complete", 4, filled=4, fill=4, position=-4)
    with pytest.raises(AssertionError, match="other side"):
        within_the_position([*ENTERED, cut, oversold], MINE, SIDES)


def test_a_sale_by_hand_against_the_stop_leaves_the_stop_room_for_the_units_sold():
    room = within_the_position([*ENTERED, SOLD_4], MINE, SIDES)
    assert room == {SBIN: SaleRoom(sold=4, beyond=4, overtaken={"s1": 10})}


def test_a_stop_placed_after_a_sale_by_hand_for_more_than_is_left_is_found():
    # Finding 2 of unit 14 round 2: the room is the overtaken exits', not the leg's.
    placed = change(3, "s1", "SELL", "SL-M", "trigger pending", 10, position=6)
    with pytest.raises(AssertionError, match="working exits"):
        within_the_position([FILLED, SOLD_4, placed], MINE, SIDES)


def test_a_stop_raised_after_a_sale_by_hand_beyond_its_units_then_is_found():
    small = [FILLED, change(1, "s1", "SELL", "SL-M", "trigger pending", 6, position=10)]
    sold_6 = sold(2, 6, position=4)
    raised = change(3, "s1", "SELL", "SL-M", "trigger pending", 10, position=4)
    within_the_position([*small, sold_6], MINE, SIDES)
    with pytest.raises(AssertionError, match="working exits"):
        within_the_position([*small, sold_6, raised], MINE, SIDES)


def test_the_stop_filling_through_zero_after_a_sale_by_hand_is_found():
    # Finding 1 of unit 14 round 1: the sale by hand excuses none of the session's fills.
    with pytest.raises(AssertionError, match="other side"):
        within_the_position([*ENTERED, SOLD_4, STOP_FILLED], MINE, SIDES)


def test_a_decided_end_lets_the_overtaken_stop_fill_through_zero_by_the_units_sold():
    assert within_the_position([*ENTERED, SOLD_4, STOP_FILLED], MINE, SIDES, OVERSOLD) == {}


def test_a_decided_end_does_not_let_an_exit_placed_after_the_sale_fill_through_zero():
    cancelled = change(3, "s1", "SELL", "SL-M", "cancelled", 10, position=6)
    exit_ = change(4, "x", "SELL", "MARKET", "complete", 10, filled=10, fill=10, position=-4)
    with pytest.raises(AssertionError, match="other side"):
        within_the_position([*ENTERED, SOLD_4, cancelled, exit_], MINE, SIDES, OVERSOLD)


def test_a_decided_end_does_not_let_the_stop_fill_deeper_than_the_units_sold():
    sold_2 = sold(2, 2, position=8)
    trailed = change(3, "s1", "SELL", "SL-M", "trigger pending", 10, position=8)
    raised = change(4, "s1", "SELL", "SL-M", "trigger pending", 12, position=8)
    deeper = change(5, "s1", "SELL", "SL-M", "complete", 12, filled=12, fill=12, position=-4)
    changes = [*ENTERED, sold_2, trailed, raised, deeper]
    with pytest.raises(AssertionError, match="other side"):
        within_the_position(changes, MINE, SIDES, OVERSOLD, within=False)


def test_an_exit_the_session_adds_beside_a_stop_a_sale_by_hand_left_too_large_is_found():
    trailed = change(3, "s1", "SELL", "SL-M", "trigger pending", 10, position=6)
    exit_ = change(4, "x", "SELL", "MARKET", "open", 6, position=6)
    within_the_position([*ENTERED, SOLD_4, trailed], MINE, SIDES)
    with pytest.raises(AssertionError, match="working exits"):
        within_the_position([*ENTERED, SOLD_4, trailed, exit_], MINE, SIDES)


def test_once_the_session_fits_the_stop_to_a_sale_by_hand_the_room_is_gone():
    fitted = change(3, "s1", "SELL", "SL-M", "trigger pending", 6, position=6)
    exit_ = change(4, "x", "SELL", "MARKET", "open", 6, position=6)
    assert within_the_position([*ENTERED, SOLD_4, fitted], MINE, SIDES) == {}
    with pytest.raises(AssertionError, match="working exits"):
        within_the_position([*ENTERED, SOLD_4, fitted, exit_], MINE, SIDES)


def test_a_session_buy_back_while_another_client_took_the_account_short_is_no_break():
    sold_15 = sold(2, 15, position=-5)
    cancelled = change(3, "s1", "SELL", "SL-M", "cancelled", 10, position=-5)
    bought = change(4, "b", "BUY", "MARKET", "complete", 2, filled=2, fill=2, position=-3)
    within_the_position([*ENTERED, sold_15, cancelled, bought], MINE, SIDES)


def test_a_session_sale_deeper_on_the_side_another_client_took_the_account_to_is_found():
    sold_15 = sold(2, 15, position=-5)
    deeper = change(3, "s1", "SELL", "SL-M", "complete", 10, filled=10, fill=10, position=-15)
    with pytest.raises(AssertionError, match="other side"):
        within_the_position([*ENTERED, sold_15, deeper], MINE, SIDES)


def test_a_waived_check_is_not_asserted():
    exit_ = change(2, "x", "SELL", "MARKET", "open", 10, position=10)
    within_the_position([*ENTERED, exit_], MINE, SIDES, within=False)


# A reversal: the entry "r" sells 20 against the long of 10, once its stop is cancelled.
STOP_CANCELLED = change(2, "s1", "SELL", "SL-M", "cancelled", 10, position=10)
REVERSAL = change(3, "r", "SELL", "MARKET", "open", 20, position=10)
REVERSED = change(4, "r", "SELL", "MARKET", "complete", 20, filled=20, fill=20, position=-10)
#: The session's entries and adds, by broker order ID.
ENTRIES = frozenset({"e", "r"})


def test_a_reversal_holds_as_an_exit_only_the_position_it_closes():
    assert (
        within_the_position([*ENTERED, STOP_CANCELLED, REVERSAL], MINE, SIDES, entries=ENTRIES)
        == {}
    )


def test_a_reversal_placed_while_the_stop_still_works_is_found():
    early = change(2, "r", "SELL", "MARKET", "open", 20, position=10)
    with pytest.raises(AssertionError, match="working exits"):
        within_the_position([*ENTERED, early], MINE, SIDES, entries=ENTRIES)


def test_a_reversals_fill_moves_the_leg_to_the_other_side():
    changes = [*ENTERED, STOP_CANCELLED, REVERSAL, REVERSED]
    assert within_the_position(changes, MINE, SIDES, entries=ENTRIES) == {}


def test_an_exit_filled_through_zero_is_found_beside_a_reversal():
    # The same sale of 20 as an exit (not an entry of the session's) crosses zero.
    changes = [*ENTERED, STOP_CANCELLED, REVERSAL, REVERSED]
    with pytest.raises(AssertionError, match="other side"):
        within_the_position(changes, MINE, SIDES, within=False)


def test_after_a_reversal_the_new_legs_stop_is_its_exit():
    stop = change(5, "s2", "BUY", "SL-M", "trigger pending", 10, position=-10)
    changes = [*ENTERED, STOP_CANCELLED, REVERSAL, REVERSED, stop]
    assert within_the_position(changes, MINE, SIDES, entries=ENTRIES) == {}


def test_after_a_reversal_an_exit_beside_the_new_legs_stop_is_found():
    stop = change(5, "s2", "BUY", "SL-M", "trigger pending", 10, position=-10)
    exit_ = change(6, "x", "BUY", "MARKET", "open", 10, position=-10)
    changes = [*ENTERED, STOP_CANCELLED, REVERSAL, REVERSED, stop, exit_]
    with pytest.raises(AssertionError, match="working exits"):
        within_the_position(changes, MINE, SIDES, entries=ENTRIES)


def test_the_legs_side_at_the_end_is_the_one_its_last_reversal_left():
    changes = [*ENTERED, STOP_CANCELLED, REVERSAL, REVERSED]
    assert sides_of(changes, ENTRIES) == {SBIN: "SELL"}


def test_an_entry_on_the_other_side_after_the_leg_closed_moves_the_leg_there():
    # The SBIN case of the probe runs: a leg closed by its stop, then an entry on the other
    # side from flat (a short after the long), its order working and then filled.
    stopped = change(2, "s1", "SELL", "SL-M", "complete", 10, filled=10, fill=10, position=0)
    short = change(3, "e2", "SELL", "MARKET", "open", 10, position=0)
    shorted = change(4, "e2", "SELL", "MARKET", "complete", 10, filled=10, fill=10, position=-10)
    changes = [*ENTERED, stopped, short, shorted]
    assert within_the_position(changes, MINE, SIDES, entries=frozenset({"e", "e2"})) == {}
