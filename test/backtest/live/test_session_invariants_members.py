"""The session invariants restated per member of an instrument group, on order changes and
runs written by hand (18-multi-strategy 5.1, unit M8): I3 per member at every change
(`session_invariants_members`), and I1 and I7 per member at the end
(`session_invariants_end`). Each finds what the rule of the group cannot, and nothing
more.

The session's client is "s", another client of the account "o"; "A" and "B" trade SBIN in
one session, A long 100 and B long 50 (the broker's net 150), each with a stop for its own
units unless a test says otherwise.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from session_invariants_changes import within_the_position
from session_invariants_end import at_the_end, no_orphan
from session_invariants_members import members_within
from session_trace import Closing, Member, OrderChange, Run

from openalgo.backtest._live._types import Mode

MINE = frozenset({"s"})
SBIN = (Mode.LIVE, "SBIN", "NSE")


def order(
    seq: int,
    order_id: str,
    strategy: str,
    action: str,
    price_type: str,
    *,
    quantity: int,
    position: int,
    members: tuple[Member, ...],
    status: str = "trigger pending",
    filled: int = 0,
    client: str = "s",
) -> OrderChange:
    """An order of SBIN as it stood after a change, the broker's net after it and the
    members' books then."""
    fill = filled if status == "complete" else 0
    return OrderChange(
        seq, seq, Mode.LIVE, order_id, client, strategy, "SBIN", "NSE", action, price_type,
        status, quantity, filled, fill, 500.0 if fill else 0.0, position, members, MINE,
    )  # fmt: skip


def member(strategy: str, units: int, *, adopted: int = 0) -> Member:
    return Member(strategy, units, adopted, ())


#: Both members hold their units; the trace reads their books at every change.
HOLDING = (member("A", 100), member("B", 50))
A_STOP = order(0, "sa", "A", "SELL", "SL-M", quantity=100, position=150, members=HOLDING)
B_STOP = order(1, "sb", "B", "SELL", "SL-M", quantity=50, position=150, members=HOLDING)


# ----- I3 per member, at every change -------------------------------------------------------


def test_each_members_stop_for_its_own_units_is_within_its_position():
    members_within([A_STOP, B_STOP], MINE)


def test_a_members_stop_sized_for_the_whole_group_is_found_though_the_group_holds_it():
    # A's stop of 150 against A's 100, with no stop of B's: the broker's net covers it.
    a_for_both = order(0, "sa", "A", "SELL", "SL-M", quantity=150, position=150, members=HOLDING)
    within_the_position([a_for_both], MINE, {SBIN: "BUY"})
    with pytest.raises(AssertionError, match="I3: A's working closing orders"):
        members_within([a_for_both], MINE)


def test_a_flat_members_stop_on_another_members_position_is_found():
    b_flat = (member("A", 100), member("B", 0))
    b_stop = order(0, "sb", "B", "SELL", "SL-M", quantity=50, position=100, members=b_flat)
    within_the_position([b_stop], MINE, {SBIN: "BUY"})
    with pytest.raises(AssertionError, match="I3: B's"):
        members_within([b_stop], MINE)


def test_a_flat_members_entry_is_no_closing_order():
    b_flat = (member("A", 100), member("B", 0))
    entry = order(0, "eb", "B", "BUY", "MARKET", quantity=50, position=100, members=b_flat,
                  status="open")  # fmt: skip
    members_within([entry], MINE)


def test_a_reversal_holds_as_an_exit_only_the_position_it_closes():
    # A sells 200 against its long of 100 (an entry, a reversal), its stop cancelled first.
    a_only = (member("A", 100),)
    reversal = order(0, "ra", "A", "SELL", "MARKET", quantity=200, position=100, members=a_only,
                     status="open")  # fmt: skip
    members_within([reversal], MINE, entries=frozenset({"ra"}))
    with pytest.raises(AssertionError, match="I3"):
        members_within([reversal], MINE)


def test_a_stop_overtaken_by_its_members_adoption_stands_until_its_cut():
    # Another client sells 60; B adopts 50 (flat) and A 10 (90 left) before their stops are
    # cut: each stop may stand beyond its member's position by what that member adopted.
    sale = order(2, "h1", "manual", "SELL", "MARKET", quantity=60, position=90, client="o",
                 members=HOLDING, status="complete", filled=60)  # fmt: skip
    adopted = (member("A", 90, adopted=-10), member("B", 0, adopted=-50))
    b_cut = order(3, "sb", "B", "SELL", "SL-M", quantity=50, position=90, members=adopted,
                  status="cancelled")  # fmt: skip
    a_cut = order(4, "sa", "A", "SELL", "SL-M", quantity=90, position=90, members=adopted)
    members_within([A_STOP, B_STOP, sale, b_cut, a_cut], MINE)


def test_a_stop_raised_after_its_members_adoption_is_found():
    adopted = (member("A", 90, adopted=-10), member("B", 50))
    a_after = order(2, "sa", "A", "SELL", "SL-M", quantity=100, position=140, members=adopted)
    raised = order(3, "sa", "A", "SELL", "SL-M", quantity=110, position=140, members=adopted)
    members_within([A_STOP, B_STOP, a_after], MINE)
    with pytest.raises(AssertionError, match="I3: A's"):
        members_within([A_STOP, B_STOP, a_after, raised], MINE)


def test_the_room_of_an_adoption_ends_once_the_members_exits_fit_it():
    adopted = (member("A", 90, adopted=-10), member("B", 50))
    a_after = order(2, "sa", "A", "SELL", "SL-M", quantity=100, position=140, members=adopted)
    a_cut = order(3, "sa", "A", "SELL", "SL-M", quantity=90, position=140, members=adopted)
    raised = order(4, "sa", "A", "SELL", "SL-M", quantity=100, position=140, members=adopted)
    with pytest.raises(AssertionError, match="I3: A's"):
        members_within([A_STOP, B_STOP, a_after, a_cut, raised], MINE)


def test_another_clients_order_tagged_with_a_members_name_is_not_its_closing_order():
    tagged = order(2, "x1", "B", "SELL", "LIMIT", quantity=500, position=150, members=HOLDING,
                   status="open", client="o")  # fmt: skip
    members_within([A_STOP, B_STOP, tagged], MINE)


# ----- I1 and I7 per member, at the end ------------------------------------------------------


def closing(strategy: str, units: float, *order_ids: str, covers: int) -> Closing:
    return Closing(strategy, "SBIN", "NSE", units, True, frozenset(order_ids), covers)


def run_of() -> Run:
    session = SimpleNamespace(
        parts=SimpleNamespace(mode=Mode.LIVE),
        traders=(),
        reconcile=SimpleNamespace(guard=SimpleNamespace(left=list)),
    )
    return Run(session, MINE)


TRACE = SimpleNamespace(unprotected={}, waived={}, kept_stops={}, oversold={})


def test_a_stop_of_a_member_whose_own_leg_is_flat_is_found_at_the_end():
    # B is flat in its kernel while A holds the broker's 100: B's stop works for nothing
    # of B's (I1 per member), though the instrument is not flat.
    legs = {"A": closing("A", 100, "sa", covers=100), "B": closing("B", 0, "sb", covers=0)}
    b_stop = order(1, "sb", "B", "SELL", "SL-M", quantity=50, position=100, members=None)
    a_stop = order(0, "sa", "A", "SELL", "SL-M", quantity=100, position=100, members=None)
    with pytest.raises(AssertionError, match="flat position"):
        no_orphan(run_of(), legs, [a_stop, b_stop], {SBIN: 100}, {}, {})


def test_each_member_covered_by_its_own_stop_at_the_end_is_protected():
    legs = {"A": closing("A", 100, "sa", covers=100), "B": closing("B", 50, "sb", covers=50)}
    at_the_end(TRACE, run_of(), legs, [A_STOP, B_STOP], {SBIN: 150}, {SBIN: "BUY"}, {})


def test_a_member_whose_stop_is_missing_is_found_though_another_members_stop_covers_it():
    # B's stop is gone at the broker, B's supervisor still counts it confirmed, and A's
    # stop of 100 alone would cover B's 50: per member, B is not protected (I7).
    legs = {"A": closing("A", 100, "sa", covers=100), "B": closing("B", 50, "sb", covers=50)}
    with pytest.raises(AssertionError, match="B ends with 150 units"):
        at_the_end(TRACE, run_of(), legs, [A_STOP], {SBIN: 150}, {SBIN: "BUY"}, {})
