"""A shared instrument's rules against the broker's net position, on their own
(`_live/_group_rules.py`, `_live/_restart_groups.py`; 18-multi-strategy 4.6, 4.8 and 4.10,
unit M6): the group's kind, what a restart does with the group, and the words that name
it.

The comparison of a session is tested on the fake OpenAlgo (`test_session_shared_*.py`).
Expected values are worked by hand: signed units, long above 0; "A" holds 100 and "B" 50
of SBIN unless a row says otherwise.
"""

from __future__ import annotations

import pytest

from openalgo.backtest._live._group_rules import (
    GroupFigure,
    and_join,
    group_kind,
    group_restart_kind,
)
from openalgo.backtest._live._groups import InstrumentGroup
from openalgo.backtest._live._reconcile_rules import Kind, RestartKind
from openalgo.backtest._live._restart_groups import judge_group

SBIN = InstrumentGroup("SBIN", "NSE", "MIS", ("A", "B"))


@pytest.mark.parametrize(
    ("group", "broker", "unexplained", "kind"),
    [
        (150, 150, 0, Kind.AGREE),
        (150, None, 0, Kind.UNREADABLE),
        (150, 200, 50, Kind.MORE),
        (0, 30, 30, Kind.MORE),
        (150, -30, -180, Kind.OTHER_SIDE),
        (150, 90, 0, Kind.EXPLAINED_SHARED),
        (150, 0, 0, Kind.EXPLAINED_SHARED),
        (150, 90, -60, Kind.UNEXPLAINED),
        (150, 0, -150, Kind.UNEXPLAINED),
    ],
    ids=[
        "agree",
        "unreadable",
        "more",
        "more of a flat group",
        "other side",
        "fewer, explained: nothing adopted",
        "flat, explained: nothing adopted",
        "fewer, unexplained",
        "flat, unexplained",
    ],
)
def test_the_groups_kind_adopts_nothing_before_the_allocation(group, broker, unexplained, kind):
    assert group_kind(group, broker, unexplained) is kind


@pytest.mark.parametrize(
    ("members", "broker", "unexplained", "verdict"),
    [
        ((100, 50), 150, 0, RestartKind.RESUME),
        ((100, 0), 100, 0, RestartKind.RESUME),
        ((0, 0), 0, 0, RestartKind.RESUME),
        ((0, 0), 30, 30, RestartKind.UNKNOWN_POSITION),
        ((100, 50), 0, -150, RestartKind.UNEXPLAINED_FLAT),
        ((100, 50), 0, 0, RestartKind.DIFFERS),
        ((100, 50), 90, 0, RestartKind.DIFFERS),
        ((100, 50), 90, -60, RestartKind.DIFFERS),
        ((100, 0), 150, 50, RestartKind.DIFFERS),
        ((100, 50), -30, -180, RestartKind.DIFFERS),
    ],
    ids=[
        "the sum agrees",
        "one member flat, the sum agrees",
        "every member flat, the broker flat",
        "every member flat, the broker holds: refused",
        "the broker flat, nothing explains it: refused",
        "the broker flat, explained",
        "fewer, explained",
        "fewer, unexplained",
        "one member flat, the broker holds more",
        "the other side",
    ],
)
def test_a_restart_judges_the_members_sum_against_the_brokers_net(
    members, broker, unexplained, verdict
):
    assert group_restart_kind(members, broker, unexplained) is verdict


def test_a_restart_of_a_group_whose_sum_agrees_resumes_with_nothing_to_act_on():
    start = judge_group(SBIN, {"A": 100, "B": 50}, 150, 0, 0)
    assert (start.restart, start.kind, start.refusal) == (RestartKind.RESUME, Kind.AGREE, None)
    assert start.figure == GroupFigure(("A", "B"), 150)


def test_a_restart_of_a_group_names_the_instrument_and_every_member_when_it_refuses():
    unknown = judge_group(SBIN, {"A": 0, "B": 0}, 30, 30, 0)
    assert unknown.refusal == (
        "OpenAlgo's books show 30 units of SBIN NSE, which A and B trade, and the journal "
        "holds none for them: close them or start with another symbol"
    )
    # Sold by hand: the trades of other orders explain the broker's flat book.
    sold = judge_group(SBIN, {"A": 100, "B": 50}, 0, -150, 0)
    assert sold.refusal is None
    gone = judge_group(SBIN, {"A": 100, "B": 50}, 0, 0, 0)
    assert gone.refusal == (
        "the journal holds 150 units of SBIN NSE for A and B together (A 100, B 50), "
        "OpenAlgo's books show none and no trade explains it: check the account in OpenAlgo"
    )


def test_a_restart_of_a_group_after_a_sale_by_hand_differs_and_adopts_nothing():
    # 60 sold by hand while the session was down: the other orders' trades explain it.
    start = judge_group(SBIN, {"A": 100, "B": 50}, 90, -60, 0)
    assert (start.restart, start.kind) == (RestartKind.DIFFERS, Kind.EXPLAINED_SHARED)


def test_a_member_moved_away_is_left_out_of_its_groups_sum():
    start = judge_group(SBIN, {"A": 100}, 100, 0, 0)
    assert (start.restart, start.figure) == (RestartKind.RESUME, GroupFigure(("A",), 100))


@pytest.mark.parametrize(
    ("names", "words"),
    [((), ""), (("A",), "A"), (("A", "B"), "A and B"), (("A", "B", "C"), "A, B and C")],
)
def test_names_are_joined_in_words(names, words):
    assert and_join(names) == words


def test_a_figure_names_the_other_members_in_call_order():
    assert GroupFigure(("A", "B", "C"), 0).others("B") == ("A", "C")
