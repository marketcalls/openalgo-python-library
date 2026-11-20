"""A shared instrument's comparison with the broker's net, its pure rules
(`_live/_reconcile_groups.py`; 18-multi-strategy 4.6 and 4.8, unit M6): the group's kind,
and each member's share of what the broker holds while the group is guarded.

The comparison of a session is tested on the fake OpenAlgo (`test_session_shared_*.py`).
Expected values are worked by hand: signed units, long above 0.
"""

from __future__ import annotations

import pytest

from openalgo.backtest._live._reconcile_groups import group_kind, share_of
from openalgo.backtest._live._reconcile_rules import Kind


@pytest.mark.parametrize(
    ("broker", "group", "member", "share"),
    [
        # The broker holds the sum: each member may close all it holds.
        (150, 150, 100, 100),
        (150, 150, 50, 50),
        # 60 sold by hand of a long 150 (A 100, B 50): A may close 40, B nothing.
        (90, 150, 100, 40),
        (90, 150, 50, 0),
        # The same short.
        (-90, -150, -100, -40),
        (-90, -150, -50, 0),
        # Flat or the other side at the broker: the broker's figure, which leaves no room.
        (0, 150, 100, 0),
        (-30, 150, 100, -30),
    ],
)
def test_a_members_share_of_what_the_broker_holds(broker, group, member, share):
    assert share_of(broker, group, member) == share


def test_the_shares_together_never_exceed_what_the_broker_holds():
    for broker in range(0, 160, 10):
        shares = share_of(broker, 150, 100) + share_of(broker, 150, 50)
        assert shares <= broker


@pytest.mark.parametrize(
    ("group", "broker", "unexplained", "kind"),
    [
        (150, 150, 0, Kind.AGREE),
        (150, None, 0, Kind.UNREADABLE),
        (150, 200, 50, Kind.MORE),
        (0, 30, 30, Kind.MORE),
        (150, -30, -180, Kind.OTHER_SIDE),
        (150, 90, 0, Kind.UNEXPLAINED),
        (150, 0, 0, Kind.UNEXPLAINED),
        (150, 90, -60, Kind.UNEXPLAINED),
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
    ],
)
def test_the_groups_kind_adopts_nothing_before_the_allocation(group, broker, unexplained, kind):
    assert group_kind(group, broker, unexplained) is kind
