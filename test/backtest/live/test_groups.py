"""The instrument groups of a live session and where a member stands, their pure parts
(`_live/_groups.py`; 18-multi-strategy 4.1, 4.11, 4.13 and 4.14a item 1, unit M6).

The gates as a session sets them are tested on the fake OpenAlgo
(`test_session_shared_gates.py`). Expected values are written by hand.
"""

from __future__ import annotations

import pytest

from openalgo.backtest._live._groups import (
    LONG,
    SHORT,
    InstrumentGroup,
    Stand,
    instrument_groups,
    shared_lines,
)


def test_each_instrument_has_a_group_of_its_strategies_in_call_order():
    groups = instrument_groups(
        [
            ("SbinTrend", "SBIN", "NSE"),
            ("InfyTrend", "INFY", "NSE"),
            ("SbinBreakout", "SBIN", "NSE"),
            ("SbinBse", "SBIN", "BSE"),
        ]
    )
    assert groups == (
        InstrumentGroup("SBIN", "NSE", "MIS", ("SbinTrend", "SbinBreakout")),
        InstrumentGroup("INFY", "NSE", "MIS", ("InfyTrend",)),
        InstrumentGroup("SBIN", "BSE", "MIS", ("SbinBse",)),
    )
    assert [group.shared for group in groups] == [True, False, False]


@pytest.mark.parametrize(
    ("stand", "resulting", "reversing"),
    [
        (Stand(None, frozenset()), frozenset(), False),
        (Stand(LONG, frozenset()), {LONG}, False),
        (Stand(None, frozenset({SHORT})), {SHORT}, False),
        (Stand(LONG, frozenset({LONG})), {LONG}, False),
        (Stand(LONG, frozenset({SHORT})), {SHORT}, True),
        (Stand(SHORT, frozenset({LONG})), {LONG}, True),
    ],
    ids=["flat", "long", "entering short", "adding long", "reversing long", "reversing short"],
)
def test_where_a_member_stands_once_its_opening_orders_fill(stand, resulting, reversing):
    assert (stand.resulting, stand.reversing) == (resulting, reversing)


def test_the_banner_names_each_shared_instrument_with_its_members_then_the_rule():
    groups = (
        InstrumentGroup("SBIN", "NSE", "MIS", ("A", "B", "C")),
        InstrumentGroup("INFY", "NSE", "MIS", ("D",)),
    )
    assert shared_lines(groups) == [
        "SBIN (NSE, MIS): A, B and C share one broker position; each holds its own stop; "
        "they trade one side at a time and with market orders only",
        "Never trade a session's symbol by hand while it runs",
    ]


def test_no_shared_instrument_adds_no_banner_line():
    assert shared_lines((InstrumentGroup("INFY", "NSE", "MIS", ("D",)),)) == []
