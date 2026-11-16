"""Which settled slices a read of the broker's candles releases in the catch-up
(`_live/_plan_release.py`; 16-session-design 2.5 B and C, unit 7): with its candle, as
a minute without a trade when a later candle proves the broker has published past it,
marked missing once the candle checks' last read (close plus 7 s) has passed, and
otherwise held for a later read.

Slices are the minutes of Monday 2026-11-16 opening from 10:10 to 10:13 IST; a slice is
settled at a read begun 2 s or more after its close. A read's candles are given by open.
"""

from __future__ import annotations

import pytest
from plan_samples import MINUTE_NS, NS, ist

from openalgo.backtest._live._bars_candle import candle_bar
from openalgo.backtest._live._plan_release import (
    MISSING,
    NO_TRADE,
    WITH_CANDLE,
    missing_note,
    releases,
)

#: The minutes opening at 10:10, 10:11, 10:12 and 10:13 IST, as (open, close).
MINUTES = tuple((ist(10, 10 + n), ist(10, 11 + n)) for n in range(4))
SETTLED_AFTER_NS = 2 * NS


def candles_of(*minutes: int) -> dict:
    """Broker candles of the minutes opening at 10:`minute` IST, by open."""
    found = {}
    for minute in minutes:
        row = {"ts_open": ist(10, minute), "open": 500.0, "high": 501.0, "low": 499.0}
        found[ist(10, minute)] = candle_bar(
            "SBIN", "NSE", row | {"close": 500.5, "volume": 10.0}, MINUTE_NS
        )
    return found


def released(read_at: int, *minutes: int) -> list[tuple[int, str]]:
    """(the minute past 10 of each slice released, how) by a read begun at `read_at`
    holding the candles of `minutes`."""
    settled = [minute for minute in MINUTES if minute[1] + SETTLED_AFTER_NS <= read_at]
    return [
        (10 + (release.opened - ist(10, 10)) // MINUTE_NS, release.kind)
        for release in releases(settled, candles_of(*minutes), read_at)
    ]


@pytest.mark.parametrize(
    ("read_at", "minutes", "expected"),
    [
        pytest.param(
            ist(10, 14, 3),
            (10, 11, 12, 13),
            [(10, WITH_CANDLE), (11, WITH_CANDLE), (12, WITH_CANDLE), (13, WITH_CANDLE)],
            id="every candle read",
        ),
        pytest.param(
            ist(10, 14, 3),
            (10, 12, 13),
            [(10, WITH_CANDLE), (11, NO_TRADE), (12, WITH_CANDLE), (13, WITH_CANDLE)],
            id="a later candle proves a minute without a trade",
        ),
        pytest.param(
            ist(10, 14, 3),
            (10, 11, 12),
            [(10, WITH_CANDLE), (11, WITH_CANDLE), (12, WITH_CANDLE)],
            id="the last candle not published 3 s after its close is waited for",
        ),
        pytest.param(
            ist(10, 14, 7),
            (10, 11, 12),
            [(10, WITH_CANDLE), (11, WITH_CANDLE), (12, WITH_CANDLE), (13, MISSING)],
            id="the last candle still not published 7 s after its close is missing",
        ),
        pytest.param(
            ist(10, 13, 5),
            (),
            [(10, MISSING), (11, MISSING)],
            id="no candle at all: the slices past their last read are missing, the rest wait",
        ),
    ],
)
def test_a_read_releases_each_settled_slice_by_what_it_holds(read_at, minutes, expected):
    assert released(read_at, *minutes) == expected


def test_a_slice_released_without_its_candle_has_no_bar():
    found = releases(MINUTES, candles_of(10, 12, 13), ist(10, 14, 3))
    assert [(release.kind, release.bar) for release in found if release.bar is None] == [
        (NO_TRADE, None)
    ]


def test_the_note_of_slices_given_up_on_names_their_minutes_and_the_last_read():
    note = missing_note("EmaTrend", "SBIN", "NSE", [ist(10, 13)], none_today=False)
    assert note == (
        "EmaTrend: 1 minute of NSE:SBIN (2026-11-16 10:13 IST) marked missing, since no "
        "broker candle of them was read by 7 s after their close: no decision is made on "
        "a bar holding one until its candle is read"
    )


def test_the_note_of_a_read_with_no_candle_of_today_says_what_to_check():
    marked = [ist(9, 15), ist(9, 16), ist(9, 17)]
    note = missing_note("EmaTrend", "SBIN", "NSE", marked, none_today=True)
    assert "3 minutes of NSE:SBIN (2026-11-16 09:15 IST to 2026-11-16 09:17 IST)" in note
    assert "check the broker's login in OpenAlgo and that the broker serves today's" in note
