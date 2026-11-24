"""How long the GTT parts wait for a leg's units to settle (`_live/_gtt_reads.py`,
`BookRead.waited_out`; the lead's ruling L1 on N3a's review round 4): `UNITS_WAIT_S` (90 s)
from the later of when the wait began and the newest own fill on the leg.

The leg is SBIN NSE CNC; the wait begins at 1 000 s. Expected answers are worked by hand
from each row's times.
"""

from __future__ import annotations

import pytest

from openalgo.backtest._live._gtt_reads import UNITS_WAIT_S, BookRead

NS = 10**9
LEG = ("SBIN", "NSE", "CNC")
BEGAN_NS = 1_000 * NS


@pytest.mark.parametrize(
    ("fills_ns", "read_s", "over"),
    [
        ({}, 1_000 + UNITS_WAIT_S - 2.0, False),
        ({}, 1_000 + UNITS_WAIT_S, True),
        ({LEG: 900 * NS}, 1_000 + UNITS_WAIT_S, True),
        ({LEG: 1_020 * NS}, 1_000 + UNITS_WAIT_S, False),
        ({LEG: 1_020 * NS}, 1_020 + UNITS_WAIT_S, True),
        ({("SBIN", "NSE", "NRML"): 1_020 * NS}, 1_000 + UNITS_WAIT_S, True),
    ],
    ids=[
        "no own fill, 88 s on",
        "no own fill, 90 s on",
        "a fill before the wait began, 90 s on",
        "a fill 20 s into the wait, 90 s after the wait began",
        "a fill 20 s into the wait, 90 s after it",
        "a fill on another leg",
    ],
)
def test_the_wait_for_a_legs_units_ends_90_s_after_it_began_or_its_newest_own_fill(
    fills_ns, read_s, over
):
    read = BookRead(rows={}, read_ns=round(read_s * NS), fills_ns=fills_ns)

    assert read.waited_out(LEG, BEGAN_NS) is over
