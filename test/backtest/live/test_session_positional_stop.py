"""What Ctrl+C does with CNC and NRML positions in a live session, on the fake OpenAlgo
(20-positional-live 10.2, decision 11.8; the lead's ruling N3b-1): by default it exits the
MIS positions and keeps the CNC and NRML ones with their GTT stops; `on_stop="flatten"`
passed exits every one; `on_stop="keep"` keeps every one. An MIS-only session behaves as
before, its banner unchanged, and a session with a CNC strategy says in its banner which
applies.

The market, the strategies and the books are `positional_samples`': `MisHold` and
`CncHold` both buy 10 SBIN at their first decision (14:42:01 IST, filled at 500.00), the
MIS leg with an SL-M stop at 497.00 and the CNC leg with a stop GTT at 497.00; Ctrl+C
comes at 14:46.
"""

from __future__ import annotations

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import STOP_AT, CncHold, MisHold, gtt_world, gtts

from openalgo.backtest._live._session_banner import ON_STOP_LINES, POSITIONAL_STOPS_LINE


@pytest.mark.parametrize(
    ("on_stop", "held", "gtt"),
    [(None, 10, "active"), ("flatten", 0, "cancelled"), ("keep", 20, "active")],
    ids=["default", "flatten", "keep"],
)
def test_ctrl_c_flattens_mis_and_keeps_cnc_by_default_and_as_on_stop_says_otherwise(
    make_fake_openalgo, monkeypatch, tmp_path, on_stop, held, gtt
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(STOP_AT)

    outcome = world.run([MisHold, CncHold], on_stop=on_stop)

    # The MIS leg's and the CNC leg's 10 units each, net at the fake.
    assert world.fake.position("SBIN", "NSE") == held
    assert [status for status, *_ in gtts(world)] == [gtt]
    assert outcome.stopped_by == "ctrl_c"
    assert ON_STOP_LINES[on_stop] in world.lines


def test_an_mis_only_session_flattens_at_ctrl_c_by_default_with_its_banner_as_before(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(STOP_AT)

    outcome = world.run(MisHold)

    assert world.fake.position("SBIN", "NSE") == 0
    assert world.fake.gtts() == []
    assert outcome.stopped_by == "ctrl_c"
    assert POSITIONAL_STOPS_LINE not in world.lines
    assert not set(ON_STOP_LINES.values()) & set(world.lines)
