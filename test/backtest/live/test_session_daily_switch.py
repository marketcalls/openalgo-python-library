"""A change of OpenAlgo's analyzer switch between a daily strategy's decision before the open
and the open, on the fake OpenAlgo (design 20-positional-live sections 8 and 10.4,
16-session-design 6.2; the lead's rulings 9b and 9c of review round 4 of unit N5a): the
mode guard locks and the session sends nothing more, so the orders the decision left
waiting for the open never go. The mode-switch alert names them, as the result's warnings
do, and a session with no MIS strategy is never told of the broker's MIS square-off.

The market and the strategy are `daily_samples`': the session starts at 09:05 IST on Monday
2026-11-16; `Recorded` (CNC) decides Friday's daily bar and buys 10 SBIN at market, an
order that waits for the 09:15 open. The switch changes at 09:10; the session is stopped
(Ctrl+C) at 09:30. Each test runs from the broker's book and from the sandbox's.
"""

from __future__ import annotations

import datetime as dt
import logging

import pytest
from daily_samples import Recorded, close_of, daily_world
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import MODES
from session_samples import alerts, ist

from openalgo.backtest._live._result_build import session_result
from openalgo.backtest._live._types import LOGGER_NAME, Mode

FRIDAY = dt.date(2026, 11, 13)


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


def switched_before_the_open(make_fake_openalgo, monkeypatch, tmp_path, *, analyzer):
    """The session of `Recorded` (module docstring) whose analyzer switch changes at 09:10."""
    world = daily_world(make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer)
    world.trace.waive("books agree", "the switch shows the other book, which holds none")
    world.script.at(ist(9, 10), lambda: world.fake.set_analyzer(not analyzer))
    world.stop_at(ist(9, 30))
    return world


@MODES
def test_a_switch_before_the_open_names_the_decisions_orders_it_drops_and_no_mis_square_off(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer
):
    world = switched_before_the_open(make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer)

    outcome = world.run(Recorded)

    mode = "SANDBOX" if analyzer else "LIVE"
    assert [bar_close for _, bar_close, _ in Recorded.decided] == [close_of(FRIDAY)]
    dropped = (
        "Recorded: the mode switch drops the orders of its decision on its D bar closing on "
        f"2026-11-13 (BUY 10 MARKET) before they went to the broker; a start today in {mode} "
        "mode does not decide that bar again"
    )
    assert alerts(caplog, "error") == [
        f"{mode} (Recorded): OpenAlgo's mode changed: the session sends nothing more. "
        f"Positions as last known: no open position. {dropped}. Stop the session with Ctrl+C"
    ]
    assert dropped in session_result(outcome).warnings
    assert world.fake.orders(Mode.SANDBOX if analyzer else Mode.LIVE) == []
