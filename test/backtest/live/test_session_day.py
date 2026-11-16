"""A whole live session on the fake OpenAlgo, end to end (`_live/session.py`;
16-session-design 3.1, 3.4 and plan step 6, unit 8): the start, the countdown, trading,
the square-off by the clock and the end of the day, under the test seam.

The market and the script are `session_samples`': the session starts at 14:40:05 IST on
Monday 2026-11-16, SBIN trades at 500.00 every 10 s, NSE's MIS entries close at 14:50
and its square-off is at 15:00.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import BuyAndHold, Hold, alerts, ist, rig

from openalgo.backtest._live._types import LOGGER_NAME


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return rig(make_fake_openalgo, monkeypatch, tmp_path)


def test_a_session_buys_sells_and_ends_flat_at_the_end_of_the_day(world):
    outcome = world.run(BuyAndHold)
    orders = world.orders()
    assert [(order.action, order.quantity) for order in orders] == [("BUY", 10), ("SELL", 10)]
    assert world.fake.position("SBIN", "NSE") == 0
    assert outcome.stopped_by == "session_end"
    assert outcome.open_positions == ()


def test_the_session_ends_at_the_square_off_once_flat(world):
    outcome = world.run(BuyAndHold)
    assert ist(15, 0) <= outcome.ended_ns < ist(15, 0, 5)


def test_the_first_decision_is_the_close_of_the_first_whole_minute_watched(world):
    world.run(BuyAndHold)
    first = world.orders()[0]
    # Watching began in the 14:40 minute (after the countdown); its close seeds only, so
    # the first decision is at 14:42:00 plus the slice's grace.
    assert ist(14, 42) <= first.placed_ns < ist(14, 42, 5)


def test_a_position_left_open_is_squared_off_by_the_clock(world, caplog):
    outcome = world.run(Hold)
    sell = world.orders()[-1]
    assert (sell.action, sell.quantity) == ("SELL", 10)
    assert ist(15, 0) <= sell.placed_ns < ist(15, 0, 5)
    assert outcome.stopped_by == "session_end"
    assert alerts(caplog, "entry") and alerts(caplog, "exit")


def test_a_position_still_open_after_the_square_off_is_alerted_each_minute_until_the_close(
    world, caplog
):
    def refuse_exits():
        for _ in range(100):
            world.fake.reject_next("RMS: blocked", price_type="MARKET")

    world.script.at(ist(14, 59, 50), refuse_exits)
    outcome = world.run(Hold)
    not_flat = [text for text in alerts(caplog, "error") if "still not flat" in text]
    assert len(not_flat) >= 2
    assert outcome.stopped_by == "session_end"
    assert ist(15, 35) <= outcome.ended_ns < ist(15, 36)
    assert [(p.symbol, p.units) for p in outcome.open_positions] == [("SBIN", 10.0)]
