"""The fault rows of a session's start that no earlier unit's test covers, on the fake
OpenAlgo (16-session-design 7.2 row F47, unit 14): OpenAlgo's Telegram bot stopped when a
session starts.

The market is `stops_samples`': SBIN on NSE on Monday 2026-11-16 from 14:40:05 IST, 10 units
every 10 s at 500.00. `StopHold` buys 10 SBIN at 14:42:01 with an SL-M stop at 497.00.
Expected values come from the scenario and the fake's books.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import ist
from stops_samples import StopHold, stop_world

from openalgo.backtest._live._types import LOGGER_NAME, Mode
from openalgo.backtest.errors import PreflightError


@pytest.fixture
def caplogged(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return caplog


def test_F47_a_live_session_with_the_telegram_bot_stopped_fails_its_preflight(
    make_fake_openalgo, monkeypatch, tmp_path, caplogged
):
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.fake.set_telegram_bot(False)
    with pytest.raises(PreflightError) as raised:
        world.run(StopHold)
    [failure] = [text for text in raised.value.failures if "alert" in text]
    assert "a LIVE session needs its alerts" in failure
    assert world.fake.log.calls("placeorder") == []
    assert world.fake.telegram_sent == []


def test_F47_a_sandbox_session_with_the_telegram_bot_stopped_runs_and_counts_its_alerts(
    make_fake_openalgo, monkeypatch, tmp_path, caplogged
):
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path, analyzer=True)
    world.fake.set_telegram_bot(False)
    world.stop_at(ist(14, 44))
    outcome = world.run(StopHold, on_stop="keep")
    assert outcome.mode == Mode.SANDBOX.value
    assert any("this SANDBOX session runs without alerts" in w for w in outcome.warnings)
    assert [o.price_type for o in world.fake.orders(Mode.SANDBOX)] == ["MARKET", "SL-M"]
    assert outcome.summary.undelivered_alerts > 0
    assert world.fake.telegram_sent == []
