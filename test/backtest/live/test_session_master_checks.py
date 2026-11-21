"""A live session's start against OpenAlgo's master contract and market holidays
(18-multi-strategy 12.2; unit M11): the expiry of each futures and options contract, and
the package calendar against OpenAlgo's holidays, on the fake OpenAlgo.

The market is `session_samples`' and `stops_samples`': the session starts at 14:40:05 IST
on Monday 2026-11-16; SBIN trades on NSE, and the NIFTY call of 29 December 2026 on NFO,
whose reference tables' expiry is 2026-12-29. The fake's holiday table is the 2026 one
OpenAlgo ships (`fake_openalgo.holidays`), which agrees with the package calendar unless a
test changes it.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from fake_openalgo.holidays import HOLIDAYS_2026
from live_samples import live, seam
from session_samples import BuyAndHold, ist, rig
from stops_samples import OPTION, OptionHold, option_world

from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest.errors import PreflightError


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    made = rig(make_fake_openalgo, monkeypatch, tmp_path)
    seam(made, monkeypatch)
    return made


def events(world, kind: str) -> list[dict]:
    return [event for run in world.trace.runs for event in run.events if event["kind"] == kind]


def test_an_equity_session_reads_the_holidays_once_and_no_expiry(world):
    world.stop_at(ist(14, 41))
    live(world, BuyAndHold)
    calls = [entry.what for entry in world.fake.log.calls()]
    assert calls.count("market/holidays") == 1
    assert calls.count("symbol") == 1


def test_a_later_day_openalgo_does_not_close_is_a_warning_and_an_event(world):
    world.fake.set_holidays(row for row in HOLIDAYS_2026 if row["date"] != "2026-12-25")
    world.stop_at(ist(14, 41))
    result = live(world, BuyAndHold)
    words = (
        "NSE on 2026-12-25: the package calendar is closed, OpenAlgo's holidays opens 09:15 "
        "to 15:30 IST"
    )
    [event] = events(world, "calendar_disagrees")
    assert event["days"] == [words]
    assert any(words in note for note in result.warnings)
    assert any(words in line for line in world.lines)


def test_a_day_openalgo_closes_today_refuses_the_start(world):
    today = {
        "date": "2026-11-16",
        "description": "A closure only OpenAlgo has",
        "holiday_type": "TRADING_HOLIDAY",
        "closed_exchanges": ["NSE"],
        "open_exchanges": [],
    }
    world.fake.set_holidays([*HOLIDAYS_2026, today])
    with pytest.raises(PreflightError, match="disagree about today"):
        live(world, BuyAndHold)
    assert world.fake.orders() == []


@pytest.fixture
def options(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    made = option_world(make_fake_openalgo, monkeypatch, tmp_path)
    seam(made, monkeypatch)
    return made


def test_an_option_whose_master_contract_expiry_disagrees_refuses_the_start(options):
    options.fake.set_master_expiry(OPTION, "NFO", "22-DEC-26")
    with pytest.raises(PreflightError, match="gives the expiry 2026-12-22"):
        live(options, OptionHold)
    assert options.fake.orders() == []


def test_an_option_whose_master_contract_gives_no_expiry_starts_with_a_warning(options):
    options.fake.set_master_expiry(OPTION, "NFO", "")
    options.stop_at(ist(14, 41))
    result = live(options, OptionHold)
    assert any(f"gives NFO:{OPTION} no expiry" in note for note in result.warnings)
