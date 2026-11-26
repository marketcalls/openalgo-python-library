"""A CNC leg under `positional_stops="rearm_daily"` carried to the next day through the
start's restore of its day-close record, on the fake OpenAlgo (design 20-positional-live
sections 4.2 and 6, decision P5; the lead's integration ruling 7b of review round 4 of unit
N5a): its stop is a day stop order, so nothing rests at the broker overnight and the start
places no GTT; the start warns that the leg has no stop until the open and names it in the
banner, the day stop goes at the open and not before, the close warns again (a WARN alert
and a Warning line of the daily summary) and its carried-legs alert never says a stop rests
at the broker, and the funds check counts the leg like any carried position.

The days are `carry_samples`': on Monday 2026-11-16 `CncKept` buys 10 SBIN CNC at 500.00
at 14:42:01 IST with a stop 3 points below (497.00), a day SL-M, and the session ends at
NSE's close with the leg carried. Tuesday's session starts at 09:05 IST, before the 09:15
open, SBIN trading at 500.00, and is stopped (Ctrl+C) at 09:30. Each test runs in the
broker's book and the sandbox's. Expected values come from the scenario.
"""

from __future__ import annotations

import logging

import pytest
from carry_samples import DAY1, DAY2, DAY3, CncKept, held, missed_day, on, to_day
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import MODES, calls, gtt_world
from session_samples import StopFlag, alerts

from openalgo.backtest._live._result_build import session_result
from openalgo.backtest._live._types import LOGGER_NAME

REARM = {"positional_stops": "rearm_daily"}
#: Tuesday's start, before the open, and NSE's open.
START, OPEN = (9, 5, 0), on(DAY2, 9, 15)
#: The design's sentence for rearm_daily (20-positional-live 6), written out.
SENTENCE = (
    "Positions held overnight have no stop at the broker from the close until your next "
    "start; start bt.live before the open on every trading day you hold a position."
)
#: The calls that send anything to the broker.
SENDS = (
    "placeorder",
    "modifyorder",
    "cancelorder",
    "placegttorder",
    "modifygttorder",
    "cancelgttorder",
)


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


def carried_rearm(make_fake_openalgo, monkeypatch, tmp_path, *, analyzer):
    """Monday's session of `CncKept` under rearm_daily run to NSE's close (module
    docstring), then the move to Tuesday's 09:05 start, stopped at 09:30."""
    world = gtt_world(
        make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer, trades_until=(15, 32)
    )
    assert world.run(CncKept, **REARM).stopped_by == "session_end"
    world.day = DAY1
    world.stop, world.feed = StopFlag(), None
    to_day(world, DAY2, start=START)
    world.stop_at(on(DAY2, 9, 30))
    return world


def preflight(caplog, name):
    """The message of the preflight check `name` of the latest start."""
    tag = f"preflight {name}: "
    found = [r.getMessage() for r in caplog.records if r.getMessage().startswith(tag)]
    return found[-1][len(tag) :]


@MODES
def test_a_rearm_daily_leg_restored_on_tuesday_gets_its_day_stop_at_the_open_only(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer
):
    world = carried_rearm(make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer)
    caplog.clear()

    outcome = world.run(CncKept, **REARM)

    tuesday = [c for c in calls(world, *SENDS) if c.ts_ns >= on(DAY2, *START)]
    # No GTT, and nothing before the open: the day stop, an SL-M for the 10 units at
    # 497.00, goes as the session reaches its open.
    assert [c.what for c in tuesday] == ["placeorder"]
    assert tuesday[0].ts_ns >= OPEN
    (stop,) = world.orders()
    assert (stop.action, stop.price_type, stop.quantity, stop.product, stop.trigger_price) == (
        "SELL",
        "SL-M",
        10,
        "CNC",
        497.0,
    )
    assert world.fake.gtts(world.fake.mode) == []
    assert held(world) == 10
    assert outcome.stopped_by == "ctrl_c"
    assert not alerts(caplog, "error")


@MODES
def test_the_start_warns_of_the_leg_without_a_stop_until_the_open_and_names_it_in_the_banner(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer
):
    world = carried_rearm(make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer)

    outcome = world.run(CncKept, **REARM)

    waits = "CncKept 10 SBIN NSE (CNC), stop 497.00: its day stop waits for the open"
    assert f"Warning: {SENTENCE}" in world.lines
    assert f"Warning: {waits}" in world.lines
    assert (
        f"CncKept: continued from its day-close record of {DAY1}: 10 SBIN NSE, stop 497.00: "
        "its day stop waits for the open"
    ) in world.lines
    assert {SENTENCE, waits} <= set(session_result(outcome).warnings)


@MODES
def test_the_close_warns_of_the_carried_leg_and_its_alert_says_no_stop_rests_overnight(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer
):
    world = carried_rearm(make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer)
    caplog.clear()

    outcome = world.run(CncKept, **REARM)

    mode = "SANDBOX" if analyzer else "LIVE"
    line = (
        "CncKept 10 SBIN NSE (CNC), stop 497.00: its day stop works at the broker until the "
        "close, then no stop until your next start"
    )
    assert alerts(caplog, "warn") == [
        f'{mode} (CncKept): positional_stops="rearm_daily": {line}. {SENTENCE}'
    ]
    assert not [text for text in alerts(caplog, "session") if "rearm_daily" in text]
    summary = session_result(outcome).summary_text().splitlines()
    assert f"Warning     : {line}" in summary
    carried = (
        f"CncKept SBIN NSE 10 units entered on {DAY1} at 500.00, stop 497.00: its day stop "
        "works at the broker until the close, then no stop until your next start"
    )
    assert f"Carried     : {carried}" in summary
    assert f"{mode} (CncKept): Carried overnight:\n{carried}" in alerts(caplog, "session")


@MODES
def test_the_funds_check_counts_the_carried_rearm_daily_leg(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer
):
    # By hand: the CNC leg holds its cost, 10 x 500.00 = 5,000.00 rupees; the capital of
    # 1,00,000.00 rupees less it leaves 95,000.00 to find in the cash available: the fake
    # broker's 2,50,000.00, the sandbox's 1,00,00,000.00.
    world = carried_rearm(make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer)

    world.run(CncKept, **REARM)

    available = "1,00,00,000.00" if analyzer else "2,50,000.00"
    assert preflight(caplog, "funds") == (
        f"{available} rupees of available cash cover the capital of 1,00,000.00 rupees less "
        "the 5,000.00 rupees the carried positions hold (CncKept SBIN CNC 5,000.00), "
        "95,000.00 rupees"
    )


@MODES
def test_a_rearm_daily_legs_missed_days_are_named_once_with_no_gtt_words(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer
):
    # Monday's leg is carried; no session runs on Tuesday; Wednesday's start names Tuesday
    # once, in the start warning of rearm_daily legs, and the banner has no line of a stop
    # GTT that fired then: the leg has none (the lead's ruling N5c-4 (g)).
    world = gtt_world(
        make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer, trades_until=(15, 32)
    )
    assert world.run(CncKept, **REARM).stopped_by == "session_end"
    world.day = DAY1
    world.stop, world.feed = StopFlag(), None
    missed_day(world, DAY2, ())
    to_day(world, DAY3, start=START)
    world.stop_at(on(DAY3, 9, 30))

    world.run(CncKept, **REARM)

    named = [line for line in world.lines if str(DAY2) in line]
    assert named == [
        f"Warning: CncKept: no session ran on {DAY2}, the trading days since its last close "
        f"on {DAY1}"
    ]
