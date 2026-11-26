"""The carry scenarios of 20-positional-live 13.3 over the day boundary, on the fake OpenAlgo
(the lead's ruling N5a-6): the user's own holdings beyond the strategy's units on the next
day (one-sided: the user's units are never sold), holdings short of them (a mismatch:
entries locked, nothing exited), the sandbox's T+1 move of the CNC units from the day's
position book into the holdings (read where the broker shows them), and a gap through the
target at the next day's open (the engine's target exit, its stop GTT cancelled first) and
through the stop (the broker fires the stop GTT, whose child is followed as the exit).
The other carry rows are the scenarios of `test_session_positional_next_day`,
`test_session_carry_missed_day`, `test_session_carry_start` and `test_carry_expired`.

The days and the strategies are `carry_samples`': on Monday `CncKept` buys 10 SBIN CNC at
500.00 at 14:42:01 IST with its stop GTT at 497.00 (child limit 492.00), `CncKeptTrail` the
same with a trail 3 points below and a target at 506.00; the legs are carried at NSE's
close. Tuesday's session starts at 14:40:05 IST (09:00:05 before the open where a test says
so) and is stopped at 14:46 unless a test says otherwise.
"""

from __future__ import annotations

import logging

import pytest
from carry_samples import DAY2, CncKept, CncKeptTrail, carried_world, held, on, to_day
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import calls, gtts, sells
from session_samples import alerts

from openalgo.backtest._live._types import LOGGER_NAME, Mode

DAY2_START = on(DAY2, 14, 40, 5)
SENDS = ("placeorder", "modifyorder", "cancelorder", "placegttorder", "modifygttorder")


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


def tuesday(world, strategy=CncKept, *, stop=(14, 46)):
    to_day(world, DAY2)
    world.stop_at(on(DAY2, *stop))
    return world.run(strategy)


def test_the_users_own_holdings_beyond_the_carried_units_are_never_sold(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    world = carried_world(make_fake_openalgo, monkeypatch, tmp_path, CncKeptTrail)
    world.trace.waive("books agree", "the user's own 50 CNC units are no difference (7.2)")
    to_day(world, DAY2)
    world.fake.set_holding("SBIN", "NSE", 60, average_price=480.0)
    world.script.price = lambda now_ns: 507.0 if now_ns >= on(DAY2, 14, 42) else 500.0

    outcome = world.run(CncKeptTrail)

    # The target's exit sells the strategy's 10 units only; its GTT was for 10.
    assert sells(world) == [("MARKET", 10, "complete")]
    assert held(world) == 50
    assert not alerts(caplog, "reconcile_mismatch")
    assert outcome.open_positions == ()


def test_holdings_short_of_the_carried_units_are_a_mismatch_with_nothing_exited(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    world = carried_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.trace.waive("books agree", "4 of the strategy's units sold by hand overnight")
    world.trace.waive("protected", "its stop GTT is for 10 units, the broker holds 6")
    to_day(world, DAY2)
    world.fake.set_holding("SBIN", "NSE", 6, average_price=500.0)
    world.stop_at(on(DAY2, 14, 46))

    outcome = world.run(CncKept)

    assert [c for c in calls(world, *SENDS, "cancelgttorder") if c.ts_ns >= DAY2_START] == []
    assert alerts(caplog, "reconcile_mismatch")
    assert any(lock.reason == "reconciliation" for lock in outcome.locks)
    assert held(world) == 6
    assert gtts(world) == [("active", 10, 497.0, "SELL")]


def test_the_sandboxs_cnc_units_move_into_the_holdings_and_are_read_there(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    # Monday's 10 units are the day's CNC position; Tuesday they are holdings (T+1), the
    # position book's line empty: the broker's units of the leg are both together.
    world = carried_world(make_fake_openalgo, monkeypatch, tmp_path, analyzer=True)
    assert world.fake.position("SBIN", "NSE", Mode.SANDBOX) == 10

    outcome = tuesday(world)

    assert world.fake.holding("SBIN", "NSE", Mode.SANDBOX) == 10
    assert world.fake.position("SBIN", "NSE", Mode.SANDBOX) == 0
    assert not alerts(caplog, "reconcile_mismatch")
    assert [c for c in calls(world, *SENDS) if c.ts_ns >= DAY2_START] == []
    (position,) = outcome.open_positions
    assert (position.units, position.stop_confirmed) == (10, True)


def test_a_gap_through_the_target_at_the_next_days_open_exits_after_the_stop_gtts_cancel(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    world = carried_world(make_fake_openalgo, monkeypatch, tmp_path, CncKeptTrail)
    to_day(world, DAY2, start=(9, 0, 5))
    opens = on(DAY2, 9, 15)
    world.script.price = lambda now_ns: 507.0 if now_ns >= opens else 500.0
    world.stop_at(on(DAY2, 9, 20))

    world.run(CncKeptTrail)

    (cancel,) = [c for c in calls(world, "cancelgttorder") if c.ts_ns >= opens]
    sold = [
        c
        for c in calls(world, "placeorder")
        if c.ts_ns >= opens and c.detail["request"]["action"] == "SELL"
    ]
    assert [c.ts_ns >= cancel.ts_ns for c in sold] == [True]
    assert sells(world) == [("MARKET", 10, "complete")]
    assert held(world) == 0
    assert not alerts(caplog, "error")


def test_a_gap_through_the_stop_at_the_next_days_open_fires_the_gtt_whose_child_exits(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    # SBIN opens at 495.00, through the 497.00 trigger: the broker fires the stop GTT and
    # its child (a LIMIT sale at 492.00) fills; the engine follows the child as the leg's
    # exit and sends nothing of its own.
    world = carried_world(make_fake_openalgo, monkeypatch, tmp_path)
    to_day(world, DAY2, start=(9, 0, 5))
    opens = on(DAY2, 9, 15)
    world.script.price = lambda now_ns: 495.0 if now_ns >= opens else 500.0
    world.stop_at(on(DAY2, 9, 20))

    outcome = world.run(CncKept)

    assert [c for c in calls(world, *SENDS, "cancelgttorder") if c.ts_ns >= opens] == []
    assert sells(world) == [("LIMIT", 10, "complete")]
    assert held(world) == 0
    (fill,) = [fill for fill in outcome.fills if fill.action == "SELL"]
    assert fill.units == 10 and fill.flags == ()
    assert not alerts(caplog, "error")
