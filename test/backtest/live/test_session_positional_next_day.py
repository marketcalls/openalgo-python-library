"""A CNC leg carried to the next trading day, on the fake OpenAlgo (20-positional-live 2 and
4.2; the lead's ruling N5a-1): the next day's start continues it from its day-close record
instead of refusing it (the refusal of unit N3b is retired), with its stop GTT taken back,
never placed a second time; the session then trails the stop GTT and exits on the target as
on the day of the entry.

The days and the strategies are `carry_samples`': on Monday `CncKeptTrail` buys 10 SBIN CNC
at 500.00 at 14:42:01 IST with its stop GTT at 497.00 (3 points below), a trail 3 points
below the best price and a target at 506.00, and the leg is carried at NSE's close. On
Tuesday the session starts at 14:40:05 IST; SBIN trades at 500.00, at 503.00 from 14:41
(the trail moves the stop to 500.00) and at 507.00 from 14:44 (past the target).
"""

from __future__ import annotations

import datetime as dt
import logging

import pytest
from carry_samples import (
    DAY1,
    DAY2,
    CncKept,
    CncKeptTrail,
    carried_world,
    held,
    journal_of,
    on,
    to_day,
)
from fake_openalgo import BROKER
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import BOOKS, MODES, calls, gtt_world, gtts, sells
from restart_samples import crash_at, crashing, restart, run_to_crash
from session_samples import alerts
from stops_samples import at_price

from openalgo.backtest._live._journal_gtt_rows import GttTerms
from openalgo.backtest._live._result_build import session_result
from openalgo.backtest._live._types import LOGGER_NAME, Mode, Purpose
from openalgo.backtest.errors import PreflightError

#: Tuesday's session start.
DAY2_START = on(DAY2, 14, 40, 5)
#: The calls that place, change or cancel a GTT.
GTT_CALLS = ("placegttorder", "modifygttorder", "cancelgttorder")


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


def day2(world):
    """Tuesday's market (module docstring) and its session to NSE's close."""
    to_day(world, DAY2)
    at_price(world, on(DAY2, 14, 41), 503.0)
    at_price(world, on(DAY2, 14, 44), 507.0)
    return world.run(CncKeptTrail)


@MODES
@BOOKS
def test_a_carried_leg_continues_with_its_stop_gtt_trailed_and_exited_on_its_target(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    world = carried_world(
        make_fake_openalgo, monkeypatch, tmp_path, CncKeptTrail, history=history, analyzer=analyzer
    )
    assert gtts(world) == [("active", 10, 497.0, "SELL")]

    outcome = day2(world)

    assert outcome.stopped_by == "session_end"
    later = [c for c in calls(world, *GTT_CALLS) if c.ts_ns >= DAY2_START]
    # Taken back, never placed again: trailed once, then cancelled before the target's exit.
    assert [c.what for c in later] == ["modifygttorder", "cancelgttorder"]
    assert [round(g.trigger, 2) for g in world.fake.gtts(world.fake.mode)] == [500.0]
    assert gtts(world)[0][0] == "cancelled"
    assert sells(world) == [("MARKET", 10, "complete")]
    assert held(world) == 0
    (trade,) = session_result(outcome).results["CncKeptTrail"].trades.itertuples()
    assert (trade.entry_time.date(), trade.exit_time.date()) == (DAY1, DAY2)
    assert not alerts(caplog, "error")


def test_a_restart_on_the_next_day_takes_the_carried_stop_gtt_back_by_its_saved_binding(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    # The day's saved state holds the carried GTT under the day's kernel id, which no
    # journal GTT of the day has: its saved binding names the journal GTT of Monday.
    world = carried_world(make_fake_openalgo, monkeypatch, tmp_path, CncKeptTrail)
    world.trace.waive("protected", "a restart: the earlier start's windows are its own")
    world.trace.waive("books agree", "a restart: the earlier start's reads are its own")
    crashing(monkeypatch)
    to_day(world, DAY2)
    at_price(world, on(DAY2, 14, 41), 503.0)
    at_price(world, on(DAY2, 14, 50), 507.0)
    crash_at(world, on(DAY2, 14, 45))
    run_to_crash(world, CncKeptTrail)

    outcome = restart(world, CncKeptTrail)

    assert outcome.stopped_by == "session_end"
    later = [c for c in calls(world, *GTT_CALLS) if c.ts_ns >= DAY2_START]
    assert [c.what for c in later] == ["modifygttorder", "cancelgttorder"]
    assert sells(world) == [("MARKET", 10, "complete")]
    assert held(world) == 0
    assert not alerts(caplog, "error")


def test_a_capital_other_than_the_carried_positions_is_refused_naming_the_one_to_pass(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = carried_world(make_fake_openalgo, monkeypatch, tmp_path)
    to_day(world, DAY2)

    with pytest.raises(PreflightError) as refused:
        world.run(CncKept, capital=200_000)

    assert refused.value.failures == (
        f"CncKept: what it carries from {DAY1} was entered with capital=100000, and this "
        "start passes capital=200000: the engine continues a carried position only with the "
        "capital it was entered with, so start it with capital=100000",
    )
    assert [c for c in calls(world, *GTT_CALLS, "placeorder") if c.ts_ns >= DAY2_START] == []


def test_a_strategy_whose_version_changed_is_carried_with_the_banner_saying_so(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = carried_world(make_fake_openalgo, monkeypatch, tmp_path)
    (monday,) = world.first.summary.versions
    to_day(world, DAY2)
    world.stop_at(on(DAY2, 14, 46))

    outcome = world.run(CncKept, params={"qty": 20})

    (tuesday,) = outcome.summary.versions
    assert monday != tuesday
    assert (
        f"CncKept: its version changed since its day-close record of {DAY1}, from {monday} to "
        f"{tuesday}: the position it carries keeps the levels and rules saved with it, and "
        "new parameters apply to new entries"
    ) in world.lines
    # Carried with its saved stop: nothing more is bought or placed.
    assert [c for c in calls(world, *GTT_CALLS, "placeorder") if c.ts_ns >= DAY2_START] == []
    assert held(world) == 10


#: What a start refused for what a strategy holds from an earlier day asks of the operator.
CHECK_BY_HAND = (
    "check the position and its GTTs in OpenAlgo, close or cancel them by hand, then start again"
)


def test_a_session_that_crashed_before_its_close_record_refuses_the_next_days_start(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # Continuing from the last save of an earlier day is not built (the report's departure
    # from the lead's ruling N5a-1 (5)): the next day's start is refused, nothing sent, the
    # stop GTT resting at the broker.
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path, trades_until=(15, 32))
    world.trace.waive("protected", "the session crashed: its windows end with it")
    crashing(monkeypatch)
    crash_at(world, on(DAY1, 14, 46))
    run_to_crash(world, CncKept)
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts()]
    world.day = DAY1
    to_day(world, DAY2)

    with pytest.raises(PreflightError) as refused:
        world.run(CncKept)

    assert refused.value.failures == (
        f"CncKept: its session of {DAY1} stopped without its day-close record (the process "
        "ended before the close) while holding a position or a GTT, which this version "
        f"continues only from a day-close record; its stop GTT {trigger_id} of {DAY1} on "
        f"NSE:SBIN (CNC) has not ended and no day-close record carries it: {CHECK_BY_HAND}",
    )
    assert [c for c in calls(world, *GTT_CALLS, "placeorder") if c.ts_ns >= DAY2_START] == []
    assert gtts(world) == [("active", 10, 497.0, "SELL")]


def test_a_gtt_of_an_earlier_day_no_record_carries_refuses_the_strategy(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path)
    earlier = DAY1 - dt.timedelta(days=3)
    with journal_of(world) as journal:
        journal.start_session(earlier, Mode.LIVE, BROKER)
        terms = GttTerms("BUY", 490.0, 490.0, 5)
        seq = journal.record_gtt(
            "CncKept", 1, 1, "SBIN", "NSE", "CNC", terms, purpose=Purpose.ENTRY
        )
        journal.bind_gtt(seq, "261113000001")

    with pytest.raises(PreflightError) as refused:
        world.run(CncKept)

    assert refused.value.failures == (
        f"CncKept: its GTT order 261113000001 of {earlier} on NSE:SBIN (CNC) has not ended "
        f"and no day-close record carries it: {CHECK_BY_HAND}",
    )
    assert calls(world, *GTT_CALLS, "placeorder") == []
