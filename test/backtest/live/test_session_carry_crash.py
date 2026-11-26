"""A session that crashed before its day-close record, continued on a later trading day from its
last save and the journal's updates newer than it, on the fake OpenAlgo (20-positional-live 4.2
step 2; the lead's rulings N5c-2 and N5c-3 (a), (d), (e) and (h)), in sandbox and broker
modes and on both GTT book shapes.

The days and the strategies are `carry_samples`' and `crash_samples`': on Monday a strategy
buys 10 SBIN CNC at 500.00 at 14:42:01 IST with its stop GTT at 497.00 (child limit 492.00);
`CncKeptTrail` also trails it 3 points below the best price, with a target at 506.00. The
session crashes at 14:46: its last save is the 14:44 slice's release, at 14:45:01, and nothing
is journaled after it. A later day's session starts at 14:40:05 IST.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3

import pytest
from carry_samples import (
    DAY1,
    DAY2,
    DAY3,
    DAY4,
    CncKept,
    CncKeptTrail,
    carried_world,
    held,
    journal_of,
    missed_day,
    on,
    to_day,
)
from crash_samples import GTT_CALLS, SENDS, crashed_world, rebuilt_line
from fake_openalgo import BROKER
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import BOOKS, GTT_LIMIT, MODES, calls, gtt_world, gtts, sells
from restart_samples import crash_at, crashing, run_to_crash
from session_samples import StopFlag, alerts
from stops_samples import at_price

from openalgo.backtest._live._carry_rebuild import REBUILT_KEY
from openalgo.backtest._live._carry_start import BINDINGS_KEY, CONFIRMED_KEY
from openalgo.backtest._live._result_build import session_result
from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest._live.journal import JOURNAL_FILE_NAME

#: Tuesday's and Thursday's session starts.
DAY2_START, DAY4_START = on(DAY2, 14, 40, 5), on(DAY4, 14, 40, 5)
#: Tuesday's trades through the stop GTT's trigger, with no session running.
THROUGH_THE_STOP = (((10, 0), 495.0), ((10, 1), 495.0))
BOTH_FLAGS = ("price_unverified", "missed_day")


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


def continued_line(name, day, held_words):
    """The banner's line of a leg continued from a record of `day`."""
    return f"{name}: continued from its day-close record of {day}: {held_words}"


def stop_gtt_words(trigger_id):
    """The banner's words of the 10 units held with their stop GTT at 497.00 confirmed."""
    return (
        f"10 SBIN NSE, stop GTT {trigger_id} confirmed at the broker, trigger 497.00 for 10 units"
    )


# (a) ------------------------------------------------------------------------------------------


@MODES
@BOOKS
def test_a_crash_on_the_entry_day_is_continued_with_its_stop_gtt_trailed_and_exited_on_target(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    world = crashed_world(
        make_fake_openalgo, monkeypatch, tmp_path, CncKeptTrail, history=history, analyzer=analyzer
    )
    to_day(world, DAY2)
    # SBIN at 503.00 from 14:41 moves the trail's stop to 500.00; 507.00 from 14:44 passes
    # the 506.00 target.
    at_price(world, on(DAY2, 14, 41), 503.0)
    at_price(world, on(DAY2, 14, 44), 507.0)

    outcome = world.run(CncKeptTrail)

    assert outcome.stopped_by == "session_end"
    assert rebuilt_line("CncKeptTrail", DAY1, "14:45:01", "14:45:01", 0) in world.lines
    later = [c.what for c in calls(world, *GTT_CALLS) if c.ts_ns >= DAY2_START]
    # Taken back, never placed a second time: trailed once, then cancelled before the exit.
    assert later == ["modifygttorder", "cancelgttorder"]
    assert [round(g.trigger, 2) for g in world.fake.gtts(world.fake.mode)] == [500.0]
    assert sells(world) == [("MARKET", 10, "complete")]
    assert held(world) == 0
    (trade,) = session_result(outcome).results["CncKeptTrail"].trades.itertuples()
    assert (trade.entry_time.date(), trade.exit_time.date()) == (DAY1, DAY2)
    assert (trade.exit_reason, trade.exit_price_rupees) == ("leg_target", 507.0)
    assert not alerts(caplog, "error")


def test_the_rebuilt_record_is_the_crashed_days_record_marked_rebuilt(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = crashed_world(make_fake_openalgo, monkeypatch, tmp_path, CncKept)
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts(world.fake.mode)]
    to_day(world, DAY2)
    world.stop_at(on(DAY2, 14, 41))

    world.run(CncKept)

    record = record_of_day(world, "CncKept", DAY1)
    assert record[REBUILT_KEY] == {
        "stopped_ns": on(DAY1, 14, 45, 1),
        "saved_ns": on(DAY1, 14, 45, 1),
        "updates": 0,
        "named": [],
    }
    ((position,),) = (record["positions"],)
    assert position["quantity_units"] == 10
    with journal_of(world) as journal:
        bound = [journal.gtt(row["seq"]).trigger_id for row in record[BINDINGS_KEY]]
    assert bound == [trigger_id]
    # Its last reconciliation read before the save (10 units, settled) came after a read of
    # the GTT book had shown the GTT resting: it confirms the GTT, as a close's read would.
    assert record[CONFIRMED_KEY] == [int(record["gtts"][0]["gtt"])]


def record_of_day(world, strategy, day):
    """`strategy`'s day-close record of `day` in the fake's ledger, read from the journal's
    file (a later record of the strategy does not replace it)."""
    path = world.folder / "live" / BROKER / JOURNAL_FILE_NAME
    with contextlib.closing(sqlite3.connect(path)) as connection:
        row = connection.execute(
            "SELECT carried FROM day_closes WHERE ledger = ? AND strategy = ? AND day = ?",
            (world.fake.mode.value, strategy, day.isoformat()),
        ).fetchone()
    return json.loads(row[0])


# (d) ------------------------------------------------------------------------------------------


@MODES
@BOOKS
def test_a_crash_then_a_stop_gtt_fired_on_a_missed_day_is_recorded_at_its_limit_with_both_flags(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    world = crashed_world(
        make_fake_openalgo, monkeypatch, tmp_path, CncKept, history=history, analyzer=analyzer
    )
    world.trace.waive("books agree", "the stop GTT fired while no session ran")
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts(world.fake.mode)]
    # No session on Tuesday or Wednesday; SBIN falls through the trigger on Tuesday.
    missed_day(world, DAY2, THROUGH_THE_STOP)
    missed_day(world, DAY3, ())
    assert held(world) == 0
    to_day(world, DAY4)
    world.stop_at(on(DAY4, 14, 50))

    outcome = world.run(CncKept)

    assert [c for c in calls(world, *SENDS) if c.ts_ns >= DAY4_START] == []
    (fill,) = outcome.fills
    assert (fill.action, fill.units, fill.price, fill.flags) == ("SELL", 10, GTT_LIMIT, BOTH_FLAGS)
    mode = world.fake.mode.name
    assert alerts(caplog, "warn") == [
        f"{mode} (CncKept): CncKept: stop GTT {trigger_id} of SBIN NSE fired on a trading day "
        f"no session ran: its 10 units are recorded sold at the child's limit, {GTT_LIMIT:.2f}, "
        "flagged price_unverified, missed_day; the contract note holds the true price"
    ]
    assert rebuilt_line("CncKept", DAY1, "14:45:01", "14:45:01", 0) in world.lines
    assert (
        f"CncKept: no session ran on {DAY2}, {DAY3}, since its day-close record of {DAY1}: a "
        "stop GTT that fired then is recorded at its child's limit, flagged price_unverified "
        "and missed_day, and targets reached then were missed"
    ) in world.lines
    assert not alerts(caplog, "error")


# (e) ------------------------------------------------------------------------------------------


@MODES
@BOOKS
def test_a_crash_on_a_day_that_continued_a_record_is_continued_again_the_next_day(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    world = carried_world(
        make_fake_openalgo, monkeypatch, tmp_path, CncKept, history=history, analyzer=analyzer
    )
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts(world.fake.mode)]
    world.trace.waive("protected", "the session crashed: its windows end with it")
    crashing(monkeypatch)
    to_day(world, DAY2)
    crash_at(world, on(DAY2, 14, 46))
    run_to_crash(world, CncKept)
    world.stop, world.feed = StopFlag(), None
    to_day(world, DAY3)
    world.stop_at(on(DAY3, 14, 46))

    outcome = world.run(CncKept)

    assert outcome.stopped_by == "ctrl_c"
    assert rebuilt_line("CncKept", DAY2, "14:45:01", "14:45:01", 0) in world.lines
    assert continued_line("CncKept", DAY2, stop_gtt_words(trigger_id)) in world.lines
    # One stop GTT all along, placed on Monday: nothing placed again on Tuesday or Wednesday.
    assert len(calls(world, "placegttorder")) == 1
    assert gtts(world) == [("active", 10, 497.0, "SELL")]
    (position,) = outcome.open_positions
    assert (position.units, position.stop_confirmed) == (10, True)
    assert not alerts(caplog, "error")


# (h) ------------------------------------------------------------------------------------------


@MODES
def test_a_kept_ctrl_c_then_a_restart_that_crashed_the_same_day_is_continued_the_next_day(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer
):
    world = gtt_world(
        make_fake_openalgo, monkeypatch, tmp_path, trades_until=(15, 32), analyzer=analyzer
    )
    world.trace.waive("protected", "the session crashed: its windows end with it")
    world.trace.waive("books agree", "a restart: the earlier start's reads are its own")
    world.stop_at(on(DAY1, 14, 44))
    assert world.run(CncKept, on_stop="keep").stopped_by == "ctrl_c"
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts(world.fake.mode)]
    # The restart that day resumes the leg, then crashes at 14:50: its last save is the
    # 14:48 slice's release, at 14:49:01.
    crashing(monkeypatch)
    crash_at(world, on(DAY1, 14, 50))
    world.stop, world.feed = StopFlag(), None
    run_to_crash(world, CncKept)
    world.day, world.stop, world.feed = DAY1, StopFlag(), None
    to_day(world, DAY2)
    world.stop_at(on(DAY2, 14, 46))

    outcome = world.run(CncKept)

    assert outcome.stopped_by == "ctrl_c"
    assert rebuilt_line("CncKept", DAY1, "14:49:01", "14:49:01", 0) in world.lines
    assert continued_line("CncKept", DAY1, stop_gtt_words(trigger_id)) in world.lines
    assert len(calls(world, "placegttorder")) == 1
    assert gtts(world) == [("active", 10, 497.0, "SELL")]
    assert not alerts(caplog, "error")
