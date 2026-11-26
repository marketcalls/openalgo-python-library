"""A same-day restart of a session holding a CNC leg with its stop GTT, on the fake OpenAlgo
(16-session-design 3.6; 20-positional-live 3; the lead's ruling N3b-2 and review finding
1): the restarted strategy's GTT map takes back each GTT the kernel restores, joined to the
journal's GTT by the kernel's GTT id, so the kernel's later GTT commands reach the broker.

The market is `positional_samples`': SBIN trades every 10 s at 500.00 until 15:32 IST
unless a test moves it. `FlatTarget` and `FlatTrail` buy 10 SBIN CNC whenever they hold
none, have no order working and SBIN is below 501.00, with a stop GTT 3 points below the
fill (497.00 for a fill at 500.00) and a target 2 points above it (502.00) or a trail.
`CncTarget` buys 10 at its first decision of each start. `CncKept` and `OptionKept` (one
lot of the NIFTY call NRML, its stop GTT at 110.00) buy once, the entry kept in
`self.state`. A restart past the MIS square-off (NSE 15:00, NFO 15:10) squares off no
CNC or NRML leg: it is carried at the close (the lead's ruling N3b-4, review round 2
finding 1). Expected values come from the fake's books and the scenario.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo import BROKER
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import (
    GTT_TRIGGER,
    MODES,
    CncHold,
    CncTarget,
    calls,
    gtt_world,
    gtts,
    sells,
)
from restart_samples import crash_at, crashing, restart, run_to_crash
from session_samples import NS, alerts, ist
from stops_samples import (
    OPTION,
    OPTION_LOT,
    STOP_POINTS,
    OptionHold,
    at_price,
    events,
    option_world,
)

import openalgo.backtest as bt
from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest._live.journal import JOURNAL_FILE_NAME, GttState, Journal

#: Below it a flat strategy buys again; the target's 503.00 is above it.
BUY_BELOW = 501.0
#: The target's price, past the 502.00 target.
TARGET_PRICE = 503.0
#: A restart the same day past the MIS square-off of NSE (15:00) and of NFO (15:10).
PAST_SQUARE_OFF = ist(15, 15)
#: The option's stop GTT: 10 points below its fill at 120.00.
OPTION_TRIGGER = 110.0


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


class FlatTarget(bt.Strategy):
    """Buys 10 SBIN CNC with a stop 3 points below and a target 2 above whenever it holds
    none, has no order working and the bar closed below `BUY_BELOW`."""

    symbol, exchange, interval, product = "SBIN", "NSE", "1m", "CNC"

    def on_bar(self, bar):
        if not self.position.qty and not self.orders and bar.close < BUY_BELOW:
            self.buy(10, stop=bt.points(STOP_POINTS), target=bt.points(2))


class FlatTrail(bt.Strategy):
    """FlatTarget with a stop that trails 3 points below the best price, and no target."""

    symbol, exchange, interval, product = "SBIN", "NSE", "1m", "CNC"

    def on_bar(self, bar):
        if not self.position.qty and not self.orders and bar.close < BUY_BELOW:
            self.buy(10, stop=bt.points(STOP_POINTS), trail=bt.trail_points(STOP_POINTS))


class CncKept(CncHold):
    """CncHold whose entry is kept in `self.state`: a restart buys nothing more."""

    def on_bar(self, bar):
        if not self.state.get("bought"):
            self.state["bought"] = True
            self.buy(self.qty, **type(self).rules)


class OptionKept(OptionHold):
    """OptionHold in NRML whose entry is kept in `self.state`."""

    product = "NRML"

    def on_bar(self, bar):
        if not self.state.get("bought"):
            self.state["bought"] = True
            self.buy(OPTION_LOT, **type(self).rules)


def world_to_the_close(make_fake_openalgo, monkeypatch, tmp_path, analyzer):
    world = gtt_world(
        make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer, trades_until=(15, 32)
    )
    world.trace.waive("protected", "a restart: the earlier start's windows are its own")
    world.trace.waive("books agree", "a restart: the earlier start's reads are its own")
    return world


def errors_of(caplog):
    return [a for a in alerts(caplog, "error") if "stops on an error" in a]


@MODES
def test_ctrl_c_keeps_the_leg_and_a_restart_cancels_its_stop_gtt_before_the_target_exit(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer
):
    world = world_to_the_close(make_fake_openalgo, monkeypatch, tmp_path, analyzer)
    world.stop_at(ist(14, 46))
    world.run(FlatTarget)
    assert gtts(world) == [("active", 10, GTT_TRIGGER, "SELL")]
    restarted_at = ist(14, 47)
    at_price(world, ist(14, 50), TARGET_PRICE)

    outcome = restart(world, FlatTarget, stop_ns=ist(14, 55))

    assert outcome.stopped_by == "ctrl_c"
    assert errors_of(caplog) == []
    (cancel,) = calls(world, "cancelgttorder")
    sales = [c for c in calls(world, "placeorder") if c.detail["request"]["action"] == "SELL"]
    assert [c.ts_ns > restarted_at for c in sales] == [True]
    assert cancel.ts_ns <= sales[0].ts_ns
    assert gtts(world) == [("cancelled", 10, GTT_TRIGGER, "SELL")]
    assert sells(world) == [("MARKET", 10, "complete")]
    assert world.fake.position("SBIN", "NSE") == 0
    assert len(calls(world, "placegttorder")) == 1


def test_a_restarted_trailing_stop_gtt_is_modified_once_a_minute_at_most(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    world = world_to_the_close(make_fake_openalgo, monkeypatch, tmp_path, False)
    world.stop_at(ist(14, 44))
    world.run(FlatTrail)
    ramp_from, ramp_to = ist(14, 46), ist(14, 48, 30)

    def price(now_ns):
        if now_ns < ramp_from:
            return 500.0
        return 500.0 + 0.5 * ((min(now_ns, ramp_to) - ramp_from) // (10 * NS) + 1)

    world.script.price = price

    outcome = restart(world, FlatTrail, stop_ns=ist(14, 50))

    assert outcome.stopped_by == "ctrl_c"
    assert errors_of(caplog) == []
    modifies = [c.ts_ns for c in calls(world, "modifygttorder")]
    assert modifies
    assert all(b - a >= 60 * NS for a, b in zip(modifies, modifies[1:], strict=False))
    ((status, units, trigger, _),) = gtts(world)
    assert (status, units) == ("active", 10)
    assert trigger > GTT_TRIGGER
    assert len(calls(world, "placegttorder")) == 1


def test_a_restarted_strategy_that_adds_raises_its_stop_gtt_to_the_whole_position(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    world = world_to_the_close(make_fake_openalgo, monkeypatch, tmp_path, False)
    world.stop_at(ist(14, 44))
    world.run(CncTarget)

    outcome = restart(world, CncTarget, stop_ns=ist(14, 48))

    # The restarted CncTarget buys 10 more at its first decision: one stop GTT for both.
    assert outcome.stopped_by == "ctrl_c"
    assert errors_of(caplog) == []
    assert world.fake.position("SBIN", "NSE") == 20
    assert gtts(world) == [("active", 20, GTT_TRIGGER, "SELL")]
    assert len(calls(world, "placegttorder")) == 1


@MODES
def test_a_crash_before_the_loop_took_the_placed_answer_resumes_with_the_gtt_confirmed(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer
):
    # The state was saved while the stop GTT's place waited for its answer; the journal
    # bound it before the crash: the restart confirms it and follows it, never placing it
    # again, and the target then cancels it before its exit.
    world = world_to_the_close(make_fake_openalgo, monkeypatch, tmp_path, analyzer)
    crashing(monkeypatch)
    crash_at(world, ist(14, 42, 30))
    run_to_crash(world, FlatTarget)
    at_price(world, ist(14, 44), TARGET_PRICE)

    outcome = restart(world, FlatTarget, stop_ns=ist(14, 47))

    assert outcome.stopped_by == "ctrl_c"
    assert errors_of(caplog) == []
    assert len(calls(world, "placegttorder")) == 1
    assert len(calls(world, "cancelgttorder")) == 1
    assert sells(world) == [("MARKET", 10, "complete")]
    assert world.fake.position("SBIN", "NSE") == 0


def test_a_gtt_the_journal_cannot_explain_at_a_restart_locks_entries_and_sends_nothing_of_it(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    # The journal says the stop GTT's cancel was sent when the session stopped, the saved
    # state that it rests: the restart cannot tell which holds, so it locks the
    # strategy's entries, alerts, and sends nothing about that GTT (the target's cancel
    # included), exiting nothing.
    world = world_to_the_close(make_fake_openalgo, monkeypatch, tmp_path, False)
    world.stop_at(ist(14, 46))
    world.run(FlatTarget)
    path = world.folder / "live" / BROKER / JOURNAL_FILE_NAME
    with Journal.open(
        path, account=BROKER, broker=BROKER, mode=world.fake.mode, now=world.fake.clock.now
    ) as journal:
        (gtt,) = journal.open_gtts()
        journal.set_gtt_state(gtt.seq, GttState.CANCELLING, note="a test's cancel under way")
    at_price(world, ist(14, 50), TARGET_PRICE)

    outcome = restart(world, FlatTarget, stop_ns=ist(14, 55))

    assert outcome.stopped_by == "ctrl_c"
    assert errors_of(caplog) == []
    (alert,) = [a for a in alerts(caplog, "error") if "cannot be taken back by the restart" in a]
    assert f"stop GTT {gtt.trigger_id}" in alert and "nothing is exited" in alert
    assert ("FlatTarget", "gtt_unresolved") in {
        (lock.strategy, lock.reason) for lock in outcome.locks
    }
    assert calls(world, "cancelgttorder") == []
    assert sells(world) == []
    assert gtts(world) == [("active", 10, GTT_TRIGGER, "SELL")]


def test_a_restart_that_ends_the_day_flat_leaves_no_record_of_the_kept_leg(
    make_fake_openalgo, monkeypatch, tmp_path
):
    world = world_to_the_close(make_fake_openalgo, monkeypatch, tmp_path, False)
    world.stop_at(ist(14, 46))
    world.run(FlatTarget)
    assert len(day_closes(world)["FlatTarget"].carried["positions"]) == 1
    at_price(world, ist(14, 50), TARGET_PRICE)

    outcome = restart(world, FlatTarget)

    assert outcome.stopped_by == "session_end"
    assert world.fake.position("SBIN", "NSE") == 0
    assert not day_closes(world)["FlatTarget"].carried.get("positions")


@MODES
def test_a_restart_the_same_day_locks_entries_again_after_a_refused_stop_gtt(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer
):
    # The broker refuses FlatTarget's first stop GTT: the leg is exited at market and its
    # entries locked for the day (decision 11.7). The restart sets that lock again, so
    # FlatTarget, flat below 501.00, buys nothing more today.
    world = world_to_the_close(make_fake_openalgo, monkeypatch, tmp_path, analyzer)
    world.fake.refuse_next_gtt("Trigger price is too close to the last price")
    world.stop_at(ist(14, 46))
    first = world.run(FlatTarget)
    assert ("FlatTarget", "gtt_refused") in {(lock.strategy, lock.reason) for lock in first.locks}
    assert world.fake.position("SBIN", "NSE") == 0

    outcome = restart(world, FlatTarget, stop_ns=ist(14, 52))

    locked = [lock for lock in outcome.locks if lock.reason == "gtt_refused"]
    assert [lock.lifted_ns for lock in locked] == [None]
    buys = [c for c in calls(world, "placeorder") if c.detail["request"]["action"] == "BUY"]
    assert len(buys) == 1
    assert world.fake.position("SBIN", "NSE") == 0
    assert len(calls(world, "placegttorder")) == 1


def kept_until_past_the_square_off(world, strategy):
    """Run `strategy` until a Ctrl+C at 14:46 keeps its leg, then move the clock on to
    `PAST_SQUARE_OFF`."""
    world.stop_at(ist(14, 46))
    world.run(strategy)
    world.script.sleep((PAST_SQUARE_OFF - world.fake.clock.now()) / NS)


@MODES
def test_a_restart_past_the_mis_square_off_carries_a_cnc_leg_to_the_close(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer
):
    world = world_to_the_close(make_fake_openalgo, monkeypatch, tmp_path, analyzer)
    kept_until_past_the_square_off(world, CncKept)

    outcome = restart(world, CncKept)

    assert outcome.stopped_by == "session_end"
    assert events(outcome, "restart_square_off") == []
    assert [e["settle"] for e in events(outcome, "gtt_restored")] == ["take_back"]
    assert calls(world, "cancelgttorder") == []
    assert sells(world) == []
    assert world.fake.position("SBIN", "NSE") == 10
    assert gtts(world) == [("active", 10, GTT_TRIGGER, "SELL")]
    (position,) = outcome.open_positions
    assert position.stop_confirmed
    assert len(day_closes(world)["CncKept"].carried["positions"]) == 1


@MODES
def test_a_restart_past_the_mis_square_off_carries_an_nrml_leg_to_the_close(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer
):
    world = option_world(
        make_fake_openalgo, monkeypatch, tmp_path, trades_until=(15, 45), analyzer=analyzer
    )
    world.trace.waive("protected", "a restart: the earlier start's windows are its own")
    world.trace.waive("books agree", "a restart: the earlier start's reads are its own")
    kept_until_past_the_square_off(world, OptionKept)

    outcome = restart(world, OptionKept)

    assert outcome.stopped_by == "session_end"
    assert events(outcome, "restart_square_off") == []
    assert calls(world, "cancelgttorder") == []
    assert [o for o in world.orders() if o.action == "SELL"] == []
    assert world.fake.position(OPTION, "NFO") == OPTION_LOT
    assert gtts(world) == [("active", OPTION_LOT, OPTION_TRIGGER, "SELL")]
    assert len(day_closes(world)["OptionKept"].carried["positions"]) == 1


def day_closes(world):
    """The day-close records of the fake's mode's ledger, by strategy."""
    path = world.folder / "live" / BROKER / JOURNAL_FILE_NAME
    with Journal.open(
        path, account=BROKER, broker=BROKER, mode=world.fake.mode, now=world.fake.clock.now
    ) as journal:
        return journal.day_closes()


def test_a_stop_gtt_a_restart_could_not_take_back_is_never_reported_confirmed(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    # The journal says the stop GTT's cancel was sent, the saved state that it rests: the
    # restart cannot tell which holds, so neither the lines nor the result call it
    # confirmed, and the operator is sent to the books first.
    world = world_to_the_close(make_fake_openalgo, monkeypatch, tmp_path, False)
    world.stop_at(ist(14, 46))
    world.run(FlatTarget)
    path = world.folder / "live" / BROKER / JOURNAL_FILE_NAME
    with Journal.open(
        path, account=BROKER, broker=BROKER, mode=world.fake.mode, now=world.fake.clock.now
    ) as journal:
        (gtt,) = journal.open_gtts()
        journal.set_gtt_state(gtt.seq, GttState.CANCELLING, note="a test's cancel under way")

    outcome = restart(world, FlatTarget, stop_ns=ist(14, 55))

    (position,) = outcome.open_positions
    assert position.stop_confirmed is False
    assert position.note.startswith(
        f"stop GTT {gtt.trigger_id} not confirmed at the broker: the restart could not tell "
        "whether it still rests; check the order book and the position"
    )
