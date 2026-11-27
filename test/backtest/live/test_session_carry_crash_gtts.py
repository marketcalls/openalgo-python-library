"""A session that crashed before its day-close record with one of its GTTs between two of the
journal's words, continued on a later trading day, on the fake OpenAlgo (20-positional-live
3.2, 3.3 and 4.2 step 2; the lead's ruling N5c-2; review round 1 of unit N5c, findings 1 and
7), in sandbox and broker modes and on both GTT book shapes.

The days and the strategies are `carry_samples`' and `crash_samples`': on Monday `CncKept`
buys 10 SBIN CNC at 500.00 at 14:42:01 IST with its stop GTT at 497.00 (child limit 492.00).
Tuesday's session starts at 14:40:05 IST, SBIN at 500.00, and is stopped at 14:46.
"""

from __future__ import annotations

import logging

import pytest
from carry_samples import (
    DAY1,
    DAY2,
    DAY3,
    CncKept,
    CncKeptTrail,
    carried_world,
    held,
    journal_of,
    on,
    to_day,
)
from crash_samples import GTT_CALLS, SENDS, crashed_world, rebuilt_line
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import BOOKS, MODES, calls, gtt_world, gtts
from restart_samples import crash_at, crashing, run_to_crash
from session_samples import NS, StopFlag, alerts
from stops_samples import at_price
from test_session_carry_crash_more import CncLimitEntry, journal_the_fill
from test_session_carry_settle_place import tuesday_until_a_replace_crashes

from openalgo.backtest._live._journal_gtt_rows import GttState
from openalgo.backtest._live._result_build import session_result
from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest._live._worker_gtts import WorkerGtts

DAY2_START, DAY3_START = on(DAY2, 14, 40, 5), on(DAY3, 14, 40, 5)


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


def fired_with_its_child_unbound(
    make_fake_openalgo, monkeypatch, tmp_path, *, child_fills=False, **options
):
    """Monday's session whose stop GTT fires on a gap to 490.00 at 14:44:40, through its
    trigger and its child's 492.00 limit, so the child (a SELL LIMIT) does not fill. The
    session takes the fire and saves at 14:45:01, and the process dies at 14:45:02, before
    the order worker binds the child in the journal (its bind is made a no-op): the save
    holds the GTT fired (Triggered), the journal's GTT says triggered. With `child_fills`,
    SBIN trades at 493.00 after the crash, which fills the child. The exchange's close
    cancels it otherwise. SBIN trades at 500.00 again from Tuesday's start."""
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path, trades_until=(15, 32), **options)
    world.trace.waive("protected", "the session crashed: its windows end with it")
    world.trace.waive("no orphan", "the session crashed with the fired GTT's child working")
    world.trace.waive("books agree", "the session crashed before the child was bound")
    crashing(monkeypatch)
    monkeypatch.setattr(WorkerGtts, "_bind_child", lambda self, request: None)
    at_price(world, on(DAY1, 14, 44, 40), 490.0)
    crash_at(world, on(DAY1, 14, 45, 2))
    run_to_crash(world, CncKept)
    if child_fills:
        world.fake.trade("SBIN", "NSE", 493.0, volume=10.0)
    client = world.fake.client()
    for order in world.fake.orders(world.fake.mode):
        if order.status == "open":
            client.cancelorder(order_id=order.order_id, strategy=order.strategy)
    at_price(world, DAY2_START, 500.0)
    world.day, world.stop, world.feed = DAY1, StopFlag(), None
    return world


@MODES
@BOOKS
def test_a_stop_gtt_fired_with_its_child_unbound_at_the_crash_is_carried_and_placed_again_once(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    world = fired_with_its_child_unbound(
        make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer, history=history
    )
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts(world.fake.mode)]
    with journal_of(world) as journal:
        assert [(g.seq, g.state) for g in journal.open_gtts()] == [(1, GttState.TRIGGERED)]
    assert held(world) == 10
    to_day(world, DAY2)
    world.stop_at(on(DAY2, 14, 46))

    outcome = world.run(CncKept)

    assert outcome.stopped_by == "ctrl_c"
    assert rebuilt_line("CncKept", DAY1, "14:45:01", "14:45:01", 0) in world.lines
    assert (
        f"CncKept: its stop GTT {trigger_id} fired before the session stopped, its child order "
        "not recorded: the position is carried without that GTT, and this start's read of "
        "the books judges what the child filled"
    ) in world.lines
    # The 10 units the books show held get one stop GTT, placed once; no order is sent.
    later = [c for c in calls(world, *SENDS) if c.ts_ns >= DAY2_START]
    assert [(c.what, c.detail["request"]["quantity"]) for c in later] == [("placegttorder", 10)]
    assert [g for g in gtts(world) if g[0] == "active"] == [("active", 10, 497.0, "SELL")]
    assert held(world) == 10
    with journal_of(world) as journal:
        (fired,) = [state for state in journal.gtt_states(1) if state[0] is GttState.FIRED]
        assert [g.seq for g in journal.open_gtts()] == [2]
    assert fired[1] == (
        "fired before the session stopped: the day-close record rebuilt after the stop carries "
        "the position without it"
    )
    assert not alerts(caplog, "error")


@MODES
@BOOKS
def test_a_stop_gtt_fired_with_its_child_unbound_that_filled_after_the_crash_locks_entries(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    # The child filled before the exchange's close: the broker holds none of the 10 units
    # the rebuilt record carries, a reconciliation mismatch at the start: entries locked,
    # nothing exited, no stop GTT placed for units the account does not hold.
    world = fired_with_its_child_unbound(
        make_fake_openalgo,
        monkeypatch,
        tmp_path,
        child_fills=True,
        analyzer=analyzer,
        history=history,
    )
    world.trace.waive("isolation", "the child of the fired GTT filled with no session running")
    assert held(world) == 0
    to_day(world, DAY2)
    world.stop_at(on(DAY2, 14, 46))

    outcome = world.run(CncKept)

    assert [c for c in calls(world, *GTT_CALLS, *SENDS) if c.ts_ns >= DAY2_START] == []
    assert any(lock.reason == "reconciliation" for lock in outcome.locks)
    assert held(world) == 0


@MODES
@BOOKS
def test_a_stop_gtt_found_fired_with_its_child_missed_after_the_save_is_recorded_at_its_limit(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    # Monday's session crashes at 14:46 (its last save at 14:45:01, the GTT resting in
    # it). Its follower had found the stop GTT fired, the leg's 10 units sold with no child
    # order found, and journaled it at 14:50 (as `_gtt_outcomes.missed` does), before the
    # loop took it. The account also holds 50 SBIN of the user's own, so the books never
    # show the leg closed. Tuesday's rebuild applies the fire and its 10 units at the
    # child's limit, 492.00, flagged price_unverified: the record carries nothing, and
    # nothing is sent.
    world = crashed_world(
        make_fake_openalgo, monkeypatch, tmp_path, CncKept, analyzer=analyzer, history=history
    )
    world.trace.waive("books agree", "the stop GTT fired after the crash")
    world.trace.waive("isolation", "the fired GTT's sale is another client's order here")
    (trigger_id,) = [gtt.trigger_id for gtt in world.fake.gtts(world.fake.mode)]
    world.script.sleep((on(DAY1, 14, 50) - world.fake.clock.now()) / NS)
    world.fake.set_gtt_status(trigger_id, "triggered")
    world.fake.client().placeorder(
        strategy="manual",
        symbol="SBIN",
        action="SELL",
        exchange="NSE",
        price_type="MARKET",
        product="CNC",
        quantity=10,
    )
    world.fake.set_holding("SBIN", "NSE", 50, average_price=480.0)
    assert held(world) == 50
    with journal_of(world) as journal:
        journal.set_gtt_state(1, GttState.TRIGGERED, note="gone from the GTT book")
        journal.set_gtt_state(1, GttState.FIRED, note="fired, its child missed: 10 units at 492.00")
    to_day(world, DAY2)
    world.stop_at(on(DAY2, 14, 46))

    outcome = world.run(CncKept)

    assert outcome.stopped_by == "ctrl_c"
    assert rebuilt_line("CncKept", DAY1, "14:50:00", "14:45:01", 2) in world.lines
    assert (
        f"CncKept: its stop GTT {trigger_id}: it fired after the save with its child order not "
        "found: its 10 units are recorded at its limit, 492.00, flagged price_unverified"
    ) in world.lines
    assert [c for c in calls(world, *SENDS) if c.ts_ns >= DAY2_START] == []
    assert outcome.open_positions == ()
    assert held(world) == 50
    assert not alerts(caplog, "error")


@MODES
@BOOKS
def test_a_crash_in_a_replaces_place_continued_the_next_day_places_the_replacement_once(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    # Tuesday's trail moves the stop to 500.00; the modify times out, so the GTT is
    # cancelled and placed again on the modify's terms, and the process dies inside that
    # place, before it reaches OpenAlgo: the save holds the GTT's modification waiting, the
    # journal its replacement's place under way. Wednesday's rebuild carries it on the terms
    # asked, and the start settles the place by the GTT book: none of its shape rests and
    # the leg's units have not moved, so it is placed once.
    world = carried_world(
        make_fake_openalgo, monkeypatch, tmp_path, CncKeptTrail, history=history, analyzer=analyzer
    )
    tuesday_until_a_replace_crashes(world, monkeypatch)
    world.day, world.stop, world.feed = DAY2, StopFlag(), None
    to_day(world, DAY3)
    world.stop_at(on(DAY3, 14, 46))

    outcome = world.run(CncKeptTrail)

    assert outcome.stopped_by == "ctrl_c"
    assert (
        "CncKeptTrail: its stop GTT g1: the place that replaces it after its cancel was under "
        "way when the session stopped"
    ) in world.lines
    later = [c for c in calls(world, *GTT_CALLS) if c.ts_ns >= DAY3_START]
    assert [(c.what, c.detail["request"]["triggerprice_sl"]) for c in later] == [
        ("placegttorder", 500.0)
    ]
    assert [g for g in gtts(world) if g[0] == "active"] == [("active", 10, 500.0, "SELL")]
    (position,) = outcome.open_positions
    assert (position.units, position.stop_confirmed) == (10, True)
    assert held(world) == 10
    assert not alerts(caplog, "error")


@MODES
def test_a_leg_carried_without_its_stop_gtt_and_held_back_by_a_mismatch_is_counted_unprotected(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer
):
    # Monday's LIMIT entry fills after the 14:45:01 save and its fill is journaled at
    # 14:45:30; the process dies at 14:46 before its stop GTT is placed. 4 of the 10 units
    # are sold by hand. Tuesday's start carries the 10 units without a stop GTT, finds the
    # books at 6 (a mismatch: entries locked, nothing exited) and holds the stop GTT back:
    # the leg is unprotected with its stop held from the start, and the warning says why,
    # not that it was opened with no stop.
    world = gtt_world(
        make_fake_openalgo, monkeypatch, tmp_path, trades_until=(14, 47), analyzer=analyzer
    )
    world.trace.waive("protected", "the session crashed: its windows end with it")
    world.trace.waive("no orphan", "the session crashed with its entry working")
    world.trace.waive("books agree", "4 units sold by hand after the crash")
    crashing(monkeypatch)
    crash_at(world, on(DAY1, 14, 46))
    run_to_crash(world, CncLimitEntry)
    world.fake.trade("SBIN", "NSE", 494.0, volume=10.0)
    (entry,) = [o for o in world.fake.orders(world.fake.mode) if o.action == "BUY"]
    journal_the_fill(world, entry, entry.value / entry.filled, on(DAY1, 14, 45, 30))
    world.fake.client().placeorder(
        strategy="manual",
        symbol="SBIN",
        action="SELL",
        exchange="NSE",
        price_type="MARKET",
        product="CNC",
        quantity=4,
    )
    world.day, world.stop, world.feed = DAY1, StopFlag(), None
    to_day(world, DAY2)
    world.stop_at(on(DAY2, 14, 46))

    outcome = world.run(CncLimitEntry)

    assert [c for c in calls(world, *SENDS) if c.ts_ns >= DAY2_START] == []
    assert held(world) == 6
    (window,) = session_result(outcome).unprotected.itertuples()
    assert (window.units, window.reason, window.from_ns) == (10, "stop held", DAY2_START)
    assert (
        "CncLimitEntry: its position of 10 SBIN NSE has no stop at the broker: its stop GTT "
        "is held back while OpenAlgo's books show another position (a reconciliation mismatch)"
    ) in outcome.warnings
    assert not any("opened with no stop at the broker" in text for text in outcome.warnings)
    assert any(lock.reason == "reconciliation" for lock in outcome.locks)
