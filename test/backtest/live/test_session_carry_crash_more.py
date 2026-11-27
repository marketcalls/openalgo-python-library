"""More sessions that crashed before their day-close record, continued on a later trading day, on
the fake OpenAlgo (20-positional-live 3.1, 4.2 step 2, 6, 7.3 and 8; the lead's rulings N5c-2
and N5c-3 (b), (c), (f), (g) and (i)), in sandbox and broker modes, on both GTT book shapes
where GTTs are involved.

The days and the strategies are `carry_samples`' and `crash_samples`': on Monday a strategy
buys 10 SBIN CNC at 500.00 at 14:42:01 IST (or a LIMIT entry at 495.00 that works) with a
stop 3 points below, held by a SINGLE GTT (497.00, child limit 492.00) or, under
`positional_stops="rearm_daily"`, by a day SL-M. The session crashes; a later day's session
starts at 14:40:05 IST (09:05 where a test says so). Expected values come from the scenario
and the fake's books.
"""

from __future__ import annotations

import datetime as dt
import functools
import logging

import pytest
from carry_samples import (
    DAY1,
    DAY2,
    FUTURE,
    FUTURE_LOT,
    CncKept,
    FutKept,
    held,
    journal_of,
    on,
    to_day,
)
from crash_samples import GTT_CALLS, SENDS, crashed_world, rebuilt_line
from daily_samples import UNITS, close_of, daily_world, open_of
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import BOOKS, MODES, calls, gtt_world, gtts
from restart_samples import Crash, crash_at, crashing, run_to_crash
from session_samples import NS, StopFlag, alerts, ist
from stops_samples import STOP_POINTS

import openalgo.backtest as bt
from openalgo.backtest._live._gtt_calls import GttCalls
from openalgo.backtest._live._types import LOGGER_NAME, OrderStatus, OrderUpdate

DAY2_START = on(DAY2, 14, 40, 5)
#: The days of the daily strategy's scenario (`daily_samples`).
FRIDAY, MONDAY, TUESDAY = dt.date(2026, 11, 13), dt.date(2026, 11, 16), dt.date(2026, 11, 17)
#: When Monday's stop GTT is cancelled by hand, so the engine places it again and the
#: process dies inside that place.
BY_HAND = on(DAY1, 14, 44)


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


class CncLimitEntry(CncKept):
    """CncKept whose entry is a DAY LIMIT buy at 495.00, below SBIN's 500.00."""

    def on_bar(self, bar):
        if not self.state.get("bought"):
            self.state["bought"] = True
            self.buy(self.qty, price=495.0, **type(self).rules)


# (b) ------------------------------------------------------------------------------------------


@MODES
@BOOKS
def test_a_crash_with_the_entrys_fill_journaled_after_the_save_is_carried_with_all_its_units(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    # Monday's LIMIT entry fills at the broker and the order worker journals the fill at
    # 14:45:30, after the 14:45:01 save, before the loop saves again: the process dies at
    # 14:46. Tuesday's start applies the fill (one later update), carries the 10 units, and
    # places the leg's stop GTT, which no session placed: 3 points below the fill's price.
    world = gtt_world(
        make_fake_openalgo,
        monkeypatch,
        tmp_path,
        trades_until=(14, 47),
        analyzer=analyzer,
        history=history,
    )
    world.trace.waive("protected", "the session crashed: its windows end with it")
    world.trace.waive("no orphan", "the session crashed with its entry working")
    world.trace.waive("books agree", "the entry filled while the crashed session waited")
    crashing(monkeypatch)
    crash_at(world, on(DAY1, 14, 46))
    run_to_crash(world, CncLimitEntry)
    world.fake.trade("SBIN", "NSE", 494.0, volume=10.0)
    (entry,) = [o for o in world.fake.orders(world.fake.mode) if o.action == "BUY"]
    price = entry.value / entry.filled
    journal_the_fill(world, entry, price, on(DAY1, 14, 45, 30))
    world.day, world.stop, world.feed = DAY1, StopFlag(), None
    to_day(world, DAY2)
    world.stop_at(on(DAY2, 14, 46))

    outcome = world.run(CncLimitEntry)

    assert rebuilt_line("CncLimitEntry", DAY1, "14:45:30", "14:45:01", 1) in world.lines
    (position,) = outcome.open_positions
    assert position.units == 10
    (placed,) = [c for c in calls(world, *GTT_CALLS) if c.ts_ns >= DAY2_START]
    request = placed.detail["request"]
    assert (placed.what, request["quantity"]) == ("placegttorder", 10)
    assert round(float(request["triggerprice_sl"]), 2) == round(price - 3, 2)
    assert not alerts(caplog, "error")


def journal_the_fill(world, entry, price, at_ns):
    """Journal the broker's word of `entry` filled whole at `price` (rupees), received at
    `at_ns`, as the order worker does."""
    update = OrderUpdate(
        order_id=entry.order_id,
        symbol="SBIN",
        exchange="NSE",
        action="BUY",
        quantity=10,
        filled_quantity=10,
        pending_quantity=0,
        average_price=price,
        status=OrderStatus.COMPLETE,
        rejection_reason="",
        mode=world.fake.mode,
        ts_recv_ns=at_ns,
        source="stream",
    )
    with journal_of(world) as journal:
        journal.record_update(update)


# (c) ------------------------------------------------------------------------------------------


def crash_in_place(*, reached: bool):
    """The process dies inside the next GTT place: after the call reached OpenAlgo
    (`reached`) or before it was sent; later places go as usual."""
    original = GttCalls.place
    armed = [True]

    def place(calls_, request):
        if not armed[0]:
            return original(calls_, request)
        armed[0] = False
        if reached:
            original(calls_, request)
        raise Crash("the process dies inside a GTT place")

    return place


def crashed_in_the_stop_gtts_place(make_fake_openalgo, monkeypatch, tmp_path, *, reached, **kw):
    """Monday's session of `CncKept` whose stop GTT is cancelled by hand at 14:44: the
    engine places it again, and the process dies inside that place (`reached` or not)."""
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path, trades_until=(15, 32), **kw)
    world.trace.waive("protected", "the session crashed: its windows end with it")
    crashing(monkeypatch)
    world.script.at(BY_HAND, lambda: cancel_by_hand(world))
    world.script.at(
        BY_HAND + NS,
        lambda: monkeypatch.setattr(GttCalls, "place", crash_in_place(reached=reached)),
    )
    run_to_crash(world, CncKept)
    world.day, world.stop, world.feed = DAY1, StopFlag(), None
    return world


def cancel_by_hand(world):
    """Another client cancels the stop GTT resting at the broker."""
    (resting,) = [g.trigger_id for g in world.fake.gtts(world.fake.mode) if g.status == "active"]
    world.fake.set_gtt_status(resting, "cancelled")


@MODES
@BOOKS
def test_a_crash_inside_a_stop_gtt_place_that_reached_the_broker_binds_it_placing_none_again(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    world = crashed_in_the_stop_gtts_place(
        make_fake_openalgo, monkeypatch, tmp_path, reached=True, analyzer=analyzer, history=history
    )
    world.trace.waive("books agree", "a restart: the earlier start's reads are its own")
    to_day(world, DAY2)
    world.stop_at(on(DAY2, 14, 46))

    world.run(CncKept)

    assert [c for c in calls(world, *GTT_CALLS) if c.ts_ns >= DAY2_START] == []
    assert [g for g in gtts(world) if g[0] == "active"] == [("active", 10, 497.0, "SELL")]
    assert not alerts(caplog, "error")


@MODES
@BOOKS
def test_a_crash_inside_a_stop_gtt_place_that_never_reached_the_broker_places_it_once(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    world = crashed_in_the_stop_gtts_place(
        make_fake_openalgo, monkeypatch, tmp_path, reached=False, analyzer=analyzer, history=history
    )
    to_day(world, DAY2)
    world.stop_at(on(DAY2, 14, 46))

    world.run(CncKept)

    later = [c for c in calls(world, *GTT_CALLS) if c.ts_ns >= DAY2_START]
    assert [(c.what, c.detail["request"]["quantity"]) for c in later] == [("placegttorder", 10)]
    assert [g for g in gtts(world) if g[0] == "active"] == [("active", 10, 497.0, "SELL")]
    assert not alerts(caplog, "error")


@MODES
@BOOKS
def test_a_crash_inside_a_stop_gtt_place_then_units_sold_by_hand_is_unresolved_placing_nothing(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    # 4 of the 10 units are sold by hand after the crash: the place that never reached the
    # broker cannot be placed again while the leg's units moved by something else.
    world = crashed_in_the_stop_gtts_place(
        make_fake_openalgo, monkeypatch, tmp_path, reached=False, analyzer=analyzer, history=history
    )
    world.trace.waive("books agree", "4 units sold by hand after the crash")
    world.fake.client().placeorder(
        strategy="manual",
        symbol="SBIN",
        action="SELL",
        exchange="NSE",
        price_type="MARKET",
        product="CNC",
        quantity=4,
    )
    to_day(world, DAY2)
    world.stop_at(on(DAY2, 14, 46))

    outcome = world.run(CncKept)

    assert [c for c in calls(world, *SENDS) if c.ts_ns >= DAY2_START] == []
    assert held(world) == 6
    mode = world.fake.mode.name
    assert (
        f"{mode} (CncKept): CncKept: stop GTT place: its place was unanswered and the position "
        "went from 10 to 6 units since, not by its units alone: it may have landed and fired, "
        "so it is not placed again. Its entries are locked and nothing is exited: check the "
        "GTT and order books in OpenAlgo for SBIN NSE, settle it by hand, then start the "
        "session again"
    ) in alerts(caplog, "error")
    held_locks = {lock.reason for lock in outcome.locks if lock.lifted_ns is None}
    assert {"gtt_unresolved", "reconciliation"} <= held_locks


# (g) ------------------------------------------------------------------------------------------


REARM = {"positional_stops": "rearm_daily"}
#: The design's sentence for rearm_daily (20-positional-live 6), written out.
SENTENCE = (
    "Positions held overnight have no stop at the broker from the close until your next "
    "start; start bt.live before the open on every trading day you hold a position."
)


@MODES
def test_a_crashed_rearm_daily_leg_gets_its_day_stop_at_the_next_open_with_both_warnings(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer
):
    world = gtt_world(
        make_fake_openalgo, monkeypatch, tmp_path, trades_until=(15, 32), analyzer=analyzer
    )
    world.trace.waive("protected", "the session crashed: its windows end with it")
    crashing(monkeypatch)
    crash_at(world, on(DAY1, 14, 46))
    run_to_crash(world, CncKept, **REARM)
    # The exchange's close cancels the day stop the crashed session left working.
    client = world.fake.client()
    for order in world.fake.orders(world.fake.mode):
        if order.status == "trigger pending":
            client.cancelorder(order_id=order.order_id, strategy=order.strategy)
    world.day, world.stop, world.feed = DAY1, StopFlag(), None
    to_day(world, DAY2, start=(9, 5, 0))
    world.stop_at(on(DAY2, 15, 35))
    caplog.clear()

    world.run(CncKept, **REARM)

    later = [c for c in calls(world, *SENDS) if c.ts_ns >= on(DAY2, 9, 5)]
    # No GTT; the day stop, an SL-M for the 10 units at 497.00, goes at the open, and the
    # close cancels it.
    assert [c.what for c in later] == ["placeorder", "cancelorder"]
    assert float(later[0].detail["request"]["trigger_price"]) == 497.0
    assert later[0].ts_ns >= on(DAY2, 9, 15)
    assert world.fake.gtts(world.fake.mode) == []
    assert f"Warning: {SENTENCE}" in world.lines
    assert rebuilt_line("CncKept", DAY1, "14:45:01", "14:45:01", 0) in world.lines
    assert any(
        SENTENCE in text or "no stop at the broker" in text for text in alerts(caplog, "warn")
    )


# (i) ------------------------------------------------------------------------------------------


@MODES
@BOOKS
def test_a_crashed_nrml_future_whose_position_changed_after_the_crash_is_a_mismatch(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    # Monday's lot of the future is sold by hand after the crash, its stop GTT left
    # resting: Tuesday's start finds the broker flat with no GTT gone to explain it, a
    # reconciliation mismatch: entries locked, nothing exited, nothing placed.
    world = crashed_world(
        make_fake_openalgo,
        monkeypatch,
        tmp_path,
        FutKept,
        future=FUTURE,
        analyzer=analyzer,
        history=history,
    )
    world.trace.waive("books agree", "the lot was sold by hand after the crash")
    world.fake.client().placeorder(
        strategy="manual",
        symbol=FUTURE,
        action="SELL",
        exchange="NFO",
        price_type="MARKET",
        product="NRML",
        quantity=FUTURE_LOT,
    )
    assert world.fake.position(FUTURE, "NFO") == 0
    to_day(world, DAY2)
    world.stop_at(on(DAY2, 14, 46))

    outcome = world.run(FutKept)

    assert [c for c in calls(world, *SENDS) if c.ts_ns >= DAY2_START] == []
    assert alerts(caplog, "reconcile_mismatch")
    assert any(lock.reason == "reconciliation" for lock in outcome.locks)
    assert [g[:2] for g in gtts(world)] == [("active", FUTURE_LOT)]
    assert world.fake.position(FUTURE, "NFO") == 0


# (f) ------------------------------------------------------------------------------------------


class DailyInOut(bt.Strategy):
    """A daily CNC strategy on SBIN: at its first decision it buys `UNITS` with a stop
    `STOP_POINTS` below (kept in `self.state`), at a later one holding them it exits. Each
    decision is recorded in the class's `decided`: (the bar's close, the units held then)."""

    symbol, exchange, interval, product = "SBIN", "NSE", "D", "CNC"
    decided: list[tuple[int, float]] = []

    def on_bar(self, bar):
        type(self).decided.append((bar.ts_close, self.position.qty))
        if self.position.qty:
            self.exit()
        elif not self.state.get("bought"):
            self.state["bought"] = True
            self.buy(UNITS, stop=bt.points(STOP_POINTS))


@MODES
@BOOKS
def test_a_daily_strategy_crashed_after_its_decisions_orders_decides_the_next_bar_once(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    # Monday's session starts at 09:05, decides Friday's daily bar, buys 10 SBIN CNC at the
    # open with its stop GTT, and crashes at 09:30 (its last save at 09:29:01). Tuesday's
    # start continues the leg, decides Monday's bar holding it, and exits at the open: no
    # bar is decided twice, and no second stop GTT is placed.
    DailyInOut.decided.clear()
    build = functools.partial(make_fake_openalgo, gtt_history=history)
    world = daily_world(build, monkeypatch, tmp_path, analyzer=analyzer)
    world.trace.waive("protected", "the session crashed: its windows end with it")
    crashing(monkeypatch)
    crash_at(world, ist(9, 30))
    run_to_crash(world, DailyInOut)
    assert DailyInOut.decided == [(close_of(FRIDAY), 0)]
    script, fake = world.script, world.fake
    script.sleep((close_of(MONDAY) + 300 * NS - fake.clock.now()) / NS)
    fake.end_day()
    tuesday_start = ist(9, 5, day=TUESDAY)
    script.sleep((tuesday_start - fake.clock.now()) / NS)
    script.trade()
    script.trades(open_of(TUESDAY), ist(15, 2, day=TUESDAY))
    world.stop, world.feed = StopFlag(), None
    world.stop_at(ist(9, 30, day=TUESDAY))

    outcome = world.run(DailyInOut)

    assert DailyInOut.decided[1:] == [(close_of(MONDAY), UNITS)]
    assert rebuilt_line("DailyInOut", MONDAY, "09:29:01", "09:29:01", 0) in world.lines
    later = [c.what for c in calls(world, *SENDS) if c.ts_ns >= tuesday_start]
    assert later == ["cancelgttorder", "placeorder"]
    assert outcome.open_positions == ()
    assert not alerts(caplog, "error")
