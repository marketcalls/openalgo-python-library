"""A daily CNC strategy carried overnight in `bt.live`, on the fake OpenAlgo (design
20-positional-live sections 4.2 and 8; the lead's integration ruling 7a of review round 4 of
unit N5a): the start of a later day restores the position and its stop GTT first (no second
place), and only then decides the latest daily bar before the open, so the decision sees the
position it carries; an exit decided then goes at the open, its stop GTT cancelled first. A
second start that day decides nothing again and places no second stop GTT.

The market is `daily_samples`': SBIN on NSE. On Monday 2026-11-16 the session starts at
09:05 IST, before the open; `DailyInOut` decides Friday's daily bar, buys 10 SBIN CNC at the
open (09:15, at 500.00) with a stop 3 points below, held by the broker as a SINGLE GTT (sell
10 at 497.00), and is stopped (Ctrl+C) at 09:30: the CNC leg is kept and its day-close
record written. SBIN trades at 500.00 every 10 s until 15:02, which makes Monday's daily
candle. The fake's day then ends (the CNC units move into the holdings, the GTT stays), and
Tuesday's session starts at 09:05: holding the position, the strategy exits at Monday's
bar. Each test runs in the broker's book and the sandbox's, on both GTT book shapes.
Expected values come from the scenario and the fake's books.
"""

from __future__ import annotations

import datetime as dt
import functools
import logging

import pytest
from daily_samples import OPEN, UNITS, close_of, daily_world, open_of
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import BOOKS, GTT_TRIGGER, MODES, calls, gtts
from restart_samples import restart
from session_samples import NS, StopFlag, alerts, ist
from stops_samples import STOP_POINTS

import openalgo.backtest as bt
from openalgo.backtest._live._result_build import session_result
from openalgo.backtest._live._types import LOGGER_NAME

MONDAY, TUESDAY = dt.date(2026, 11, 16), dt.date(2026, 11, 17)
FRIDAY = dt.date(2026, 11, 13)
#: Where each day's session starts, IST (before NSE's 09:15 open).
START = (9, 5, 0)
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


def monday(make_fake_openalgo, monkeypatch, tmp_path, *, analyzer, history):
    """Monday's session of `DailyInOut` (module docstring), stopped at 09:30."""
    DailyInOut.decided.clear()
    build = functools.partial(make_fake_openalgo, gtt_history=history)
    world = daily_world(build, monkeypatch, tmp_path, analyzer=analyzer)
    world.stop_at(ist(9, 30))
    assert world.run(DailyInOut).stopped_by == "ctrl_c"
    return world


def to_tuesday(world):
    """Play the rest of Monday (SBIN's trades to 15:02), end the fake's day and move to
    Tuesday's start, SBIN trading 10 units every 10 s at 500.00 from its open to 15:02."""
    script, fake = world.script, world.fake
    script.sleep((close_of(MONDAY) + 300 * NS - fake.clock.now()) / NS)
    fake.end_day()
    script.sleep((ist(*START, day=TUESDAY) - fake.clock.now()) / NS)
    script.trade()
    script.trades(open_of(TUESDAY), ist(15, 2, day=TUESDAY))
    world.stop, world.feed = StopFlag(), None


#: The stop's line of the exit Tuesday's first start decided before the open and left
#: unsent (the lead's ruling N5c-4 (j)), written out.
KEPT_EXIT = (
    "DailyInOut: the stop leaves the orders of its decision on its D bar closing on "
    "2026-11-16 (SELL 10 MARKET) unsent: a restart today sends them, at the open or at once "
    "after it, flagged late_decision, since that bar is not decided again; with no restart "
    "today they are never sent"
)


def sent_since(world, instant_ns):
    """The endpoint of each call that sends anything, made at or after `instant_ns`."""
    return [c.what for c in calls(world, *SENDS) if c.ts_ns >= instant_ns]


@MODES
@BOOKS
def test_monday_decides_before_the_open_buys_at_the_open_and_carries_its_stop_gtt(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer, history
):
    world = monday(make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer, history=history)

    assert DailyInOut.decided == [(close_of(FRIDAY), 0)]
    assert sent_since(world, 0) == ["placeorder", "placegttorder"]
    (bought,) = world.orders()
    assert (bought.action, bought.quantity, bought.product, bought.status) == (
        "BUY",
        UNITS,
        "CNC",
        "complete",
    )
    assert bought.placed_ns >= OPEN
    assert gtts(world) == [("active", UNITS, GTT_TRIGGER, "SELL")]
    # Its banner line says when it decides and when its orders go, not an intraday entry
    # cut-off (the lead's ruling N5c-4 (b)).
    (line,) = [line for line in world.lines if line.startswith("DailyInOut: NSE:SBIN every D")]
    assert (
        ", decides each D bar before the open, on the bar that closed, and its orders go at "
        "the open, no square-off: the position is carried overnight"
    ) in line
    assert "entries until" not in line


@MODES
@BOOKS
def test_tuesday_restores_the_leg_then_decides_mondays_bar_and_exits_at_the_open(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    world = monday(make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer, history=history)
    to_tuesday(world)
    world.stop_at(ist(9, 30, day=TUESDAY))
    caplog.clear()

    outcome = world.run(DailyInOut)

    # Monday's bar was decided holding the 10 units restored from Monday's record.
    assert DailyInOut.decided[1:] == [(close_of(MONDAY), UNITS)]
    tuesday_start = ist(*START, day=TUESDAY)
    # No second stop GTT: Monday's is taken back, then cancelled before the exit goes.
    assert sent_since(world, tuesday_start) == ["cancelgttorder", "placeorder"]
    (sold,) = [o for o in world.orders() if o.action == "SELL"]
    assert (sold.price_type, sold.quantity, sold.product, sold.status) == (
        "MARKET",
        UNITS,
        "CNC",
        "complete",
    )
    assert open_of(TUESDAY) <= sold.placed_ns
    (cancel,) = calls(world, "cancelgttorder")
    assert cancel.ts_ns <= sold.placed_ns
    assert [g for g in gtts(world) if g[0] == "active"] == []
    assert outcome.open_positions == ()
    assert not alerts(caplog, "error")
    assert not alerts(caplog, "reconcile_mismatch")


@MODES
@BOOKS
def test_a_second_start_on_tuesday_decides_nothing_again_and_places_no_second_stop_gtt(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer, history
):
    # Tuesday's first start decides Monday's bar before the open (an exit) and is stopped
    # (Ctrl+C) at 09:08: the stop keeps the CNC leg with its stop GTT, and the exit waits
    # for the open in the state it saves, unsent. A second start at 09:10 resumes that
    # state: it decides nothing again, places no second stop GTT, and sends the exit
    # decided at the open, the stop GTT cancelled first.
    world = monday(make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer, history=history)
    to_tuesday(world)
    world.stop_at(ist(9, 8, day=TUESDAY))
    first = world.run(DailyInOut)
    assert first.stopped_by == "ctrl_c"
    assert sent_since(world, ist(*START, day=TUESDAY)) == []
    # The stop names the exit it leaves unsent and what becomes of it (N5c-4 (j)).
    assert KEPT_EXIT in session_result(first).warnings
    assert f"Warning: {KEPT_EXIT}" in world.lines
    world.script.sleep((ist(9, 10, day=TUESDAY) - world.fake.clock.now()) / NS)
    caplog.clear()

    second = restart(world, DailyInOut, stop_ns=ist(9, 30, day=TUESDAY))

    assert DailyInOut.decided[1:] == [(close_of(MONDAY), UNITS)]
    assert (
        "Warning: DailyInOut: its D bar closing on 2026-11-16 was decided by an earlier start "
        "today and is not decided again today"
    ) in world.lines
    assert sent_since(world, ist(*START, day=TUESDAY)) == ["cancelgttorder", "placeorder"]
    (sold,) = world.orders()
    assert (sold.action, sold.price_type, sold.quantity, sold.status) == (
        "SELL",
        "MARKET",
        UNITS,
        "complete",
    )
    assert open_of(TUESDAY) <= sold.placed_ns
    assert [g for g in gtts(world) if g[0] == "active"] == []
    assert second.open_positions == ()
    assert not alerts(caplog, "error")


@MODES
def test_a_restart_after_the_open_sends_the_exit_a_stop_left_unsent_flagged_late_decision(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer
):
    # Tuesday's first start decides Monday's bar before the open (an exit) and is stopped
    # at 09:08, the exit left unsent. The restart at 09:20, after the open, sends it at
    # once, flagged late_decision, as a start after the open sends its decision's orders;
    # the bar is not decided again.
    world = monday(make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer, history=True)
    to_tuesday(world)
    world.stop_at(ist(9, 8, day=TUESDAY))
    assert world.run(DailyInOut).stopped_by == "ctrl_c"
    world.script.sleep((ist(9, 20, day=TUESDAY) - world.fake.clock.now()) / NS)
    restarted_at = world.fake.clock.now()

    second = restart(world, DailyInOut, stop_ns=ist(9, 40, day=TUESDAY))

    assert DailyInOut.decided[1:] == [(close_of(MONDAY), UNITS)]
    (sold,) = [o for o in world.orders() if o.action == "SELL"]
    assert (sold.price_type, sold.quantity, sold.status) == ("MARKET", UNITS, "complete")
    assert sold.placed_ns >= restarted_at
    (fill,) = [fill for fill in second.fills if fill.action == "SELL"]
    assert "late_decision" in fill.flags
    assert not alerts(caplog, "error")


@MODES
def test_a_restart_in_the_opens_first_minute_sends_the_exit_left_unsent_flagged_late_decision(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer
):
    # As above, the restart at 09:15:30, before its catch-up has released the 09:15
    # slice: the open has passed, so the exit goes at once flagged late_decision, as a
    # start at 09:15:30 decides late (the lead's ruling N5c-4 (j); review finding 4).
    world = monday(make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer, history=True)
    to_tuesday(world)
    world.stop_at(ist(9, 8, day=TUESDAY))
    assert world.run(DailyInOut).stopped_by == "ctrl_c"
    world.script.sleep((ist(9, 15, 30, day=TUESDAY) - world.fake.clock.now()) / NS)

    second = restart(world, DailyInOut, stop_ns=ist(9, 40, day=TUESDAY))

    assert DailyInOut.decided[1:] == [(close_of(MONDAY), UNITS)]
    (sold,) = [o for o in world.orders() if o.action == "SELL"]
    assert (sold.price_type, sold.quantity, sold.status) == ("MARKET", UNITS, "complete")
    (fill,) = [fill for fill in second.fills if fill.action == "SELL"]
    assert "late_decision" in fill.flags
    assert not alerts(caplog, "error")
