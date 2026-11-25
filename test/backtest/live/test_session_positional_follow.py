"""A CNC leg's stop GTT followed through a live session, on the fake OpenAlgo
(20-positional-live 2, 3.1 to 3.4; the lead's ruling N3b-2): a trailing stop modified at
most once a minute, a child that does not fill through a base bar cancelled and the
position exited at market, a GTT whose fate cannot be told (unresolved, or two orders that
could each be its child) locking the strategy's entries with an ERROR alert and never an
exit, a short future's stop sent in the trigger field above the price, and a strategy's
GTT entry resting, firing and filling.

The market, the strategies and the books are `positional_samples`': the entry fills at
500.00 at 14:42:01 IST, the stop GTT sells 10 at 497.00 or below (child limit 492.00),
Ctrl+C at 14:46 keeps the CNC position with its GTT.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import (
    BOOKS,
    GTT_LIMIT,
    MODES,
    STOP_AT,
    CncHold,
    CncTrail,
    CncTwoTargets,
    calls,
    gtt_world,
    gtts,
    sells,
)
from session_samples import NS, alerts, ist
from stops_samples import at_price

import openalgo.backtest as bt
from openalgo.backtest._live._types import LOGGER_NAME

#: The trail's ramp: SBIN rises 0.50 a trade (every 10 s) from 14:43 to 14:45:30.
RAMP_FROM, RAMP_TO, RAMP_STEP = ist(14, 43), ist(14, 45, 30), 0.5
#: Below the child's 492.00 limit: the child rests unfilled.
BELOW_LIMIT = 490.0
#: Past the first target (502.00), short of the second (504.00).
FIRST_TARGET = 502.5
MOVE_AT = ist(14, 43, 30)


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


def run(world, strategy, **arguments):
    world.stop_at(STOP_AT)
    return world.run(strategy, **arguments)


def ramp(world):
    """SBIN's price rises `RAMP_STEP` a trade from `RAMP_FROM` to `RAMP_TO`, then holds."""

    def price(now_ns):
        if now_ns < RAMP_FROM:
            return 500.0
        steps = (min(now_ns, RAMP_TO) - RAMP_FROM) // (10 * NS)
        return 500.0 + RAMP_STEP * (steps + 1)

    world.script.price = price


@BOOKS
@MODES
def test_a_trailing_stop_gtt_is_modified_at_most_once_a_minute(
    make_fake_openalgo, monkeypatch, tmp_path, history, analyzer
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path, history=history, analyzer=analyzer)
    ramp(world)

    run(world, CncTrail)

    modifies = [entry.ts_ns for entry in calls(world, "modifygttorder")]
    assert modifies, "the trail moved the stop"
    gaps = [later - earlier for earlier, later in zip(modifies, modifies[1:], strict=False)]
    assert all(gap >= 60 * NS for gap in gaps)
    ((status, units, trigger, _),) = gtts(world)
    assert (status, units) == ("active", 10)
    assert trigger > 497.0


@BOOKS
def test_a_child_not_filled_through_a_base_bar_is_cancelled_and_the_position_exited(
    make_fake_openalgo, monkeypatch, tmp_path, history
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path, history=history)
    at_price(world, MOVE_AT, BELOW_LIMIT)

    outcome = run(world, CncHold)

    assert sells(world) == [("LIMIT", 10, "cancelled"), ("MARKET", 10, "complete")]
    (child,) = [o for o in world.orders() if o.price_type == "LIMIT"]
    assert child.price == GTT_LIMIT
    assert world.fake.position("SBIN", "NSE") == 0
    trades = outcome.outputs["CncHold"]["trades"]
    assert list(trades["exit_reason"]) == ["leg_sl"]
    # The engine's exit of what the child left is flagged as such (20-positional-live 3.2).
    fills = outcome.outputs["CncHold"]["fills"]
    sold = [list(flags) for side, flags in zip(fills["side"], fills["flags"]) if side == "SELL"]
    assert sold == [["gtt_child_unfilled"]]


@BOOKS
@MODES
def test_a_partial_target_cuts_the_stop_gtt_to_the_units_left_before_its_exit(
    make_fake_openalgo, monkeypatch, tmp_path, history, analyzer
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path, history=history, analyzer=analyzer)
    at_price(world, MOVE_AT, FIRST_TARGET)

    run(world, CncTwoTargets)

    order = [entry.what for entry in calls(world, "modifygttorder", "placeorder")]
    # The entry, the stop GTT cut to 5, then the first target's exit of 5.
    assert order == ["placeorder", "modifygttorder", "placeorder"]
    assert sells(world) == [("MARKET", 5, "complete")]
    assert [(status, units) for status, units, *_ in gtts(world)] == [("active", 5)]
    assert world.fake.position("SBIN", "NSE") == 5


def test_an_unresolved_stop_gtt_locks_entries_with_an_error_alert_and_exits_nothing(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path)
    # The decided end of the lead's ruling N3b-2: an unresolved GTT locks and alerts, and
    # the leg is the user's to settle by hand.
    world.trace.left_unprotected_by_decision("CncHold", "N3b-2 unresolved GTT")
    # Another client of the account rests a GTT of the stop's own shape before the entry;
    # then the stop GTT's place reaches the broker and its answer is lost: the book shows
    # two GTTs it could be, and the engine never guesses between them.
    same_shape = {
        "strategy": "manual",
        "symbol": "SBIN",
        "action": "SELL",
        "exchange": "NSE",
        "trigger_type": "SINGLE",
        "product": "CNC",
        "quantity": 10,
        "price_type": "LIMIT",
        "price": GTT_LIMIT,
        "triggerprice_sl": 497.0,
        "triggerprice_tg": 0.0,
    }
    other = world.fake.client()
    world.script.at(ist(14, 41, 30), lambda: other.placegttorder(**same_shape))
    timeout = world.fake.faults.timeout
    world.script.at(ist(14, 41, 31), lambda: timeout("placegttorder", reached_broker=True))

    outcome = run(world, CncHold)

    (alert,) = [a for a in alerts(caplog, "error") if "nothing is exited" in a]
    assert "stop GTT place" in alert
    assert "Its entries are locked" in alert
    assert sells(world) == []
    assert world.fake.position("SBIN", "NSE") == 10
    assert ("CncHold", "gtt_unresolved") in {(lock.strategy, lock.reason) for lock in outcome.locks}


def test_two_orders_that_could_each_be_a_fired_gtts_child_lock_entries_and_exit_nothing(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path, history=False)
    world.trace.waive("books agree", "the child fills unbound: the mismatch is the decided end")
    other = world.fake.client()
    hand_sale = {
        "strategy": "manual",
        "symbol": "SBIN",
        "action": "SELL",
        "exchange": "NSE",
        "price_type": "LIMIT",
        "product": "CNC",
        "quantity": 10,
        "price": 510.0,
    }
    world.script.at(ist(14, 43, 31), lambda: other.placeorder(**hand_sale))
    at_price(world, ist(14, 43, 32), 496.0)

    run(world, CncHold)

    (alert,) = [a for a in alerts(caplog, "error") if "could each be its child" in a]
    assert "Its entries are locked and nothing is exited" in alert
    session_sells = [o for o in world.orders() if o.action == "SELL" and o.price_type == "MARKET"]
    assert session_sells == []


class ShortFuture(bt.Strategy):
    """Sells one lot of the NIFTY future NRML at its first decision, with a stop 10 points
    above: its stop GTT buys when the future trades at the trigger or above."""

    symbol, exchange, interval, product = "NIFTY29DEC26FUT", "NFO", "1m", "NRML"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.sell(65, stop=bt.points(10))


def test_a_short_nrml_futures_stop_is_sent_in_the_trigger_field_above_the_price(
    make_fake_openalgo, monkeypatch, tmp_path
):
    from plan_samples import candles

    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path, trades_until=None)
    fake = world.fake
    future = "NIFTY29DEC26FUT"
    fake.add_instrument(
        future, "NFO", lot_size=65, tick_size=0.05, instrument_type="FUT", expiry="29-DEC-26"
    )
    fake.seed_candles(future, "NFO", candles(today_until=(14, 40)))
    fake.trade(future, "NFO", 500.0, volume=65)
    for instant in range(ist(14, 40, 15), ist(15, 12), 10 * NS):
        world.script.at(instant, lambda: fake.trade(future, "NFO", 500.0, volume=65))

    run(world, ShortFuture)

    (placed,) = calls(world, "placegttorder")
    request = placed.detail["request"]
    assert (request["action"], request["triggerprice_sl"], request["triggerprice_tg"]) == (
        "BUY",
        0.0,
        510.0,
    )
    assert fake.position(future, "NFO") == -65


class GttEntry(bt.Strategy):
    """Rests a GTT buy of 10 SBIN CNC at 499.00 at its first decision, with a stop 3 points
    below the fill."""

    symbol, exchange, interval, product = "SBIN", "NSE", "1m", "CNC"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(qty=10, price=499.0, validity="gtt", stop=bt.points(3))


@MODES
def test_a_strategys_gtt_entry_rests_fires_and_fills_then_its_stop_gtt_is_placed(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer)
    at_price(world, MOVE_AT, 498.5)

    outcome = run(world, GttEntry)

    buys = [o for o in world.orders() if o.action == "BUY"]
    assert [(o.price_type, o.quantity, o.status) for o in buys] == [("LIMIT", 10, "complete")]
    assert world.fake.position("SBIN", "NSE") == 10
    statuses = [(status, action) for status, _, _, action in gtts(world)]
    assert statuses == [("triggered", "BUY"), ("active", "SELL")]
    assert outcome.stopped_by == "ctrl_c"


@MODES
def test_a_gtt_entry_refused_at_placement_is_rejected_with_no_exit_and_no_lock(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer)
    world.fake.refuse_next_gtt("Trigger price is too close to the last price")

    outcome = run(world, GttEntry)

    assert world.orders() == []
    assert not [a for a in alerts(caplog, "error") if "GttEntry" in a and "GTT" in a]
    (reject,) = [a for a in alerts(caplog, "reject") if "GttEntry" in a]
    assert "the broker refused the GTT order of SBIN NSE" in reject
    assert "Trigger price is too close to the last price" in reject
    assert "exited" not in reject
    assert "gtt_refused" not in {lock.reason for lock in outcome.locks}


@MODES
def test_a_gtt_entry_cancelled_by_hand_ends_as_an_order_and_is_not_placed_again(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer)

    def cancel_by_hand():
        world.fake.set_gtt_status(world.fake.gtts()[0].trigger_id, "cancelled")

    world.script.at(MOVE_AT, cancel_by_hand)

    run(world, GttEntry)

    assert len(calls(world, "placegttorder")) == 1
    (reject,) = [a for a in alerts(caplog, "reject") if "GttEntry" in a]
    assert "GTT order" in reject and "ended at the broker" in reject
    assert "stop GTT" not in reject and "places it again" not in reject
