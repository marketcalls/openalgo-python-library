"""A CNC leg's stop GTT in a live session, on the fake OpenAlgo (20-positional-live 2 and
3; the lead's ruling N3b-2): the kernel's GTT commands reach the broker as the order
worker's GTT requests, and each answer and finding comes back as its kernel call.

The market, the strategies and the books are `positional_samples`': the entry fills at
500.00 at 14:42:01 IST, the stop GTT sells 10 when SBIN trades at 497.00 or below (child
limit 492.00), and Ctrl+C at 14:46 keeps the CNC position with its GTT (the default
`on_stop`). Each test runs on both GTT book shapes and, where the sandbox can play it, in
the broker's book and the sandbox's.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import (
    BOOKS,
    GTT_LIMIT,
    GTT_TRIGGER,
    MODES,
    STOP_AT,
    CncHold,
    CncTarget,
    calls,
    gtt_world,
    gtts,
    sells,
)
from session_samples import alerts, ist
from stops_samples import at_price, events

from openalgo.backtest._live._types import LOGGER_NAME

#: SBIN's prices of the moves: through the target (502.00), through the stop's trigger
#: (497.00) above the child's limit (492.00), and through the child's limit.
THROUGH_TARGET, THROUGH_STOP, BELOW_LIMIT = 503.0, 496.0, 490.0
#: When the price moves: the first trade at or after it is at 14:43:35.
MOVE_AT = ist(14, 43, 30)


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


def run(world, strategy, **arguments):
    world.stop_at(STOP_AT)
    return world.run(strategy, **arguments)


@BOOKS
@MODES
def test_a_cnc_entrys_fill_places_one_stop_gtt_that_the_broker_confirms(
    make_fake_openalgo, monkeypatch, tmp_path, history, analyzer
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path, history=history, analyzer=analyzer)

    outcome = run(world, CncHold)

    assert len(calls(world, "placegttorder")) == 1
    assert gtts(world) == [("active", 10, GTT_TRIGGER, "SELL")]
    answers = [event["answer"] for event in events(outcome, "gtt") if "answer" in event]
    assert answers == ["placed"]
    assert world.fake.position("SBIN", "NSE") == 10
    assert outcome.stopped_by == "ctrl_c"


@BOOKS
@MODES
def test_a_target_cancels_the_stop_gtt_then_exits_once_it_is_cancelled(
    make_fake_openalgo, monkeypatch, tmp_path, history, analyzer
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path, history=history, analyzer=analyzer)
    at_price(world, MOVE_AT, THROUGH_TARGET)

    run(world, CncTarget)

    order = [entry.what for entry in calls(world, "cancelgttorder", "placeorder")]
    # The entry, the GTT's cancel, then the exit: never the exit before the cancel.
    assert order == ["placeorder", "cancelgttorder", "placeorder"]
    assert [status for status, *_ in gtts(world)] == ["cancelled"]
    assert sells(world) == [("MARKET", 10, "complete")]
    assert world.fake.position("SBIN", "NSE") == 0


@BOOKS
@MODES
def test_the_stop_gtt_fires_and_its_childs_fill_closes_the_trade_at_the_childs_price(
    make_fake_openalgo, monkeypatch, tmp_path, history, analyzer
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path, history=history, analyzer=analyzer)
    at_price(world, MOVE_AT, THROUGH_STOP)

    outcome = run(world, CncHold)

    (child,) = [o for o in world.orders() if o.action == "SELL"]
    assert (child.price_type, child.quantity, child.status) == ("LIMIT", 10, "complete")
    trades = outcome.outputs["CncHold"]["trades"]
    assert list(trades["exit_reason"]) == ["leg_sl"]
    assert list(trades["exit_price_rupees"]) == [child.average_price]
    assert world.fake.position("SBIN", "NSE") == 0


@BOOKS
def test_a_stop_gtt_refused_at_placement_exits_the_leg_and_locks_the_day(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, history
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path, history=history)
    world.fake.refuse_next_gtt("Trigger price is too close to the last price")

    outcome = run(world, CncHold)

    assert sells(world) == [("MARKET", 10, "complete")]
    assert world.fake.position("SBIN", "NSE") == 0
    trades = outcome.outputs["CncHold"]["trades"]
    assert list(trades["exit_reason"]) == ["manual"]
    (refused,) = [a for a in alerts(caplog, "error") if "refused the stop GTT" in a]
    assert '"Trigger price is too close to the last price"' in refused
    assert "entries are locked for the day" in refused


@BOOKS
def test_a_stop_gtt_cancelled_by_hand_is_placed_again(
    make_fake_openalgo, monkeypatch, tmp_path, history
):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path, history=history)
    first = "261116000001"
    world.script.at(ist(14, 43), lambda: world.fake.set_gtt_status(first, "cancelled"))

    run(world, CncHold)

    statuses = [(status, units) for status, units, *_ in gtts(world)]
    assert statuses == [("cancelled", 10), ("active", 10)]
    assert len(calls(world, "placegttorder")) == 2
    assert world.fake.position("SBIN", "NSE") == 10


def test_the_gtt_childs_limit_is_the_brokers_child_limit(make_fake_openalgo, monkeypatch, tmp_path):
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path)

    run(world, CncHold)

    (gtt,) = world.fake.gtts()
    assert (gtt.trigger, gtt.price) == (GTT_TRIGGER, GTT_LIMIT)
