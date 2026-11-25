"""The exit guard and a CNC leg's stop GTT, on the fake OpenAlgo (16-session-design 5.4 "As
built"; 20-positional-live 7.2; the lead's ruling N3b-4).

While OpenAlgo's books show fewer units of a leg than the engine holds with no trade to
explain it, the guard holds back every order that would close more than the broker holds.
For a CNC or NRML leg the stop is a GTT: the guard holds the cancel of the stop GTT while
the engine holds a position (a target's or a strategy's exit asks for it first, and that
exit would be held), so the GTT stays at the broker protecting what it holds, and it holds
a stop GTT placed again for more units than the broker holds. Such a place was never handed
to the order worker, so the 10-second deadline of a stop GTT not confirmed has not begun for
it: nothing exits the leg, and once the books agree it goes and the leg stays protected
(review round 2 finding 3 of unit N3b).

The market and the strategies are `positional_samples`': `CncHold` buys 10 SBIN CNC at
14:42:01, filled at 500.00, with a stop GTT at 497.00; `CncTarget` adds a target at
502.00. At 14:43 five of the strategy's units go with no trade (the holdings answer -5, so
OpenAlgo holds 5). Expected values come from the fake's books and the scenario.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import (
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

#: When five of the strategy's ten units go with no trade, and OpenAlgo's holdings then.
FALL_AT, BROKER_UNITS = ist(14, 43), 5


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


def first_trigger(world):
    """The broker's trigger ID of the first GTT of the fake's book of its mode."""
    return world.fake.gtts()[0].trigger_id


def guarded_world(make_fake_openalgo, monkeypatch, tmp_path, analyzer):
    """A world whose holdings of SBIN fall by 5 units with no trade at `FALL_AT`."""
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path, analyzer=analyzer)
    world.trace.waive("protected", "the guard keeps a stop GTT beyond what the broker holds")
    world.trace.waive("books agree", "the broker's units fall with no trade")
    world.script.at(FALL_AT, lambda: world.fake.set_holding("SBIN", "NSE", -BROKER_UNITS))
    world.stop_at(STOP_AT)
    return world


@MODES
def test_a_target_on_a_guarded_leg_keeps_the_stop_gtt_and_sends_no_exit(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer
):
    world = guarded_world(make_fake_openalgo, monkeypatch, tmp_path, analyzer)
    at_price(world, ist(14, 44), 503.0)

    outcome = world.run(CncTarget)

    assert events(outcome, "exit_guard")[0]["broker_units"] == BROKER_UNITS
    # The target asks for the stop GTT's cancel first: it is held, so the GTT stays and
    # the exit that waits for its answer is never made.
    assert calls(world, "cancelgttorder") == []
    assert gtts(world) == [("active", 10, GTT_TRIGGER, "SELL")]
    assert sells(world) == []
    held = [e for e in events(outcome, "exit_held") if e.get("gtt") is not None]
    assert [(e["gtt"], e["held"]) for e in held] == [(1, "cancel")]
    errors = alerts(caplog, "error")
    named = f"stop GTT {first_trigger(world)} of SBIN NSE"
    assert any(named in a and "held back" in a for a in errors)


@MODES
def test_a_stop_gtt_ended_by_hand_on_a_guarded_leg_is_not_placed_again_beyond_the_broker(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer
):
    world = guarded_world(make_fake_openalgo, monkeypatch, tmp_path, analyzer)
    world.trace.waive("exits within the position", "the GTT is cancelled by hand")
    world.script.at(
        ist(14, 44), lambda: world.fake.set_gtt_status(first_trigger(world), "cancelled")
    )

    outcome = world.run(CncHold)

    # The engine places its stop again for its 10 units; the broker holds 5.
    assert [c.ts_ns < ist(14, 44) for c in calls(world, "placegttorder")] == [True]
    assert gtts(world) == [("cancelled", 10, GTT_TRIGGER, "SELL")]
    held = [e for e in events(outcome, "exit_held") if e.get("gtt") is not None]
    assert [(e["gtt"], e["held"]) for e in held] == [(2, "request")]


def cancelled_by_hand_at(world, instant_ns):
    """The first GTT is cancelled by hand at `instant_ns`: the engine places its stop again
    for its 10 units, which the guard holds while the broker holds 5."""
    world.trace.waive("exits within the position", "the GTT is cancelled by hand")

    def cancel_by_hand():
        world.fake.set_gtt_status(first_trigger(world), "cancelled")

    world.script.at(instant_ns, cancel_by_hand)


@MODES
def test_a_stop_gtt_place_the_guard_holds_starts_no_unprotected_deadline(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer
):
    # The place was never handed to the order worker: the deadline of a stop GTT not
    # confirmed runs from that hand-over, so nothing exits the leg and no alert says the
    # GTT was accepted.
    world = guarded_world(make_fake_openalgo, monkeypatch, tmp_path, analyzer)
    cancelled_by_hand_at(world, ist(14, 44))

    outcome = world.run(CncHold)

    assert events(outcome, "stop_unconfirmed") == []
    assert not [a for a in alerts(caplog, "error") if "not confirmed working" in a]
    assert calls(world, "cancelgttorder") == []
    assert sells(world) == []


@MODES
def test_a_held_stop_gtt_place_goes_once_the_books_agree_and_the_leg_stays_protected(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer
):
    world = guarded_world(make_fake_openalgo, monkeypatch, tmp_path, analyzer)
    cancelled_by_hand_at(world, ist(14, 44))
    agree_at = ist(14, 45)
    world.script.at(agree_at, lambda: world.fake.set_holding("SBIN", "NSE", 0))

    outcome = world.run(CncHold)

    assert [c.ts_ns > agree_at for c in calls(world, "placegttorder")] == [False, True]
    assert events(outcome, "stop_unconfirmed") == []
    assert calls(world, "cancelgttorder") == []
    assert sells(world) == []
    assert world.fake.position("SBIN", "NSE") == 10
    assert gtts(world) == [
        ("cancelled", 10, GTT_TRIGGER, "SELL"),
        ("active", 10, GTT_TRIGGER, "SELL"),
    ]


@MODES
def test_the_held_cancel_goes_once_the_books_agree_and_the_target_exit_follows_it(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer
):
    world = guarded_world(make_fake_openalgo, monkeypatch, tmp_path, analyzer)
    at_price(world, ist(14, 44), 503.0)
    agree_at = ist(14, 44, 30)
    world.script.at(agree_at, lambda: world.fake.set_holding("SBIN", "NSE", 0))

    world.run(CncTarget)

    (cancel,) = calls(world, "cancelgttorder")
    (sale,) = [c for c in calls(world, "placeorder") if c.detail["request"]["action"] == "SELL"]
    assert agree_at < cancel.ts_ns <= sale.ts_ns
    assert gtts(world) == [("cancelled", 10, GTT_TRIGGER, "SELL")]
    assert sells(world) == [("MARKET", 10, "complete")]
    assert world.fake.position("SBIN", "NSE") == 0
