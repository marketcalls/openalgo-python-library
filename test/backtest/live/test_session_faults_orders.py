"""The fault rows unit 8 ran without leg rules whose Assertion names a stop, run with one
(16-session-design 7.2 rows F10, F36 and F60, unit 14): a slow placement, the analyzer
switch turned off in a sandbox session, and an entry the order book cannot tell apart.

The market is `stops_samples`': SBIN on NSE on Monday 2026-11-16 from 14:40:05 IST, 10 units
every 10 s at 500.00. `StopHold` buys 10 SBIN at 14:42:01 with an SL-M stop at 497.00. The
session invariants (`session_invariants`) run after each test. Expected values come from
the scenario and the fake's books.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, alerts, ist
from stops_samples import STOP_TRIGGER, StopHold, stop_world

from openalgo.backtest._live._types import LOGGER_NAME, Mode


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return stop_world(make_fake_openalgo, monkeypatch, tmp_path)


def test_F10_a_slow_entry_answer_leaves_no_false_unprotected_exit_and_the_stop_confirmed(world):
    world.fake.faults.slow("placeorder", 5)
    # SBIN trades every 2 s as well, so the clock's jump inside the call leaves no 10 s
    # without a tick, which the risk watch would take for a dead feed (6.5).
    for instant in range(ist(14, 41), ist(14, 46), 2 * NS):
        world.script.at(instant, world.script.trade)
    world.stop_at(ist(14, 45))
    outcome = world.run(StopHold, on_stop="keep")
    assert [(o.price_type, o.status) for o in world.orders()] == [
        ("MARKET", "complete"),
        ("SL-M", "trigger pending"),
    ]
    assert world.fake.log.calls("cancelorder") == []
    (left,) = outcome.open_positions
    assert (left.units, left.stop_confirmed) == (10, True)
    assert world.orders()[1].trigger_price == STOP_TRIGGER


def test_F36_after_a_sandbox_session_sees_live_mode_nothing_reaches_the_live_book(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path, analyzer=True)
    world.script.at(ist(14, 43, 30), lambda: world.fake.set_analyzer(False))
    world.stop_at(ist(14, 47))
    outcome = world.run(StopHold)
    assert (outcome.mode, outcome.stopped_by) == ("sandbox", "mode_breach")
    assert world.fake.orders(Mode.LIVE) == []
    sandbox = [(o.price_type, o.status) for o in world.fake.orders(Mode.SANDBOX)]
    assert sandbox == [("MARKET", "complete"), ("SL-M", "trigger pending")]
    # Nothing is read after the lock: the sandbox's stop is the one known at the breach.
    [left] = outcome.open_positions
    assert (left.units, left.stop_confirmed, left.note) == (10, True, "known as of the mode breach")
    [breach] = [t for t in alerts(caplog, "error") if "mode changed" in t]
    assert f"stop {sandbox_stop(world)} confirmed at the broker" in breach


def sandbox_stop(world) -> str:
    return next(o.order_id for o in world.fake.orders(Mode.SANDBOX) if o.price_type == "SL-M")


@pytest.fixture
def ambiguous(world):
    """F60: the session's entry gets an HTML 502 though it reached the broker, and another
    client places the same order half a second later; before the run, which Ctrl+C stops
    at 14:45 with on_stop="keep"."""
    world.fake.faults.bad_gateway_page("placeorder", reached_broker=True)
    other = world.fake.client()
    world.script.at(
        ist(14, 42, 2) + NS // 2,
        lambda: other.placeorder(
            strategy="StopHold",
            symbol="SBIN",
            action="BUY",
            exchange="NSE",
            price_type="MARKET",
            product="MIS",
            quantity=10,
        ),
    )
    world.stop_at(ist(14, 45))
    # The engine holds no position of an entry it cannot bind, so it places no stop: the
    # session's 10 units at the broker have none, as the ERROR alert and the open
    # position say (7.1 lists F60).
    world.trace.leave_unprotected("StopHold", "F60")
    return world


def test_F60_an_entry_the_book_cannot_tell_apart_gets_no_stop_and_holds_the_instrument(
    ambiguous, caplog
):
    world = ambiguous
    world.run(StopHold, on_stop="keep")
    named = [text for text in alerts(caplog, "error") if "cannot be told apart" in text]
    assert len(named) == 1
    assert [o.price_type for o in world.orders()] == ["MARKET", "MARKET"]
    assert world.fake.log.calls("modifyorder") == world.fake.log.calls("cancelorder") == []
    assert world.fake.position("SBIN", "NSE") == 20


def test_F60_the_ambiguous_entrys_units_are_an_open_position_with_no_stop(ambiguous):
    outcome = ambiguous.run(StopHold, on_stop="keep")
    (left,) = outcome.open_positions
    assert (left.strategy, left.units, left.stop_order_id, left.stop_confirmed) == (
        "StopHold",
        10,
        "",
        False,
    )
    assert "cannot be told apart in the order book from another order" in left.note
    assert outcome.summary.flat is False


def test_F60_the_ambiguous_entrys_units_count_unprotected_until_the_end(ambiguous):
    outcome = ambiguous.run(StopHold, on_stop="keep")
    (window,) = outcome.unprotected
    assert (window.strategy, window.units, window.reason) == ("StopHold", 10, "ambiguous order")
    assert ist(14, 42, 2) <= window.from_ns <= ist(14, 45)
    assert outcome.summary.unprotected_seconds == pytest.approx(window.seconds(outcome.ended_ns))
    assert outcome.summary.unprotected_seconds > 0
