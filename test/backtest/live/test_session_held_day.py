"""An instrument held for the day starts no square-off at the broker, for a strategy alone
on it as for the members of a group (16-session-design 4.9 and 6.2; 2.4.0 item P3-3, the
rule of 18-multi-strategy F105).

The market is `stops_samples`': SBIN on NSE on Monday 2026-11-16 from 14:40:05 IST, 10 units
every 10 s at 500.00. `StopThenRestingAdd` buys 10 SBIN with a stop 3 points below at
14:42:01 (an SL-M at 497.00), then at 14:43:01 rests an add below the market: a LIMIT buy
of 10 at 490.00. Half a second after that decision another client places the same order,
and the answer to the session's own is lost after it reached the broker: the order book
cannot tell the two apart, so nothing more can be sent for SBIN today. The session runs
past NSE's MIS square-off (15:00) and two Ctrl+C at 15:01 and 15:02 end it. Neither LIMIT
buy fills, so the broker holds the 10 units the engine holds. The session invariants run
after each test. Expected values come from the scenario and the fake's books.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, alerts, ist
from stops_samples import STOP_POINTS, stop_world

import openalgo.backtest as bt
from openalgo.backtest._live._types import LOGGER_NAME

UNITS = 10
ADD_LIMIT = 490.0
SQUARE_OFF = ist(15, 0)


class StopThenRestingAdd(bt.Strategy):
    """Buys `UNITS` with a stop at its first decision; rests a LIMIT add of `UNITS` at
    `ADD_LIMIT` at its second."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(UNITS, stop=bt.points(STOP_POINTS))
        elif self.calls == 2:
            self.buy(UNITS, price=ADD_LIMIT)


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path)
    other = world.fake.client()

    def the_same_add_by_another_client_then_lost():
        other.placeorder(
            strategy="manual",
            symbol="SBIN",
            action="BUY",
            exchange="NSE",
            price_type="LIMIT",
            product="MIS",
            quantity=UNITS,
            price=ADD_LIMIT,
        )
        world.fake.faults.bad_gateway_page("placeorder", reached_broker=True)

    world.script.at(ist(14, 43, 1) + NS // 2, the_same_add_by_another_client_then_lost)
    world.stop_at(ist(15, 1))
    world.stop_at(ist(15, 2))
    return world


def stop_of(world):
    (stop,) = [o for o in world.orders() if o.price_type == "SL-M"]
    return stop


def test_the_add_is_ambiguous_so_sbin_is_held_for_the_day(world, caplog):
    world.run(StopThenRestingAdd)
    named = [text for text in alerts(caplog, "error") if "cannot be told apart" in text]
    assert len(named) == 1 and "(add)" in named[0]


def test_the_square_off_cancels_no_stop_and_sends_no_exit(world):
    world.run(StopThenRestingAdd)

    assert world.fake.log.calls("cancelorder") == []
    session = [c for c in world.fake.log.calls("placeorder") if c.ts_ns >= SQUARE_OFF]
    assert session == []
    assert (stop_of(world).status, world.fake.position("SBIN", "NSE")) == (
        "trigger pending",
        UNITS,
    )


def held(stop_id: str) -> str:
    """What the note and the square-off alert say of SBIN held for the day by the add, and
    what they ask of the operator: the stop works at the broker, so it is cancelled before
    a sale by hand (which it would otherwise follow with its own), and the add may work
    too."""
    return (
        "SBIN NSE is held for the day by StopThenRestingAdd's add order, intent "
        "20261116-live-1-StopThenRestingAdd-o3, which the order book cannot tell apart from "
        f"another order: nothing can be sent for it today, so cancel its stop {stop_id} at "
        "the broker, then check the order book, cancelling what else still works, then "
        "close it by hand in OpenAlgo; the broker's own MIS square-off remains the backstop"
    )


def test_the_stop_is_reported_working_and_the_note_says_the_instrument_is_held(world):
    outcome = world.run(StopThenRestingAdd)

    # The row counts the add's units too, which may be at the broker with no stop, so it
    # is not called confirmed for them all (16-session-design 4.9, as F60's entry); the
    # note names the stop confirmed for the engine's 10, not one being cancelled.
    (row,) = outcome.open_positions
    stop = stop_of(world).order_id
    assert (row.units, row.stop_order_id, row.stop_confirmed) == (2 * UNITS, stop, False)
    assert "being cancelled" not in row.note
    assert f"stop {stop} confirmed at the broker; {held(stop)}" in row.note


def test_one_error_alert_at_the_square_off_says_nothing_is_sent_and_why(world, caplog):
    world.run(StopThenRestingAdd)

    (said,) = [text for text in alerts(caplog, "error") if "no square-off is sent" in text]
    stop = stop_of(world).order_id
    assert f"StopThenRestingAdd SBIN NSE {UNITS} units: no square-off is sent for it" in said
    # It names the stop to cancel before a sale by hand, as the position's note does.
    assert said.endswith(f"stop {stop} confirmed at the broker; {held(stop)}")
    assert not [text for text in alerts(caplog, "error") if "cannot be sent now" in text]
