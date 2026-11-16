"""The unit 8 fault rows that name a stop, run again with one (16-session-design 7.2 rows
F01, F03 to F05, F07, F09, F29 to F33, F35, F62 and F89; unit 10): a leg whose stop the
broker holds through lost, repeated, late and silent order words, a rate-limited exit, a
mode breach and a strategy's exception.

`StopRoundTrip` buys 10 SBIN at 14:42:01 IST with a stop at 497.00 and exits three bars
later (`stops_samples`). Every test replays the fake's order timeline against the
invariants of 16-session-design 7.1.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import alerts, ist
from stops_samples import STOP_POINTS, STOP_TRIGGER, StopHold, check_invariants, events, stop_world

import openalgo.backtest as bt
from openalgo.backtest._live._types import LOGGER_NAME, Mode


class StopRoundTrip(StopHold):
    """StopHold that exits its position at its fourth decision (14:45:01)."""

    def on_bar(self, bar):
        super().on_bar(bar)
        if self.calls == 4 and self.position.qty:
            self.exit()


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return stop_world(make_fake_openalgo, monkeypatch, tmp_path)


def round_trip(world) -> list[tuple]:
    """Each live order at the fake: action, units, price type, status."""
    return [(o.action, o.quantity, o.price_type, o.status) for o in world.orders()]


#: A round trip with a stop: the entry, its stop cancelled for the exit, the exit.
ROUND_TRIP = [
    ("BUY", 10, "MARKET", "complete"),
    ("SELL", 10, "SL-M", "cancelled"),
    ("SELL", 10, "MARKET", "complete"),
]


@pytest.mark.parametrize(
    "fault",
    [
        pytest.param(lambda f: f.bad_gateway_page("placeorder", reached_broker=True), id="F01"),
        pytest.param(lambda f: f.timeout("placeorder", reached_broker=True), id="F03"),
        pytest.param(
            lambda f: f.gateway_error("placeorder", status=504, reached_broker=True), id="F04"
        ),
        pytest.param(lambda f: f.drop_connection("placeorder", reached_broker=True), id="F05"),
        pytest.param(lambda f: f.lose_answer("placeorder"), id="F07"),
    ],
)
def test_an_entry_bound_from_the_book_gets_one_stop_for_its_fill(world, fault):
    fault(world.fake.faults)
    outcome = world.run(StopRoundTrip)

    assert round_trip(world) == ROUND_TRIP
    stop = world.orders()[1]
    assert stop.trigger_price == STOP_TRIGGER
    assert world.fake.position("SBIN", "NSE") == 0
    assert outcome.stopped_by == "session_end"
    check_invariants(world.timeline, strategy="StopRoundTrip")


def test_F09_a_rate_limited_exit_after_the_stops_cancel_is_asked_for_again_and_fills(world):
    world.script.at(ist(14, 45, 2), lambda: world.fake.faults.rate_limit("placeorder"))
    world.run(StopRoundTrip)

    sells = [
        c for c in world.fake.log.calls("placeorder") if c.detail["request"]["action"] == "SELL"
    ]
    assert [c.detail["request"]["pricetype"] for c in sells] == ["SL-M", "MARKET", "MARKET"]
    assert world.fake.position("SBIN", "NSE") == 0
    check_invariants(world.timeline, strategy="StopRoundTrip")


@pytest.fixture
def baseline(make_fake_openalgo, monkeypatch, tmp_path_factory):
    """The trades of a run with no fault: (units, entry price, exit price, exit reason)."""
    folder = tmp_path_factory.mktemp("baseline")
    return trades_of(stop_world(make_fake_openalgo, monkeypatch, folder).run(StopRoundTrip))


def trades_of(outcome) -> list[tuple]:
    table = outcome.outputs["StopRoundTrip"]["trades"]
    columns = ("entry_units", "entry_price_rupees", "exit_price_rupees", "exit_reason")
    return list(zip(*(table[column] for column in columns), strict=True))


def test_F29_with_the_order_stream_silent_the_stop_is_confirmed_by_its_polls(world, baseline):
    world.fake.faults.silence_order_updates()
    outcome = world.run(StopRoundTrip)

    assert trades_of(outcome) == baseline
    assert round_trip(world) == ROUND_TRIP
    check_invariants(world.timeline, strategy="StopRoundTrip")


def test_F30_an_entry_fill_whose_words_are_lost_still_places_the_stop(world, baseline):
    world.fake.faults.drop_next_order_update()
    world.fake.faults.drop_next_order_update()
    outcome = world.run(StopRoundTrip)

    assert trades_of(outcome) == baseline
    assert round_trip(world) == ROUND_TRIP
    check_invariants(world.timeline, strategy="StopRoundTrip")


def test_F31_an_entry_word_that_arrives_twice_places_one_stop(world, caplog):
    world.fake.faults.duplicate_next_order_update()
    world.fake.faults.duplicate_next_order_update()
    world.run(StopRoundTrip)

    assert round_trip(world) == ROUND_TRIP
    assert len(alerts(caplog, "entry")) == 1
    check_invariants(world.timeline, strategy="StopRoundTrip")


def test_F32_part_fills_released_newest_first_end_with_one_stop_of_the_whole(world):
    world.fake.faults.hold_order_updates()
    world.fake.split_next_fill(4)
    world.script.price = lambda now_ns: 501.0 if now_ns >= ist(14, 42, 5) else 500.0
    world.script.at(
        ist(14, 42, 30), lambda: world.fake.faults.release_order_updates(newest_first=True)
    )
    world.stop_at(ist(14, 44))
    world.run(StopHold, on_stop="keep")

    (stop,) = [o for o in world.orders() if o.price_type == "SL-M"]
    assert (stop.quantity, stop.status) == (10, "trigger pending")
    check_invariants(world.timeline, strategy="StopHold")


def test_F33_a_fill_reported_at_price_0_and_priced_by_the_tradebook_places_the_stop(world):
    world.fake.faults.zero_price_next_fill()
    world.stop_at(ist(14, 44))
    outcome = world.run(StopHold, on_stop="keep")

    fills = events(outcome, "fill")
    assert (fills[0]["price"], fills[0]["estimated"]) == (500.0, False)
    assert [o.price_type for o in world.orders()] == ["MARKET", "SL-M"]
    check_invariants(world.timeline, strategy="StopHold")


def test_F35_after_a_mode_breach_the_live_stop_still_works_and_the_alert_names_it(world, caplog):
    world.script.at(ist(14, 43, 30), lambda: world.fake.set_analyzer(True))
    world.stop_at(ist(14, 47))
    outcome = world.run(StopRoundTrip)

    stop = world.fake.orders(Mode.LIVE)[1]
    assert (stop.price_type, stop.status) == ("SL-M", "trigger pending")
    assert world.fake.position("SBIN", "NSE", Mode.LIVE) == 10
    assert outcome.stopped_by == "mode_breach"
    named = [t for t in alerts(caplog, "error") if "mode changed" in t]
    assert len(named) == 1 and f"stop {stop.order_id} confirmed at the broker" in named[0]


class RaisesHolding(StopHold):
    """StopHold that raises at its third decision."""

    def on_bar(self, bar):
        super().on_bar(bar)
        if self.calls == 3:
            raise RuntimeError("the strategy's own error")


def test_F62_a_strategys_exception_cancels_the_stop_exits_then_is_raised(world):
    with pytest.raises(RuntimeError, match="the strategy's own error"):
        world.run(RaisesHolding)

    assert round_trip(world) == ROUND_TRIP
    assert world.fake.position("SBIN", "NSE") == 0
    check_invariants(world.timeline, strategy="RaisesHolding")


class BuysWhenFlat(bt.Strategy):
    """Buys 10 SBIN with a stop at every decision at which it holds none."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def on_bar(self, bar):
        if not self.position.qty:
            self.buy(10, stop=bt.points(STOP_POINTS))


def test_F89_an_entry_whose_fill_is_late_blocks_a_second_and_gets_one_stop(world):
    world.fake.faults.hold_order_updates()
    for _ in range(90):
        world.fake.faults.gateway_error("orderstatus", status=503)
    world.script.at(ist(14, 43, 30), world.fake.faults.release_order_updates)
    world.stop_at(ist(14, 46))
    outcome = world.run(BuysWhenFlat, on_stop="keep")

    assert [o.quantity for o in world.orders() if o.action == "BUY"] == [10]
    assert "live_in_flight" in outcome.outputs["BuysWhenFlat"]["orders"]["reject_code"]
    (stop,) = [o for o in world.orders() if o.price_type == "SL-M"]
    assert (stop.quantity, stop.status) == (10, "trigger pending")
    check_invariants(world.timeline, strategy="BuysWhenFlat")
