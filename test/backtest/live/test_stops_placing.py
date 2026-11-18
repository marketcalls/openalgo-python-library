"""The broker-held stop of a live session, placed and confirmed, on the fake OpenAlgo
(`_live/stops.py`; 16-session-design 4.1, 4.2, 4.7, 4.10 and 7.2 rows F11 to F18, F24,
F79 and F81): the stop the fill of an entry places, its shape, its confirmation and the
unprotected windows, the deadline of a stop never confirmed, a stop rejected, lost,
cancelled by someone else, and the stop's units after a part fill or an add.

The strategy is `StopHold` (buys 10 SBIN at 14:42:01 IST with a stop at 497.00) or the
option's `OptionHold` (`stops_samples`); the session runs to the end of the day unless a
test stops it. Every test replays the fake's order timeline against the invariants of
16-session-design 7.1.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, alerts, ist
from stops_samples import (
    OPTION,
    OPTION_LOT,
    STOP_TRIGGER,
    OptionHold,
    StopHold,
    at_price,
    check_invariants,
    option_world,
    stop_world,
)

import openalgo.backtest as bt
from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest._live.stops import UNPROTECTED_EXIT_AFTER_S


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return stop_world(make_fake_openalgo, monkeypatch, tmp_path)


@pytest.fixture
def option(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return option_world(make_fake_openalgo, monkeypatch, tmp_path)


def shapes(world) -> list[tuple]:
    """Each live order at the fake: action, units, price type, trigger, limit, status."""
    return [
        (o.action, o.quantity, o.price_type, o.trigger_price, o.price, o.status)
        for o in world.orders()
    ]


def windows(outcome) -> list[tuple[str, float]]:
    """The unprotected windows: (reason, seconds)."""
    return [(w.reason, w.seconds(outcome.ended_ns)) for w in outcome.unprotected]


def test_an_entrys_fill_places_its_stop_at_the_broker_and_the_square_off_cancels_it(world):
    outcome = world.run(StopHold)

    assert shapes(world) == [
        ("BUY", 10, "MARKET", 0.0, 0.0, "complete"),
        ("SELL", 10, "SL-M", STOP_TRIGGER, 0.0, "cancelled"),
        ("SELL", 10, "MARKET", 0.0, 0.0, "complete"),
    ]
    stop = world.orders()[1]
    assert stop.placed_ns - world.orders()[0].placed_ns <= 3 * NS
    assert outcome.stopped_by == "session_end"
    check_invariants(world.timeline, strategy="StopHold")


def test_a_stop_is_confirmed_by_its_first_working_word_and_the_window_closes_then(world):
    world.stop_at(ist(14, 44))
    outcome = world.run(StopHold, on_stop="keep")

    assert [w.reason for w in outcome.unprotected] == ["stop unconfirmed"]
    (window,) = outcome.unprotected
    assert 0 < window.seconds(outcome.ended_ns) <= 3
    assert outcome.summary.unprotected_seconds == window.seconds(outcome.ended_ns)
    (left,) = outcome.open_positions
    assert (left.units, left.stop_order_id, left.stop_confirmed) == (
        10.0,
        world.orders()[1].order_id,
        True,
    )
    assert "confirmed at the broker" in left.note
    check_invariants(world.timeline, strategy="StopHold")


def test_F11_a_stop_rejected_after_acceptance_exits_at_market_at_once(world, caplog):
    world.fake.reject_next("trigger price invalid", price_type="SL-M")

    outcome = world.run(StopHold)

    assert [row[2:] for row in shapes(world)] == [
        ("MARKET", 0.0, 0.0, "complete"),
        ("SL-M", STOP_TRIGGER, 0.0, "rejected"),
        ("MARKET", 0.0, 0.0, "complete"),
    ]
    stop, exit_ = world.orders()[1:]
    assert exit_.placed_ns - stop.placed_ns <= 2 * NS
    assert world.fake.position("SBIN", "NSE") == 0
    rejects = alerts(caplog, "reject")
    assert any(stop.order_id in text and "trigger price invalid" in text for text in rejects)
    assert sum(seconds for _, seconds in windows(outcome)) <= 5
    check_invariants(world.timeline, strategy="StopHold")


def test_F12_an_option_stop_refused_at_placement_exits_at_market_at_once(option):
    option.fake.reject_next("RMS: trigger too far", price_type="SL", at_placement=True)

    option.run(OptionHold)

    assert [(o.action, o.price_type, o.status) for o in option.orders()] == [
        ("BUY", "MARKET", "complete"),
        ("SELL", "MARKET", "complete"),
    ]
    assert option.fake.position(OPTION, "NFO") == 0
    check_invariants(option.timeline, strategy="OptionHold")


def test_F13_an_options_stop_is_a_stop_limit_5_percent_beyond_its_trigger_on_the_tick(option):
    option.stop_at(ist(14, 44))
    option.run(OptionHold, on_stop="keep")

    stop = option.orders()[1]
    # 10 points below the fill at 120.00 is 110.00; 5 percent below that, 104.50.
    assert (stop.action, stop.quantity, stop.price_type) == ("SELL", OPTION_LOT, "SL")
    assert (stop.trigger_price, stop.price) == (110.0, 104.5)
    assert all(o.price_type != "SL-M" for o in option.orders())
    check_invariants(option.timeline, strategy="OptionHold")


def test_F14_a_stop_accepted_and_never_confirmed_exits_the_leg_after_10_seconds(world, caplog):
    def no_word_comes():
        world.fake.faults.silence_order_updates()
        for _ in range(12):
            world.fake.faults.gateway_error("orderstatus", status=503)

    world.script.at(ist(14, 42, 3) + NS // 2, no_word_comes)
    outcome = world.run(StopHold)

    stop = world.orders()[1]
    cancels = world.fake.log.calls("cancelorder")
    assert stop.placed_ns + UNPROTECTED_EXIT_AFTER_S * NS <= cancels[0].ts_ns
    assert cancels[0].ts_ns <= stop.placed_ns + (UNPROTECTED_EXIT_AFTER_S + 2) * NS
    assert world.fake.position("SBIN", "NSE") == 0
    unconfirmed = [s for reason, s in windows(outcome) if reason == "stop unconfirmed"]
    assert unconfirmed and unconfirmed[0] >= UNPROTECTED_EXIT_AFTER_S
    assert any("not confirmed working" in text for text in alerts(caplog, "error"))
    check_invariants(world.timeline, strategy="StopHold")


def test_F15_a_stop_placement_unanswered_holds_the_leg_alerts_at_once_and_binds_later(
    world, caplog
):
    world.script.at(
        ist(14, 42, 3) + NS // 2,
        lambda: world.fake.faults.timeout("placeorder", reached_broker=True),
    )
    world.stop_at(ist(14, 47))
    outcome = world.run(StopHold, on_stop="keep")

    assert [(o.price_type, o.status) for o in world.orders()] == [
        ("MARKET", "complete"),
        ("SL-M", "trigger pending"),
    ]
    assert any("got no answer" in text and "(stop)" in text for text in alerts(caplog, "error"))
    # The call timed out after 120 s: the leg counts unprotected until the order book
    # bound the stop and its word could be known as the stop's.
    assert sum(seconds for _, seconds in windows(outcome)) >= 120
    assert [c.detail["request"]["pricetype"] for c in world.fake.log.calls("placeorder")] == [
        "MARKET",
        "SL-M",
    ]
    (left,) = outcome.open_positions
    assert left.stop_confirmed
    check_invariants(world.timeline, strategy="StopHold")


def test_F16_a_stop_placement_that_never_reached_the_broker_is_placed_again(world, caplog):
    world.script.at(
        ist(14, 42, 3) + NS // 2, lambda: world.fake.faults.bad_gateway_page("placeorder")
    )
    world.stop_at(ist(14, 47))
    outcome = world.run(StopHold, on_stop="keep")

    stops = [o for o in world.orders() if o.price_type == "SL-M"]
    assert [(o.quantity, o.status) for o in stops] == [(10, "trigger pending")]
    assert [o.action for o in world.orders()] == ["BUY", "SELL"]
    assert any("(stop)" in text and "not placed" in text for text in alerts(caplog, "reject"))
    assert outcome.open_positions[0].stop_confirmed
    check_invariants(world.timeline, strategy="StopHold")


def test_F17_a_part_fill_gets_a_stop_for_its_units_then_a_modify_to_the_whole(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # No trade from 14:42 to 14:42:20: the entry fills 4 units as it arrives and the rest
    # at the next trade, once the stop for the 4 is confirmed.
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path, trades_until=(14, 42))
    world.script.trades(ist(14, 42, 20), ist(15, 2))
    world.fake.split_next_fill(4)
    world.stop_at(ist(14, 44))
    outcome = world.run(StopHold, on_stop="keep")

    stop = world.orders()[1]
    assert (stop.price_type, stop.quantity, stop.status) == ("SL-M", 10, "trigger pending")
    placed = [
        c for c in world.fake.log.calls("placeorder") if c.detail["request"]["action"] == "SELL"
    ]
    assert int(placed[0].detail["request"]["quantity"]) == 4
    modifies = world.fake.log.calls("modifyorder")
    assert [int(c.detail["request"]["quantity"]) for c in modifies] == [10]
    assert [w.reason for w in outcome.unprotected] == ["stop unconfirmed", "stop unconfirmed"]
    check_invariants(world.timeline, strategy="StopHold")


class AddsOnce(StopHold):
    """StopHold that adds 5 units at its third decision."""

    def on_bar(self, bar):
        super().on_bar(bar)
        if self.calls == 3:
            self.buy(5, **type(self).rules)


def test_F18_an_add_modifies_the_stop_up_to_the_position(world):
    world.stop_at(ist(14, 46))
    world.run(AddsOnce, on_stop="keep")

    (stop,) = [o for o in world.orders() if o.price_type == "SL-M"]
    assert (stop.quantity, stop.status) == (15, "trigger pending")
    assert world.fake.position("SBIN", "NSE") == 15
    check_invariants(world.timeline, strategy="AddsOnce")


def test_F24_a_stops_status_flips_change_nothing(world):
    world.stop_at(ist(14, 52))
    world.run(StopHold, on_stop="keep")

    assert world.fake.log.calls("modifyorder") == []
    assert world.fake.log.calls("cancelorder") == []
    polled = [e for e in world.fake.log.calls("orderstatus")]
    assert polled, "the resting stop is polled, so its status reads open between stream words"
    check_invariants(world.timeline, strategy="StopHold")


def test_F79_a_stop_cancelled_by_another_client_is_placed_again_once(world, caplog):
    other = world.fake.client()

    def cancel_the_stop():
        stop = world.orders()[1]
        other.cancelorder(order_id=stop.order_id, strategy="StopHold")

    world.script.at(ist(14, 43, 30), cancel_the_stop)
    world.stop_at(ist(14, 46))
    outcome = world.run(StopHold, on_stop="keep")

    stops = [(o.quantity, o.status) for o in world.orders() if o.price_type == "SL-M"]
    assert stops == [(10, "cancelled"), (10, "trigger pending")]
    assert any(
        "cancelled at the broker with no request" in text for text in alerts(caplog, "reject")
    )
    assert [w.reason for w in outcome.unprotected][-1] == "stop unconfirmed"
    check_invariants(world.timeline, strategy="StopHold")


def test_F81_an_add_whose_answer_is_lost_counts_its_units_unprotected_until_bound(world, caplog):
    world.script.at(
        ist(14, 44, 1) - NS // 2,
        lambda: world.fake.faults.timeout("placeorder", reached_broker=True),
    )
    world.stop_at(ist(14, 50))
    outcome = world.run(AddsOnce, on_stop="keep")

    (stop,) = [o for o in world.orders() if o.price_type == "SL-M"]
    assert (stop.quantity, stop.status) == (15, "trigger pending")
    assert "unbound add" in [w.reason for w in outcome.unprotected]
    assert any("the add of 5 SBIN" in text for text in alerts(caplog, "error"))
    check_invariants(world.timeline, strategy="AddsOnce")


def test_an_entry_without_a_stop_rule_is_warned_about_and_never_counted_unprotected(world):
    class NoStop(StopHold):
        rules: dict = {}

    world.stop_at(ist(14, 44))
    outcome = world.run(NoStop, on_stop="keep")

    assert outcome.unprotected == ()
    assert any("no stop at the broker" in warning for warning in outcome.warnings)
    assert any(
        line.startswith("Warning: NoStop: a position of 10 SBIN NSE was opened with no stop")
        for line in world.lines
    )


def test_a_trail_moves_the_brokers_stop_and_a_stop_moved_alert_says_so(world, caplog):
    class Trails(StopHold):
        rules = {"stop": bt.points(3), "trail": bt.trail_points(3)}

    at_price(world, ist(14, 43), 505.0)
    world.stop_at(ist(14, 45))
    world.run(Trails, on_stop="keep")

    (stop,) = [o for o in world.orders() if o.price_type == "SL-M"]
    assert stop.trigger_price == 502.0
    assert any("moved to 502.00" in text for text in alerts(caplog, "stop_moved"))
    check_invariants(world.timeline, strategy="Trails")
