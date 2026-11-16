"""Exits of a leg whose stop the broker holds, on the fake OpenAlgo (`_live/stops.py`,
`_live/_session_exits.py`; 16-session-design 4.3 to 4.5, 4.8 and 7.2 rows F19 to F22, F25
to F28, F43 to F45, F66, F67, F69, F82, F83 and F91): one exit owner (cancel, then exit),
the partial exit that cuts the stop first, a stop that fills while it is being
cancelled, cancels refused, timed out or lost, trailing by modifying the stop, exits that
fail, the square-off with no tick or refused, the daily loss on ticks, a second trade of
the day and the freeze quantity.

The strategies buy 10 SBIN at 14:42:01 IST with a stop at 497.00 (`stops_samples`), or one
lot of the option. Every test replays the fake's order timeline against the invariants of
16-session-design 7.1: never a stop and an exit working beyond the position, never short.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo import BROKER
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import DAY, NS, alerts, ist
from stops_samples import (
    OPTION,
    OPTION_LOT,
    STOP_POINTS,
    OptionHold,
    StopHold,
    TargetHold,
    at_price,
    check_invariants,
    events,
    option_world,
    stop_world,
)

import openalgo.backtest as bt
from openalgo.backtest._live._session_exits import EXIT_ATTEMPTS, EXIT_RETRY_DELAYS_S
from openalgo.backtest._live._stops_cancels import CANCEL_RETRY_DELAYS_S
from openalgo.backtest._live._types import LOGGER_NAME, Mode
from openalgo.backtest._live.journal import Journal

#: When the target of `TargetHold` is reached: SBIN trades at 502.00 from 14:44.
TARGET_AT = ist(14, 44)


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return stop_world(make_fake_openalgo, monkeypatch, tmp_path)


def journal_stop(world, outcome, strategy: str) -> str | None:
    """The stop the journal records for `strategy`'s SBIN leg today."""
    journal = Journal.open(
        outcome.paths["journal"],
        account=BROKER,
        broker=BROKER,
        mode=Mode.LIVE,
        now=world.fake.clock.now,
    )
    with journal:
        return journal.stop_of(strategy, "SBIN", "NSE", day=DAY)


def sells(world) -> list[tuple]:
    """Each SELL at the fake: units, price type, status."""
    return [(o.quantity, o.price_type, o.status) for o in world.orders() if o.action == "SELL"]


def calls_of(world, endpoint: str) -> list:
    return world.fake.log.calls(endpoint)


def target_reached(world) -> None:
    at_price(world, TARGET_AT, 502.0)


# ----- one exit owner: cancel, then exit ---------------------------------------------


def test_F20_a_target_reached_on_a_tick_cancels_the_stop_then_exits(world):
    target_reached(world)
    outcome = world.run(TargetHold)

    assert sells(world) == [(10, "SL-M", "cancelled"), (10, "MARKET", "complete")]
    stop, exit_ = [o for o in world.orders() if o.action == "SELL"]
    (cancel,) = calls_of(world, "cancelorder")
    assert TARGET_AT <= cancel.ts_ns < exit_.placed_ns < TARGET_AT + 15 * NS
    assert world.fake.position("SBIN", "NSE") == 0
    assert outcome.outputs["TargetHold"]["trades"]["exit_reason"] == ["leg_target"]
    check_invariants(world.timeline, strategy="TargetHold")


def test_F21_a_stop_that_fills_while_its_cancel_is_sent_closes_the_leg_with_no_exit(world):
    target_reached(world)

    def the_broker_fills_the_stop():
        world.fake.faults.hold_order_updates()
        stop = next(o for o in world.orders() if o.price_type == "SL-M")
        world.fake.fill(stop.order_id)

    world.script.at(ist(14, 44, 5), the_broker_fills_the_stop)
    world.script.at(ist(14, 45), world.fake.faults.release_order_updates)
    outcome = world.run(TargetHold)

    assert sells(world) == [(10, "SL-M", "complete")]
    assert len(calls_of(world, "cancelorder")) == 1
    assert world.fake.position("SBIN", "NSE") == 0
    assert outcome.open_positions == ()
    check_invariants(world.timeline, strategy="TargetHold")


def test_F22_a_cancel_that_times_out_is_settled_by_a_read_before_the_exit_goes(world):
    # The stream is silent, so only a read can tell the session the stop was cancelled.
    target_reached(world)

    def cancel_times_out():
        world.fake.faults.silence_order_updates()
        world.fake.faults.timeout("cancelorder", reached_broker=True)

    world.script.at(ist(14, 44, 4), cancel_times_out)
    world.run(TargetHold)

    assert sells(world) == [(10, "SL-M", "cancelled"), (10, "MARKET", "complete")]
    stop, exit_ = [o for o in world.orders() if o.action == "SELL"]
    (cancel,) = calls_of(world, "cancelorder")
    reads = [c for c in calls_of(world, "orderstatus") if c.ts_ns >= cancel.ts_ns]
    assert reads[0].detail["request"]["orderid"] == stop.order_id
    assert reads[0].ts_ns < exit_.placed_ns
    assert world.fake.position("SBIN", "NSE") == 0
    check_invariants(world.timeline, strategy="TargetHold")


def test_F66_a_rate_limited_cancel_is_sent_again_a_second_after_its_read(world):
    target_reached(world)
    world.script.at(ist(14, 44, 4), lambda: world.fake.faults.rate_limit("cancelorder"))
    world.run(TargetHold)

    first, second = calls_of(world, "cancelorder")
    exit_ = world.orders()[-1]
    assert second.ts_ns - first.ts_ns >= CANCEL_RETRY_DELAYS_S[0] * NS
    assert second.ts_ns < exit_.placed_ns
    assert sells(world) == [(10, "SL-M", "cancelled"), (10, "MARKET", "complete")]
    assert world.fake.position("SBIN", "NSE") == 0
    check_invariants(world.timeline, strategy="TargetHold")


def test_a_cancel_refused_three_times_is_alerted_and_sent_again_every_5_seconds(world, caplog):
    target_reached(world)

    def refuse_three():
        for _ in range(3):
            world.fake.faults.rate_limit("cancelorder")

    world.script.at(ist(14, 44, 4), refuse_three)
    world.run(TargetHold)

    times = [call.ts_ns for call in calls_of(world, "cancelorder")]
    gaps = [(b - a) / NS for a, b in zip(times, times[1:], strict=False)]
    assert len(times) == 4
    assert all(gap >= delay for gap, delay in zip(gaps, CANCEL_RETRY_DELAYS_S, strict=True))
    refused = [t for t in alerts(caplog, "error") if "was refused 3 times" in t]
    assert len(refused) == 1 and "(HTTP 429:" in refused[0]
    assert sells(world) == [(10, "SL-M", "cancelled"), (10, "MARKET", "complete")]
    check_invariants(world.timeline, strategy="TargetHold")


def test_F67_a_cancel_refused_while_the_login_is_expired_leaves_the_stop_working(world, caplog):
    # The login expires right after the tick that reaches the target (the feed stops
    # ticking while it is expired), so the stop's cancel meets the expired login.
    target_reached(world)
    world.script.at(ist(14, 44, 5), world.fake.faults.expire_session)
    world.script.at(ist(14, 45), world.fake.faults.restore_session)
    world.run(TargetHold)

    stop = next(o for o in world.orders() if o.price_type == "SL-M")
    exit_ = world.orders()[-1]
    assert sells(world) == [(10, "SL-M", "cancelled"), (10, "MARKET", "complete")]
    assert exit_.placed_ns > ist(14, 45)
    assert any(
        stop.order_id in text and "not confirmed" in text for text in alerts(caplog, "error")
    )
    assert world.fake.position("SBIN", "NSE") == 0
    check_invariants(world.timeline, strategy="TargetHold")


def test_F83_a_cancel_whose_answer_is_lost_is_settled_by_a_read_and_exits(world):
    target_reached(world)
    world.script.at(ist(14, 44, 4), lambda: world.fake.faults.lose_answer("cancelorder"))
    world.run(TargetHold)

    assert len(calls_of(world, "cancelorder")) == 1
    assert sells(world) == [(10, "SL-M", "cancelled"), (10, "MARKET", "complete")]
    assert world.fake.position("SBIN", "NSE") == 0
    check_invariants(world.timeline, strategy="TargetHold")


# ----- partial exits: the stop cut first -------------------------------------------------


class HalfAtThird(StopHold):
    """StopHold that exits half its position at its third decision (14:44:01)."""

    def on_bar(self, bar):
        super().on_bar(bar)
        if self.calls == 3:
            self.exit(fraction=0.5)


def test_F19_a_partial_exit_cuts_the_stop_down_before_it_goes(world):
    world.stop_at(ist(14, 46))
    world.run(HalfAtThird, on_stop="keep")

    (modify,) = calls_of(world, "modifyorder")
    sale = next(o for o in world.orders() if o.price_type == "MARKET" and o.action == "SELL")
    assert int(modify.detail["request"]["quantity"]) == 5
    assert modify.ts_ns < sale.placed_ns
    assert sells(world) == [(5, "SL-M", "trigger pending"), (5, "MARKET", "complete")]
    assert world.fake.position("SBIN", "NSE") == 5
    check_invariants(world.timeline, strategy="HalfAtThird")


class TwoTargets(StopHold):
    """StopHold with two targets: half at 501.00, the rest at 502.00."""

    rules = {
        "stop": bt.points(STOP_POINTS),
        "targets": [(bt.points(1), 0.5), (bt.points(2), 0.5)],
    }


def test_F19_two_targets_cut_the_stop_for_the_first_then_cancel_it_for_the_second(world):
    at_price(world, ist(14, 44), 501.0)
    at_price(world, ist(14, 45), 502.0)
    world.run(TwoTargets)

    assert sells(world) == [
        (5, "SL-M", "cancelled"),
        (5, "MARKET", "complete"),
        (5, "MARKET", "complete"),
    ]
    assert world.fake.position("SBIN", "NSE") == 0
    check_invariants(world.timeline, strategy="TwoTargets")


def test_a_rejected_partial_target_is_not_sent_again_and_the_stop_covers_the_position_again(
    world, caplog
):
    # Decided for unit 10: no retry turns a partial target into a full exit.
    at_price(world, ist(14, 44), 501.0)
    world.script.at(
        ist(14, 43, 50), lambda: world.fake.reject_next("RMS: blocked", price_type="MARKET")
    )
    world.stop_at(ist(14, 44, 50))
    outcome = world.run(TwoTargets, on_stop="keep")

    assert sells(world) == [(10, "SL-M", "trigger pending"), (5, "MARKET", "rejected")]
    assert [int(c.detail["request"]["quantity"]) for c in calls_of(world, "modifyorder")] == [5, 10]
    assert any("RMS: blocked" in text for text in alerts(caplog, "reject"))
    assert outcome.open_positions[0].stop_confirmed
    check_invariants(world.timeline, strategy="TwoTargets")


# ----- trailing: modifying the broker's stop ----------------------------------------------


class Trails(StopHold):
    """StopHold whose stop trails 3 points below the best price."""

    rules = {"stop": bt.points(STOP_POINTS), "trail": bt.trail_points(3)}


def rising(world, *, start: int, step: float, until: int) -> None:
    """SBIN rises by `step` rupees each trade from `start` until `until`, then holds."""

    def price(now_ns: int) -> float:
        if now_ns < start:
            return 500.0
        trades = (min(now_ns, until) - start) // (10 * NS)
        return round(500.0 + step * (trades + 1), 2)

    world.script.price = price


def test_F25_trailing_modifies_are_a_quarter_percent_and_5_seconds_apart(world):
    rising(world, start=ist(14, 43), step=0.5, until=ist(14, 46))
    world.stop_at(ist(14, 48))
    world.run(Trails, on_stop="keep")

    modifies = calls_of(world, "modifyorder")
    triggers = [float(c.detail["request"]["trigger_price"]) for c in modifies]
    assert len(modifies) >= 3
    for before, after in zip([497.0, *triggers], triggers, strict=False):
        assert after - before >= before * 0.0025
    for first, second in zip(modifies, modifies[1:], strict=False):
        assert second.ts_ns - first.ts_ns >= 5 * NS
    (stop,) = [o for o in world.orders() if o.price_type == "SL-M"]
    high = 500.0 + 0.5 * 19
    assert stop.trigger_price == triggers[-1] and high - 3 - stop.trigger_price < 503 * 0.0025
    check_invariants(world.timeline, strategy="Trails")


def kernel_trigger(outcome, name: str = "Trails") -> float:
    orders = outcome.outputs[name]["orders"]
    stops = [i for i, kind in enumerate(orders["order_type"]) if kind == "SL-M"]
    return orders["trigger_rupees"][stops[-1]]


@pytest.mark.parametrize(
    "fault",
    [
        pytest.param(lambda f: f.timeout("modifyorder", reached_broker=True), id="timeout"),
        pytest.param(
            lambda f: f.bad_gateway_page("modifyorder", reached_broker=True), id="bad_gateway"
        ),
        pytest.param(lambda f: f.lose_answer("modifyorder"), id="F82_lost_answer"),
    ],
)
def test_F26_a_trailing_modify_unanswered_is_settled_by_a_read_of_its_terms(world, fault):
    at_price(world, ist(14, 43), 505.0)
    world.script.at(ist(14, 42, 50), lambda: fault(world.fake.faults))
    world.stop_at(ist(14, 44, 30))
    outcome = world.run(Trails, on_stop="keep")

    (modify,) = calls_of(world, "modifyorder")
    (stop,) = [o for o in world.orders() if o.price_type == "SL-M"]
    assert stop.trigger_price == 502.0
    assert kernel_trigger(outcome) == stop.trigger_price
    reads = [c for c in calls_of(world, "orderstatus") if c.ts_ns > modify.ts_ns]
    assert reads and calls_of(world, "cancelorder") == []
    check_invariants(world.timeline, strategy="Trails")


def test_F27_a_trailing_modify_refused_keeps_the_old_trigger_until_the_next_is_accepted(world):
    at_price(world, ist(14, 43), 505.0)
    world.script.at(ist(14, 42, 50), lambda: world.fake.faults.rate_limit("modifyorder"))
    world.stop_at(ist(14, 44, 30))
    outcome = world.run(Trails, on_stop="keep")

    refused, accepted = calls_of(world, "modifyorder")
    assert accepted.ts_ns - refused.ts_ns >= 5 * NS
    (stop,) = [o for o in world.orders() if o.price_type == "SL-M"]
    assert stop.trigger_price == 502.0 == kernel_trigger(outcome)
    check_invariants(world.timeline, strategy="Trails")


# ----- exits that fail ------------------------------------------------------------------


class ExitsThenBuys(StopHold):
    """StopHold that exits at its third decision (14:44:01) and tries to buy again at its
    fifth."""

    def on_bar(self, bar):
        super().on_bar(bar)
        if self.calls == 3:
            self.exit()
        elif self.calls == 5:
            self.buy(5)


def test_F28_an_exit_rejected_five_times_halts_entries_and_places_the_stop_again(world, caplog):
    def refuse_five():
        for _ in range(EXIT_ATTEMPTS):
            world.fake.reject_next("RMS blocked", price_type="MARKET")

    world.script.at(ist(14, 43, 50), refuse_five)
    world.stop_at(ist(14, 47))
    outcome = world.run(ExitsThenBuys, on_stop="keep")

    exits = [o for o in world.orders() if (o.action, o.price_type) == ("SELL", "MARKET")]
    gaps = [(b.placed_ns - a.placed_ns) / NS for a, b in zip(exits, exits[1:], strict=False)]
    assert [o.status for o in exits] == ["rejected"] * EXIT_ATTEMPTS
    assert all(gap >= delay for gap, delay in zip(gaps, EXIT_RETRY_DELAYS_S, strict=True))
    stops = [(o.status) for o in world.orders() if o.price_type == "SL-M"]
    assert stops == ["cancelled", "trigger pending"]
    assert any("failed 5 times" in text for text in alerts(caplog, "error"))
    assert "live_lock" in outcome.outputs["ExitsThenBuys"]["orders"]["reject_code"]
    assert journal_stop(world, outcome, "ExitsThenBuys") == world.orders()[-1].order_id
    check_invariants(world.timeline, strategy="ExitsThenBuys")


def test_F43_the_square_off_comes_by_the_clock_with_no_tick_and_cancels_the_stop_first(
    make_fake_openalgo, monkeypatch, tmp_path
):
    option = option_world(make_fake_openalgo, monkeypatch, tmp_path, trades_until=(15, 9))
    option.run(OptionHold)

    stop, exit_ = [o for o in option.orders() if o.action == "SELL"]
    (cancel,) = calls_of(option, "cancelorder")
    assert (stop.price_type, stop.status) == ("SL", "cancelled")
    assert ist(15, 10, 1) <= cancel.ts_ns < exit_.placed_ns < ist(15, 10, 10)
    assert option.fake.position(OPTION, "NFO") == 0
    check_invariants(option.timeline, strategy="OptionHold")


def test_F44_a_refused_square_off_is_asked_for_again_with_alerts_until_flat(world, caplog):
    def refuse_five():
        for _ in range(EXIT_ATTEMPTS):
            world.fake.reject_next("RMS: square-off blocked", price_type="MARKET")

    world.script.at(ist(14, 59, 50), refuse_five)
    outcome = world.run(StopHold)

    assert any("still not flat after square-off" in t for t in alerts(caplog, "error"))
    assert world.fake.position("SBIN", "NSE") == 0
    assert outcome.stopped_by == "session_end"
    check_invariants(world.timeline, strategy="StopHold")


def test_a_square_off_refused_after_the_retries_places_the_stop_again_each_time(world):
    # Five refusals use the schedule up; the kernel's own re-send at 15:01:01 is refused
    # too, and the one at 15:02:01 fills.
    def refuse_six():
        for _ in range(EXIT_ATTEMPTS + 1):
            world.fake.reject_next("RMS: square-off blocked", price_type="MARKET")

    world.script.at(ist(14, 59, 50), refuse_six)
    world.run(StopHold)

    stops = [o.status for o in world.orders() if o.price_type == "SL-M"]
    assert stops == ["cancelled", "cancelled", "cancelled"]
    exits = [o.status for o in world.orders() if o.price_type == "MARKET" and o.action == "SELL"]
    assert exits == ["rejected"] * (EXIT_ATTEMPTS + 1) + ["complete"]
    assert world.fake.position("SBIN", "NSE") == 0
    check_invariants(world.timeline, strategy="StopHold")


class BuysWhenFlat(StopHold):
    """Buys 10 SBIN with a stop at every decision at which it holds none."""

    def on_bar(self, bar):
        self.calls += 1
        if not self.position.qty:
            self.buy(self.qty, **type(self).rules)


def test_F45_the_daily_loss_on_ticks_exits_mid_bar_and_locks_entries(world):
    at_price(world, ist(14, 43, 30), 497.5)
    outcome = world.run(BuysWhenFlat, max_daily_loss=20)

    exit_ = next(o for o in world.orders() if (o.action, o.price_type) == ("SELL", "MARKET"))
    assert ist(14, 43, 30) <= exit_.placed_ns < ist(14, 44)
    assert [o.action for o in world.orders() if o.action == "BUY"] == ["BUY"]
    assert world.fake.position("SBIN", "NSE") == 0
    reasons = outcome.outputs["BuysWhenFlat"]["trades"]["exit_reason"]
    assert reasons == ["daily_loss_limit"]
    check_invariants(world.timeline, strategy="BuysWhenFlat")


# ----- the day's second trade, and the freeze quantity -----------------------------------


class TwoTrades(TargetHold):
    """TargetHold that buys again with a stop once flat after its target."""

    def on_bar(self, bar):
        self.calls += 1
        if not self.position.qty and self.calls in (1, 4):
            self.buy(self.qty, **type(self).rules)


def test_F69_a_second_trade_of_the_day_records_its_own_stop(world):
    at_price(world, ist(14, 43), 502.0)
    at_price(world, ist(14, 44), 500.0)
    world.stop_at(ist(14, 47))
    outcome = world.run(TwoTrades, on_stop="keep")

    stops = [(o.order_id, o.status) for o in world.orders() if o.price_type == "SL-M"]
    assert [status for _, status in stops] == ["cancelled", "trigger pending"]
    assert journal_stop(world, outcome, "TwoTrades") == stops[-1][0]
    check_invariants(world.timeline, strategy="TwoTrades")


class ScalesIn(OptionHold):
    """Buys 20 lots of the option with a stop 5 points below, then 10 lots more at its
    third decision."""

    rules = {"stop": bt.points(5)}

    def on_bar(self, bar):
        self.calls += 1
        if self.calls in (1, 3):
            self.buy(OPTION_LOT * (20 if self.calls == 1 else 10), **type(self).rules)


def test_F91_an_add_that_would_pass_the_freeze_quantity_is_rejected(
    make_fake_openalgo, monkeypatch, tmp_path
):
    option = option_world(make_fake_openalgo, monkeypatch, tmp_path, price=10.0)
    option.stop_at(ist(14, 46))
    outcome = option.run(ScalesIn, on_stop="keep")

    assert [o.quantity for o in option.orders() if o.action == "BUY"] == [OPTION_LOT * 20]
    assert "freeze_qty" in outcome.outputs["ScalesIn"]["orders"]["reject_code"]
    (stop,) = [o for o in option.orders() if o.price_type == "SL"]
    assert stop.quantity == OPTION_LOT * 20 <= 1800
    check_invariants(option.timeline, strategy="ScalesIn")


def test_the_stop_unfilled_event_is_not_written_when_nothing_crossed(world):
    world.stop_at(ist(14, 44))
    outcome = world.run(StopHold, on_stop="keep")

    assert events(outcome, "stop_crossed") == []
    assert events(outcome, "repriced") == []
