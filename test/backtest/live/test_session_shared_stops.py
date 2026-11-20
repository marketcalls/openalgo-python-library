"""Each member's broker-held stop on a shared instrument, on the fake OpenAlgo
(`_live/stops.py`, `_live/_groups.py`; 18-multi-strategy 4.3 to 4.5 and 4.9, rows F94,
F96, F97, F104, F114 and F116, unit M6): a member's target, stops, trailing and
square-off act on its own orders only, ticks and quotes reach every member, and one
member's unanswered placement holds the other's stop with a reason of its own.

The market and the strategies are `shared_samples`': SBIN at 500.00, A (`SbinTrend`, 100
units, stop 497.00, target 502.00) and B (`SbinBreakout`, 50 units, stop 497.00) buy at
14:42:01 IST, A first. Every test runs the session invariants, the sum rule (I4), the
isolation (I5) and one side (I6) among them.
"""

from __future__ import annotations

import json
import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, TRADE_EVERY_S, BuyAndHold, Hold, alerts, ist
from shared_samples import A_UNITS, B_UNITS, SbinBreakout, SbinTrend, sbin_orders, terms
from stops_samples import STOP_POINTS, StopHold, at_price, stop_world

import openalgo.backtest as bt
from openalgo.backtest._live._session_resolved import AMBIGUOUS_LOCK
from openalgo.backtest._live._stops_cover import INSTRUMENT_HELD
from openalgo.backtest._live._types import LOGGER_NAME

#: A's target is reached: SBIN trades at 502.00 from 14:44.
TARGET_AT = ist(14, 44)
#: The square-off of NSE's MIS positions.
SQUARE_OFF = ist(15, 0)
STOP_TYPES = ("SL", "SL-M")


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return stop_world(make_fake_openalgo, monkeypatch, tmp_path)


def calls_on(world, endpoint: str, order_id: str) -> list:
    """The fake's calls of `endpoint` on broker order `order_id`."""
    calls = world.fake.log.calls(endpoint)
    return [c for c in calls if c.detail["request"].get("orderid") == order_id]


def stop_of(world, strategy: str):
    """`strategy`'s first stop at the fake."""
    return next(o for o in sbin_orders(world, strategy) if o.price_type in STOP_TYPES)


def never_stop_and_exit_together(world, strategy: str) -> None:
    """At no change at the fake does `strategy` have a stop and an exit working at once."""
    working: dict[str, object] = {}
    for change in world.trace.changes:
        if change.strategy != strategy:
            continue
        working[change.order_id] = change
        live = [c for c in working.values() if c.working]
        stops = [c for c in live if c.price_type in STOP_TYPES]
        exits = [c for c in live if c.price_type not in STOP_TYPES and c.action == "SELL"]
        assert not (stops and exits), (strategy, change)


def test_F94_a_target_of_one_member_cancels_its_stop_then_exits_it_alone(world):
    at_price(world, TARGET_AT, 502.0)
    outcome = world.run([SbinTrend, SbinBreakout])

    assert terms(sbin_orders(world, "SbinTrend")) == [
        ("BUY", A_UNITS, "MARKET", "complete"),
        ("SELL", A_UNITS, "SL-M", "cancelled"),
        ("SELL", A_UNITS, "MARKET", "complete"),
    ]
    assert outcome.outputs["SbinTrend"]["trades"]["exit_reason"] == ["leg_target"]
    b_stop = stop_of(world, "SbinBreakout")
    assert calls_on(world, "modifyorder", b_stop.order_id) == []
    # B's stop is cancelled only for its own square-off.
    assert all(c.ts_ns >= SQUARE_OFF for c in calls_on(world, "cancelorder", b_stop.order_id))
    never_stop_and_exit_together(world, "SbinTrend")


def test_F94_the_net_never_falls_below_b_while_b_holds_its_units(world):
    at_price(world, TARGET_AT, 502.0)
    world.run([SbinTrend, SbinBreakout])

    # Every change but B's own (its square-off exit fills before its kernel hears it).
    for change in world.trace.changes:
        held = {m.strategy: m.units for m in change.members or ()}
        if held.get("SbinBreakout") == B_UNITS and change.strategy != "SbinBreakout":
            assert change.position >= B_UNITS, change


class StopHalfPointCloser(StopHold):
    """A B of 50 units with its stop 2.5 points below (497.50)."""

    qty = B_UNITS
    rules = {"stop": bt.points(2.5)}


class StopHundred(StopHold):
    """An A of 100 units with its stop 3 points below (497.00)."""

    qty = A_UNITS


def test_F96_both_stops_filled_by_one_trade_close_both_legs_and_nothing_is_sent(world):
    fall = ist(14, 44)
    at_price(world, fall, 496.0)
    outcome = world.run([StopHundred, StopHalfPointCloser])

    assert terms(sbin_orders(world, "StopHundred"))[1] == ("SELL", A_UNITS, "SL-M", "complete")
    assert terms(sbin_orders(world, "StopHalfPointCloser"))[1] == (
        "SELL",
        B_UNITS,
        "SL-M",
        "complete",
    )
    assert world.fake.position("SBIN", "NSE") == 0
    # The stops' fills close both legs: no order is placed after the fall.
    assert [o for o in world.orders() if o.placed_ns >= fall] == []
    for name in ("StopHundred", "StopHalfPointCloser"):
        assert outcome.outputs[name]["trades"]["exit_reason"] == ["leg_sl"]


def test_F97_the_square_off_cancels_then_exits_each_member_and_ends_flat(world):
    outcome = world.run([SbinTrend, SbinBreakout])

    for name, units in (("SbinTrend", A_UNITS), ("SbinBreakout", B_UNITS)):
        assert terms(sbin_orders(world, name)) == [
            ("BUY", units, "MARKET", "complete"),
            ("SELL", units, "SL-M", "cancelled"),
            ("SELL", units, "MARKET", "complete"),
        ]
        exit_ = sbin_orders(world, name)[-1]
        assert SQUARE_OFF <= exit_.placed_ns < SQUARE_OFF + 10 * NS
        assert outcome.outputs[name]["trades"]["exit_reason"] == ["scheduler"]
        never_stop_and_exit_together(world, name)
    assert world.fake.position("SBIN", "NSE") == 0
    assert outcome.open_positions == ()


class Trails(StopHold):
    """A of 100 units whose stop trails 3 points below the best price."""

    qty = A_UNITS
    rules = {"stop": bt.points(STOP_POINTS), "trail": bt.trail_points(3)}


class TrailsTooB(StopHold):
    """B of 50 units whose stop trails 4 points below the best price."""

    qty = B_UNITS
    rules = {"stop": bt.points(4), "trail": bt.trail_points(4)}


def rising(world, *, start: int, step: float, until: int) -> None:
    """SBIN rises by `step` rupees each trade from `start` until `until`, then holds."""

    def price(now_ns: int) -> float:
        if now_ns < start:
            return 500.0
        trades = (min(now_ns, until) - start) // (TRADE_EVERY_S * NS)
        return round(500.0 + step * (trades + 1), 2)

    world.script.price = price


def test_F114_each_trailing_stop_is_modified_by_its_own_member_to_its_own_level(world):
    rising(world, start=ist(14, 43), step=0.5, until=ist(14, 46))
    world.stop_at(ist(14, 48))
    outcome = world.run([Trails, TrailsTooB], on_stop="keep")

    # The best price is 509.50 (19 rises of 0.50): each kernel trails its own distance.
    for name, distance in (("Trails", 3.0), ("TrailsTooB", 4.0)):
        stop = stop_of(world, name)
        modifies = calls_on(world, "modifyorder", stop.order_id)
        assert len(modifies) >= 2
        assert all(c.detail["request"]["strategy"] == name for c in modifies)
        orders = outcome.outputs[name]["orders"]
        stops = [i for i, kind in enumerate(orders["order_type"]) if kind == "SL-M"]
        # The broker's trigger is the member's kernel level (the session kept the stop).
        assert stop.trigger_price == orders["trigger_rupees"][stops[-1]]
        assert 509.5 - distance - stop.trigger_price < 509.5 * 0.0025
        for first, second in zip(modifies, modifies[1:], strict=False):
            assert second.ts_ns - first.ts_ns >= 5 * NS


def test_F116_a_dead_feed_quotes_each_shared_instrument_once_for_both_members(world):
    dead_at, back_at = ist(14, 43, 30), ist(14, 44, 30)
    at_price(world, ist(14, 43, 40), 502.0)
    world.script.at(dead_at, world.fake.faults.feed_dead)
    world.script.at(back_at, world.fake.faults.feed_alive)
    world.stop_at(ist(14, 46))
    outcome = world.run([SbinTrend, SbinBreakout], on_stop="keep")

    reads = [c.ts_ns for c in world.fake.log.calls("quotes") if dead_at <= c.ts_ns < back_at]
    gaps = {later - earlier for earlier, later in zip(reads, reads[1:], strict=False)}
    assert len(reads) >= 5 and gaps == {2 * NS}
    # A's target is reached on a quote: A exits alone, B keeps its position and its stop.
    a_exit = sbin_orders(world, "SbinTrend")[-1]
    assert (a_exit.action, a_exit.price_type) == ("SELL", "MARKET")
    assert dead_at < a_exit.placed_ns < back_at
    assert outcome.outputs["SbinTrend"]["trades"]["exit_reason"] == ["leg_target"]
    b_stop = stop_of(world, "SbinBreakout")
    assert (b_stop.status, calls_on(world, "cancelorder", b_stop.order_id)) == (
        "trigger pending",
        [],
    )


class EntersSecond(StopHold):
    """An A that buys 50 SBIN at its second decision (14:43:01) with a stop 3 points below,
    the shape of B's entry."""

    qty = B_UNITS

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 2:
            self.buy(self.qty, **type(self).rules)


def test_F104_one_members_unanswered_entry_holds_the_others_stop_until_bound(world, caplog):
    faults = world.fake.faults

    def keep_b_unheard():
        faults.hold_order_updates()
        for _ in range(60):
            faults.gateway_error("orderstatus")

    def release_b():
        faults.release_order_updates()

    def a_unanswered():
        # A's entry reaches the broker with no answer, and the order book cannot be read
        # for a while, so A's intent stays unbound when B's fill is heard.
        faults.bad_gateway_page("placeorder", reached_broker=True)
        for _ in range(8):
            faults.gateway_error("orderbook")

    world.script.at(ist(14, 41, 59), keep_b_unheard)
    world.script.at(ist(14, 43), a_unanswered)
    world.script.at(ist(14, 43, 5), release_b)
    outcome = world.run([EntersSecond, SbinBreakout])

    a_entry, b_entry = (sbin_orders(world, name)[0] for name in ("EntersSecond", "SbinBreakout"))
    b_stop = stop_of(world, "SbinBreakout")
    # B's stop is placed once A's entry, of the same shape as B's, is bound to A's order.
    resolved = [e for e in events_of(outcome, "resolved") if e["strategy"] == "EntersSecond"]
    assert [(e["resolution"], e["order_id"]) for e in resolved] == [("bound", a_entry.order_id)]
    assert b_entry.order_id != a_entry.order_id
    assert resolved[0]["ts_ns"] <= b_stop.placed_ns
    windows = [w for w in outcome.unprotected if w.strategy == "SbinBreakout"]
    assert INSTRUMENT_HELD in [w.reason for w in windows]
    held = [t for t in alerts(caplog, "error") if "SbinBreakout: the stop of SBIN" in t]
    assert held and "EntersSecond's order" in held[0]


def test_an_ambiguous_order_on_a_shared_instrument_locks_every_members_entries(world, caplog):
    world.fake.faults.bad_gateway_page("placeorder", reached_broker=True)
    other = world.fake.client()

    def same_order_by_another_client():
        other.placeorder(
            strategy="BuyAndHold",
            symbol="SBIN",
            action="BUY",
            exchange="NSE",
            price_type="MARKET",
            product="MIS",
            quantity=10,
        )

    world.script.at(ist(14, 42, 2) + NS // 2, same_order_by_another_client)
    world.stop_at(ist(14, 45))
    outcome = world.run([BuyAndHold, Hold], on_stop="keep")

    locks = [(e["strategy"], e["reason"]) for e in events_of(outcome, "lock")]
    assert ("BuyAndHold", AMBIGUOUS_LOCK) in locks and ("Hold", AMBIGUOUS_LOCK) in locks
    (named,) = [text for text in alerts(caplog, "error") if "cannot be told apart" in text]
    assert "its entries and those of Hold are locked" in named


def events_of(outcome, kind: str) -> list[dict]:
    """The run's event-stream lines of `kind`."""
    with open(outcome.paths["events"], encoding="utf-8") as stream:
        return [event for line in stream if (event := json.loads(line))["kind"] == kind]
