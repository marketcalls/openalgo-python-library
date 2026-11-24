"""A live session's GTTs through the binding (`Engine.session`;
rust/oa_py/src/backtest/engine/gtts.rs and live_state.rs; design 20-positional-live, unit
N1): a CNC leg's stop as a "place_gtt" command, modified by a trail, cancelled before a
target's exit, fired with its child filled, refused; the answers that do not fit refused
by kind; and a session carried to the next trading day through JSON.

No server is involved: the test plays the venue. The session is the NSE day 2025-07-01 on
1-minute slices from 09:15 IST, a 1-minute strategy, bars flat at 500.00; the CNC buy
(order 1) is decided at the close of minute 0 and filled at 500.00 at the open of minute
1, so a stop 3 points below is a GTT (g1) triggering at 497.00 with its child selling 1
percent below it, 492.03 down to the tick: 492.00.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

import numpy as np
import pytest

from openalgo.backtest._calendar_convert import native_attr, to_day
from openalgo.backtest.calendar import default_calendar
from openalgo.backtest.errors import VenueRefusalError

DAY = dt.date(2025, 7, 1)
NEXT_DAY = dt.date(2025, 7, 2)
MINUTE_NS = 60 * 1_000_000_000
SECOND_NS = 1_000_000_000
SLICES = 375
NO_BARS = (np.array([], dtype=np.int64), *(np.array([], dtype=np.float64) for _ in range(5)))
ENTRY, G1 = 1, 1
SBIN = {
    "symbol": "SBIN",
    "exchange": "NSE",
    "kind": "EQ",
    "lot_size": 1,
    "tick_size": 0.05,
    "multiplier": 1.0,
    "freeze_qty": None,
    "last_trading_day": None,
}
STOP_3 = {"stop": ("points", 3.0, None)}
STOP_TERMS = {
    "instrument": 0,
    "product": "CNC",
    "side": "SELL",
    "quantity_units": 10.0,
    "trigger_price": 497.0,
    "price": 492.0,
    "trigger_below": True,
}


def open_ns(day: dt.date) -> int:
    """09:15 IST of `day` (03:45 UTC), UTC nanoseconds."""
    return int(dt.datetime(day.year, day.month, day.day, 3, 45, tzinfo=dt.UTC).timestamp()) * 10**9


def at(minute: int, second: int = 0, day: dt.date = DAY) -> int:
    return open_ns(day) + minute * MINUTE_NS + second * SECOND_NS


def session_config(day: dt.date = DAY) -> dict[str, Any]:
    """A cost-free, slippage-free 1-minute run on 1-minute bars of `day`."""
    return {
        "first_day": to_day(day),
        "last_day": to_day(day),
        "base_nanos": MINUTE_NS,
        "strategy_interval": MINUTE_NS,
        "fill": "next_open",
        "quantity_mode": "venue",
        "capital": 1_000_000.0,
        "costs": None,
        "slippage": ("zero", 0.0, 0),
        "risk": {
            "risk_basis": "gross",
            "after_risk_exit": "lock_session",
            "round_levels": True,
            "strict_config": True,
        },
        "deliver_events": False,
    }


def new_session(day: dt.date = DAY) -> Any:
    engine_class = native_attr("Engine")
    calendar = default_calendar()._cal
    return engine_class.session(calendar, session_config(day), [SBIN], [NO_BARS])


def release_minute(engine: Any, minute: int, day: dt.date = DAY) -> tuple:
    """Hand the session the flat bar of `minute`, release it and return its next pause."""
    assert engine.advance()[0] == "waiting"
    engine.add_bar(0, (at(minute, day=day), 500.0, 500.0, 500.0, 500.0, 1_000.0))
    engine.release(minute + 1)
    return engine.advance()


def cnc_buy(rules: dict[str, Any]) -> dict[str, Any]:
    return {
        "instrument": 0,
        "side": "BUY",
        "quantity": {"kind": "units", "value": 10.0},
        "order_type": "MARKET",
        "validity": "DAY",
        "product": "CNC",
        "reduce_only": False,
        "reverse": False,
        "rules": rules,
    }


def holding(rules: dict[str, Any]) -> Any:
    """A session that bought 10 SBIN in CNC with `rules` at 500.00 at the open of minute
    1, waiting before minute 1, the commands of the fill left to take."""
    engine = new_session()
    assert release_minute(engine, 0)[0] == "bar_close"
    outcome, order_id, _ = engine.submit(cnc_buy(rules))
    assert (outcome, order_id) == ("submitted", ENTRY)
    assert engine.advance()[0] == "waiting"
    engine.take_commands()
    engine.apply_fill(ENTRY, 10.0, 500.0, at(1))
    return engine


def protected(rules: dict[str, Any] | None = None) -> Any:
    """`holding` with STOP_3 (and `rules`), g1's place taken and confirmed."""
    engine = holding({**STOP_3, **(rules or {})})
    engine.take_commands()
    engine.apply_gtt_placed(G1)
    return engine


def placed(engine: Any) -> list[dict[str, Any]]:
    return [command for command in engine.take_commands() if command["command"] == "place"]


def test_a_cnc_entry_that_fills_places_its_stop_as_a_gtt():
    engine = holding(STOP_3)
    expected = {"command": "place_gtt", "gtt_id": G1, "order_id": None, **STOP_TERMS}
    assert engine.take_commands() == [expected]


def test_a_confirmed_gtt_rests_and_is_queried_by_id_and_by_position():
    engine = holding(STOP_3)
    engine.take_commands()
    assert engine.gtt(G1)["state"] == "Requested"
    slice_index, slice_ts, step = engine.apply_gtt_placed(G1)
    assert slice_ts == at(1)
    view = engine.gtt(G1)
    assert (view["state"], view["purpose"], view["order_id"]) == ("Active", "stop", None)
    assert (view["pending"], view["child_order_id"]) == (None, None)
    assert {key: view[key] for key in STOP_TERMS} == STOP_TERMS
    assert [g["gtt_id"] for g in engine.gtts(0, "CNC")] == [G1]
    assert engine.gtts(0, "MIS") == []
    assert engine.gtt(9) is None


def test_a_trail_modifies_the_stop_gtt_and_the_venue_confirms_it():
    # 502.00 arms the 2-point trail: the stop 500.00, its child 495.00.
    engine = protected({"trail": (("points", 2.0, None), None)})
    engine.push_tick(0, 502.0, at(1, 5))
    moved = {**STOP_TERMS, "trigger_price": 500.0, "price": 495.0}
    assert engine.take_commands() == [{"command": "modify_gtt", "gtt_id": G1, **moved}]
    assert engine.gtt(G1)["pending"] == moved
    engine.apply_gtt_modified(G1)
    assert engine.gtt(G1)["trigger_price"] == 500.0


def test_a_target_cancels_the_stop_gtt_and_exits_once_the_cancel_is_confirmed():
    engine = protected({"target": ("points", 2.0, None)})
    engine.push_tick(0, 502.0, at(1, 5))
    assert engine.take_commands() == [{"command": "cancel_gtt", "gtt_id": G1, **STOP_TERMS}]
    engine.apply_gtt_cancelled(G1)
    (target_exit,) = placed(engine)
    assert (target_exit["role"], target_exit["quantity_units"]) == ("exit", 10.0)
    assert engine.gtt(G1)["state"] == "Canceled"


def test_a_fired_gtt_is_followed_as_its_child_order():
    engine = protected()
    child = engine.apply_gtt_triggered(G1, at(1, 30))
    assert child == 2
    order = engine.order(child)
    assert (order["side"], order["order_type"], order["limit_rupees"]) == ("SELL", "LIMIT", 492.0)
    assert engine.gtt(G1)["child_order_id"] == child
    engine.apply_fill(child, 10.0, 497.0, at(1, 31))
    assert engine.position(0, "CNC")["quantity_units"] == 0.0
    assert engine.gtt(G1)["state"] == "ChildFilled"


def test_a_refused_stop_gtt_exits_the_leg_and_locks_entries():
    engine = holding(STOP_3)
    engine.take_commands()
    engine.apply_gtt_refused(G1)
    (exit_order,) = placed(engine)
    assert (exit_order["role"], exit_order["reason"]) == ("risk_exit", "stop_rejected")
    assert engine.entry_locks() == [native_attr("GTT_REFUSED_LOCK")] == ["gtt_refused"]


def test_answers_that_do_not_fit_are_refused_by_kind_and_change_nothing():
    engine = protected()
    with pytest.raises(VenueRefusalError) as unknown:
        engine.apply_gtt_placed(9)
    assert unknown.value.kind == "unknown_gtt"
    with pytest.raises(VenueRefusalError) as second:
        engine.apply_gtt_placed(G1)
    assert second.value.kind == "gtt_state"
    assert engine.gtt(G1)["state"] == "Active"
    assert engine.take_commands() == []


def test_a_gtt_order_against_the_position_held_is_not_built_in_a_session():
    # A GTT sale of 10 at 495.00 beside g1, not reduce-only: both GTTs could fire at the
    # broker and sell 20 against the 10 held.
    engine = protected()
    assert release_minute(engine, 1)[0] == "bar_close"
    sale = {**cnc_buy({}), "side": "SELL", "order_type": "LIMIT", "price": 495.0}
    with pytest.raises(NotImplementedError, match="GTT order against the position held"):
        engine.submit({**sale, "validity": "GTT"})
    assert engine.take_commands() == []


def test_a_stop_gtt_left_resting_on_a_flat_position_keeps_entries_out_and_is_cancelled_again():
    # The broker sells the 10 by itself and refuses g1's cancellation: a new CNC buy is
    # rejected live_in_flight, and the cancellation goes again once the session reaches
    # the next minute.
    engine = protected()
    engine.apply_external_fill(0, "CNC", "SELL", 10.0, 499.0, at(1, 40), "manual")
    (cancel,) = engine.take_commands()
    assert (cancel["command"], cancel["gtt_id"]) == ("cancel_gtt", G1)
    engine.apply_gtt_cancel_refused(G1)
    assert release_minute(engine, 1)[0] == "bar_close"
    outcome, _, code = engine.submit(cnc_buy(STOP_3))
    assert (outcome, code) == ("rejected", "live_in_flight")
    assert engine.advance()[0] == "waiting"
    (again,) = engine.take_commands()
    assert (again["command"], again["gtt_id"]) == ("cancel_gtt", G1)


def test_a_gtt_that_ends_is_named_by_one_of_the_kernels_ends():
    engine = protected()
    with pytest.raises(Exception, match="GTT end"):
        engine.apply_gtt_ended(G1, "lapsed")
    engine.apply_gtt_ended(G1, "expired")
    (again,) = engine.take_commands()
    assert (again["command"], again["gtt_id"]) == ("place_gtt", 2)


def test_the_gtt_vocabularies_are_the_kernels():
    assert native_attr("GTT_ENDS") == ("expired", "rejected")
    states = native_attr("GTT_STATES")
    assert {"Requested", "Active", "ModifyPending", "CancelPending", "VenueEnded"} <= set(states)


def test_a_session_carried_through_json_restores_into_the_next_trading_day():
    # Day 1 closes holding 10 with g1 resting; its carried state goes through JSON and
    # into day 2's session: the position, its leg and g1 (renumbered g1) come back.
    engine = protected()
    for minute in range(1, SLICES):
        release_minute(engine, minute)
    assert engine.advance()[0] == "finished"
    carried = json.loads(json.dumps(engine.carried_state()))
    assert carried["kind"] == "carried"
    tomorrow = new_session(NEXT_DAY)
    ids = tomorrow.restore_carried_state(carried)
    assert ids == {"gtts": {G1: 1}, "orders": {}}
    assert tomorrow.position(0, "CNC")["quantity_units"] == 10.0
    view = tomorrow.gtt(1)
    assert (view["state"], view["trigger_price"]) == ("Active", 497.0)
    with pytest.raises(Exception, match="later trading day"):
        new_session(DAY).restore_carried_state(carried)
