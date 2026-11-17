"""A live session's saved state and the broker's own fills through the binding
(`Engine.live_state`, `Engine.restore_live_state`, `Engine.apply_external_fill`;
rust/oa_py/src/backtest/engine/live_state.rs): the state is plain data that survives a
JSON round trip, a session built for a restart takes it once after its catch-up and goes
on as the saved one would, and a fill the broker made with no order of the session's is
recorded or refused with its kind.

No server is involved: the test plays the venue. The session is the NSE day 2025-07-01 on
1-minute slices from 09:15 IST, a 1-minute strategy, bars flat at 500.00; the buy of 10
(order 1) is decided at the close of minute 0 and filled at 500.00 at the open of minute
1, with a stop 3 points below it at 497.00 (order 2). The state is saved while the
session waits before minute 1.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from typing import Any

import numpy as np
import pytest

from openalgo.backtest._calendar_convert import native_attr, to_day
from openalgo.backtest.calendar import default_calendar
from openalgo.backtest.errors import VenueRefusalError

DAY = dt.date(2025, 7, 1)
MINUTE_NS = 60 * 1_000_000_000
SECOND_NS = 1_000_000_000
# 09:15 IST is 03:45 UTC.
OPEN_NS = int(dt.datetime(2025, 7, 1, 3, 45, tzinfo=dt.UTC).timestamp()) * 10**9
NO_BARS = (np.array([], dtype=np.int64), *(np.array([], dtype=np.float64) for _ in range(5)))
ENTRY, STOP = 1, 2
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


def session_config() -> dict[str, Any]:
    """A cost-free, slippage-free 1-minute run on 1-minute bars of DAY."""
    return {
        "first_day": to_day(DAY),
        "last_day": to_day(DAY),
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


def bar(minute: int) -> tuple[int, float, float, float, float, float]:
    """The bar of minute `minute` after 09:15: flat at 500.00."""
    return (OPEN_NS + minute * MINUTE_NS, 500.0, 500.0, 500.0, 500.0, 1_000.0)


def at(minute: int, second: int = 0) -> int:
    """`second` seconds after the open of minute `minute`, UTC nanoseconds."""
    return OPEN_NS + minute * MINUTE_NS + second * SECOND_NS


def new_session() -> Any:
    engine_class = native_attr("Engine")
    return engine_class.session(default_calendar()._cal, session_config(), [SBIN], [NO_BARS])


def caught_up() -> Any:
    """A session built for a restart, caught up to the wait before minute 1: the bar of
    minute 0 added and released, nothing decided."""
    engine = new_session()
    assert engine.advance()[0] == "waiting"
    engine.add_bar(0, bar(0))
    engine.release(1)
    assert engine.advance()[0] == "bar_close"
    assert engine.advance()[0] == "waiting"
    return engine


def holding() -> Any:
    """The session of the module docs, waiting before minute 1 with its stop placed."""
    engine = caught_up_with_buy()
    engine.take_commands()
    engine.apply_fill(ENTRY, 10, 500.0, at(1))
    assert [command["role"] for command in engine.take_commands()] == ["leg_stop"]
    return engine


def caught_up_with_buy() -> Any:
    """A session that bought 10 with a stop 3 points below at the close of minute 0."""
    engine = new_session()
    assert engine.advance()[0] == "waiting"
    engine.add_bar(0, bar(0))
    engine.release(1)
    assert engine.advance()[0] == "bar_close"
    buy = {
        "instrument": 0,
        "side": "BUY",
        "quantity": {"kind": "units", "value": 10},
        "order_type": "MARKET",
        "validity": "DAY",
        "product": "MIS",
        "reduce_only": False,
        "reverse": False,
        "rules": {"stop": ("points", 3.0, None)},
    }
    assert engine.submit(buy)[:2] == ("submitted", ENTRY)
    assert engine.advance()[0] == "waiting"
    return engine


def test_the_saved_state_is_plain_data_that_survives_a_json_round_trip():
    state = holding().live_state()
    text = json.dumps(state, allow_nan=False)
    assert json.loads(text) == state
    assert state["layout"] == 2
    assert (state["slice"], state["stage"]) == (1, "ready")
    [position] = state["positions"]
    assert (position["product"], position["quantity_units"]) == ("MIS", 10.0)
    [stop] = state["orders"]
    request = stop["request"]
    assert (stop["order"], stop["role"], request["order_type"]) == (STOP, "leg_stop", "SL-M")
    assert request["trigger_rupees"] == 497.0


def test_a_restarted_session_takes_the_state_read_back_from_json_and_renumbers_its_orders():
    saved = json.loads(json.dumps(holding().live_state()))
    restarted = caught_up()
    assert restarted.restore_live_state(saved) == {STOP: 1}
    stop = restarted.order(1)
    assert (stop["quantity_units"], stop["trigger_rupees"]) == (10.0, 497.0)
    assert restarted.position(0, "MIS")["quantity_units"] == 10.0


def test_a_restarted_session_goes_on_as_the_uninterrupted_one():
    uninterrupted = holding()
    restarted = caught_up()
    ids = restarted.restore_live_state(uninterrupted.live_state())
    for engine, stop in ((uninterrupted, STOP), (restarted, ids[STOP])):
        engine.push_tick(0, 496.0, at(1, 20))
        engine.apply_fill(stop, 10, 497.0, at(1, 21))
    assert restarted.account() == uninterrupted.account()
    assert restarted.live_state() == uninterrupted.live_state()


def with_field(state: dict[str, Any], path: tuple[str, ...], value: Any) -> dict[str, Any]:
    """A copy of `state` with the value at `path` replaced."""
    copy = json.loads(json.dumps(state))
    *parents, last = path
    target = copy
    for key in parents:
        target = target[key]
    target[last] = value
    return copy


@pytest.mark.parametrize(
    ("what", "path", "value", "message"),
    [
        ("another layout", ("layout",), 3, "layout 3"),
        ("a number that is not finite", ("account", "capital_rupees"), math.nan, "finite"),
        ("a value that is not plain data", ("desk", "entry_locks"), {"starting"}, "set"),
        ("an order that was not saved working", ("desk", "cancelling"), [9], "cancelling"),
    ],
)
def test_a_state_that_cannot_be_read_is_refused_naming_the_field(what, path, value, message):
    state = with_field(holding().live_state(), path, value)
    restarted = caught_up()
    with pytest.raises(native_attr("NativeEngineInputError"), match=message):
        restarted.restore_live_state(state)
    assert restarted.working_orders() == [], f"{what}: nothing restored"


def test_a_state_nested_far_deeper_than_any_saved_one_is_refused_and_the_process_lives():
    nested: list[Any] = []
    for _ in range(5_000):
        nested = [nested]
    state = holding().live_state()
    state["instruments"] = nested
    restarted = caught_up()
    with pytest.raises(native_attr("NativeEngineInputError"), match="nested deeper than"):
        restarted.restore_live_state(state)
    assert restarted.working_orders() == []


def test_a_session_that_cannot_take_a_state_now_refuses_it():
    state = holding().live_state()
    restored = caught_up()
    restored.restore_live_state(state)
    with pytest.raises(native_attr("NativeEngineError"), match="restored already"):
        restored.restore_live_state(state)
    with pytest.raises(native_attr("NativeEngineError"), match="orders or positions"):
        caught_up_with_buy().restore_live_state(state)


def test_the_broker_selling_part_of_the_position_is_recorded_and_the_stop_cut():
    engine = holding()
    stamp = engine.apply_external_fill(0, "MIS", "SELL", 4, 499.0, at(1, 10), "manual")
    assert stamp == (1, at(1), 3)
    assert engine.position(0, "MIS")["quantity_units"] == 6.0
    [cut] = engine.take_commands()
    assert (cut["command"], cut["order_id"], cut["quantity_units"]) == ("modify", STOP, 6.0)


@pytest.mark.parametrize(
    ("what", "side", "units"),
    [("a buy on the long", "BUY", 1), ("a sale through zero", "SELL", 11)],
)
def test_a_broker_fill_that_would_not_only_close_or_reduce_is_refused(what, side, units):
    engine = holding()
    with pytest.raises(VenueRefusalError) as raised:
        engine.apply_external_fill(0, "MIS", side, units, 499.0, at(1, 10), "manual")
    assert raised.value.kind == "beyond_position", what
    assert engine.position(0, "MIS")["quantity_units"] == 10.0


@pytest.mark.parametrize(
    ("side", "reason", "message"),
    [("HOLD", "manual", "side"), ("SELL", "broker", "exit reason")],
)
def test_an_unknown_side_or_reason_of_a_broker_fill_is_refused(side, reason, message):
    with pytest.raises(native_attr("NativeEngineInputError"), match=message):
        holding().apply_external_fill(0, "MIS", side, 1, 499.0, at(1, 10), reason)
