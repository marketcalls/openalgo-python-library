"""A live session's legs through the binding (`Engine.session`;
rust/oa_py/src/backtest/engine/legs.rs): the broker holds a leg's stop as an order in the
segment's shape, kept on the leg's units and trigger; the session's ticks (`push_tick`)
ratchet the trail, exit a target and mark the position; a stop that ends without a fill
protects the leg again; and the session's own calls (`protect`, `reprice_exit`, an exit at
a limit) say what they did.

No server is involved: the test plays the venue. The session is the NSE day 2025-07-01 on
1-minute slices from 09:15 IST, a 1-minute strategy, bars flat at 500.00; the buy (order
1) is decided at the close of minute 0 and filled at 500.00 at the open of minute 1, so a
stop 3 points below it is at 497.00 and the kernel's first stop order is order 2.
"""

from __future__ import annotations

import datetime as dt
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
NIFTY_CALL = {
    "symbol": "NIFTY31JUL2525000CE",
    "exchange": "NFO",
    "kind": "OPT",
    "lot_size": 75,
    "tick_size": 0.05,
    "multiplier": 1.0,
    "freeze_qty": None,
    "last_trading_day": to_day(dt.date(2025, 7, 31)),
}
STOP_3 = {"stop": ("points", 3.0, None)}


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


def new_session(instrument: dict[str, Any] = SBIN) -> Any:
    engine_class = native_attr("Engine")
    calendar = default_calendar()._cal
    return engine_class.session(calendar, session_config(), [instrument], [NO_BARS])


def release_minute(engine: Any, minute: int) -> tuple:
    """Hand the session the bar of `minute`, release it and return its next pause."""
    assert engine.advance()[0] == "waiting"
    engine.add_bar(0, bar(minute))
    engine.release(minute + 1)
    return engine.advance()


def order(side: str, units: float, **fields: Any) -> dict[str, Any]:
    return {
        "instrument": 0,
        "side": side,
        "quantity": {"kind": "units", "value": units},
        "order_type": "MARKET",
        "validity": "DAY",
        "product": "MIS",
        "reduce_only": False,
        "reverse": False,
        **fields,
    }


def holding(
    rules: dict[str, Any] | None,
    units: float = 10,
    price: float = 500.0,
    instrument: dict[str, Any] = SBIN,
) -> Any:
    """A session that bought `units` with `rules` at `price` at the open of minute 1,
    waiting before minute 1, the commands of the fill left to take."""
    engine = new_session(instrument)
    assert release_minute(engine, 0)[0] == "bar_close"
    outcome, order_id, _ = engine.submit(order("BUY", units, rules=rules))
    assert (outcome, order_id) == ("submitted", ENTRY)
    assert engine.advance()[0] == "waiting"
    engine.take_commands()
    engine.apply_fill(ENTRY, units, price, at(1))
    return engine


def holding_with_stop(rules: dict[str, Any] | None = None) -> Any:
    """Holding 10 with the stop 3 points below (and `rules`), the stop's place taken."""
    engine = holding({**STOP_3, **(rules or {})})
    engine.take_commands()
    return engine


def placed(engine: Any) -> list[dict[str, Any]]:
    return [command for command in engine.take_commands() if command["command"] == "place"]


@pytest.mark.parametrize("product", ["CNC", "NRML"])
def test_a_stop_rule_on_a_carry_product_is_not_built_in_a_session(product):
    engine = new_session()
    assert release_minute(engine, 0)[0] == "bar_close"
    with pytest.raises(NotImplementedError, match="CNC or NRML"):
        engine.submit(order("BUY", 10, product=product, rules=STOP_3))


@pytest.mark.parametrize(
    ("instrument", "units", "price", "shape"),
    [
        (SBIN, 10, 500.0, ("SL-M", None, 497.0)),
        # 5 percent below a sell stop at 120.30 is 114.285, down to the tick 114.25.
        (NIFTY_CALL, 75, 123.3, ("SL", 114.25, 120.3)),
    ],
    ids=["cash equity: SL-M", "an option: SL 5 percent beyond the trigger"],
)
def test_the_fill_that_opens_a_leg_places_its_stop_in_the_segments_shape(
    instrument, units, price, shape
):
    engine = holding(STOP_3, units=units, price=price, instrument=instrument)
    (stop,) = placed(engine)
    assert (stop["order_id"], stop["side"], stop["quantity_units"]) == (STOP, "SELL", units)
    assert (stop["role"], stop["reason"]) == ("leg_stop", None)
    assert (stop["order_type"], stop["price"], stop["trigger_price"]) == shape


def test_a_tick_through_the_stop_sends_nothing_the_broker_holds_it():
    engine = holding_with_stop()
    engine.push_tick(0, 496.0, at(1, 5))
    assert engine.take_commands() == []
    assert engine.position(0, "MIS")["quantity_units"] == 10.0


def test_a_tick_at_the_target_cancels_the_stop_then_exits():
    engine = holding_with_stop({"target": ("points", 2.0, None)})
    engine.push_tick(0, 502.0, at(1, 5))
    assert engine.take_commands() == [{"command": "cancel", "order_id": STOP}]
    engine.apply_cancel(STOP)
    (target_exit,) = placed(engine)
    assert (target_exit["role"], target_exit["quantity_units"]) == ("exit", 10.0)


def test_a_trail_moves_the_brokers_stop_by_a_modification():
    engine = holding_with_stop({"trail": (("points", 2.0, None), None)})
    engine.push_tick(0, 502.0, at(1, 5))
    (modify,) = engine.take_commands()
    assert modify["command"] == "modify" and modify["order_id"] == STOP
    assert (modify["order_type"], modify["trigger_price"], modify["quantity_units"]) == (
        "SL-M",
        500.0,
        10.0,
    )


@pytest.mark.parametrize("price", [0.0, -1.0, math.nan])
def test_a_tick_that_is_not_a_price_is_an_input_error(price):
    engine = holding_with_stop()
    with pytest.raises(native_attr("NativeEngineInputError"), match="tick"):
        engine.push_tick(0, price, at(1, 5))


def test_a_tick_at_a_bar_close_is_refused_with_its_kind():
    engine = new_session()
    assert release_minute(engine, 0)[0] == "bar_close"
    with pytest.raises(VenueRefusalError) as raised:
        engine.push_tick(0, 500.0, at(1))
    assert raised.value.kind == "at_bar_close"


@pytest.mark.parametrize(
    ("end", "role", "reason"),
    [
        (lambda engine: engine.apply_reject(STOP, "venue"), "risk_exit", "stop_rejected"),
        (lambda engine: engine.apply_cancel(STOP), "leg_stop", None),
    ],
    ids=["rejected: exit at market", "cancelled by the broker: placed again"],
)
def test_a_stop_that_ends_without_a_fill_protects_its_leg(end, role, reason):
    engine = holding_with_stop()
    end(engine)
    (protecting,) = placed(engine)
    assert (protecting["role"], protecting["reason"], protecting["quantity_units"]) == (
        role,
        reason,
        10.0,
    )


def test_a_partial_exit_cuts_the_stop_down_before_it_goes():
    engine = holding_with_stop()
    assert release_minute(engine, 1)[0] == "bar_close"
    reduce = order("SELL", 0, quantity={"kind": "fraction", "value": 0.5}, reduce_only=True)
    assert engine.submit(reduce)[0] == "submitted"
    (cut,) = engine.take_commands()
    assert (cut["command"], cut["order_id"], cut["quantity_units"]) == ("modify", STOP, 5.0)
    assert engine.advance()[0] == "waiting"
    engine.apply_modify(STOP)
    (partial,) = placed(engine)
    assert (partial["role"], partial["quantity_units"]) == ("exit", 5.0)


def test_a_plain_market_sale_beside_the_stop_goes_once_the_stop_is_cancelled():
    engine = holding_with_stop()
    assert release_minute(engine, 1)[0] == "bar_close"
    outcome, sale, _ = engine.submit(order("SELL", 10))
    assert outcome == "submitted"
    assert engine.take_commands() == [{"command": "cancel", "order_id": STOP}]
    assert engine.advance()[0] == "waiting"
    engine.apply_cancel(STOP)
    (sent,) = placed(engine)
    assert (sent["order_id"], sent["role"], sent["quantity_units"]) == (sale, "exit", 10.0)


def test_a_stop_fill_beyond_the_position_is_refused_as_would_reverse():
    # A plain sale resting beside the stop fills first; the broker then fills the stop.
    engine = holding_with_stop()
    assert release_minute(engine, 1)[0] == "bar_close"
    outcome, sale, _ = engine.submit(order("SELL", 10, order_type="LIMIT", price=500.0))
    assert outcome == "submitted"
    assert engine.advance()[0] == "waiting"
    engine.apply_fill(sale, 10, 500.0, at(2))
    with pytest.raises(VenueRefusalError) as raised:
        engine.apply_fill(STOP, 10, 497.0, at(2, 5))
    assert raised.value.kind == "would_reverse"
    assert engine.position(0, "MIS")["quantity_units"] == 0.0


def test_protect_says_what_it_did():
    assert holding_with_stop().protect(0, "MIS") == "stop_works"
    assert holding(None).protect(0, "MIS") == "no_stop_level"
    engine = holding_with_stop()
    engine.request_exit(0, "MIS", "kill_switch")
    assert engine.protect(0, "MIS") == "exiting"
    engine.apply_cancel(STOP)
    (kill,) = placed(engine)
    engine.apply_reject(kill["order_id"], "venue")
    assert engine.protect(0, "MIS") == "placed"
    (stop,) = placed(engine)
    assert (stop["role"], stop["trigger_price"], stop["quantity_units"]) == (
        "leg_stop",
        497.0,
        10.0,
    )


def test_reprice_exit_modifies_the_stop_to_a_limit_for_its_total_units():
    engine = holding_with_stop()
    engine.apply_fill(STOP, 4, 497.0, at(1, 5))
    assert engine.reprice_exit(STOP, 495.0) is True
    (reprice,) = engine.take_commands()
    assert (reprice["command"], reprice["order_type"], reprice["price"]) == (
        "modify",
        "LIMIT",
        495.0,
    )
    assert (reprice["trigger_price"], reprice["quantity_units"]) == (None, 10.0)


def test_reprice_exit_refuses_an_order_that_is_neither_a_stop_nor_an_engine_exit():
    engine = holding_with_stop()
    assert release_minute(engine, 1)[0] == "bar_close"
    outcome, add, _ = engine.submit(order("BUY", 10, order_type="LIMIT", price=495.0))
    assert outcome == "submitted"
    with pytest.raises(native_attr("NativeEngineInputError"), match="neither"):
        engine.reprice_exit(add, 495.0)


def test_reprice_exit_refuses_an_order_that_has_ended_or_a_limit_off_the_tick():
    engine = holding_with_stop()
    with pytest.raises(VenueRefusalError) as ended:
        engine.reprice_exit(ENTRY, 495.0)
    assert ended.value.kind == "order_ended"
    with pytest.raises(native_attr("NativeEngineInputError"), match="tick"):
        engine.reprice_exit(STOP, 495.03)


def test_an_exit_at_a_limit_cancels_the_stop_then_goes_as_a_limit_order():
    engine = holding_with_stop()
    asked = engine.request_exit(0, "MIS", "stop_unfilled", limit_rupees=490.0)
    assert asked == "waits_for_cancels"
    assert engine.take_commands() == [{"command": "cancel", "order_id": STOP}]
    engine.apply_cancel(STOP)
    (limit_exit,) = placed(engine)
    assert (limit_exit["order_type"], limit_exit["price"]) == ("LIMIT", 490.0)
    assert (limit_exit["role"], limit_exit["reason"]) == ("risk_exit", "stop_unfilled")


def test_protect_waits_while_a_resting_sale_is_cancelled_for_the_stops_room():
    engine = holding_with_stop()
    engine.request_exit(0, "MIS", "kill_switch")
    engine.apply_cancel(STOP)
    (kill,) = placed(engine)
    engine.apply_reject(kill["order_id"], "venue")
    assert release_minute(engine, 1)[0] == "bar_close"
    target = order("SELL", 10, order_type="LIMIT", price=520.0, reduce_only=True)
    outcome, sale, _ = engine.submit(target)
    assert outcome == "submitted"
    assert engine.advance()[0] == "waiting"
    engine.take_commands()
    assert engine.protect(0, "MIS") == "waits_for_cancels"
    assert engine.take_commands() == [{"command": "cancel", "order_id": sale}]
    engine.apply_cancel(sale)
    (stop,) = placed(engine)
    assert (stop["role"], stop["quantity_units"]) == ("leg_stop", 10.0)


def test_the_protect_vocabulary_is_the_kernels():
    assert native_attr("PROTECT_OUTCOMES") == (
        "placed",
        "waits_for_cancels",
        "stop_works",
        "exiting",
        "no_stop_level",
        "flat",
        "market_closed",
    )
