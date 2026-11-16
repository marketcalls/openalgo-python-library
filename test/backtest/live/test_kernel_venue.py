"""The kernel's live session through the binding (`Engine.session`): its venue is
external, so every order goes out as a command dict and the venue's fills, rejections and
cancellations come back (rust/oa_py/src/backtest/engine/venue.rs).

No server is involved: the test plays the venue. The session is the NSE day 2025-07-01
on 1-minute slices from 09:15 IST, one instrument (SBIN), a 1-minute strategy.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pytest

from openalgo.backtest._calendar_convert import native_attr, to_day
from openalgo.backtest.calendar import default_calendar

DAY = dt.date(2025, 7, 1)
MINUTE_NS = 60 * 1_000_000_000
# 09:15 IST is 03:45 UTC.
OPEN_NS = int(dt.datetime(2025, 7, 1, 3, 45, tzinfo=dt.UTC).timestamp()) * 10**9
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
NO_BARS = (np.array([], dtype=np.int64), *(np.array([], dtype=np.float64) for _ in range(5)))


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


def new_session() -> Any:
    engine_class = native_attr("Engine")
    return engine_class.session(default_calendar()._cal, session_config(), [SBIN], [NO_BARS])


def new_backtest() -> Any:
    """A backtest of the bar of minute 0."""
    ts_open, *prices = bar(0)
    arrays = (
        np.array([ts_open], dtype=np.int64),
        *(np.array([price], dtype=np.float64) for price in prices),
    )
    return native_attr("Engine")(default_calendar()._cal, session_config(), [SBIN], [arrays])


def release_minute(engine: Any, minute: int) -> tuple:
    """Hand the session the bar of `minute`, release it and return its next pause."""
    assert engine.advance()[0] == "waiting"
    engine.add_bar(0, bar(minute))
    engine.release(minute + 1)
    return engine.advance()


def buy(units: float, **fields: Any) -> dict[str, Any]:
    return {
        "instrument": 0,
        "side": "BUY",
        "quantity": {"kind": "units", "value": units},
        "order_type": "MARKET",
        "validity": "DAY",
        "product": "MIS",
        "reduce_only": False,
        "reverse": False,
        **fields,
    }


def session_with_order(request: dict[str, Any]) -> tuple[Any, int]:
    """A session that submitted `request` at the close of minute 0 and waits before
    minute 1; the order's id."""
    engine = new_session()
    assert release_minute(engine, 0)[0] == "bar_close"
    outcome, order_id, _ = engine.submit(request)
    assert outcome == "submitted"
    assert engine.advance()[0] == "waiting"
    return engine, order_id


def test_a_session_names_its_venue_and_a_backtest_its_own():
    assert new_session().venue() == "external"
    assert new_backtest().venue() == "simulated"


def test_an_order_is_handed_to_the_venue_as_a_place_command():
    engine, order_id = session_with_order(buy(100, order_type="LIMIT", price=499.5))
    assert engine.take_commands() == [
        {
            "command": "place",
            "order_id": order_id,
            "instrument": 0,
            "product": "MIS",
            "side": "BUY",
            "quantity_units": 100.0,
            "order_type": "LIMIT",
            "price": 499.5,
            "trigger_price": None,
        }
    ]
    assert engine.take_commands() == []


def test_the_venues_fills_are_recorded_at_its_price_and_add_up():
    engine, order_id = session_with_order(buy(100))
    first_minute = OPEN_NS + MINUTE_NS
    assert engine.apply_fill(order_id, 40.0, 500.10, first_minute) == (1, first_minute, 2)
    assert engine.order(order_id)["status"] == "PartiallyFilled"
    engine.apply_fill(order_id, 60.0, 500.20, first_minute + 1)
    assert engine.order(order_id)["status"] == "Filled"
    position = engine.position(0, "MIS")
    assert position["quantity_units"] == 100.0
    assert position["average_price_rupees"] == pytest.approx(500.16)


def test_a_cancel_is_sent_and_ends_the_order_once_the_venue_confirms_it():
    engine, order_id = session_with_order(buy(10, order_type="LIMIT", price=450.0))
    assert release_minute(engine, 1)[0] == "bar_close"
    assert engine.cancel(order_id) == "Submitted"
    assert engine.take_commands()[-1] == {"command": "cancel", "order_id": order_id}
    assert engine.advance()[0] == "waiting"
    engine.apply_cancel(order_id)
    assert engine.order(order_id)["status"] == "Canceled"


def test_a_rejection_carries_its_reject_code():
    engine, order_id = session_with_order(buy(10))
    engine.apply_reject(order_id, "venue")
    order = engine.order(order_id)
    assert (order["status"], order["reject_code"]) == ("Rejected", "venue")


def test_an_unknown_reject_code_is_an_input_error():
    engine, order_id = session_with_order(buy(10))
    with pytest.raises(native_attr("NativeEngineInputError"), match="unknown reject code"):
        engine.apply_reject(order_id, "broker_said_no")


def test_a_fill_of_more_units_than_the_order_has_left_is_an_input_error():
    engine, order_id = session_with_order(buy(10))
    with pytest.raises(native_attr("NativeEngineInputError"), match="10 units left"):
        engine.apply_fill(order_id, 11.0, 500.0, OPEN_NS + MINUTE_NS)


def test_a_result_for_a_backtest_is_refused_as_a_call_it_cannot_take():
    with pytest.raises(native_attr("NativeEngineError"), match="fills its own orders") as raised:
        new_backtest().apply_cancel(1)
    assert not isinstance(raised.value, native_attr("NativeEngineInputError"))


def test_a_stop_on_an_order_in_a_session_is_not_built_yet():
    engine = new_session()
    assert release_minute(engine, 0)[0] == "bar_close"
    with_stop = buy(10, rules={"stop": ("points", 4.0, None)})
    with pytest.raises(NotImplementedError, match="in a live session"):
        engine.submit(with_stop)
