"""The kernel's live session through the binding (`Engine.session`): its venue is
external, so every order goes out as a command dict and the venue's fills, rejections,
cancellations and answers come back (rust/oa_py/src/backtest/engine/venue.rs); a result
the kernel refuses raises `VenueRefusalError` with its kind.

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
from openalgo.backtest.errors import VENUE_REFUSAL_KINDS, VenueRefusalError

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
            "role": "entry",
            "reason": None,
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


def refusal_kind(call: Any, *args: Any) -> str:
    """The kind of the VenueRefusalError that `call(*args)` raises."""
    with pytest.raises(VenueRefusalError) as raised:
        call(*args)
    assert raised.value.kind in VENUE_REFUSAL_KINDS
    return raised.value.kind


def test_a_fill_of_more_units_than_the_order_has_left_is_refused_by_kind():
    engine, order_id = session_with_order(buy(10))
    kind = refusal_kind(engine.apply_fill, order_id, 11.0, 500.0, OPEN_NS + MINUTE_NS)
    assert kind == "too_many_units"
    assert engine.order(order_id)["filled_units"] == 0.0


def test_a_result_for_an_order_that_has_ended_is_refused_as_a_race():
    engine, order_id = session_with_order(buy(10))
    engine.apply_fill(order_id, 10.0, 500.0, OPEN_NS + MINUTE_NS)
    assert refusal_kind(engine.apply_cancel, order_id) == "order_ended"
    assert refusal_kind(engine.apply_reject, 99, "venue") == "unknown_order"


def test_a_result_for_a_backtest_is_refused_by_kind():
    assert refusal_kind(new_backtest().apply_cancel, 1) == "not_external"


def test_the_refusal_kinds_are_the_kernels():
    assert frozenset(native_attr("VENUE_REFUSAL_KINDS")) == VENUE_REFUSAL_KINDS


def test_a_reduce_only_fill_beyond_the_position_is_refused_for_the_excess():
    # Long 10 with a reduce-only LIMIT sale of 10 resting; a plain sale of 4 fills first,
    # then the broker fills the resting 10: 6 close the position, 4 are refused.
    engine, bought = session_with_order(buy(10))
    engine.apply_fill(bought, 10.0, 500.0, OPEN_NS + MINUTE_NS)
    assert release_minute(engine, 1)[0] == "bar_close"
    sell = {"side": "SELL", "quantity": {"kind": "units", "value": 10.0}}
    _, resting, _ = engine.submit(buy(10, order_type="LIMIT", price=505.0, reduce_only=True) | sell)
    _, sale, _ = engine.submit(buy(4) | {"side": "SELL"})
    assert engine.advance()[0] == "waiting"
    second_minute = OPEN_NS + 2 * MINUTE_NS
    engine.apply_fill(sale, 4.0, 500.0, second_minute)
    kind = refusal_kind(engine.apply_fill, resting, 10.0, 505.0, second_minute + 1)
    assert kind == "would_reverse"
    assert engine.position(0, "MIS")["quantity_units"] == 0.0
    assert engine.order(resting)["filled_units"] == 6.0


def test_a_cancellation_the_venue_refuses_leaves_the_order_working():
    engine, order_id = session_with_order(buy(10, order_type="LIMIT", price=450.0))
    assert release_minute(engine, 1)[0] == "bar_close"
    engine.cancel(order_id)
    assert engine.advance()[0] == "waiting"
    engine.apply_cancel_refused(order_id)
    assert engine.order(order_id)["status"] == "Submitted"
    assert refusal_kind(engine.apply_cancel_refused, order_id) == "not_pending"


def test_a_modification_takes_effect_once_the_venue_confirms_it():
    engine, order_id = session_with_order(buy(10, order_type="LIMIT", price=450.0))
    assert release_minute(engine, 1)[0] == "bar_close"
    engine.modify(order_id, buy(20, order_type="LIMIT", price=455.0))
    commands = engine.take_commands()
    assert commands[-1]["command"] == "modify"
    assert engine.order(order_id)["quantity_units"] == 10.0
    assert engine.advance()[0] == "waiting"
    engine.apply_modify(order_id)
    assert engine.order(order_id)["quantity_units"] == 20.0


def test_a_stop_on_an_order_in_a_session_is_not_built_yet():
    engine = new_session()
    assert release_minute(engine, 0)[0] == "bar_close"
    with_stop = buy(10, rules={"stop": ("points", 4.0, None)})
    with pytest.raises(NotImplementedError, match="in a live session"):
        engine.submit(with_stop)
