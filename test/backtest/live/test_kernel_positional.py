"""A live session's positional rules of unit N4 through the binding (`Engine.session`;
rust/oa_py/src/backtest/engine/setup.rs, venue.rs, live_state.rs and pre_open.rs; design
20-positional-live sections 6, 8 and 9): `positional_stops="rearm_daily"` carried over a
night, a derivatives position at expiry, and a daily strategy's decision before the open.

No server is involved: the test plays the venue. Sessions run on 1-minute slices from 09:15
IST with no charges and no slippage, bars flat at 500.00.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

import numpy as np
import pytest

from openalgo.backtest._calendar_convert import native_attr, to_day
from openalgo.backtest.calendar import default_calendar

TUESDAY = dt.date(2025, 7, 1)
WEDNESDAY = dt.date(2025, 7, 2)
THURSDAY = dt.date(2025, 7, 3)
EXPIRY = dt.date(2025, 7, 31)
AFTER_EXPIRY = dt.date(2025, 8, 1)
MINUTE_NS = 60 * 1_000_000_000
SECOND_NS = 1_000_000_000
SLICES = 375
# The NFO square-off of 2025-07-31, 15:10 IST: the open of minute 355.
SQUARE_OFF = 355
NO_BARS = (np.array([], dtype=np.int64), *(np.array([], dtype=np.float64) for _ in range(5)))
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
ENTRY = 1


def open_ns(day: dt.date) -> int:
    """09:15 IST of `day` (03:45 UTC), UTC nanoseconds."""
    return int(dt.datetime(day.year, day.month, day.day, 3, 45, tzinfo=dt.UTC).timestamp()) * 10**9


def close_ns(day: dt.date) -> int:
    """15:30 IST of `day` (10:00 UTC), UTC nanoseconds."""
    return open_ns(day) + SLICES * MINUTE_NS


def at(minute: int, day: dt.date, second: int = 0) -> int:
    return open_ns(day) + minute * MINUTE_NS + second * SECOND_NS


def session(
    day: dt.date,
    instrument: dict[str, Any] = SBIN,
    *,
    positional_stops: str | None = None,
    interval: Any = MINUTE_NS,
) -> Any:
    """A cost-free, slippage-free session of `day` on 1-minute slices."""
    risk: dict[str, Any] = {
        "risk_basis": "gross",
        "after_risk_exit": "lock_session",
        "round_levels": True,
        "strict_config": True,
    }
    if positional_stops is not None:
        risk["positional_stops"] = positional_stops
    config = {
        "first_day": to_day(day),
        "last_day": to_day(day),
        "base_nanos": MINUTE_NS,
        "strategy_interval": interval,
        "fill": "next_open",
        "quantity_mode": "venue",
        "capital": 10_000_000.0,
        "costs": None,
        "slippage": ("zero", 0.0, 0),
        "risk": risk,
        "deliver_events": False,
    }
    engine_class = native_attr("Engine")
    return engine_class.session(default_calendar()._cal, config, [instrument], [NO_BARS])


def release(engine: Any, minute: int, day: dt.date, close: float = 500.0) -> tuple:
    """Hand the session the bar of `minute`, opening at 500.00 and closing at `close`,
    release it and return its next pause."""
    assert engine.advance()[0] == "waiting"
    low = min(close, 500.0)
    engine.add_bar(0, (at(minute, day), 500.0, max(close, 500.0), low, close, 1_000.0))
    engine.release(minute + 1)
    return engine.advance()


def order(side: str, units: float, product: str, rules: dict[str, Any]) -> dict[str, Any]:
    return {
        "instrument": 0,
        "side": side,
        "quantity": {"kind": "units", "value": units},
        "order_type": "MARKET",
        "validity": "DAY",
        "product": product,
        "reduce_only": False,
        "reverse": False,
        "rules": rules,
    }


def bought(engine: Any, day: dt.date, units: float, product: str, rules: dict) -> Any:
    """`engine` after a buy of `units` at 500.00 at the open of minute 1 of `day`."""
    assert release(engine, 0, day)[0] == "bar_close"
    assert engine.submit(order("BUY", units, product, rules))[:2] == ("submitted", ENTRY)
    assert engine.advance()[0] == "waiting"
    engine.take_commands()
    engine.apply_fill(ENTRY, units, 500.0, at(1, day))
    return engine


def to_close(
    engine: Any, day: dt.date, first: int, last_close: float = 500.0
) -> list[dict[str, Any]]:
    """Run `engine` from minute `first` to its close, the last bar closing at
    `last_close`, confirming each cancellation the close sends; the commands of the
    close."""
    for minute in range(first, SLICES):
        release(engine, minute, day, last_close if minute == SLICES - 1 else 500.0)
    # The last slice paused at its close; its step 8 is the session's close.
    waits = engine.advance()[0] == "waiting"
    closed = engine.take_commands()
    for command in closed:
        if command["command"] == "cancel":
            engine.apply_cancel(command["order_id"])
        if command["command"] == "cancel_gtt":
            engine.apply_gtt_cancelled(command["gtt_id"])
    if waits:
        assert engine.advance()[0] == "finished"
    return closed


def places(engine: Any) -> list[dict[str, Any]]:
    return [command for command in engine.take_commands() if command["command"] == "place"]


# positional_stops="rearm_daily" -----------------------------------------------------------


def rearm_carried() -> dict[str, Any]:
    """Tuesday closed holding 10 SBIN in CNC with a 3-point stop under rearm_daily; its
    carried state through JSON."""
    engine = bought(session(TUESDAY, positional_stops="rearm_daily"), TUESDAY, 10.0, "CNC", STOP_3)
    (stop,) = places(engine)
    assert (stop["role"], stop["order_type"], stop["trigger_price"]) == ("leg_stop", "SL-M", 497.0)
    closed = to_close(engine, TUESDAY, 1)
    assert closed == [{"command": "cancel", "order_id": stop["order_id"]}]
    return json.loads(json.dumps(engine.carried_state()))


def test_a_rearm_daily_stop_is_a_day_order_and_the_carried_state_holds_no_stop():
    carried = rearm_carried()
    assert (carried["orders"], carried["gtts"]) == ([], [])
    assert len(carried["legs"]) == 1


def test_a_rearm_daily_leg_waits_for_the_open_then_its_stop_is_placed_again():
    wednesday = session(WEDNESDAY, positional_stops="rearm_daily")
    ids = wednesday.restore_carried_state(rearm_carried())
    assert ids == {"gtts": {}, "orders": {}, "expired": []}
    assert wednesday.protect(0, "CNC") == "waits_for_open"
    assert wednesday.take_commands() == []
    wednesday.push_tick(0, 501.0, at(0, WEDNESDAY, 1))
    (stop,) = places(wednesday)
    assert (stop["role"], stop["side"], stop["trigger_price"]) == ("leg_stop", "SELL", 497.0)
    assert stop["flags"] == []


def test_an_unknown_positional_stops_is_refused():
    with pytest.raises(native_attr("NativeEngineInputError"), match="positional_stops"):
        session(TUESDAY, positional_stops="overnight")


# Expiry ---------------------------------------------------------------------------------


def future(symbol: str, exit_day: dt.date | None) -> dict[str, Any]:
    """An NFO future of lot 75 whose last trading day is 2025-07-31."""
    return {
        **SBIN,
        "symbol": symbol,
        "exchange": "NFO",
        "kind": "FUT",
        "lot_size": 75,
        "last_trading_day": to_day(EXPIRY),
        "positional_exit_day": None if exit_day is None else to_day(exit_day),
    }


def test_a_stock_future_is_exited_at_the_square_off_of_its_exit_day():
    stock = future("SBIN31JUL25FUT", EXPIRY)
    engine = bought(session(EXPIRY, stock), EXPIRY, 75.0, "NRML", {})
    for minute in range(1, SQUARE_OFF):
        release(engine, minute, EXPIRY)
    assert places(engine) == []
    # The 15:09 slice closes: the session reaches the 15:10 open.
    assert engine.advance()[0] == "waiting"
    (exit_order,) = places(engine)
    assert (exit_order["role"], exit_order["reason"]) == ("risk_exit", "expiry_exit")
    assert (exit_order["side"], exit_order["quantity_units"]) == ("SELL", 75.0)


def test_an_expired_index_future_is_recorded_exited_at_its_last_close_the_next_day():
    # Bought at 500.00; the expiry day's last bar closes at 512.00, apart from every other
    # price, so the exit's 75 x 12.00 = 900.00 shows that last close was used.
    index = future("NIFTY31JUL25FUT", None)
    engine = bought(session(EXPIRY, index), EXPIRY, 75.0, "NRML", STOP_3)
    (place_gtt,) = engine.take_commands()
    engine.apply_gtt_placed(place_gtt["gtt_id"])
    closed = to_close(engine, EXPIRY, 1, last_close=512.0)
    assert [command["command"] for command in closed] == ["cancel_gtt"]
    carried = json.loads(json.dumps(engine.carried_state()))
    assert carried["last_closes"] == [{"instrument": 0, "close_rupees": 512.0}]
    tomorrow = session(AFTER_EXPIRY, index)
    ids = tomorrow.restore_carried_state(carried)
    assert ids["expired"] == [0]
    position = tomorrow.position(0, "NRML")
    assert (position["quantity_units"], position["realized_rupees"]) == (0.0, 900.0)
    assert tomorrow.take_commands() == []


# The pre-open decision ---------------------------------------------------------------------


def daily_bar(day: dt.date, close: float) -> tuple:
    """The broker's daily candle of `day` as the binding takes it."""
    return (0, open_ns(day), close_ns(day), 500.0, 510.0, 490.0, close, 1_000.0)


def test_a_daily_strategy_decides_the_previous_sessions_bar_before_the_open():
    engine = session(WEDNESDAY, interval="D")
    before_open = open_ns(WEDNESDAY) - 30 * MINUTE_NS
    decided = engine.pre_open_decision([daily_bar(TUESDAY, 500.0)], None, before_open)
    assert (decided["bar_close"], decided["late"]) == (close_ns(TUESDAY), False)
    kind, decided_at, bars, _, _ = decided["pause"]
    assert (kind, decided_at, len(bars)) == ("bar_close", close_ns(TUESDAY), 1)
    assert engine.submit(order("BUY", 10.0, "CNC", {}))[:2] == ("submitted", ENTRY)
    assert engine.take_commands() == []
    assert engine.advance()[0] == "waiting"
    engine.push_tick(0, 501.0, at(0, WEDNESDAY, 1))
    (entry,) = places(engine)
    assert (entry["order_id"], entry["role"], entry["flags"]) == (ENTRY, "entry", [])


def test_a_bar_decided_already_is_not_decided_again():
    engine = session(WEDNESDAY, interval="D")
    before_open = open_ns(WEDNESDAY) - 30 * MINUTE_NS
    bars = [daily_bar(TUESDAY, 500.0)]
    assert engine.pre_open_decision(bars, close_ns(TUESDAY), before_open) is None


def test_a_decision_after_the_open_is_sent_at_once_flagged_late_decision():
    engine = session(WEDNESDAY, interval="D")
    after_open = open_ns(WEDNESDAY) + 5 * MINUTE_NS
    decided = engine.pre_open_decision([daily_bar(TUESDAY, 500.0)], None, after_open)
    assert decided["late"] is True
    engine.submit(order("BUY", 10.0, "CNC", {}))
    (entry,) = places(engine)
    assert entry["flags"] == ["late_decision"]
    assert engine.order_flags(ENTRY) == ["late_decision"]
