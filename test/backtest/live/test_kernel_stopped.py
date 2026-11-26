"""A session of an earlier day that stopped without its close, rebuilt through the binding
(`Engine.restore_stopped_state`; rust/oa_py/src/backtest/engine/live_state.rs; design
20-positional-live 4.2 step 2, unit N5c item N5c-1): its last `live_state()`, through JSON
as the journal keeps it, given to a session built for its day, then the venue's results
after the save, `stop_session()` and `carried_state()`.

No server is involved: the test plays the venue. Sessions run on 1-minute slices from 09:15
IST with no charges and no slippage, bars flat at 500.00. 10 SBIN are bought in CNC at the
09:16 open at 500.00 with a stop 3 points below (497.00), held by a SINGLE GTT whose child's
limit is 1 percent below the trigger on the tick below (492.00).
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
MINUTE_NS = 60 * 1_000_000_000
SECOND_NS = 1_000_000_000
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
ENTRY = 1
STOP_GTT = 1


def at(minute: int, second: int = 0) -> int:
    """`minute` minutes and `second` seconds after 09:15 IST on Tuesday, UTC nanoseconds."""
    opens = dt.datetime(TUESDAY.year, TUESDAY.month, TUESDAY.day, 3, 45, tzinfo=dt.UTC)
    return int(opens.timestamp()) * 10**9 + minute * MINUTE_NS + second * SECOND_NS


def session(capital: float = 10_000_000.0) -> Any:
    """A cost-free, slippage-free session of Tuesday on 1-minute slices."""
    config = {
        "first_day": to_day(TUESDAY),
        "last_day": to_day(TUESDAY),
        "base_nanos": MINUTE_NS,
        "strategy_interval": MINUTE_NS,
        "fill": "next_open",
        "quantity_mode": "venue",
        "capital": capital,
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
    engine_class = native_attr("Engine")
    return engine_class.session(default_calendar()._cal, config, [SBIN], [NO_BARS])


def release(engine: Any, minute: int) -> tuple:
    """Hand the session the flat bar of `minute`, release it and return its next pause."""
    assert engine.advance()[0] == "waiting"
    engine.add_bar(0, (at(minute), 500.0, 500.0, 500.0, 500.0, 1_000.0))
    engine.release(minute + 1)
    return engine.advance()


def entry_sent() -> Any:
    """Tuesday's session with the buy of 10 SBIN CNC of the 09:15 close sent, not filled."""
    engine = session()
    assert release(engine, 0)[0] == "bar_close"
    request = {
        "instrument": 0,
        "side": "BUY",
        "quantity": {"kind": "units", "value": 10.0},
        "order_type": "MARKET",
        "validity": "DAY",
        "product": "CNC",
        "reduce_only": False,
        "reverse": False,
        "rules": {"stop": ("points", 3.0, None)},
    }
    assert engine.submit(request)[:2] == ("submitted", ENTRY)
    assert engine.advance()[0] == "waiting"
    (place,) = engine.take_commands()
    assert (place["command"], place["order_id"]) == ("place", ENTRY)
    return engine


def through_json(state: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(state))


def stopped_and_carried(engine: Any) -> dict[str, Any]:
    engine.stop_session()
    return through_json(engine.carried_state())


def holding_with_stop_gtt() -> Any:
    """`entry_sent` with the entry filled at 500.00 and its stop GTT placed and confirmed."""
    engine = entry_sent()
    engine.apply_fill(ENTRY, 10.0, 500.0, at(1))
    (place,) = engine.take_commands()
    assert (place["command"], place["gtt_id"]) == ("place_gtt", STOP_GTT)
    engine.apply_gtt_placed(STOP_GTT)
    return engine


def test_a_saved_leg_with_its_stop_gtt_is_carried_as_the_session_stopped_after_its_save():
    engine = holding_with_stop_gtt()
    saved = through_json(engine.live_state())
    expected = stopped_and_carried(engine)
    rebuilt = session()

    assert rebuilt.restore_stopped_state(saved, to_day(WEDNESDAY), []) == {}

    carried = stopped_and_carried(rebuilt)
    assert carried == expected
    ((gtt,),) = (carried["gtts"],)
    assert (gtt["terms"]["trigger_rupees"], gtt["terms"]["limit_rupees"]) == (497.0, 492.0)


def test_a_fill_after_the_save_is_carried_and_the_stop_gtt_no_session_placed_is_not():
    saved = through_json(entry_sent().live_state())
    rebuilt = session()
    assert rebuilt.restore_stopped_state(saved, to_day(WEDNESDAY), []) == {ENTRY: ENTRY}

    rebuilt.apply_fill(ENTRY, 10.0, 500.0, at(1, 5))

    carried = stopped_and_carried(rebuilt)
    ((position,),) = (carried["positions"],)
    assert (position["quantity_units"], position["average_price_rupees"]) == (10.0, 500.0)
    assert carried["gtts"] == []


def test_a_stop_gtt_whose_place_never_reached_the_broker_is_not_carried():
    engine = entry_sent()
    engine.apply_fill(ENTRY, 10.0, 500.0, at(1))
    engine.take_commands()
    saved = through_json(engine.live_state())
    rebuilt = session()

    rebuilt.restore_stopped_state(saved, to_day(WEDNESDAY), [STOP_GTT])

    assert stopped_and_carried(rebuilt)["gtts"] == []


def test_a_save_of_the_day_it_is_carried_to_is_refused_as_input():
    saved = through_json(holding_with_stop_gtt().live_state())

    with pytest.raises(native_attr("NativeEngineInputError"), match="another session"):
        session().restore_stopped_state(saved, to_day(TUESDAY), [])


def test_a_save_of_another_capital_is_refused_as_input():
    saved = through_json(holding_with_stop_gtt().live_state())

    with pytest.raises(native_attr("NativeEngineInputError"), match="capital"):
        session(capital=5_000_000.0).restore_stopped_state(saved, to_day(WEDNESDAY), [])


def test_an_unsent_gtt_that_waits_for_no_answer_is_refused():
    saved = through_json(holding_with_stop_gtt().live_state())

    with pytest.raises(native_attr("NativeEngineError")):
        session().restore_stopped_state(saved, to_day(WEDNESDAY), [STOP_GTT])
