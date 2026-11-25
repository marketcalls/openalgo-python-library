"""The flags a live session's venue reports with a fill, through the binding
(`Engine.apply_fill(order_id, units, price, at, flags=())`;
rust/oa_py/src/backtest/engine/venue.rs; design 20-positional-live 3.3 and 4.3, unit N4f
item N4f-1): "price_unverified" and "missed_day" on the fill and on its trade, the names
the binding exposes, an unknown name refused with nothing recorded, and the engine's own
flags refused from a venue.

No server is involved: the test plays the venue. The session is the NSE day 2025-07-01 on
1-minute slices from 09:15 IST, bars flat at 500.00; a CNC buy of 10 (order 1) is decided
at the close of minute 0 and filled at 500.00 at the open of minute 1 with a stop 3
points below, a GTT (g1) triggering at 497.00 whose child sells at 492.00. The venue
fires g1 at 09:16:30; its child (order 2) is recorded at its 492.00 limit.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pytest

from openalgo.backtest._calendar_convert import native_attr, to_day
from openalgo.backtest.calendar import default_calendar
from openalgo.backtest.errors import VenueRefusalError

DAY = dt.date(2025, 7, 1)
MINUTE_NS = 60 * 1_000_000_000
SECOND_NS = 1_000_000_000
NO_BARS = (np.array([], dtype=np.int64), *(np.array([], dtype=np.float64) for _ in range(5)))
ENTRY, CHILD, G1 = 1, 2, 1
CHILD_LIMIT = 492.0
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
CONFIG = {
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


def at(minute: int, second: int = 0) -> int:
    """`second` seconds into minute `minute` of the session (09:15 IST is 03:45 UTC)."""
    open_ns = int(dt.datetime(2025, 7, 1, 3, 45, tzinfo=dt.UTC).timestamp()) * 10**9
    return open_ns + minute * MINUTE_NS + second * SECOND_NS


def child_fired() -> Any:
    """The session whose stop GTT g1 the venue fired at 09:16:30, its child unfilled."""
    engine = native_attr("Engine").session(default_calendar()._cal, CONFIG, [SBIN], [NO_BARS])
    assert engine.advance()[0] == "waiting"
    engine.add_bar(0, (at(0), 500.0, 500.0, 500.0, 500.0, 1_000.0))
    engine.release(1)
    assert engine.advance()[0] == "bar_close"
    buy = {
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
    assert engine.submit(buy)[:2] == ("submitted", ENTRY)
    assert engine.advance()[0] == "waiting"
    engine.take_commands()
    engine.apply_fill(ENTRY, 10.0, 500.0, at(1))
    engine.take_commands()
    engine.apply_gtt_placed(G1)
    assert engine.apply_gtt_triggered(G1, at(1, 30)) == CHILD
    return engine


def finished(engine: Any) -> dict[str, Any]:
    engine.stop_session()
    return engine.finish()


def test_a_missed_childs_fill_carries_its_flag_in_the_fills_and_the_trades_tables():
    # Recorded at the child's 492.00 limit: a loss of (492.00 - 500.00) x 10 = 80.00.
    engine = child_fired()
    engine.apply_fill(CHILD, 10.0, CHILD_LIMIT, at(1, 31), flags=("price_unverified",))
    output = finished(engine)
    fills, trades = output["fills"], output["trades"]
    assert fills["flags"] == [[], ["price_unverified"]]
    assert fills["price_rupees"][1] == CHILD_LIMIT
    assert trades["flags"] == [["price_unverified"]]
    assert trades["gross_pnl_rupees"] == [-80.0]


def test_a_stop_that_fired_on_a_missed_day_carries_both_names():
    engine = child_fired()
    flags = ["missed_day", "price_unverified"]
    engine.apply_fill(CHILD, 10.0, CHILD_LIMIT, at(1, 31), flags)
    output = finished(engine)
    assert output["fills"]["flags"][1] == ["price_unverified", "missed_day"]
    assert output["trades"]["flags"] == [["price_unverified", "missed_day"]]


def test_a_fill_without_flags_is_called_as_before_and_carries_none():
    engine = child_fired()
    engine.apply_fill(CHILD, 10.0, 497.0, at(1, 31))
    output = finished(engine)
    assert output["fills"]["flags"] == [[], []]
    assert output["trades"]["flags"] == [[]]


def test_an_unknown_flag_name_is_refused_and_nothing_is_recorded():
    engine = child_fired()
    error = native_attr("NativeEngineInputError")
    with pytest.raises(error, match='unknown fill flag "unverified"'):
        engine.apply_fill(CHILD, 10.0, CHILD_LIMIT, at(1, 31), flags=("unverified",))
    assert engine.order(CHILD)["filled_units"] == 0.0
    assert engine.position(0, "CNC")["quantity_units"] == 10.0


def test_one_of_the_engines_own_flags_is_refused_from_the_venue_by_kind():
    engine = child_fired()
    with pytest.raises(VenueRefusalError) as refused:
        engine.apply_fill(CHILD, 10.0, CHILD_LIMIT, at(1, 31), flags=("stop_crossed",))
    assert refused.value.kind == "invalid_fill"
    assert engine.order(CHILD)["filled_units"] == 0.0


def test_the_binding_names_every_fill_flag_and_the_two_a_venue_reports():
    assert native_attr("FILL_FLAGS") == (
        "filled_at_last_close",
        "gtt_child_unfilled",
        "external",
        "stop_crossed",
        "late_decision",
        "price_unverified",
        "missed_day",
    )
    assert native_attr("VENUE_FILL_FLAGS") == ("price_unverified", "missed_day")
